"""拘束ブレークラインの前処理と検証。

拘束ブレークラインはメッシュ辺に一致させる線である。gmsh へそのまま渡すと
次の 3 つで破綻するため、ここで整えてから渡す。

1. 頂点が密すぎる。測量線は 1 m 間隔のこともあり、頂点間隔がそのまま要素辺長に
   なって面積下限 625 m^2 を割る。間引く。
2. 線どうしの交点に節点が無い。gmsh は交差する拘束線を扱えないので、
   あらかじめ交点で分割（noding）する。
3. 線が外周四角形帯にかかる。帯は Transfinite で 1 区間 1 四角形に固定して
   いるため、線が入り込むと帯が壊れる。Γ1 の内側へ余白を取って切り取る。

参照レイヤ（OSM 等）はここへ渡さない。型で分離してある。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

from .boundary_quad_band import PolygonBand
from .config import Config
from .io_vector import ConstraintBreaklines
from .utils import get_logger

SPECIAL_LAYER = "special_edges"


@dataclass
class PreparedLine:
    """gmsh へ渡す 1 本の拘束線。"""

    coords: np.ndarray  # (n, 2)、n >= 2
    layer: str
    band_index: int
    # True なら近接・短尺を理由に落とさない（特殊辺）。
    strict: bool = False

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.coords, axis=0), axis=1).sum())


@dataclass
class PreparedBreaklines:
    lines: list[PreparedLine] = field(default_factory=list)
    input_length: float = 0.0
    clipped_length: float = 0.0
    crowded_length: float = 0.0
    close_vertex_pairs: int = 0

    def is_empty(self) -> bool:
        return not self.lines

    def by_band(self, band_index: int) -> list[PreparedLine]:
        return [ln for ln in self.lines if ln.band_index == band_index]

    @property
    def output_length(self) -> float:
        return sum(ln.length for ln in self.lines)

    def stats(self) -> dict[str, float]:
        return {
            "breakline_count": len(self.lines),
            "breakline_input_km": self.input_length / 1000.0,
            "breakline_clipped_km": self.clipped_length / 1000.0,
            "breakline_crowded_dropped_km": self.crowded_length / 1000.0,
            "breakline_constrained_km": self.output_length / 1000.0,
            "breakline_close_vertex_pairs": self.close_vertex_pairs,
        }


def _as_linestrings(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if len(geom.coords) >= 2 else []
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if len(g.coords) >= 2]
    if hasattr(geom, "geoms"):  # GeometryCollection
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_as_linestrings(g))
        return out
    return []


def _resample(coords: np.ndarray, min_spacing: float) -> np.ndarray:
    """頂点間隔が min_spacing 以上になるよう間引く。両端は必ず残す。

    両端はネットワークの分岐点でもあるため落とせない。終点が直前の採用点に
    近すぎるときは、終点ではなくその採用点を落とす。
    """
    if len(coords) <= 2:
        return coords

    keep = [0]
    for i in range(1, len(coords) - 1):
        if float(np.linalg.norm(coords[i] - coords[keep[-1]])) >= min_spacing:
            keep.append(i)
    keep.append(len(coords) - 1)

    while len(keep) > 2 and float(np.linalg.norm(coords[keep[-1]] - coords[keep[-2]])) < min_spacing:
        keep.pop(-2)
    return coords[keep]


def _dedupe(coords: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = [0]
    for i in range(1, len(coords)):
        if float(np.linalg.norm(coords[i] - coords[keep[-1]])) > tol:
            keep.append(i)
    return coords[keep]


def _crowded_runs(
    line: LineString, crowded: np.ndarray, stations: np.ndarray, min_run: float
) -> list[tuple[float, float]]:
    """近接が続く区間を [開始距離, 終了距離] で返す。

    交点では線どうしが必ず接するが、それは近接ではない。短い区間は交点まわりの
    ものとみなして残し、min_run 以上続くものだけを対象にする。
    """
    idx = np.flatnonzero(crowded)
    if len(idx) == 0:
        return []
    runs = []
    for group in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        start, end = stations[group[0]], stations[group[-1]]
        if end - start >= min_run:
            runs.append((start, end))
    return runs


def _keep_intervals(
    length: float, drop: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """全長から drop 区間を除いた残りの区間。"""
    keep = []
    cursor = 0.0
    for start, end in drop:
        if start - cursor > 0.0:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if length - cursor > 0.0:
        keep.append((cursor, length))
    return keep


def _shared_endpoint(line: LineString, other: LineString) -> float | None:
    """line と other が端点を共有しているなら、その line 上での位置を返す。

    noding 済みなので、線が繋がっているなら端点が一致している。繋がっている
    相手にすぐ近くで接するのは当たり前であって、近接ではない。
    """
    ends = [np.asarray(other.coords[0]), np.asarray(other.coords[-1])]
    for at, xy in ((0.0, np.asarray(line.coords[0])),
                   (line.length, np.asarray(line.coords[-1]))):
        if min(float(np.linalg.norm(xy - e)) for e in ends) < 1e-6:
            return at
    return None


def drop_crowded_segments(
    lines: list[PreparedLine],
    bands: list[PolygonBand],
    min_clearance: float,
    min_run: float = 20.0,
    station_step: float = 10.0,
) -> tuple[list[PreparedLine], float]:
    """近接しすぎる拘束区間を落とす。

    拘束線どうし、あるいは拘束線と外周四角形帯の内側境界 Γ1 が min_clearance より
    近づくと、その間の要素は面積下限を割る。面積下限を優先する方針なので、
    そういう区間では拘束を諦める。落とした延長を併せて返す。

    繋がっている線どうしは接続点で必ず接するので、そこは対象から外す。ただし
    外すのは接続点から min_clearance までに限る。浅い角度で交わる 2 本はその先も
    近いままで、そこには実際に薄い要素ができるため落とす必要がある。
    """
    if not lines:
        return lines, 0.0

    geoms = [LineString(ln.coords) for ln in lines]
    obstacles = list(geoms)
    # Γ1 も障害物に含める。帯に沿って走る線は帯との間に薄い要素を作るため。
    owner_of_obstacle = list(range(len(geoms)))
    for band_index, band in enumerate(bands):
        if band.interior is None:
            continue
        obstacles.append(band.interior.boundary)
        owner_of_obstacle.append(-1 - band_index)

    tree = STRtree(obstacles)

    kept: list[PreparedLine] = []
    dropped = 0.0
    for i, (line, prepared) in enumerate(zip(geoms, lines)):
        if prepared.strict:
            kept.append(prepared)
            continue
        stations = np.arange(0.0, line.length + station_step, station_step)
        pts = shapely.points(
            np.array([line.interpolate(float(s)).coords[0] for s in stations])
        )
        crowded = np.zeros(len(stations), dtype=bool)
        for station_idx, obstacle_idx in zip(*tree.query(
            pts, predicate="dwithin", distance=min_clearance
        )):
            owner = owner_of_obstacle[int(obstacle_idx)]
            if owner == i:
                continue
            if owner < 0:
                if -1 - owner != prepared.band_index:
                    continue
            else:
                at = _shared_endpoint(line, obstacles[int(obstacle_idx)])
                if at is not None and abs(stations[station_idx] - at) < min_clearance:
                    continue
            crowded[station_idx] = True

        runs = _crowded_runs(line, crowded, stations, min_run)
        # 接続点のすぐ先から近接が始まるなら、見逃した接続点まわりも巻き込んで
        # 落とす。浅い角度で分かれた又の部分がそれで、まさに薄い要素ができる。
        reach = min_clearance + station_step
        runs = [
            (0.0 if s <= reach else s,
             line.length if e >= line.length - reach else e)
            for s, e in runs
        ]
        if not runs:
            kept.append(prepared)
            continue

        for start, end in _keep_intervals(line.length, runs):
            piece = shapely.ops.substring(line, start, end)
            for ls in _as_linestrings(piece):
                kept.append(
                    PreparedLine(
                        coords=np.asarray(ls.coords, dtype=float),
                        layer=prepared.layer,
                        band_index=prepared.band_index,
                        strict=prepared.strict,
                    )
                )
        dropped += line.length - sum(e - s for s, e in _keep_intervals(line.length, runs))

    return kept, dropped


def trim_close_endpoints(
    lines: list[PreparedLine], min_clearance: float, step: float = 5.0
) -> tuple[list[PreparedLine], float]:
    """他の線の近くで途切れている端点を、間隔が空くまで後退させる。

    拘束線が別の線のすぐ手前で終わると、その端点と相手の線の間に薄い三角形が
    必ず生まれる。交点なら noding 済みで端点を共有しているはずなので、共有して
    いない近接は「繋がっていないのに近い」状態であり、拘束を後退させてよい。
    """
    if len(lines) < 2:
        return lines, 0.0

    geoms = [LineString(ln.coords) for ln in lines]
    tree = STRtree(geoms)

    def blocked(pt, i: int, anchor) -> bool:
        for j in tree.query(pt.buffer(min_clearance)):
            if int(j) == i:
                continue
            other = geoms[int(j)]
            ends = shapely.points([other.coords[0], other.coords[-1]])
            if shapely.distance(ends, anchor).min() < 1e-6:
                continue  # 端点を共有している＝正しく繋がっている
            if shapely.distance(pt, other) < min_clearance:
                return True
        return False

    out: list[PreparedLine] = []
    trimmed = 0.0
    for i, (line, prepared) in enumerate(zip(geoms, lines)):
        if prepared.strict:
            out.append(prepared)
            continue
        cuts = []
        for at_start in (True, False):
            anchor = shapely.Point(line.coords[0 if at_start else -1])
            cut = 0.0
            while cut < line.length:
                d = cut if at_start else line.length - cut
                if not blocked(line.interpolate(d), i, anchor):
                    break
                cut += step
            cuts.append(cut)

        if cuts[0] + cuts[1] <= 0.0:
            out.append(prepared)
            continue

        trimmed += min(cuts[0] + cuts[1], line.length)
        piece = shapely.ops.substring(line, cuts[0], line.length - cuts[1])
        for ls in _as_linestrings(piece):
            out.append(
                PreparedLine(
                    coords=np.asarray(ls.coords, dtype=float),
                    layer=prepared.layer,
                    band_index=prepared.band_index,
                    strict=prepared.strict,
                )
            )

    return out, trimmed


def _tidy(
    lines: list[PreparedLine], min_spacing: float, min_length: float
) -> list[PreparedLine]:
    """切り落としで生じた短い線分を均し、短くなりすぎた線を捨てる。

    近接区間の除去も端点の後退も、線を任意の位置で切る。切り口と元の頂点の間に
    min_spacing より短い線分が残ると、gmsh はそこに短い辺を作り、面積下限を割る
    三角形ができる。切った後にもう一度間引く必要がある。
    """
    out: list[PreparedLine] = []
    for ln in lines:
        coords = _resample(_dedupe(ln.coords), min_spacing)
        if len(coords) < 2:
            continue
        line = PreparedLine(
            coords=coords, layer=ln.layer, band_index=ln.band_index, strict=ln.strict,
        )
        if ln.strict or line.length >= min_length:
            out.append(line)
    return out


def _count_close_vertex_pairs(lines: list[PreparedLine], min_spacing: float) -> int:
    """別の線に属する頂点どうしが近すぎる箇所の数。

    近接した拘束線は、その間に面積下限を割る要素を強制的に作ってしまう。
    ここでは自動修正せず、件数を診断として返す。
    """
    if len(lines) < 2:
        return 0
    pts = np.vstack([ln.coords for ln in lines])
    owner = np.concatenate([np.full(len(ln.coords), i) for i, ln in enumerate(lines)])

    tree = STRtree(shapely.points(pts))
    pairs = tree.query(shapely.points(pts), predicate="dwithin", distance=min_spacing)
    a, b = pairs
    mask = (a < b) & (owner[a] != owner[b])
    return int(mask.sum())


def prepare_breaklines(
    breaklines: ConstraintBreaklines, bands: list[PolygonBand], cfg: Config
) -> PreparedBreaklines:
    """拘束ブレークラインを gmsh へ渡せる形に整える。"""
    logger = get_logger()
    out = PreparedBreaklines()
    if breaklines.is_empty():
        return out
    if not cfg.features.breaklines_as_mesh_edges:
        special_gdf = breaklines.layers.get(SPECIAL_LAYER)
        if special_gdf is None or special_gdf.empty:
            return out
        breaklines = ConstraintBreaklines(layers={SPECIAL_LAYER: special_gdf})

    bp = cfg.mesh.breakline_processing
    min_spacing = bp.resolved_min_vertex_spacing(cfg.mesh.global_min_size)
    min_length = bp.resolved_min_length(cfg.mesh.global_min_size)
    band_width = (
        cfg.mesh.boundary_quad_band.width
        if cfg.mesh.boundary_quad_band.enabled
        else 0.0
    )
    margin = bp.resolved_interior_margin(cfg.mesh.global_min_size, band_width)

    # レイヤ名を後で引けるよう、元の線を索引しておく
    originals: list[LineString] = []
    origin_layer: list[str] = []
    for name, gdf in breaklines.layers.items():
        for geom in gdf.geometry:
            for ls in _as_linestrings(geom):
                originals.append(ls)
                origin_layer.append(name)
    if not originals:
        return out

    out.input_length = float(sum(ls.length for ls in originals))
    tree = STRtree(originals)

    # 交点で分割してから、分岐点までを 1 本に繋ぎ直す
    noded = unary_union(originals)
    # linemerge は単一の LineString を受け付けないため、その場合はそのまま使う
    merged = _as_linestrings(linemerge(noded) if hasattr(noded, "geoms") else noded)

    for band_index, band in enumerate(bands):
        if band.interior is None:
            continue
        region = band.interior.buffer(-margin)
        if region.is_empty:
            continue

        for line in merged:
            mid = line.interpolate(0.5, normalized=True)
            layer = origin_layer[int(tree.nearest(mid))]
            strict = layer == SPECIAL_LAYER
            clip_region = band.interior if strict else region
            if clip_region is None or clip_region.is_empty:
                continue
            clipped = shapely.intersection(line, clip_region)
            for piece in _as_linestrings(clipped):
                out.clipped_length += piece.length
                coords = _dedupe(np.asarray(piece.coords, dtype=float))
                if len(coords) < 2:
                    continue

                simplified = LineString(coords).simplify(
                    bp.simplify_tolerance, preserve_topology=False
                )
                coords = _resample(
                    _dedupe(np.asarray(simplified.coords, dtype=float)), min_spacing
                )
                if len(coords) < 2:
                    continue
                length = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
                if not strict and length < min_length:
                    continue

                out.lines.append(
                    PreparedLine(
                        coords=coords,
                        layer=layer,
                        band_index=band_index,
                        strict=strict,
                    )
                )

    # 面積下限を優先する。近接して薄い要素を強制する区間は拘束を諦める。
    clearance = bp.resolved_min_clearance(cfg.mesh.min_element_area, min_spacing)
    out.lines, out.crowded_length = drop_crowded_segments(out.lines, bands, clearance)

    trimmed_lines, trimmed = trim_close_endpoints(out.lines, clearance)
    out.crowded_length += trimmed
    out.lines = _tidy(trimmed_lines, min_spacing, min_length)

    out.close_vertex_pairs = _count_close_vertex_pairs(out.lines, min_spacing)

    special_km = sum(ln.length for ln in out.lines if ln.strict) / 1000.0
    logger.info(
        "拘束線: %d 本 / 総延長 %.2f km（入力 %.2f km、領域内 %.2f km、うち特殊辺 %.2f km）",
        len(out.lines), out.output_length / 1000.0,
        out.input_length / 1000.0, out.clipped_length / 1000.0, special_km,
    )
    if out.crowded_length > 0.0:
        logger.warning(
            "近接のためブレークライン %.2f km の拘束を見送りました（間隔 %.0f m 未満）。"
            "面積下限 %.0f m^2 を優先。特殊辺は見送りません",
            out.crowded_length / 1000.0, clearance, cfg.mesh.min_element_area,
        )
    if out.close_vertex_pairs:
        logger.warning(
            "拘束線どうしの頂点が %d 箇所で %.0f m 以内に接近しています。"
            "その間には面積下限を割る要素が生じ得ます",
            out.close_vertex_pairs, min_spacing,
        )
    return out


def breakline_coverage(mesh, prepared: PreparedBreaklines, tol: float = 0.01) -> dict[str, float]:
    """拘束線のうち、実際にメッシュ辺に載っている割合を測る。

    gmsh は拘束線をサイズ場に応じて細分するため、線分の対応は 1 対 1 にならない。
    近傍のメッシュ辺だけを集めて緩衝域を作り、線との交差長で被覆率を求める。
    """
    if prepared.is_empty():
        return {"breakline_coverage": 1.0, "breakline_uncovered_m": 0.0}

    edges = _mesh_edges(mesh)
    segments = shapely.linestrings(
        mesh.nodes[edges].reshape(-1, 2), indices=np.repeat(np.arange(len(edges)), 2)
    )
    tree = STRtree(segments)

    total = 0.0
    covered = 0.0
    per_layer: dict[str, list[float]] = {}
    for ln in prepared.lines:
        line = LineString(ln.coords)
        near = tree.query(line.buffer(tol * 10.0))
        if len(near) == 0:
            total += line.length
            per_layer.setdefault(ln.layer, [0.0, 0.0])
            per_layer[ln.layer][1] += line.length
            continue
        band = unary_union(segments[near]).buffer(tol)
        hit = shapely.intersection(line, band).length
        total += line.length
        covered += hit
        st = per_layer.setdefault(ln.layer, [0.0, 0.0])
        st[0] += hit
        st[1] += line.length

    out = {
        "breakline_coverage": covered / total if total else 1.0,
        "breakline_uncovered_m": total - covered,
    }
    for layer, (hit, tot) in per_layer.items():
        out[f"breakline_coverage_{layer}"] = hit / tot if tot else 1.0
    return out


def _mesh_edges(mesh) -> np.ndarray:
    """メッシュの無向辺を重複なく返す。"""
    parts = []
    for elems in (mesh.triangles, mesh.quads):
        if len(elems):
            parts.append(np.stack([elems, np.roll(elems, -1, axis=1)], axis=2).reshape(-1, 2))
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.sort(np.vstack(parts), axis=1), axis=0)


def breaklines_to_geodataframe(prepared: PreparedBreaklines, crs):
    """診断・確認用に、実際に拘束した線を書き出す。"""
    import geopandas as gpd

    if prepared.is_empty():
        return gpd.GeoDataFrame({"layer": [], "band": []}, geometry=[], crs=crs)
    return gpd.GeoDataFrame(
        {
            "layer": [ln.layer for ln in prepared.lines],
            "band": [ln.band_index for ln in prepared.lines],
            "n_vertices": [len(ln.coords) for ln in prepared.lines],
        },
        geometry=[LineString(ln.coords) for ln in prepared.lines],
        crs=crs,
    )


def unconstrained_polygon(band: PolygonBand, margin: float) -> Polygon:
    """拘束線を配置してよい領域。テストと診断で使う。"""
    return band.interior.buffer(-margin) if band.interior is not None else Polygon()
