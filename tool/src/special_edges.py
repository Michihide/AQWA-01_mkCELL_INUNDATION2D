"""特殊辺ラインの読込・拘束化・メッシュ辺への対応付け。

計算範囲と特殊辺ラインを先に定め、ラインを Gmsh の拘束条件にしてから
メッシュを切る。生成後に辺をラインへ対応付け、special_edges.csv を書く。
幾何だけの拘束は `input.breaklines` のまま。
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .config import Config
from .io_vector import (
    ConstraintBreaklines,
    VectorInputError,
    _load_lines,
    load_domain,
)
from .mesh_bin_io import (
    KIND_BY_NAME,
    KIND_NAME,
    SpecialEdgeTable,
    parse_kind,
    parse_z_mode,
    resolve_kind_name,
)
from .mesh_tables import NormalizedMesh
from .utils import get_logger

_OVERLAP_LEN = 1e-3
_KIND_ALIASES = ("kind", "KIND", "type", "Type", "special_kind")
_ZC_ALIASES = ("zc", "Zc", "z_c", "ZC", "e_zc")
_H_ALIASES = ("H", "Hstr", "h", "e_H", "e_Hstr")
_ZROAD_ALIASES = ("z_road", "zroad", "Z_road", "e_zc_road")
_B_ALIASES = ("B", "Bstr", "e_Bstr", "width", "opening_width")
_COVER_ALIASES = ("cover", "T", "thickness", "overburden", "土被り")
_QGROUP_ALIASES = ("qgroup", "q_group", "Qgroup", "e_qgroup")
_ZMODE_ALIASES = ("z_mode", "zmode", "z_ref", "datum", "z_datum")
_FR_ALIASES = ("fr", "Fr", "FR", "froude", "Froude")

_KIND_PRIORITY = {
    "WALL": 6,
    "Q": 5,
    "W": 5,
    "FROUDE": 4,
    "CULVERT": 3,
    "ROAD": 2,
    "NONE": 0,
}


@dataclass
class SpecialEdgeFeature:
    """計算範囲内の特殊辺ライン 1 本。"""

    geom: LineString
    kind: str
    zc: float | None
    H: float
    z_road: float | None
    qgroup: int
    B: float | None = None
    z_mode: str = "absolute"
    cover: float | None = None
    fr: float | None = None


def split_line_at_vertices(line: LineString) -> list[LineString]:
    """折点で 2 点の線分に分ける。各区間が独立した特殊辺になる。"""
    coords = list(line.coords)
    if len(coords) < 2:
        return []
    segs: list[LineString] = []
    for a, b in zip(coords[:-1], coords[1:]):
        if a[0] == b[0] and a[1] == b[1]:
            continue
        segs.append(LineString([a, b]))
    return segs


def _as_linestrings(geom: BaseGeometry | None) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return split_line_at_vertices(geom)
    if isinstance(geom, MultiLineString):
        out: list[LineString] = []
        for g in geom.geoms:
            if isinstance(g, LineString):
                out.extend(split_line_at_vertices(g))
        return out
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_as_linestrings(g))
        return out
    return []


def _find_column(columns: list[str], preferred: str, aliases: tuple[str, ...]) -> str | None:
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    for name in (preferred, *aliases):
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _optional_int(value: object, default: int = 0) -> int:
    parsed = _optional_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _kind_from_row(value: object, *, source: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        name = resolve_kind_name(text)
    except ValueError:
        get_logger().warning("特殊辺 %s: 不明な kind %r をスキップします", source, value)
        return None
    if name == "NONE":
        return None
    if str(value).strip().upper() != name:
        get_logger().info("特殊辺 %s: kind %r を %s として読みました", source, value, name)
    return name


def _features_from_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    fixed_kind: str | None,
    cfg: Config,
    source: str,
) -> list[SpecialEdgeFeature]:
    se = cfg.input.special_edges
    columns = [str(c) for c in gdf.columns if c != "geometry"]
    col_kind = _find_column(columns, se.field_kind, _KIND_ALIASES)
    col_zc = _find_column(columns, se.field_zc, _ZC_ALIASES)
    col_h = _find_column(columns, se.field_H, _H_ALIASES)
    col_zr = _find_column(columns, se.field_z_road, _ZROAD_ALIASES)
    col_b = _find_column(columns, se.field_B, _B_ALIASES)
    col_cover = _find_column(columns, se.field_cover, _COVER_ALIASES)
    col_qg = _find_column(columns, se.field_qgroup, _QGROUP_ALIASES)
    col_zm = _find_column(columns, se.field_z_mode, _ZMODE_ALIASES)
    col_fr = _find_column(columns, se.field_fr, _FR_ALIASES)
    default_mode = parse_z_mode(se.z_mode)

    if fixed_kind is None and col_kind is None:
        fixed_kind = resolve_kind_name(se.default_kind)
        get_logger().info(
            "特殊辺 %s: kind 列が無いため default_kind=%s を使います",
            source, fixed_kind,
        )

    features: list[SpecialEdgeFeature] = []
    for i, row in gdf.iterrows():
        kind = fixed_kind
        if kind is None:
            kind = _kind_from_row(row[col_kind], source=f"{source}[{i}]")
            if kind is None:
                continue
        h_val = _optional_float(row[col_h]) if col_h else None
        try:
            z_mode = parse_z_mode(row[col_zm], default=default_mode) if col_zm else default_mode
        except ValueError:
            get_logger().warning(
                "特殊辺 %s[%s]: 不明な z_mode %r を %s として扱います",
                source, i, row[col_zm], default_mode,
            )
            z_mode = default_mode
        for line in _as_linestrings(row.geometry):
            features.append(
                SpecialEdgeFeature(
                    geom=line,
                    kind=kind,
                    zc=_optional_float(row[col_zc]) if col_zc else None,
                    H=h_val if h_val is not None else 0.0,
                    z_road=_optional_float(row[col_zr]) if col_zr else None,
                    qgroup=_optional_int(row[col_qg]) if col_qg else 0,
                    B=_optional_float(row[col_b]) if col_b else None,
                    z_mode=z_mode,
                    cover=_optional_float(row[col_cover]) if col_cover else None,
                    fr=_optional_float(row[col_fr]) if col_fr else None,
                )
            )
    return features


def load_special_edge_features(
    cfg: Config,
    target: CRS,
    clip_to: BaseGeometry | None = None,
) -> list[SpecialEdgeFeature]:
    """特殊辺ラインを target CRS へ直し、計算範囲で切って返す。"""
    logger = get_logger()
    rel = cfg.input.special_edges.file
    if not rel:
        logger.info("特殊辺ラインは未指定。special_edges.csv は空のままです")
        return []
    path = cfg.resolve(rel)
    if path is None or not path.exists():
        raise VectorInputError(f"特殊辺ラインが見つかりません: {rel}")
    gdf = _load_lines(path, target, clip_to)
    if gdf.empty:
        logger.info("特殊辺: 計算範囲内に線がありません")
        return []
    features = _features_from_gdf(gdf, fixed_kind=None, cfg=cfg, source=path.name)
    n_src = int((gdf.geometry.notna() & ~gdf.geometry.is_empty).sum())
    logger.info(
        "特殊辺 %s: 入力 %d 本を折点で %d 区間に分割（総延長 %.1f km）",
        path.name, n_src, len(features),
        sum(f.geom.length for f in features) / 1000.0,
    )
    return features


def _pair_relation(a: LineString, b: LineString) -> tuple[str, float]:
    """2 本の関係。交差は距離 0 で許す。重なりと隙間 < clearance が問題。"""
    inter = a.intersection(b)
    if inter is not None and not inter.is_empty:
        length = 0.0
        if inter.geom_type == "LineString":
            length = float(inter.length)
        elif hasattr(inter, "geoms"):
            length = sum(float(g.length) for g in inter.geoms if hasattr(g, "length"))
        if length > _OVERLAP_LEN:
            return "overlap", 0.0
        return "cross", 0.0
    return "gap", float(a.distance(b))


def special_edge_proximity_report(
    features: list[SpecialEdgeFeature],
    clearance: float,
) -> dict:
    """特殊辺どうしの距離の傾向。clearance 未満の隙間と重なりを衝突とする。"""
    geoms = [f.geom for f in features]
    n = len(geoms)
    bins = {
        "cross": 0,
        "overlap": 0,
        "gap_lt_10": 0,
        "gap_10_clearance": 0,
        "gap_clearance_50": 0,
        "gap_ge_50": 0,
    }
    conflicts: list[tuple[int, int, str, float]] = []
    shortest = float("inf")
    if n < 2:
        return {"n": n, "clearance": clearance, "bins": bins, "conflicts": conflicts, "shortest": None}

    for i, geom in enumerate(geoms):
        for j in range(i + 1, n):
            kind, dist = _pair_relation(geom, geoms[j])
            if kind == "cross":
                bins["cross"] += 1
                continue
            if kind == "overlap":
                bins["overlap"] += 1
                conflicts.append((i, j, kind, dist))
                continue
            shortest = min(shortest, dist)
            if dist < 10.0:
                bins["gap_lt_10"] += 1
            elif dist < clearance:
                bins["gap_10_clearance"] += 1
            elif dist < 50.0:
                bins["gap_clearance_50"] += 1
            else:
                bins["gap_ge_50"] += 1
            if dist < clearance:
                conflicts.append((i, j, kind, dist))

    return {
        "n": n,
        "clearance": clearance,
        "bins": bins,
        "conflicts": conflicts,
        "shortest": None if shortest == float("inf") else shortest,
    }


def log_special_edge_proximity(report: dict) -> None:
    """距離の傾向をログに出す。"""
    logger = get_logger()
    c = report["clearance"]
    b = report["bins"]
    logger.info(
        "特殊辺の近接傾向（%d 本、下限 %.0f m）: 交差 %d / 重なり %d / "
        "隙間 <10 m %d / 10–%.0f m %d / %.0f–50 m %d / ≥50 m %d",
        report["n"], c, b["cross"], b["overlap"],
        b["gap_lt_10"], c, b["gap_10_clearance"],
        c, b["gap_clearance_50"], b["gap_ge_50"],
    )
    if report["shortest"] is not None:
        logger.info("特殊辺の最短隙間: %.2f m", report["shortest"])
    for i, j, kind, dist in report["conflicts"][:12]:
        if kind == "overlap":
            logger.warning("  衝突: 線 %d と %d が重なっています", i, j)
        else:
            logger.warning("  衝突: 線 %d と %d の隙間 %.2f m（下限 %.0f m）", i, j, dist, c)
    extra = len(report["conflicts"]) - 12
    if extra > 0:
        logger.warning("  …ほか %d 組", extra)


def assert_special_edge_clearance(
    features: list[SpecialEdgeFeature],
    clearance: float,
) -> None:
    """線同士が下限より近い／重なるときは傾向を出して止める。"""
    if len(features) < 2:
        return
    report = special_edge_proximity_report(features, clearance)
    log_special_edge_proximity(report)
    n_bad = len(report["conflicts"])
    if n_bad == 0:
        return
    raise VectorInputError(
        f"特殊辺どうしが近すぎます（衝突 {n_bad} 組、下限 {clearance:.0f} m）。"
        "間隔を空けたラインを入れてから再実行してください"
    )


def pick_clear_indices(geoms: list[LineString], clearance: float) -> list[int]:
    """長い線から順に、下限を割らない添字を取る。交差は残す。"""
    ranked = sorted(range(len(geoms)), key=lambda i: geoms[i].length, reverse=True)
    kept: list[int] = []
    for i in ranked:
        ok = True
        for j in kept:
            kind, dist = _pair_relation(geoms[i], geoms[j])
            if kind == "overlap" or (kind == "gap" and dist < clearance):
                ok = False
                break
        if ok:
            kept.append(i)
    return kept


def pick_clear_special_features(
    features: list[SpecialEdgeFeature],
    clearance: float,
) -> list[SpecialEdgeFeature]:
    """長い線から順に、下限を割らない部分集合を取る。"""
    keep = pick_clear_indices([f.geom for f in features], clearance)
    return [features[i] for i in keep]


def special_features_to_breaklines(
    features: list[SpecialEdgeFeature],
    crs: CRS,
) -> ConstraintBreaklines:
    """特殊辺ラインを拘束ブレークラインと同じ入力形にする。"""
    if not features:
        return ConstraintBreaklines()
    gdf = gpd.GeoDataFrame(
        {
            "name": [f"special_{f.kind}_{i}" for i, f in enumerate(features)],
            "kind": [f.kind for f in features],
        },
        geometry=[f.geom for f in features],
        crs=crs,
    )
    return ConstraintBreaklines(layers={"special_edges": gdf})


def match_special_edges(
    tables: NormalizedMesh,
    features: list[SpecialEdgeFeature],
    match_tol: float,
) -> SpecialEdgeTable:
    """メッシュ辺を特殊辺ラインへ対応付ける。両端と中点が線上にある辺だけを採用。"""
    if not features or len(tables.edges) == 0:
        return SpecialEdgeTable()

    geoms = [f.geom for f in features]
    tree = STRtree(geoms)
    xy = np.column_stack((tables.nodes.x, tables.nodes.y))
    v1 = tables.edges.v1.astype(np.int64) - 1
    v2 = tables.edges.v2.astype(np.int64) - 1

    chosen: dict[int, tuple[int, SpecialEdgeFeature, float]] = {}
    for i in range(len(tables.edges)):
        p1 = xy[int(v1[i])]
        p2 = xy[int(v2[i])]
        mid = 0.5 * (p1 + p2)
        hits = np.asarray(tree.query(Point(mid).buffer(match_tol)), dtype=np.int64)
        best: SpecialEdgeFeature | None = None
        best_d = match_tol
        best_pri = -1
        for hi in hits:
            line = geoms[int(hi)]
            feat = features[int(hi)]
            d = float(line.distance(Point(mid)))
            d1 = float(line.distance(Point(p1)))
            d2 = float(line.distance(Point(p2)))
            if d > match_tol or d1 > match_tol or d2 > match_tol:
                continue
            pri = _KIND_PRIORITY.get(feat.kind, 0)
            if pri > best_pri or (pri == best_pri and d < best_d):
                best = feat
                best_d = d
                best_pri = pri
        if best is None:
            continue
        eid = int(tables.edges.id[i])
        prev = chosen.get(eid)
        if prev is None or best_pri > _KIND_PRIORITY.get(prev[1].kind, 0) or (
            best_pri == _KIND_PRIORITY.get(prev[1].kind, 0) and best_d < prev[2]
        ):
            chosen[eid] = (i, best, best_d)

    if not chosen:
        return SpecialEdgeTable()

    items = sorted(chosen.items(), key=lambda item: item[0])
    n = len(items)
    edge_id = np.empty(n, dtype=np.int32)
    kind = np.empty(n, dtype=np.int32)
    zc = np.empty(n, dtype=np.float64)
    H = np.empty(n, dtype=np.float64)
    z_road = np.empty(n, dtype=np.float64)
    qgroup = np.empty(n, dtype=np.int32)
    B = np.empty(n, dtype=np.float64)
    fr = np.zeros(n, dtype=np.float64)
    for j, (eid, (i, feat, _d)) in enumerate(items):
        crest = float(tables.edges.z_crest[i])
        h_val = float(feat.H) if feat.H is not None else 0.0
        if feat.z_mode == "relative":
            dz = feat.zc if feat.zc is not None else 0.0
            dzr = feat.z_road if feat.z_road is not None else dz
            zc_val = crest + dz
            zr_val = crest + dzr
        else:
            zc_val = feat.zc if feat.zc is not None else crest
            zr_val = feat.z_road if feat.z_road is not None else zc_val
        if feat.kind == "CULVERT" and feat.cover is not None:
            zr_val = zc_val + h_val + float(feat.cover)
        edge_id[j] = eid
        kind[j] = KIND_BY_NAME[feat.kind]
        zc[j] = zc_val
        H[j] = h_val
        z_road[j] = zr_val
        qgroup[j] = int(feat.qgroup)
        B[j] = float(feat.B) if feat.B is not None else 0.0
        fr[j] = float(feat.fr) if feat.fr is not None else 0.0
    return SpecialEdgeTable(edge_id, kind, zc, H, z_road, qgroup, B, fr)


def special_edges_for_mesh(
    cfg: Config,
    tables: NormalizedMesh,
    clip_to: BaseGeometry | None = None,
) -> SpecialEdgeTable:
    """設定から特殊辺を読み、正規化テーブルの辺へ対応付ける。"""
    target = CRS.from_epsg(cfg.crs.target_epsg)
    domain = clip_to
    if domain is None and cfg.input.special_edges.file:
        try:
            polys, _crs = load_domain(cfg)
            domain = MultiPolygon(polys) if len(polys) > 1 else polys[0]
        except Exception:
            domain = None
    features = load_special_edge_features(cfg, target, domain)
    tol = cfg.input.special_edges.resolved_match_tol(cfg.mesh.global_min_size)
    table = match_special_edges(tables, features, tol)
    if features:
        counts: dict[str, int] = {}
        for k in table.kind:
            name = KIND_NAME.get(int(k), str(int(k)))
            counts[name] = counts.get(name, 0) + 1
        get_logger().info(
            "特殊辺を %d 本のメッシュ辺へ対応付けました（許容 %.2f m）: %s",
            len(table), tol, counts if counts else "なし",
        )
    return table
