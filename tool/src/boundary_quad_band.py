"""外周四角形帯の生成。

外周 Γ0 を等間隔にリサンプリングし、各頂点で内向きマイター法線に沿って
オフセットした Γ1 を作る。Γ0 と Γ1 の頂点は 1 対 1 に対応するので、区間ごとに
4 辺の Plane Surface を作って Transfinite + Recombine を掛ければ、外周に接する
要素をすべて四角形にできる。

内向きオフセットは、鋭角な凸コーナーや幅の狭いくびれで自己交差する。該当頂点の
オフセット幅を局所的に縮めて解消し、それでも解消しない場合は黙って三角形へ
落とさず診断情報として記録する（指示書 8.3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LinearRing, LineString, Polygon
from shapely.strtree import STRtree

from .geometry_cleaning import polygon_rings_resampled
from .utils import get_logger

MITER_LIMIT = 3.0

# close_ends() が「真の退化」（面積がほぼゼロ）と判定する閾値 [m^2]。
# 面積下限（min_element_area）による区間除外は行わない（境界形状の再現性を
# 優先するため）。ここでは gmsh が壊れる縮退三角形だけを避ける。
_DEGENERATE_AREA_EPS = 1e-6


@dataclass
class RingBand:
    """1 つのリング（外周または穴）に沿う四角形帯。

    skipped に入れた区間には四角形を置かない。そこでは内側の三角形領域が Γ0 まで
    降りてきて、境界に三角形が接する。境界形状を変えずに済ませるための逃げ道で、
    外周をすべて四角形で覆うことより境界形状を保つことを優先する。
    """

    outer: np.ndarray  # (N, 2) Γ0
    inner: np.ndarray  # (N, 2) Γ1
    widths: np.ndarray  # (N,) 各頂点で実際に用いたオフセット幅
    is_hole: bool = False
    failed_segments: list[int] = field(default_factory=list)
    skipped: set[int] = field(default_factory=set)
    # 帯の端で幅を 0 に落とした頂点。ここでは Γ1 が Γ0 に接する。
    collapsed: set[int] = field(default_factory=set)

    @property
    def n(self) -> int:
        return len(self.outer)

    def inner_pt(self, v: int) -> np.ndarray:
        """頂点 v の内側点。幅を 0 に落とした頂点では Γ0 上の点そのもの。"""
        return self.outer[v] if v in self.collapsed else self.inner[v]

    def close_ends(self, max_passes: int = 8) -> None:
        """帯の端で幅を 0 に落とし、最後の区間を三角形で閉じる。

        端をそのままにすると、Γ1 から Γ0 へ帯幅ぶん垂直に降りる切れ込みができる。
        端の頂点で幅を 0 にして Γ1 を Γ0 に着地させる。その区間の帯要素は
        四角形ではなく三角形になるが、内側の境界は段差なく斜めに降りる。

        面積下限（min_element_area）による区間除外は行わない。境界形状の
        再現性を優先するため、閉じの三角形がどれだけ小さくてもそのまま置く。
        ただし面積がほぼゼロの真の退化（gmsh が壊れる縮退三角形）だけは、
        その区間も帯から外して 1 つ手前で閉じ直す。
        """
        n = self.n
        for _ in range(max_passes):
            if not self.skipped or len(self.skipped) >= n:
                self.collapsed = set()
                return
            self._collapse_junctions()
            small = {
                i for i in self.closing_indices()
                if _shoelace(self.cell_coords(i)) < _DEGENERATE_AREA_EPS
            }
            if not small:
                return
            self.skipped |= small

    def _collapse_junctions(self) -> None:
        n = self.n
        while True:
            self.collapsed = {
                v for i in self.skipped for v in (i, (i + 1) % n)
                if ((v - 1) % n in self.skipped) != (v in self.skipped)
            }
            # 両端を落とされた区間は潰れてしまうので帯から外す
            dead = {
                i for i in self.cell_indices()
                if i in self.collapsed and (i + 1) % n in self.collapsed
            }
            if not dead or len(self.skipped | dead) >= n:
                return
            self.skipped |= dead

    def cell_coords(self, i: int) -> np.ndarray:
        """帯要素の頂点。幅を 0 に落とした側では点が重なるので三角形になる。"""
        j = (i + 1) % self.n
        pts = [self.outer[i], self.outer[j], self.inner_pt(j), self.inner_pt(i)]
        kept = [p for k, p in enumerate(pts)
                if not np.array_equal(p, pts[k - 1])]
        return np.array(kept)

    def cell_indices(self) -> list[int]:
        """帯要素（四角形または閉じの三角形）を置く区間。"""
        return [i for i in range(self.n) if i not in self.skipped]

    def quad_indices(self) -> list[int]:
        """四角形を置く区間。"""
        return [
            i for i in self.cell_indices()
            if i not in self.collapsed and (i + 1) % self.n not in self.collapsed
        ]

    def closing_indices(self) -> list[int]:
        """三角形で閉じる区間。片側の頂点だけ幅が 0 になっている。"""
        quads = set(self.quad_indices())
        return [i for i in self.cell_indices() if i not in quads]

    def quad_coords(self, i: int) -> np.ndarray:
        j = (i + 1) % self.n
        return np.array([self.outer[i], self.outer[j], self.inner[j], self.inner[i]])

    def quad_areas(self, live_only: bool = False) -> np.ndarray:
        idx = self.quad_indices() if live_only else range(self.n)
        return np.array([_shoelace(self.cell_coords(i)) for i in idx])

    def cell_areas(self) -> np.ndarray:
        """帯要素（四角形と閉じの三角形）の面積。"""
        return np.array([_shoelace(self.cell_coords(i)) for i in self.cell_indices()])

    def interior_walk(self) -> list[tuple[str, int]]:
        """内側の三角形領域の境界を辿る順路。

        ("inner", i) は Γ1 の区間 i、("outer", i) は Γ0 の区間 i、
        ("down", i) は頂点 i で Γ1 から Γ0 へ、("up", i) はその逆を意味する。
        スキップ区間では Γ0 まで降りて境界沿いに進み、また Γ1 へ戻る。
        """
        n = self.n
        if not self.skipped:
            return [("inner", i) for i in range(n)]
        if len(self.skipped) >= n:
            return [("outer", i) for i in range(n)]

        start = next(i for i in range(n) if i not in self.skipped)
        walk: list[tuple[str, int]] = []
        i, done = start, 0
        while done < n:
            if i not in self.skipped:
                walk.append(("inner", i))
                i = (i + 1) % n
                done += 1
                continue
            # 幅を 0 に落とした頂点では Γ1 がすでに Γ0 に接しているので降りない
            if i not in self.collapsed:
                walk.append(("down", i))
            while done < n and i in self.skipped:
                walk.append(("outer", i))
                i = (i + 1) % n
                done += 1
            if i not in self.collapsed:
                walk.append(("up", i))
        return walk

    def interior_ring(self) -> np.ndarray:
        """内側領域の境界の座標列。順路の各区間の始点を並べたもの。"""
        return np.array([
            self.inner_pt(i) if kind in ("inner", "down") else self.outer[i]
            for kind, i in self.interior_walk()
        ])


@dataclass
class PolygonBand:
    """1 ポリゴン分の四角形帯と、その内側に残る三角形領域。"""

    polygon: Polygon
    exterior: RingBand
    holes: list[RingBand] = field(default_factory=list)
    interior: Polygon | None = None  # 三角形で埋める領域
    ok: bool = True
    reason: str = ""
    corner_cuts: int = 0  # 境界形状を切り落とした頂点数（既定では 0）
    max_corner_deviation: float = 0.0

    def rings(self) -> list[RingBand]:
        return [self.exterior, *self.holes]

    def quad_count(self) -> int:
        return sum(len(r.quad_indices()) for r in self.rings())

    def skipped_count(self) -> int:
        return sum(len(r.skipped) for r in self.rings())


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _miter_directions(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各頂点のマイター方向（単位）と伸長係数を返す。

    リングは外周が反時計回り、穴が時計回りに揃えられている前提。どちらの場合も
    解析領域（材料側）は進行方向の左側にあるため、左法線が帯を張る向きになる。

    Args:
        pts: (N, 2) の閉リング頂点（閉じ点なし）

    Returns:
        (dirs, scales) — オフセット量 = width * scales[i] * dirs[i]
    """
    nxt = np.roll(pts, -1, axis=0)
    edges = nxt - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    lengths[lengths == 0.0] = 1.0
    u = edges / lengths[:, None]
    normals = np.column_stack([-u[:, 1], u[:, 0]])

    prev_normals = np.roll(normals, 1, axis=0)
    bisec = normals + prev_normals
    norm = np.hypot(bisec[:, 0], bisec[:, 1])

    # 180 度折り返し（スパイク）では法線が打ち消し合う。その頂点は辺法線を使う。
    degenerate = norm < 1e-9
    bisec[degenerate] = normals[degenerate]
    norm[degenerate] = 1.0
    dirs = bisec / norm[:, None]

    cos_half = np.einsum("ij,ij->i", dirs, normals)
    cos_half = np.clip(cos_half, 1e-3, 1.0)
    scales = np.minimum(1.0 / cos_half, MITER_LIMIT)
    return dirs, scales


def _self_intersecting_vertices(ring_pts: np.ndarray) -> set[int]:
    """リングの自己交差に関与する頂点集合を返す。"""
    n = len(ring_pts)
    segments = [
        LineString([ring_pts[i], ring_pts[(i + 1) % n]]) for i in range(n)
    ]
    tree = STRtree(segments)
    bad: set[int] = set()
    for i, seg in enumerate(segments):
        for j in tree.query(seg):
            j = int(j)
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue  # 自分自身と隣接辺は共有点を持って当然
            if seg.intersects(segments[j]):
                bad.update({i, (i + 1) % n, j, (j + 1) % n})
    return bad


def build_ring_band(
    pts: np.ndarray,
    width: float,
    is_hole: bool,
    min_width_ratio: float,
    max_passes: int,
) -> RingBand:
    """1 リング分の四角形帯を作る。自己交差する頂点は幅を縮めて解消を試みる。"""
    n = len(pts)
    dirs, scales = _miter_directions(pts)
    widths = np.full(n, float(width))
    min_width = width * min_width_ratio

    for _ in range(max_passes):
        inner = pts + (widths * scales)[:, None] * dirs
        bad = _self_intersecting_vertices(inner)
        bad |= _degenerate_quad_vertices(pts, inner)
        if not bad:
            return RingBand(outer=pts, inner=inner, widths=widths, is_hole=is_hole)
        idx = np.fromiter(bad, dtype=int)
        shrinkable = widths[idx] > min_width
        if not shrinkable.any():
            break
        widths[idx[shrinkable]] *= 0.7
        widths = np.maximum(widths, min_width)

    inner = pts + (widths * scales)[:, None] * dirs
    remaining = sorted(_self_intersecting_vertices(inner) | _degenerate_quad_vertices(pts, inner))
    return RingBand(
        outer=pts, inner=inner, widths=widths, is_hole=is_hole, failed_segments=remaining
    )


def _degenerate_quad_vertices(outer: np.ndarray, inner: np.ndarray) -> set[int]:
    """面積が非正、または自己交差する四角形に関与する頂点。"""
    n = len(outer)
    bad: set[int] = set()
    for i in range(n):
        j = (i + 1) % n
        quad = np.array([outer[i], outer[j], inner[j], inner[i]])
        if _shoelace(quad) <= 0.0:
            bad.update({i, j})
            continue
        if not Polygon(quad).is_simple:
            bad.update({i, j})
    return bad


def _turn_angles(pts: np.ndarray) -> np.ndarray:
    """各頂点での進行方向の変化量（絶対値・ラジアン）。"""
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    v1 = pts - prev
    v2 = nxt - pts
    a1 = np.arctan2(v1[:, 1], v1[:, 0])
    a2 = np.arctan2(v2[:, 1], v2[:, 0])
    d = np.mod(a2 - a1 + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.hypot(*(p - a)))
    t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
    return float(np.hypot(*(p - (a + t * ab))))


def _sharp_interface_corners(
    inner: np.ndarray, min_element_area: float, max_angle_deg: float
) -> np.ndarray:
    """内側領域を三角形で埋められない Γ1 の鋭角コーナーを検出する。

    内角 θ が小さい凸コーナーでは、そこに置ける最大の三角形が
    (Q[i-1], Q[i], Q[i+1]) に限られる。その面積が下限を割るなら、どう分割しても
    面積下限を満たせない。θ が大きい（ほぼ直線の）頂点は、頂点を遠くに取れるため
    対象外とする。

    Returns:
        違反している頂点のインデックスと、その corner 三角形面積
    """
    prev = np.roll(inner, 1, axis=0)
    nxt = np.roll(inner, -1, axis=0)
    v1 = prev - inner
    v2 = nxt - inner
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(axis=1)
    # 内側は Γ1 の進行方向左側。内角は atan2(-cross, dot) を 0..2pi で測る。
    theta = np.arctan2(-cross, dot)
    theta = np.where(theta < 0.0, theta + 2.0 * np.pi, theta)

    corner_area = 0.5 * np.abs(cross)
    convex = theta < np.radians(max_angle_deg)
    return np.flatnonzero(convex & (corner_area < min_element_area))


def _skip_bad_segments(band: RingBand, max_passes: int = 8) -> int:
    """孤立した帯四角形を、境界形状を変えずに帯から外す。

    区間を帯から外すのは、内向きオフセットの自己交差で有効な四角形を作れない
    場合（`failed_segments`、呼び出し元で `skipped` に反映済み）だけである。
    面積下限や鋭角コーナーを理由にした区間除外は行わない。境界（堤防など）の
    形状再現性を優先し、面積下限は境界に接する要素には課さないため。

    ここでは、それによって両隣を挟まれてしまい帯の切れ端にしかならない孤立
    した四角形を吸収するカスケード処理のみを行う（見た目の整理）。
    """
    n = band.n
    if n == 0:
        return 0

    before = len(band.skipped)
    for _ in range(max_passes):
        added: set[int] = set()
        for i in band.quad_indices():
            # 両隣を外された四角形は、帯の切れ端にしかならない
            if (i - 1) % n in band.skipped and (i + 1) % n in band.skipped:
                added.add(i)
        added -= band.skipped
        if not added:
            break
        band.skipped |= added

    band.failed_segments = [i for i in band.failed_segments if i not in band.skipped]
    band.close_ends()
    return len(band.skipped) - before


def _merge_undersized_quads(
    pts: np.ndarray,
    width: float,
    is_hole: bool,
    min_element_area: float,
    max_corner_cut: float,
    min_width_ratio: float,
    max_passes: int,
    corner_angle_deg: float = 120.0,
    max_removals: int | None = None,
) -> tuple[np.ndarray, RingBand, int, float]:
    """面積下限を満たせない箇所の頂点を落として、帯の区間を統合する。

    頂点 i を落とすと帯四角形 i-1 と i が 1 つになり、Γ1 側のコーナーも鈍くなる。
    境界形状のずれ（コーナーの切り落とし量）が max_corner_cut を超える場合は
    落とさない。

    Returns:
        (pts, band, n_removed, max_deviation)
    """
    band = build_ring_band(pts, width, is_hole, min_width_ratio, max_passes)
    n_removed = 0
    max_dev = 0.0
    if max_removals is None:
        max_removals = len(pts)

    while n_removed < max_removals:
        if band.failed_segments or len(pts) <= 8:
            break

        n = len(pts)
        turns = _turn_angles(pts)
        candidates: list[int] = []

        areas = band.quad_areas()
        undersized = np.flatnonzero(areas < min_element_area)
        if undersized.size:
            # 面積不足を招いている側（より鋭いコーナー）の頂点から試す
            worst = int(undersized[np.argmin(areas[undersized])])
            candidates = sorted([worst, (worst + 1) % n], key=lambda k: -turns[k])
        else:
            sharp = _sharp_interface_corners(band.inner, min_element_area, corner_angle_deg)
            if sharp.size == 0:
                break
            candidates = [int(k) for k in sharp]

        removed = False
        for k in candidates:
            dev = _point_line_distance(pts[k], pts[(k - 1) % n], pts[(k + 1) % n])
            if dev > max_corner_cut:
                continue
            pts = np.delete(pts, k, axis=0)
            max_dev = max(max_dev, dev)
            n_removed += 1
            removed = True
            break
        if not removed:
            break

        band = build_ring_band(pts, width, is_hole, min_width_ratio, max_passes)

    return pts, band, n_removed, max_dev


def triangle_only_polygon_band(poly: Polygon, spacing: float) -> PolygonBand:
    """四角形帯を置かず、外周をリサンプリングした三角形領域だけを返す。"""
    exterior_pts, hole_pts = polygon_rings_resampled(poly, spacing)

    def _skipped_ring(pts: np.ndarray, is_hole: bool) -> RingBand:
        n = len(pts)
        return RingBand(
            outer=pts,
            inner=np.array(pts, copy=True),
            widths=np.zeros(n),
            is_hole=is_hole,
            skipped=set(range(n)),
        )

    return PolygonBand(
        polygon=poly,
        exterior=_skipped_ring(exterior_pts, False),
        holes=[_skipped_ring(pts, True) for pts in hole_pts],
        interior=poly,
        ok=True,
    )


def build_polygon_band(
    poly: Polygon,
    width: float,
    spacing: float,
    min_width_ratio: float = 0.25,
    max_passes: int = 12,
    min_element_area: float = 0.0,
    max_corner_cut: float | None = None,
    corner_angle_deg: float = 120.0,
) -> PolygonBand:
    """1 ポリゴンの外周四角形帯と、三角形で埋める内側領域を作る。

    四角形を置けるかどうかは、内向きオフセットが自己交差せず有効な四角形を
    作れるかだけで決まる。境界形状の再現性を優先するため、面積下限
    （min_element_area）を理由に区間を帯から外すことはしない。面積下限は
    帯四角形にも内側の三角形にも課さず、境界に接する要素は面積下限の
    チェック・修復の対象外として扱う（呼び出し元 main.py を参照）。
    max_corner_cut > 0 を明示したときに限り、境界の頂点を落として区間を
    つなぐ従来の方法も併用する（既定では無効）。
    """
    logger = get_logger()
    exterior_pts, hole_pts = polygon_rings_resampled(poly, spacing)
    if max_corner_cut is None:
        max_corner_cut = 0.0

    n_cut, max_dev = 0, 0.0
    if min_element_area > 0.0 and max_corner_cut > 0.0:
        # 明示的に許可された場合だけ、境界の頂点を落として区間をつなぐ
        exterior_pts, _, n_cut, max_dev = _merge_undersized_quads(
            exterior_pts, width, False, min_element_area, max_corner_cut,
            min_width_ratio, max_passes, corner_angle_deg,
        )
        new_hole_pts = []
        for pts in hole_pts:
            pts, _, n_rm, dev = _merge_undersized_quads(
                pts, width, True, min_element_area, max_corner_cut,
                min_width_ratio, max_passes, corner_angle_deg,
            )
            n_cut += n_rm
            max_dev = max(max_dev, dev)
            new_hole_pts.append(pts)
        hole_pts = new_hole_pts
        if n_cut:
            logger.info(
                "外周帯: 面積下限を満たすためコーナー頂点を %d 個統合"
                "（最大切り落とし %.1f m）", n_cut, max_dev,
            )

    ext_band = build_ring_band(exterior_pts, width, False, min_width_ratio, max_passes)
    hole_bands = [
        build_ring_band(pts, width, True, min_width_ratio, max_passes) for pts in hole_pts
    ]
    band = PolygonBand(polygon=poly, exterior=ext_band, holes=hole_bands)

    # 四角形を置けない区間（内向きオフセットの自己交差）は帯から外し、
    # その区間だけ三角形に譲る。境界形状を保つことを、外周をすべて四角形で
    # 覆うことより優先する。
    for ring in band.rings():
        # 内向きオフセットが解消できなかった区間も三角形へ逃がす
        ring.skipped |= {i for i in ring.failed_segments}
        ring.skipped |= {(i - 1) % ring.n for i in ring.failed_segments}
        _skip_bad_segments(ring)

    band.corner_cuts = n_cut
    band.max_corner_deviation = max_dev

    if all(len(r.skipped) >= r.n for r in band.rings()):
        logger.warning(
            "外周四角形帯: 四角形を置ける区間が無く、全周を三角形で埋めます"
            "（面積 %.2f km^2）", poly.area / 1e6,
        )

    try:
        interior = Polygon(
            ext_band.interior_ring(), [h.interior_ring() for h in hole_bands]
        )
    except Exception as exc:  # noqa: BLE001 - shapely が投げる型は環境依存
        band.ok = False
        band.reason = f"内側領域の構築に失敗: {exc}"
        return band

    if not interior.is_valid or interior.is_empty or interior.area <= 0.0:
        band.ok = False
        band.reason = "内側領域が退化しました（帯幅に対して領域が細すぎます）"
        logger.warning("外周四角形帯: %s（面積 %.2f km^2）", band.reason, poly.area / 1e6)
        return band

    band.interior = interior
    return band


def band_report(band: PolygonBand, min_element_area: float) -> dict[str, float]:
    """帯の四角形面積の統計。

    below_min / below_min_ratio は境界に接する要素の面積下限「違反」ではなく、
    参考統計であることに注意（境界形状の再現性を優先するため、これらの要素は
    面積下限のチェック・修復の対象外として扱われる）。
    """
    parts = [r.quad_areas(live_only=True) for r in band.rings()]
    parts = [p for p in parts if p.size]
    if not parts:
        return {"count": 0, "skipped": band.skipped_count()}
    areas = np.concatenate(parts)
    return {
        "count": int(areas.size),
        "skipped": band.skipped_count(),
        "area_min": float(areas.min()),
        "area_median": float(np.median(areas)),
        "area_max": float(areas.max()),
        "below_min": int((areas < min_element_area).sum()),
        "below_min_ratio": float((areas < min_element_area).mean()),
    }
