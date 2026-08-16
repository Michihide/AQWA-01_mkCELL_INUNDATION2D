"""gmsh ジオメトリの構築。

外周四角形帯は、区間ごとに 4 辺の Plane Surface を作り、全辺を numNodes=2 の
Transfinite Curve にしたうえで Transfinite Surface + Recombine を指定する。
これで区間ごとにちょうど 1 個の四角形が生成され、外周に接する要素はすべて
四角形になる。内側は Γ1 を境界とする 1 枚の Plane Surface とし、三角形で埋める。

Γ1 の各辺も numNodes=2 の Transfinite Curve なので、内側領域の境界節点は帯の
内側節点と厳密に一致し、両者は適合（conforming）する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np

from .boundary_quad_band import PolygonBand
from .utils import get_logger

# Physical Group 名
PG_QUAD_BOUNDARY = "SURFACE_QUAD_BOUNDARY"
PG_TRI_BOUNDARY = "SURFACE_TRI_BOUNDARY"
PG_TRI_INTERIOR = "SURFACE_TRI_INTERIOR"
PG_QUAD_INTERIOR = "SURFACE_QUAD_INTERIOR"
PG_DOMAIN_BOUNDARY = "CURVE_DOMAIN_BOUNDARY"
PG_BAND_INTERFACE = "CURVE_BAND_INTERFACE"
PG_BREAKLINE_PREFIX = "CURVE_BREAKLINE_"


@dataclass
class GeometryTags:
    """作成した gmsh エンティティのタグ。"""

    band_surfaces: list[int] = field(default_factory=list)
    # 帯の端を閉じる三角形。帯の一部だが四角形ではない。
    band_tri_surfaces: list[int] = field(default_factory=list)
    interior_surfaces: list[int] = field(default_factory=list)
    boundary_curves: list[int] = field(default_factory=list)
    interface_curves: list[int] = field(default_factory=list)
    breakline_curves: dict[str, list[int]] = field(default_factory=dict)
    # 内側面の番号 -> その面へ埋め込む曲線タグ
    embedded_curves: dict[int, list[int]] = field(default_factory=dict)
    # 内部で四角形化する対象の面（Phase 2 以降）
    interior_quad_surfaces: list[int] = field(default_factory=list)

    def all_surfaces(self) -> list[int]:
        return [*self.band_surfaces, *self.band_tri_surfaces, *self.interior_surfaces]

    def all_breakline_curves(self) -> list[int]:
        return [tag for curves in self.breakline_curves.values() for tag in curves]


def _add_ring_points(pts: np.ndarray) -> list[int]:
    add = gmsh.model.geo.addPoint
    return [add(float(p[0]), float(p[1]), 0.0) for p in pts]


def _add_ring_lines(point_tags: list[int]) -> list[int]:
    add = gmsh.model.geo.addLine
    n = len(point_tags)
    return [add(point_tags[i], point_tags[(i + 1) % n]) for i in range(n)]


def build_band_geometry(band: PolygonBand, tags: GeometryTags) -> None:
    """1 ポリゴン分の帯と内側領域を gmsh へ追加する。

    帯から外された区間には四角形面を作らない。内側領域の境界はそこで Γ0 まで
    降り、境界沿いに進んでから Γ1 へ戻る。Γ0 も Γ1 も numNodes=2 の Transfinite
    なので、どちらを辿っても節点間隔は同じで、帯とは適合したままになる。
    """
    geo = gmsh.model.geo
    interior_loops: list[int] = []

    for ring in band.rings():
        n = ring.n
        cells = ring.cell_indices()
        quads = set(ring.quad_indices())
        walk = ring.interior_walk()

        # スキップ区間の Γ1 と、どこからも使わない点・線は作らない。孤立した点を
        # 残すと、どの要素にも属さない節点が出力に混ざる。
        need_inner = {v for i in cells for v in (i, (i + 1) % n)}
        need_inner |= {i for kind, i in walk if kind in ("down", "up")}
        need_radial = need_inner - ring.collapsed

        outer_pts = _add_ring_points(ring.outer)
        # 幅 0 の頂点では Γ1 と Γ0 が同一点。点を分けると隙間ができる。
        inner_pts = {
            v: outer_pts[v] if v in ring.collapsed
            else geo.addPoint(*(float(c) for c in ring.inner_pt(v)), 0.0)
            for v in sorted(need_inner)
        }
        outer_lines = _add_ring_lines(outer_pts)
        inner_lines = {
            i: geo.addLine(inner_pts[i], inner_pts[(i + 1) % n]) for i in cells
        }
        radial = {v: geo.addLine(outer_pts[v], inner_pts[v]) for v in sorted(need_radial)}

        for tag in (*outer_lines, *inner_lines.values(), *radial.values()):
            geo.mesh.setTransfiniteCurve(tag, 2)

        for i in cells:
            j = (i + 1) % n
            if i in quads:
                # 反時計回り: Γ0(i->j) -> 径方向(j) -> Γ1(j->i) -> 径方向(i) の逆
                loop = [outer_lines[i], radial[j], -inner_lines[i], -radial[i]]
            elif j in ring.collapsed:
                # 帯の終わり。Γ1 が Γ0 に着地して三角形になる。
                loop = [outer_lines[i], -inner_lines[i], -radial[i]]
            else:
                # 帯の始まり
                loop = [outer_lines[i], radial[j], -inner_lines[i]]
            surf = geo.addPlaneSurface([geo.addCurveLoop(loop)])
            geo.mesh.setTransfiniteSurface(surf)
            if i in quads:
                geo.mesh.setRecombine(2, surf)
                tags.band_surfaces.append(surf)
            else:
                tags.band_tri_surfaces.append(surf)

        tags.boundary_curves.extend(outer_lines)
        tags.interface_curves.extend(inner_lines.values())

        # Γ1 は外周リングなら CCW、穴リングなら CW に並んでいる。そのまま辿れば
        # 内側領域を左手側に見ることになり、面の法線が +z を向く。
        curve_of = {
            "inner": lambda i: inner_lines[i],
            "outer": lambda i: outer_lines[i],
            "down": lambda i: -radial[i],
            "up": lambda i: radial[i],
        }
        interior_loops.append(
            geo.addCurveLoop([curve_of[kind](i) for kind, i in walk])
        )

    interior_surface = gmsh.model.geo.addPlaneSurface(interior_loops)
    tags.interior_surfaces.append(interior_surface)


def add_breakline_geometry(
    lines: list, band_index: int, tags: GeometryTags
) -> list[int]:
    """拘束線を点＋線として追加し、作成した曲線タグを返す。

    embed は synchronize の後でしか呼べないため、ここでは幾何の追加だけを行う。
    """
    geo = gmsh.model.geo
    curves: list[int] = []
    for line in lines:
        pts = [geo.addPoint(float(x), float(y), 0.0) for x, y in line.coords]
        segs = [geo.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        tags.breakline_curves.setdefault(line.layer, []).extend(segs)
        curves.extend(segs)
    if curves:
        tags.embedded_curves[band_index] = curves
    return curves


def build_geometry(bands: list[PolygonBand], breaklines=None) -> GeometryTags:
    """全ポリゴンのジオメトリを構築し、Physical Group を設定する。"""
    logger = get_logger()
    tags = GeometryTags()
    for index, band in enumerate(bands):
        if not band.ok:
            raise ValueError(f"外周四角形帯が構築できていません: {band.reason}")
        build_band_geometry(band, tags)
        if breaklines is not None:
            add_breakline_geometry(breaklines.by_band(index), index, tags)

    gmsh.model.geo.synchronize()

    # 内側面に線を埋め込むと、その線はメッシュ辺として必ず再現される
    n_embedded = 0
    for index, curves in tags.embedded_curves.items():
        gmsh.model.mesh.embed(1, curves, 2, tags.interior_surfaces[index])
        n_embedded += len(curves)

    logger.info(
        "ジオメトリ: 帯四角形面 %d, 帯端の三角形面 %d, 内側面 %d, 外周曲線 %d,"
        " 埋め込み拘束曲線 %d",
        len(tags.band_surfaces), len(tags.band_tri_surfaces),
        len(tags.interior_surfaces), len(tags.boundary_curves), n_embedded,
    )
    return tags


def set_physical_groups(tags: GeometryTags) -> dict[str, int]:
    """Physical Group を作成し、名前 -> タグの対応を返す。"""
    groups: dict[str, int] = {}

    def add(dim: int, entities: list[int], name: str) -> None:
        if not entities:
            return
        tag = gmsh.model.addPhysicalGroup(dim, entities)
        gmsh.model.setPhysicalName(dim, tag, name)
        groups[name] = tag

    add(2, tags.band_surfaces, PG_QUAD_BOUNDARY)
    add(2, tags.band_tri_surfaces, PG_TRI_BOUNDARY)
    interior_tri = [s for s in tags.interior_surfaces if s not in tags.interior_quad_surfaces]
    add(2, interior_tri, PG_TRI_INTERIOR)
    add(2, tags.interior_quad_surfaces, PG_QUAD_INTERIOR)
    add(1, tags.boundary_curves, PG_DOMAIN_BOUNDARY)
    add(1, tags.interface_curves, PG_BAND_INTERFACE)
    for name, curves in tags.breakline_curves.items():
        add(1, curves, f"{PG_BREAKLINE_PREFIX}{name.upper()}")

    return groups
