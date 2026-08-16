"""建物関連の点群生成と三角形への建物属性付与。"""

from __future__ import annotations

from typing import List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

from . import points as points_mod


def building_boundary_points(
    buildings_gdf: Optional[gpd.GeoDataFrame], spacing: float
) -> List[Point]:
    """建物ポリゴンの境界を点列化する。"""
    pts: List[Point] = []
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return pts
    for geom in buildings_gdf.geometry:
        pts.extend(points_mod.densify_polygon_boundary(geom, spacing))
    return pts


def building_buffer_points(
    buildings_gdf: Optional[gpd.GeoDataFrame],
    buffer_width: float,
    spacing: float,
) -> List[Point]:
    """建物周囲バッファのリング上に点を配置する。

    建物外側のバッファ境界を点列化することで、建物周囲の流れを安定させる。
    """
    pts: List[Point] = []
    if buildings_gdf is None or len(buildings_gdf) == 0 or buffer_width <= 0:
        return pts
    # 全建物をまとめてバッファ → 外周リングを点列化（重複領域を統合）
    union = buildings_gdf.geometry.union_all()
    buffered = union.buffer(buffer_width)
    pts.extend(points_mod.densify_polygon_boundary(buffered, spacing))
    return pts


def get_building_holes(buildings_gdf: Optional[gpd.GeoDataFrame]) -> Optional[list]:
    """穴モード用に建物ポリゴンのリストを返す。"""
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return None
    return list(buildings_gdf.geometry)


def compute_building_attributes(
    triangles_gdf: gpd.GeoDataFrame,
    buildings_gdf: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """各三角形に建物属性を付与する。

    付与する属性:
      - building_intersect: bool（建物と重なるか）
      - building_fraction: 三角形面積に対する建物重複面積割合 [0,1]
      - building_count: 重なる建物数
      - centroid_in_building: bool（重心が建物内か）
    """
    out = triangles_gdf.copy()
    n = len(out)
    if buildings_gdf is None or len(buildings_gdf) == 0:
        out["building_intersect"] = False
        out["building_fraction"] = 0.0
        out["building_count"] = 0
        out["centroid_in_building"] = False
        return out

    build_geoms = list(buildings_gdf.geometry)
    tree = STRtree(build_geoms)

    intersect_flags = np.zeros(n, dtype=bool)
    fractions = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)
    centroid_in = np.zeros(n, dtype=bool)

    for i, geom in enumerate(out.geometry.values):
        area = geom.area
        if area <= 0:
            continue
        cand = tree.query(geom)  # 候補インデックス（バウンディングボックス交差）
        overlap_area = 0.0
        cnt = 0
        for j in cand:
            bgeom = build_geoms[int(j)]
            if not geom.intersects(bgeom):
                continue
            inter = geom.intersection(bgeom)
            if inter.is_empty:
                continue
            a = inter.area
            if a > 0:
                overlap_area += a
                cnt += 1
        if cnt > 0:
            intersect_flags[i] = True
            counts[i] = cnt
            fractions[i] = min(overlap_area / area, 1.0)

        centroid = geom.centroid
        cand_c = tree.query(centroid)
        for j in cand_c:
            if build_geoms[int(j)].contains(centroid):
                centroid_in[i] = True
                break

    out["building_intersect"] = intersect_flags
    out["building_fraction"] = fractions
    out["building_count"] = counts
    out["centroid_in_building"] = centroid_in
    return out


def apply_building_mode(
    triangles_gdf: gpd.GeoDataFrame, mode: str
) -> gpd.GeoDataFrame:
    """建物モードに応じて三角形を処理する。

    - building_as_roughness: すべて残す（属性で扱う）。
    - building_as_hole: 重心が建物内の三角形を削除。境界をまたぐ三角形に warning。
    """
    out = triangles_gdf.copy()
    out["building_warning"] = False

    if mode == "building_as_hole":
        # 境界をまたぐ（交差するが重心は外）三角形に warning
        warn_mask = out["building_intersect"] & (~out["centroid_in_building"])
        out.loc[warn_mask, "building_warning"] = True
        # 重心が建物内の三角形を削除
        keep = ~out["centroid_in_building"]
        out = out[keep].reset_index(drop=True)

    return out
