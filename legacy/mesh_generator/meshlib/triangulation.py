"""Delaunay 三角形分割と境界外三角形の除去。

scipy.spatial.Delaunay を用いる。将来的に Triangle / Gmsh による
制約付き Delaunay へ差し替えやすいよう、入出力を点群配列と三角形
インデックスに限定している。
"""

from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from shapely.prepared import prep


def triangulate_points(coords: np.ndarray) -> np.ndarray:
    """点座標 (N,2) を Delaunay 分割し、三角形の頂点インデックス (M,3) を返す。"""
    if coords.shape[0] < 3:
        raise ValueError("三角形分割には最低 3 点が必要です。")
    tri = Delaunay(coords)
    return tri.simplices.copy()


def _triangle_polygon(coords: np.ndarray, idx: np.ndarray) -> Polygon:
    return Polygon([coords[idx[0]], coords[idx[1]], coords[idx[2]]])


def create_delaunay_triangles(
    points_gdf: gpd.GeoDataFrame,
    boundary_polygon,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    mode: str = "building_as_roughness",
    area_fraction_threshold: float = 0.0,
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """点群から Delaunay 三角形を生成し、解析領域外の三角形を除去する。

    境界外除去は「重心が解析ポリゴン内」かどうかで判定する（Clip しない）。
    area_fraction_threshold > 0 の場合、三角形面積のうち領域内割合が閾値以上の
    三角形のみ残す（重心判定との AND）。

    Returns:
        (triangles_gdf, coords)
          triangles_gdf: geometry と node1/node2/node3（coords へのインデックス）。
          coords: (N,2) のノード座標配列。
    """
    coords = np.column_stack([points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()])
    crs = points_gdf.crs

    simplices = triangulate_points(coords)

    prepared_boundary = prep(boundary_polygon)

    geoms = []
    node_ids = []
    for idx in simplices:
        poly = _triangle_polygon(coords, idx)
        if poly.area <= 0:
            continue
        centroid = poly.centroid
        if not prepared_boundary.contains(centroid):
            continue
        if area_fraction_threshold > 0.0:
            inter = poly.intersection(boundary_polygon)
            if inter.is_empty or (inter.area / poly.area) < area_fraction_threshold:
                continue
        geoms.append(poly)
        node_ids.append((int(idx[0]), int(idx[1]), int(idx[2])))

    if not geoms:
        empty = gpd.GeoDataFrame(
            {"node1": [], "node2": [], "node3": []},
            geometry=[],
            crs=crs,
        )
        return empty, coords

    node_arr = np.array(node_ids, dtype=int)
    tris = gpd.GeoDataFrame(
        {
            "node1": node_arr[:, 0],
            "node2": node_arr[:, 1],
            "node3": node_arr[:, 2],
        },
        geometry=geoms,
        crs=crs,
    )
    return tris, coords


def reindex_used_nodes(
    triangles_gdf: gpd.GeoDataFrame, coords: np.ndarray
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """三角形で実際に使われているノードのみに再採番する。

    Returns:
        (triangles_gdf, new_coords) — node1/2/3 が new_coords のインデックスを指す。
    """
    if len(triangles_gdf) == 0:
        return triangles_gdf, np.empty((0, 2))

    used = np.unique(
        triangles_gdf[["node1", "node2", "node3"]].to_numpy().ravel()
    )
    remap = {old: new for new, old in enumerate(used)}
    new_coords = coords[used]

    out = triangles_gdf.copy()
    for col in ("node1", "node2", "node3"):
        out[col] = out[col].map(remap).astype(int)
    return out.reset_index(drop=True), new_coords
