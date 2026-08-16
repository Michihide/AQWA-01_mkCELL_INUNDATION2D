"""三角形メッシュの node / cell / edge テーブル（隣接関係）の構築。

有限体積法で利用できるよう、各辺の左右セル・法線・境界種別を計算する。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree


def build_node_table(coords: np.ndarray) -> pd.DataFrame:
    """ノードテーブル（node_id, x, y）を作成する。"""
    n = coords.shape[0]
    return pd.DataFrame({
        "node_id": np.arange(n, dtype=int),
        "x": coords[:, 0],
        "y": coords[:, 1],
    })


def build_cell_table(triangles_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """セルテーブル（三角形ごとの属性）を作成する。"""
    cols = [
        "triangle_id", "node1", "node2", "node3",
        "centroid_x", "centroid_y", "area",
        "z_mean", "z_min", "z_max", "z_std", "z_range",
        "building_fraction",
        "edge_min", "edge_max", "aspect_ratio", "min_angle", "max_angle",
        "quality_flag", "refine_flag",
    ]
    data = {}
    n = len(triangles_gdf)
    for c in cols:
        if c in triangles_gdf:
            data[c] = triangles_gdf[c].to_numpy()
        else:
            data[c] = np.zeros(n)
    df = pd.DataFrame(data)
    df = df.rename(columns={"triangle_id": "cell_id"})
    return df


def build_edge_table(
    triangles_gdf: gpd.GeoDataFrame,
    coords: np.ndarray,
    boundary_polygon=None,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    boundary_tol: float = 1.0,
    building_near_tol: float = 5.0,
) -> pd.DataFrame:
    """エッジテーブルを作成する。

    各辺 (node_a < node_b) について左右セルを割り当て、辺長・法線・境界種別を計算。
    right_cell が無い辺を境界辺とし、outer / building_near / internal に分類する。
    """
    node1 = triangles_gdf["node1"].to_numpy()
    node2 = triangles_gdf["node2"].to_numpy()
    node3 = triangles_gdf["node3"].to_numpy()
    if "triangle_id" in triangles_gdf:
        cell_ids = triangles_gdf["triangle_id"].to_numpy()
    else:
        cell_ids = np.arange(len(triangles_gdf), dtype=int)
    centroids = np.column_stack([
        triangles_gdf["centroid_x"].to_numpy(),
        triangles_gdf["centroid_y"].to_numpy(),
    ]) if "centroid_x" in triangles_gdf else None

    # 辺 -> [(cell_id, third_node)]
    edge_map: dict[tuple[int, int], list] = defaultdict(list)

    def add_edge(a, b, cell, third):
        key = (a, b) if a < b else (b, a)
        edge_map[key].append((int(cell), int(third)))

    for i in range(len(triangles_gdf)):
        n1, n2, n3 = int(node1[i]), int(node2[i]), int(node3[i])
        cid = int(cell_ids[i])
        add_edge(n1, n2, cid, n3)
        add_edge(n2, n3, cid, n1)
        add_edge(n3, n1, cid, n2)

    # 境界判定の準備
    boundary_line = None
    if boundary_polygon is not None:
        boundary_line = boundary_polygon.boundary

    build_tree = None
    build_geoms = None
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        build_geoms = list(buildings_gdf.geometry)
        build_tree = STRtree(build_geoms)

    rows = []
    edge_id = 0
    for (a, b), cells in sorted(edge_map.items()):
        pa = coords[a]
        pb = coords[b]
        ex, ey = pb[0] - pa[0], pb[1] - pa[1]
        length = float(np.hypot(ex, ey))
        # 法線（辺に直交、左セル外向きに後で調整）
        if length > 0:
            nx, ny = ey / length, -ex / length
        else:
            nx, ny = 0.0, 0.0

        left_cell = cells[0][0]
        right_cell = cells[1][0] if len(cells) > 1 else -1

        # 左セル重心から外向きになるよう法線の向きを調整
        if centroids is not None and left_cell >= 0:
            # left_cell は cell_id。triangle_id==index 前提で参照
            try:
                lc = centroids[left_cell]
                mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
                vx, vy = mx - lc[0], my - lc[1]
                if (nx * vx + ny * vy) < 0:
                    nx, ny = -nx, -ny
            except (IndexError, TypeError):
                pass

        # 境界種別
        if right_cell >= 0:
            boundary_type = "internal"
        else:
            boundary_type = "outer"
            mid = Point((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
            if build_tree is not None:
                cand = build_tree.query(mid.buffer(building_near_tol))
                near_build = any(
                    build_geoms[int(j)].distance(mid) <= building_near_tol for j in cand
                )
                if near_build:
                    boundary_type = "building_near"
            if boundary_type == "outer" and boundary_line is not None:
                if boundary_line.distance(mid) > boundary_tol:
                    # 外周から離れた境界辺（建物穴の縁など）
                    boundary_type = "internal_boundary"

        rows.append({
            "edge_id": edge_id,
            "node_a": a,
            "node_b": b,
            "left_cell": left_cell,
            "right_cell": right_cell,
            "edge_length": length,
            "normal_x": nx,
            "normal_y": ny,
            "boundary_type": boundary_type,
        })
        edge_id += 1

    return pd.DataFrame(rows)
