"""三角形メッシュの品質指標の計算と品質フラグ付与。"""

from __future__ import annotations

import geopandas as gpd
import numpy as np


def _triangle_metrics(p0, p1, p2):
    """1 つの三角形（3 頂点座標）の各種指標を返す。"""
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p0 - p2)
    edges = np.array([a, b, c])
    perimeter = float(edges.sum())

    # 面積（外積）
    area = 0.5 * abs(
        (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    )

    edge_min = float(edges.min())
    edge_max = float(edges.max())
    edge_mean = float(edges.mean())

    # アスペクト比（最長辺 / 最短辺）
    aspect_ratio = edge_max / edge_min if edge_min > 0 else np.inf

    # コンパクトネス（4πA / P^2、正三角形で最大、円で 1）
    compactness = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # 内角（余弦定理）
    def angle(opposite, s1, s2):
        if s1 <= 0 or s2 <= 0:
            return 0.0
        cosv = (s1 ** 2 + s2 ** 2 - opposite ** 2) / (2 * s1 * s2)
        cosv = max(-1.0, min(1.0, cosv))
        return float(np.degrees(np.arccos(cosv)))

    ang_a = angle(b, a, c)  # 頂点 p0
    ang_b = angle(c, a, b)  # 頂点 p1
    ang_c = angle(a, b, c)  # 頂点 p2
    angles = np.array([ang_a, ang_b, ang_c])

    return {
        "area": float(area),
        "perimeter": perimeter,
        "edge_min": edge_min,
        "edge_max": edge_max,
        "edge_mean": edge_mean,
        "aspect_ratio": float(aspect_ratio),
        "compactness": float(compactness),
        "min_angle": float(angles.min()),
        "max_angle": float(angles.max()),
    }


def compute_quality(
    triangles_gdf: gpd.GeoDataFrame, coords: np.ndarray
) -> gpd.GeoDataFrame:
    """各三角形の品質指標・重心・triangle_id を計算して付与する。"""
    out = triangles_gdf.copy()
    n = len(out)
    if n == 0:
        for col in [
            "triangle_id", "area", "perimeter", "edge_min", "edge_max",
            "edge_mean", "aspect_ratio", "compactness", "min_angle",
            "max_angle", "centroid_x", "centroid_y",
        ]:
            out[col] = []
        return out

    node1 = out["node1"].to_numpy()
    node2 = out["node2"].to_numpy()
    node3 = out["node3"].to_numpy()

    records = {k: np.zeros(n) for k in [
        "area", "perimeter", "edge_min", "edge_max", "edge_mean",
        "aspect_ratio", "compactness", "min_angle", "max_angle",
        "centroid_x", "centroid_y",
    ]}

    for i in range(n):
        p0 = coords[node1[i]]
        p1 = coords[node2[i]]
        p2 = coords[node3[i]]
        m = _triangle_metrics(np.asarray(p0, float), np.asarray(p1, float), np.asarray(p2, float))
        for k, v in m.items():
            records[k][i] = v
        centroid = (np.asarray(p0, float) + np.asarray(p1, float) + np.asarray(p2, float)) / 3.0
        records["centroid_x"][i] = centroid[0]
        records["centroid_y"][i] = centroid[1]

    out["triangle_id"] = np.arange(n, dtype=int)
    for k, v in records.items():
        out[k] = v
    return out


def flag_quality(
    triangles_gdf: gpd.GeoDataFrame,
    min_edge_length: float,
    min_area: float,
    min_angle: float,
    max_aspect_ratio: float,
) -> gpd.GeoDataFrame:
    """品質基準を満たさない三角形にフラグを付ける。"""
    out = triangles_gdf.copy()
    out["flag_short_edge"] = out["edge_min"] < min_edge_length
    out["flag_small_area"] = out["area"] < min_area
    out["flag_small_angle"] = out["min_angle"] < min_angle
    out["flag_high_aspect"] = out["aspect_ratio"] > max_aspect_ratio
    out["quality_flag"] = (
        out["flag_short_edge"]
        | out["flag_small_area"]
        | out["flag_small_angle"]
        | out["flag_high_aspect"]
    )
    return out


def quality_summary(triangles_gdf: gpd.GeoDataFrame) -> dict:
    """品質サマリ（統計値と警告件数）を返す。"""
    if len(triangles_gdf) == 0:
        return {"n_triangles": 0}

    def _stat(col):
        if col not in triangles_gdf:
            return {}
        s = triangles_gdf[col]
        return {
            f"{col}_min": float(s.min()),
            f"{col}_max": float(s.max()),
            f"{col}_mean": float(s.mean()),
        }

    summary = {"n_triangles": int(len(triangles_gdf))}
    for col in ["area", "edge_min", "edge_max", "min_angle", "aspect_ratio", "z_range", "z_std"]:
        summary.update(_stat(col))

    for flag in ["flag_short_edge", "flag_small_area", "flag_small_angle", "flag_high_aspect", "quality_flag"]:
        if flag in triangles_gdf:
            summary[f"n_{flag}"] = int(triangles_gdf[flag].sum())
    return summary
