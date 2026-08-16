"""点群生成・点列化・近接間引きユーティリティ。

すべての点は決定論的に生成される（乱数を使用しない）。
点には種類 (type) と優先度 (priority) を付与し、間引き時に優先度の高い点を残す。
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

# 点種類ごとの優先度（指示書 6.3）
PRIORITY = {
    "boundary": 100,
    "building_boundary": 90,
    "building_buffer": 85,
    "river": 80,
    "road": 80,
    "facility": 70,
    "slope": 60,
    "refine": 50,
    "interior": 10,
}


def densify_linestring(line: LineString, interval: float) -> List[Point]:
    """LineString 上に interval 間隔で点を配置する。

    始点・終点を必ず含む。MultiLineString はモジュール内の別関数で展開。
    """
    if interval <= 0:
        raise ValueError("interval は正の値である必要があります。")
    length = line.length
    if length == 0:
        return [Point(line.coords[0])]

    n = max(int(np.floor(length / interval)), 1)
    distances = np.linspace(0.0, length, n + 1)
    return [line.interpolate(float(d)) for d in distances]


def _densify_any_line(geom, interval: float) -> List[Point]:
    """LineString / MultiLineString を点列化する。"""
    pts: List[Point] = []
    if geom is None or geom.is_empty:
        return pts
    if isinstance(geom, LineString):
        pts.extend(densify_linestring(geom, interval))
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            pts.extend(densify_linestring(part, interval))
    return pts


def densify_polygon_boundary(polygon, interval: float) -> List[Point]:
    """ポリゴンの外周・内周（穴）を点列化する。"""
    pts: List[Point] = []
    if polygon is None or polygon.is_empty:
        return pts
    polys = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
    for poly in polys:
        if not isinstance(poly, Polygon):
            continue
        pts.extend(densify_linestring(LineString(poly.exterior.coords), interval))
        for ring in poly.interiors:
            pts.extend(densify_linestring(LineString(ring.coords), interval))
    return pts


def densify_lines_gdf(gdf: Optional[gpd.GeoDataFrame], interval: float) -> List[Point]:
    """ラインの GeoDataFrame を点列化する（河川・道路向け）。"""
    pts: List[Point] = []
    if gdf is None or len(gdf) == 0:
        return pts
    for geom in gdf.geometry:
        pts.extend(_densify_any_line(geom, interval))
    return pts


def generate_regular_points_within_polygon(
    polygon,
    spacing: float,
    holes: Optional[Iterable] = None,
) -> List[Point]:
    """ポリゴンの bounding box に格子点を作り、内部の点だけ残す。

    holes（建物など）が与えられた場合、その内部の点は除外する（穴モード用）。
    """
    if spacing <= 0:
        raise ValueError("spacing は正の値である必要があります。")
    minx, miny, maxx, maxy = polygon.bounds
    # 端をまたぐように原点を spacing グリッドにスナップ（再現性のため固定原点）
    x0 = np.floor(minx / spacing) * spacing
    y0 = np.floor(miny / spacing) * spacing
    xs = np.arange(x0, maxx + spacing, spacing)
    ys = np.arange(y0, maxy + spacing, spacing)

    if len(xs) == 0 or len(ys) == 0:
        return []

    gx, gy = np.meshgrid(xs, ys)
    coords = np.column_stack([gx.ravel(), gy.ravel()])

    candidate = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in coords]
    )
    # ポリゴン内判定
    from shapely.prepared import prep

    prepared_poly = prep(polygon)
    mask = np.array([prepared_poly.contains(p) for p in candidate.geometry])
    inside = [p for p, m in zip(candidate.geometry, mask) if m]

    if holes is not None:
        hole_union = None
        hole_list = list(holes)
        if hole_list:
            from shapely.ops import unary_union

            hole_union = unary_union(hole_list)
        if hole_union is not None and not hole_union.is_empty:
            prepared_hole = prep(hole_union)
            inside = [p for p in inside if not prepared_hole.contains(p)]

    return inside


def points_to_gdf(
    points: Iterable[Point],
    point_type: str,
    crs,
    priority: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Point のリストを type / priority 付きの GeoDataFrame にする。"""
    pts = list(points)
    if priority is None:
        priority = PRIORITY.get(point_type, 10)
    if not pts:
        return gpd.GeoDataFrame(
            {"type": pd.Series(dtype="object"), "priority": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
    return gpd.GeoDataFrame(
        {
            "type": [point_type] * len(pts),
            "priority": [int(priority)] * len(pts),
        },
        geometry=pts,
        crs=crs,
    )


def merge_point_gdfs(gdfs: Iterable[gpd.GeoDataFrame], crs) -> gpd.GeoDataFrame:
    """複数の点 GeoDataFrame を結合する（空は無視）。"""
    valid = [g for g in gdfs if g is not None and len(g) > 0]
    if not valid:
        return gpd.GeoDataFrame(
            {"type": pd.Series(dtype="object"), "priority": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
    merged = pd.concat(valid, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=crs).reset_index(drop=True)


def thin_points(points_gdf: gpd.GeoDataFrame, min_distance: float) -> gpd.GeoDataFrame:
    """min_distance 未満に近接する点を間引く（優先度の高い点を残す）。

    アルゴリズム（グリッド空間ハッシュによる貪欲法・決定論的）:
      1. priority 降順 → x 昇順 → y 昇順 で点を並べる。
      2. セルサイズ = min_distance の一様グリッドに採用点を登録。
      3. 候補点の周囲 3x3 セル内の採用点との距離が min_distance 未満なら破棄。
    """
    if points_gdf is None or len(points_gdf) == 0:
        return points_gdf
    if min_distance <= 0:
        return points_gdf.reset_index(drop=True)

    g = points_gdf.copy()
    g["_x"] = g.geometry.x
    g["_y"] = g.geometry.y
    g = g.sort_values(
        ["priority", "_x", "_y"], ascending=[False, True, True]
    ).reset_index(drop=True)

    coords = g[["_x", "_y"]].to_numpy()
    cell = float(min_distance)
    min_d2 = cell * cell

    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    accepted_idx: List[int] = []

    for i in range(len(coords)):
        x, y = float(coords[i, 0]), float(coords[i, 1])
        cx, cy = int(np.floor(x / cell)), int(np.floor(y / cell))
        keep = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = grid.get((cx + dx, cy + dy))
                if not bucket:
                    continue
                for (ax, ay) in bucket:
                    if (ax - x) ** 2 + (ay - y) ** 2 < min_d2:
                        keep = False
                        break
                if not keep:
                    break
            if not keep:
                break
        if keep:
            accepted_idx.append(i)
            grid.setdefault((cx, cy), []).append((x, y))

    out = g.iloc[accepted_idx].drop(columns=["_x", "_y"]).reset_index(drop=True)
    return out
