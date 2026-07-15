"""DEM を用いた標高統計・勾配点生成。"""

from __future__ import annotations

from typing import List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import Point


def _reproject_geoms_to_dem(geoms_gdf: gpd.GeoDataFrame, dem_crs):
    """三角形ジオメトリを DEM の CRS に合わせる。"""
    if geoms_gdf.crs is None:
        return geoms_gdf
    try:
        if dem_crs is not None and not geoms_gdf.crs.equals(dem_crs):
            return geoms_gdf.to_crs(dem_crs)
    except Exception:
        pass
    return geoms_gdf


def compute_elevation_stats(
    triangles_gdf: gpd.GeoDataFrame,
    dem_path: str,
    all_touched: bool = False,
) -> gpd.GeoDataFrame:
    """各三角形内の DEM 標高統計を計算する。

    付与する属性: z_min, z_max, z_mean, z_std, z_range, z_count
    window 読み込み + geometry_mask を使用。ピクセルが取れない微小三角形は
    重心・頂点でのサンプリングにフォールバックする。
    """
    out = triangles_gdf.copy()
    n = len(out)
    z_min = np.full(n, np.nan)
    z_max = np.full(n, np.nan)
    z_mean = np.full(n, np.nan)
    z_std = np.full(n, np.nan)
    z_count = np.zeros(n, dtype=int)

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        nodata = src.nodata
        geoms_for_sampling = _reproject_geoms_to_dem(out[["geometry"]].copy(), dem_crs)
        geoms = list(geoms_for_sampling.geometry)

        for i, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            try:
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                win = win.round_offsets().round_lengths()
                # 1 ピクセル分だけ拡張して取りこぼしを防ぐ
                win = rasterio.windows.Window(
                    col_off=win.col_off - 1,
                    row_off=win.row_off - 1,
                    width=win.width + 2,
                    height=win.height + 2,
                )
                data = src.read(1, window=win, boundless=True, fill_value=nodata if nodata is not None else -9999)
                transform = src.window_transform(win)
            except Exception:
                data = None

            vals = None
            if data is not None and data.size > 0 and data.shape[0] > 0 and data.shape[1] > 0:
                try:
                    mask = geometry_mask(
                        [geom],
                        out_shape=data.shape,
                        transform=transform,
                        invert=True,
                        all_touched=all_touched,
                    )
                    sel = data[mask]
                    if nodata is not None:
                        sel = sel[sel != nodata]
                    sel = sel[np.isfinite(sel)]
                    if sel.size > 0:
                        vals = sel.astype(float)
                except Exception:
                    vals = None

            if vals is None or vals.size == 0:
                # フォールバック: 重心と頂点でサンプリング
                sample_pts = [geom.centroid] + [Point(c) for c in geom.exterior.coords[:-1]]
                coords = [(p.x, p.y) for p in sample_pts]
                sampled = np.array([v[0] for v in src.sample(coords)], dtype=float)
                if nodata is not None:
                    sampled = sampled[sampled != nodata]
                sampled = sampled[np.isfinite(sampled)]
                vals = sampled

            if vals.size > 0:
                z_min[i] = float(np.min(vals))
                z_max[i] = float(np.max(vals))
                z_mean[i] = float(np.mean(vals))
                z_std[i] = float(np.std(vals))
                z_count[i] = int(vals.size)

    out["z_min"] = z_min
    out["z_max"] = z_max
    out["z_mean"] = z_mean
    out["z_std"] = z_std
    out["z_range"] = z_max - z_min
    out["z_count"] = z_count
    return out


def generate_slope_points(
    boundary_polygon,
    dem_path: str,
    spacing: float,
    slope_threshold: float,
    target_crs,
) -> List[Point]:
    """DEM の勾配が大きい領域に追加点を生成する。

    spacing 間隔の規則格子点で勾配を評価し、閾値を超える点を返す。
    返り値は target_crs（解析座標系）の Point リスト。
    """
    pts: List[Point] = []
    if spacing <= 0:
        return pts

    minx, miny, maxx, maxy = boundary_polygon.bounds
    x0 = np.floor(minx / spacing) * spacing
    y0 = np.floor(miny / spacing) * spacing
    xs = np.arange(x0, maxx + spacing, spacing)
    ys = np.arange(y0, maxy + spacing, spacing)
    if len(xs) < 2 or len(ys) < 2:
        return pts

    gx, gy = np.meshgrid(xs, ys)
    grid_pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(gx.ravel(), gy.ravel())],
        crs=target_crs,
    )
    from shapely.prepared import prep

    prepared = prep(boundary_polygon)
    inside_mask = np.array([prepared.contains(p) for p in grid_pts.geometry])
    grid_pts = grid_pts[inside_mask].reset_index(drop=True)
    if len(grid_pts) == 0:
        return pts

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        nodata = src.nodata
        sample_gdf = grid_pts
        if dem_crs is not None and not grid_pts.crs.equals(dem_crs):
            sample_gdf = grid_pts.to_crs(dem_crs)

        coords = [(p.x, p.y) for p in sample_gdf.geometry]
        elev = np.array([v[0] for v in src.sample(coords)], dtype=float)

    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)

    # 規則格子に戻して勾配を評価
    ncol = len(xs)
    nrow = len(ys)
    z_grid = np.full((nrow, ncol), np.nan)
    # grid_pts の順序は meshgrid.ravel() のうち inside のもの → 再マップ
    full_inside = inside_mask.reshape(nrow, ncol)
    z_grid[full_inside] = elev

    dzdy, dzdx = np.gradient(z_grid, spacing, spacing)
    slope = np.hypot(dzdx, dzdy)

    refine_mask = full_inside & np.isfinite(slope) & (slope > slope_threshold)
    rows, cols = np.where(refine_mask)
    for r, c in zip(rows, cols):
        pts.append(Point(xs[c], ys[r]))
    return pts


def find_refine_triangles(
    triangles_gdf: gpd.GeoDataFrame,
    z_range_threshold: float,
    z_std_threshold: float,
    edge_max_for_refine: float,
    z_range_for_long_edge: float,
) -> np.ndarray:
    """標高分布が大きく再分割すべき三角形の bool マスクを返す。"""
    z_range = triangles_gdf.get("z_range")
    z_std = triangles_gdf.get("z_std")
    edge_max = triangles_gdf.get("edge_max")

    n = len(triangles_gdf)
    mask = np.zeros(n, dtype=bool)
    if z_range is not None:
        mask |= (z_range.to_numpy() > z_range_threshold)
    if z_std is not None:
        mask |= (z_std.to_numpy() > z_std_threshold)
    if edge_max is not None and z_range is not None:
        mask |= (
            (edge_max.to_numpy() > edge_max_for_refine)
            & (z_range.to_numpy() > z_range_for_long_edge)
        )
    mask = mask & np.isfinite(triangles_gdf.get("z_range").to_numpy())
    return mask
