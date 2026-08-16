"""要素ごとの地形指標。

各要素に含まれる DEM ピクセルから、代表標高・平面フィット RMSE・勾配方向の
ばらつきを求める。

要素数が 20 万規模になるため、要素ごとにループせず、DEM 格子を一度だけ要素番号で
ラスタライズし、以降はすべて np.bincount による区分和で計算する。平面フィットも
和の組み合わせから正規方程式を組んで一括で解く。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin

from .config import Config
from .io_raster import DemGrid
from .mesh_parser import Mesh
from .utils import get_logger


@dataclass
class TerrainReport:
    """要素ごとの地形指標。三角形 -> 四角形の順。"""

    n_samples: np.ndarray
    elevation: np.ndarray
    plane_fit_rmse: np.ndarray
    slope_magnitude: np.ndarray
    slope_direction_spread_deg: np.ndarray

    def __len__(self) -> int:
        return len(self.elevation)


def mesh_polygons(mesh: Mesh) -> np.ndarray:
    """メッシュ要素を shapely ポリゴンの配列にする。三角形 -> 四角形の順。"""
    coords = mesh.element_polygons()
    tri_n = len(mesh.triangles)
    geoms = []
    if tri_n:
        geoms.append(shapely.polygons(np.asarray(coords[:tri_n])))
    if len(mesh.quads):
        geoms.append(shapely.polygons(np.asarray(coords[tri_n:])))
    return np.concatenate(geoms) if geoms else np.empty(0, dtype=object)


def rasterize_polygons(geoms: np.ndarray, dem: DemGrid) -> np.ndarray:
    """DEM 格子の各ピクセルがどの要素に属するかを返す。

    値は要素番号 + 1。どの要素にも属さないピクセルは 0。ピクセル中心が要素内に
    あるかで判定する（rasterio の既定と同じ）。
    """
    transform = from_origin(dem.x_min, dem.y_max, dem.px, dem.px)
    return rasterize(
        ((g, i + 1) for i, g in enumerate(geoms)),
        out_shape=dem.array.shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )


def _segment_sums(labels: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(labels, weights=values, minlength=n)[:n]


def compute_terrain_metrics(mesh: Mesh, dem: DemGrid, cfg: Config) -> TerrainReport:
    """全要素の地形指標を計算する。"""
    return compute_polygon_terrain_metrics(
        mesh_polygons(mesh), mesh.element_centroids(), dem, cfg
    )


def compute_polygon_terrain_metrics(
    geoms: np.ndarray, centroids: np.ndarray, dem: DemGrid, cfg: Config
) -> TerrainReport:
    """任意のポリゴン集合に対して地形指標を計算する。

    自前のメッシュだけでなく、既存メッシュ（face.gpkg など）との比較にも使う。
    """
    logger = get_logger()
    t = cfg.terrain
    n = len(geoms)

    labels = rasterize_polygons(geoms, dem)
    z_all = dem.array.astype(np.float64)
    valid = (labels > 0) & np.isfinite(z_all)

    idx = labels[valid].astype(np.int64) - 1
    z = z_all[valid]

    rows, cols = np.nonzero(valid)
    px, py = dem.rowcol_to_xy(rows, cols)

    counts = np.bincount(idx, minlength=n)[:n].astype(np.int64)
    has_data = counts > 0
    safe_n = np.where(has_data, counts, 1).astype(np.float64)

    # 要素重心を原点にした局所座標で正規方程式を組む（条件数の悪化を避ける）
    dx = px - centroids[idx, 0]
    dy = py - centroids[idx, 1]

    s_z = _segment_sums(idx, z, n)
    s_x = _segment_sums(idx, dx, n)
    s_y = _segment_sums(idx, dy, n)
    s_xx = _segment_sums(idx, dx * dx, n)
    s_yy = _segment_sums(idx, dy * dy, n)
    s_xy = _segment_sums(idx, dx * dy, n)
    s_xz = _segment_sums(idx, dx * z, n)
    s_yz = _segment_sums(idx, dy * z, n)
    s_zz = _segment_sums(idx, z * z, n)

    mat = np.empty((n, 3, 3))
    mat[:, 0, 0] = s_xx
    mat[:, 0, 1] = mat[:, 1, 0] = s_xy
    mat[:, 0, 2] = mat[:, 2, 0] = s_x
    mat[:, 1, 1] = s_yy
    mat[:, 1, 2] = mat[:, 2, 1] = s_y
    mat[:, 2, 2] = counts
    rhs = np.column_stack([s_xz, s_yz, s_z])

    # サンプルが 3 点未満、または退化している要素は平面を決められない
    det = np.linalg.det(mat)
    scale = np.maximum(s_xx * s_yy, 1e-12)
    solvable = has_data & (counts >= 3) & (np.abs(det) > 1e-9 * scale * np.maximum(counts, 1))

    coef = np.zeros((n, 3))
    if solvable.any():
        # numpy 2 では b が (K, 3) だと行列扱いになるため、明示的に列ベクトルにする
        coef[solvable] = np.linalg.solve(mat[solvable], rhs[solvable][..., None])[..., 0]
    coef[~solvable, 2] = np.where(has_data, s_z / safe_n, 0.0)[~solvable]

    a, b, c = coef[:, 0], coef[:, 1], coef[:, 2]
    # RMSE^2 = (1/n) * sum (z - (a dx + b dy + c))^2 を各種の和から展開して求める
    resid_sq = (
        s_zz
        - 2.0 * (a * s_xz + b * s_yz + c * s_z)
        + a * a * s_xx + b * b * s_yy + counts * c * c
        + 2.0 * (a * b * s_xy + a * c * s_x + b * c * s_y)
    )
    rmse = np.sqrt(np.maximum(resid_sq, 0.0) / safe_n)
    rmse[~has_data] = 0.0
    slope_mag = np.where(solvable, np.hypot(a, b), 0.0)

    elevation = _representative_elevation(
        t.representative_elevation_method, centroids, dem, idx, z, s_z, safe_n, has_data, coef, n
    )

    spread = _slope_direction_spread(dem, valid, idx, n, t.flat_slope_threshold, counts)

    n_no_sample = int((~has_data).sum())
    if n_no_sample:
        logger.warning(
            "DEM サンプルが得られない要素が %d 個あります（重心の値で代用）", n_no_sample
        )

    return TerrainReport(
        n_samples=counts,
        elevation=elevation,
        plane_fit_rmse=np.nan_to_num(rmse, nan=0.0),
        slope_magnitude=np.nan_to_num(slope_mag, nan=0.0),
        slope_direction_spread_deg=spread,
    )


def _representative_elevation(
    method: str,
    centroids: np.ndarray,
    dem: DemGrid,
    idx: np.ndarray,
    z: np.ndarray,
    s_z: np.ndarray,
    safe_n: np.ndarray,
    has_data: np.ndarray,
    coef: np.ndarray,
    n: int,
) -> np.ndarray:
    if method == "median":
        elevation = np.full(n, np.nan)
        order = np.argsort(idx, kind="stable")
        sorted_idx, sorted_z = idx[order], z[order]
        starts = np.searchsorted(sorted_idx, np.arange(n), side="left")
        ends = np.searchsorted(sorted_idx, np.arange(n), side="right")
        for i in np.flatnonzero(has_data):
            elevation[i] = np.median(sorted_z[starts[i]:ends[i]])
    elif method == "centroid_sample":
        elevation = dem.sample(centroids[:, 0], centroids[:, 1])
    elif method == "fitted_plane_at_centroid":
        # 局所座標の原点が重心なので、切片がそのまま重心での値になる
        elevation = coef[:, 2].copy()
    else:  # area_weighted_mean（ピクセル面積は一定なので単純平均と同じ）
        elevation = s_z / safe_n

    fallback = ~has_data if method != "centroid_sample" else ~np.isfinite(elevation)
    if fallback.any():
        sampled = dem.sample(centroids[fallback, 0], centroids[fallback, 1])
        elevation[fallback] = sampled
    return elevation


def _slope_direction_spread(
    dem: DemGrid,
    valid: np.ndarray,
    idx: np.ndarray,
    n: int,
    flat_threshold: float,
    counts: np.ndarray,
) -> np.ndarray:
    """勾配方向の円周標準偏差 [deg]。勾配の大きさで重み付けする。"""
    # x は列方向で東正、y は行番号の減少方向が北
    gy, gx = np.gradient(dem.array.astype(np.float64), dem.px)
    gy = -gy

    ex = gx[valid]
    ey = gy[valid]
    mag = np.hypot(ex, ey)
    steep = np.isfinite(mag) & (mag > flat_threshold)
    if not steep.any():
        return np.zeros(n)

    sub_idx = idx[steep]
    w = mag[steep]
    angle = np.arctan2(ey[steep], ex[steep])

    w_sum = _segment_sums(sub_idx, w, n)
    c_sum = _segment_sums(sub_idx, w * np.cos(angle), n)
    s_sum = _segment_sums(sub_idx, w * np.sin(angle), n)
    n_steep = np.bincount(sub_idx, minlength=n)[:n]

    spread = np.zeros(n)
    ok = (w_sum > 0.0) & (n_steep >= 2)
    r = np.zeros(n)
    r[ok] = np.clip(np.hypot(c_sum[ok], s_sum[ok]) / w_sum[ok], 1e-12, 1.0)
    spread[ok] = np.degrees(np.sqrt(-2.0 * np.log(r[ok])))
    return spread


def terrain_violations(report: TerrainReport, cfg: Config) -> dict[str, np.ndarray]:
    """地形基準を満たさない要素のブールマスク。"""
    t = cfg.terrain
    flat = report.slope_magnitude <= t.flat_slope_threshold
    out = {
        "plane_fit_rmse": report.plane_fit_rmse > t.plane_fit_rmse_max,
        # 平坦な要素では勾配方向に意味が無いため判定から外す
        "slope_direction_spread": (~flat)
        & (report.slope_direction_spread_deg > t.slope_direction_spread_max_deg),
    }
    if t.refine_if_insufficient_dem_samples:
        out["insufficient_dem_samples"] = report.n_samples < t.minimum_dem_samples_per_element
    return out


def summarize_terrain(
    report: TerrainReport, cfg: Config, area: np.ndarray | None = None
) -> dict[str, float]:
    v = terrain_violations(report, cfg)
    n = max(len(report), 1)
    out = {
        "dem_samples_min": int(report.n_samples.min()) if len(report) else 0,
        "dem_samples_median": float(np.median(report.n_samples)) if len(report) else 0.0,
        "plane_fit_rmse_median": float(np.median(report.plane_fit_rmse)) if len(report) else 0.0,
        "plane_fit_rmse_p95": float(np.percentile(report.plane_fit_rmse, 95)) if len(report) else 0.0,
        "plane_fit_rmse_max": float(report.plane_fit_rmse.max()) if len(report) else 0.0,
        "elevation_min": float(np.nanmin(report.elevation)) if len(report) else float("nan"),
        "elevation_max": float(np.nanmax(report.elevation)) if len(report) else float("nan"),
    }
    if area is not None and len(report):
        # 面積で重み付けした地形表現誤差。細分化の費用対効果を見るための指標。
        total = float(area.sum())
        out["plane_fit_rmse_area_weighted"] = float((report.plane_fit_rmse * area).sum() / total)
        out["slope_spread_area_weighted"] = float(
            (report.slope_direction_spread_deg * area).sum() / total
        )
        viol_any = np.logical_or.reduce(list(v.values())) if v else np.zeros(n, dtype=bool)
        out["terrain_viol_area_ratio"] = float(area[viol_any].sum() / total)

    for name, mask in v.items():
        out[f"terrain_viol_{name}"] = int(mask.sum())
    out["terrain_viol_any"] = int(np.logical_or.reduce(list(v.values())).sum()) if v else 0
    out["terrain_viol_any_ratio"] = out["terrain_viol_any"] / n
    return out
