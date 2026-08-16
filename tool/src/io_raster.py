"""DEM の読み込み。

解析領域の bbox に対応するウィンドウだけを読み、作業座標系（target_epsg）の
規則格子へ再サンプリングして保持する。以降の平面フィット・代表標高は、この
メートル系の格子上で評価する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

from .config import Config
from .utils import get_logger


class RasterInputError(ValueError):
    """DEM 入力が不正。"""


@dataclass
class DemGrid:
    """作業座標系上の規則格子 DEM。

    値は `array[row, col]`。row は y の降順（north-up）。
    """

    array: np.ndarray  # (H, W) float32、NoData は nan
    x_min: float
    y_max: float
    px: float  # ピクセルサイズ [m]（x, y 共通）
    crs: CRS

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.x_min,
            self.y_max - self.height * self.px,
            self.x_min + self.width * self.px,
            self.y_max,
        )

    def xy_to_rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((np.asarray(x) - self.x_min) / self.px).astype(np.int64)
        row = np.floor((self.y_max - np.asarray(y)) / self.px).astype(np.int64)
        return row, col

    def rowcol_to_xy(self, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ピクセル中心の座標を返す。"""
        x = self.x_min + (np.asarray(col) + 0.5) * self.px
        y = self.y_max - (np.asarray(row) + 0.5) * self.px
        return x, y

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """最近傍サンプリング。範囲外は nan。"""
        row, col = self.xy_to_rowcol(x, y)
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        out = np.full(row.shape, np.nan, dtype=np.float64)
        out[inside] = self.array[row[inside], col[inside]]
        return out

    def window_indices(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        """bbox を覆う (row0, row1, col0, col1)。row1/col1 は排他的。"""
        min_x, min_y, max_x, max_y = bounds
        col0 = max(0, int(np.floor((min_x - self.x_min) / self.px)))
        col1 = min(self.width, int(np.ceil((max_x - self.x_min) / self.px)) + 1)
        row0 = max(0, int(np.floor((self.y_max - max_y) / self.px)))
        row1 = min(self.height, int(np.ceil((self.y_max - min_y) / self.px)) + 1)
        return row0, row1, col0, col1


def load_dem(cfg: Config, target: CRS, bounds: tuple[float, float, float, float],
             buffer_m: float = 500.0) -> DemGrid:
    """解析領域 bbox（作業座標系）に対応する DEM を読み込み、target CRS へ再投影する。"""
    logger = get_logger()
    path = cfg.resolve(cfg.input.dem)
    if path is None or not path.exists():
        raise RasterInputError(f"DEM が見つかりません: {cfg.input.dem}")

    min_x, min_y, max_x, max_y = bounds
    min_x -= buffer_m
    min_y -= buffer_m
    max_x += buffer_m
    max_y += buffer_m

    with rasterio.open(path) as src:
        if src.crs is None:
            raise RasterInputError(f"{path}: DEM の CRS が未設定です")
        src_crs = CRS.from_user_input(src.crs)

        # 作業座標系の bbox を DEM の CRS へ変換してウィンドウを決める
        to_src = Transformer.from_crs(target, src_crs, always_xy=True)
        xs = [min_x, max_x, min_x, max_x]
        ys = [min_y, min_y, max_y, max_y]
        sx, sy = to_src.transform(xs, ys)
        win = from_bounds(min(sx), min(sy), max(sx), max(sy), transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            raise RasterInputError("DEM と解析領域が重なりません")

        data = src.read(1, window=win, masked=True)
        # nodata が宣言されていない DEM がある（球磨川の 5 m DEM は -9999 が素通り
        # する）。標高としてあり得ない値は無効として落とす。
        data = np.ma.masked_outside(data, -1000.0, 9000.0)
        src_transform = src.window_transform(win)
        src_bounds = rasterio.windows.bounds(win, src.transform)

        # 出力解像度: DEM の原解像度をメートル換算した値（設定があればそれを使う）
        target_px = cfg.terrain.dem_target_px_m
        if target_px <= 0:
            target_px = _estimate_pixel_size_m(src_crs, target, src_bounds, win)
        target_px = max(target_px, 0.5)

        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, target, win.width, win.height, *src_bounds, resolution=target_px,
        )
        dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(
            source=data.filled(np.nan).astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    grid = DemGrid(
        array=dst,
        x_min=dst_transform.c,
        y_max=dst_transform.f,
        px=float(abs(dst_transform.a)),
        crs=target,
    )
    valid = np.isfinite(dst)
    logger.info(
        "DEM 読込: %d x %d px, %.2f m/px, 有効率 %.1f%%, 標高 %.1f - %.1f m",
        grid.height, grid.width, grid.px, 100.0 * valid.mean(),
        float(np.nanmin(dst)) if valid.any() else float("nan"),
        float(np.nanmax(dst)) if valid.any() else float("nan"),
    )
    return grid


def _estimate_pixel_size_m(
    src_crs: CRS, target: CRS, src_bounds: tuple[float, float, float, float], win
) -> float:
    """DEM 1 ピクセルの作業座標系での大きさ [m] を推定する。"""
    to_target = Transformer.from_crs(src_crs, target, always_xy=True)
    w, s, e, n = src_bounds
    (x0, x1), (y0, y1) = to_target.transform([w, e], [s, n])
    width_m = abs(x1 - x0)
    height_m = abs(y1 - y0)
    return min(width_m / max(win.width, 1), height_m / max(win.height, 1))
