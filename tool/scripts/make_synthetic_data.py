#!/usr/bin/env python3
"""単体テスト用の合成データを生成する。

実データ（球磨川）は開発の主対象だが、単体テストでは決定的で小さい入力が要る。
矩形・L 字・鋭角コーナー付きの領域と、傾斜＋段差を持つ合成 DEM を作る。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

EPSG = 6670
ORIGIN_X = -20000.0
ORIGIN_Y = -90000.0
# 合成 DEM の堤防の位置と諸元。テストからも参照する。
RIDGE_Y = ORIGIN_Y + 400.0
RIDGE_HEIGHT = 3.0
RIDGE_HALF_WIDTH = 20.0


def rectangle(width: float = 1200.0, height: float = 800.0) -> Polygon:
    return Polygon([
        (ORIGIN_X, ORIGIN_Y),
        (ORIGIN_X + width, ORIGIN_Y),
        (ORIGIN_X + width, ORIGIN_Y + height),
        (ORIGIN_X, ORIGIN_Y + height),
    ])


def l_shape() -> Polygon:
    return Polygon([
        (ORIGIN_X, ORIGIN_Y),
        (ORIGIN_X + 1400, ORIGIN_Y),
        (ORIGIN_X + 1400, ORIGIN_Y + 500),
        (ORIGIN_X + 600, ORIGIN_Y + 500),
        (ORIGIN_X + 600, ORIGIN_Y + 1200),
        (ORIGIN_X, ORIGIN_Y + 1200),
    ])


def sharp_corner() -> Polygon:
    """鋭角コーナーを含む領域。コーナー統合処理の検証に使う。"""
    return Polygon([
        (ORIGIN_X, ORIGIN_Y),
        (ORIGIN_X + 1500, ORIGIN_Y + 60),
        (ORIGIN_X + 1500, ORIGIN_Y + 700),
        (ORIGIN_X + 200, ORIGIN_Y + 700),
    ])


def with_hole() -> Polygon:
    outer = rectangle(1600.0, 1200.0).exterior
    hole = [
        (ORIGIN_X + 600, ORIGIN_Y + 450),
        (ORIGIN_X + 1000, ORIGIN_Y + 450),
        (ORIGIN_X + 1000, ORIGIN_Y + 750),
        (ORIGIN_X + 600, ORIGIN_Y + 750),
    ]
    return Polygon(outer, [hole[::-1]])


def synthetic_dem(
    bounds: tuple[float, float, float, float], px: float = 5.0
) -> tuple[np.ndarray, rasterio.Affine]:
    """一定勾配＋堤防状の段差＋緩やかな起伏を持つ DEM。"""
    min_x, min_y, max_x, max_y = bounds
    nx = int(np.ceil((max_x - min_x) / px))
    ny = int(np.ceil((max_y - min_y) / px))
    x = min_x + (np.arange(nx) + 0.5) * px
    y = max_y - (np.arange(ny) + 0.5) * px
    xx, yy = np.meshgrid(x, y)

    z = 100.0 + 0.002 * (xx - min_x) + 0.001 * (yy - min_y)
    z += RIDGE_HEIGHT * np.exp(-(((yy - RIDGE_Y) / RIDGE_HALF_WIDTH) ** 2))
    # 緩やかな起伏
    z += 0.4 * np.sin((xx - min_x) / 150.0) * np.cos((yy - min_y) / 180.0)
    return z.astype(np.float32), from_origin(min_x, max_y, px, px)


def write_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    polys = {
        "rectangle": rectangle(),
        "l_shape": l_shape(),
        "sharp_corner": sharp_corner(),
        "with_hole": with_hole(),
    }
    gdf = gpd.GeoDataFrame(
        {"name": list(polys)}, geometry=list(polys.values()), crs=f"EPSG:{EPSG}"
    )
    gdf.to_file(out_dir / "synthetic_domain.gpkg", layer="domain", driver="GPKG")

    # 拘束ブレークラインの例。堤防天端に沿う直線。
    lines = gpd.GeoDataFrame(
        {"kind": ["levee"]},
        geometry=[LineString([(ORIGIN_X + 50, RIDGE_Y), (ORIGIN_X + 1150, RIDGE_Y)])],
        crs=f"EPSG:{EPSG}",
    )
    lines.to_file(out_dir / "synthetic_breaklines.gpkg", layer="levees", driver="GPKG")

    bounds = gdf.total_bounds
    pad = 100.0
    z, transform = synthetic_dem(
        (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    )
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": z.shape[0], "width": z.shape[1],
        "transform": transform, "crs": f"EPSG:{EPSG}", "nodata": np.nan,
    }
    with rasterio.open(out_dir / "synthetic_dem.tif", "w", **profile) as dst:
        dst.write(z, 1)

    print(f"合成データを出力しました: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "tests" / "data"),
    )
    write_all(Path(parser.parse_args().out))
