"""動作確認用の合成テストデータ（解析領域・DEM・建物）を生成する。

使い方:
    python make_test_data.py [出力ディレクトリ]
既定の出力先は ./testdata（EPSG:6677 平面直角座標系）。
"""

import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "testdata")
    os.makedirs(out, exist_ok=True)
    epsg = 6677  # 平面直角座標系（メートル）

    # 解析領域: 0..400m の正方形を少し不整形にしたポリゴン
    area = Polygon([(0, 0), (400, 0), (400, 300), (250, 380), (0, 300)])
    gpd.GeoDataFrame(geometry=[area], crs=epsg).to_file(
        os.path.join(out, "analysis_area.gpkg"), driver="GPKG"
    )

    # 建物: いくつかの矩形
    buildings = [
        box(60, 60, 100, 100),
        box(150, 120, 210, 170),
        box(280, 80, 330, 140),
        box(120, 220, 170, 260),
    ]
    gpd.GeoDataFrame(geometry=buildings, crs=epsg).to_file(
        os.path.join(out, "buildings.gpkg"), driver="GPKG"
    )

    # DEM: 1m 解像度。傾斜 + 局所的な起伏（細分化が起きるように）
    res = 1.0
    xmin, ymin, xmax, ymax = -10, -10, 420, 400
    ncol = int((xmax - xmin) / res)
    nrow = int((ymax - ymin) / res)
    xs = xmin + (np.arange(ncol) + 0.5) * res
    ys = ymax - (np.arange(nrow) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    # ゆるい勾配 + 一部に急な起伏（ガウシアン丘）
    z = 0.01 * gx + 0.005 * gy
    z += 3.0 * np.exp(-(((gx - 300) ** 2 + (gy - 250) ** 2) / (2 * 30.0 ** 2)))
    z += 2.0 * np.exp(-(((gx - 100) ** 2 + (gy - 150) ** 2) / (2 * 20.0 ** 2)))
    z = z.astype("float32")

    transform = from_origin(xmin, ymax, res, res)
    with rasterio.open(
        os.path.join(out, "dem.tif"), "w", driver="GTiff",
        height=nrow, width=ncol, count=1, dtype="float32",
        crs=f"EPSG:{epsg}", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(z, 1)

    print(f"テストデータを出力しました: {out}")
    print("  - analysis_area.gpkg, buildings.gpkg, dem.tif")


if __name__ == "__main__":
    main()
