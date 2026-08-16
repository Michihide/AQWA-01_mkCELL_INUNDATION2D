from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from src.zonal_stats import (
    _pixel_area_m2,
    building_ratio_perimeter,
    raster_code_areas,
)


def test_pixel_area_projected_is_plain_unit_conversion():
    crs = CRS.from_epsg(6670)  # メートル単位の投影座標系
    area = _pixel_area_m2(crs, (5.0, 5.0), 0.0, 0.0)
    assert area == pytest.approx(25.0)


def test_pixel_area_geographic_uses_geodesic_area():
    # EPSG:4326 で 0.001 度四方は、赤道付近でおよそ 111 m 四方 ~ 12321 m^2
    crs = CRS.from_epsg(4326)
    area = _pixel_area_m2(crs, (0.001, 0.001), 0.0, 0.0)
    assert area == pytest.approx(111_320.0 ** 2 * 1e-6, rel=0.05)


def _write_geographic_raster(path, values: np.ndarray, res: float, origin=(130.0, 33.0)):
    transform = from_origin(origin[0], origin[1], res, res)
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype=values.dtype, crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(values, 1)


def test_raster_code_areas_geographic_raster(tmp_path):
    # 4x4 のラスタ。左半分がコード 10、右半分がコード 20。
    res = 0.001
    values = np.zeros((4, 4), dtype=np.uint8)
    values[:, :2] = 10
    values[:, 2:] = 20
    raster_path = tmp_path / "landuse.tif"
    _write_geographic_raster(raster_path, values, res)

    # ラスタ全体を覆うポリゴン（メッシュ CRS = ラスタと同じ EPSG:4326 を使い、
    # 単位変換をテストの本質から切り離す）。
    origin_x, origin_y = 130.0, 33.0
    poly = box(origin_x, origin_y - 4 * res, origin_x + 4 * res, origin_y)
    mesh_crs = CRS.from_epsg(4326)

    areas = raster_code_areas([poly], mesh_crs, raster_path, ["10", "20"])
    assert areas.shape == (1, 2)
    # 左右で半分ずつのコードなので、ほぼ同じ面積になる
    assert areas[0, 0] == pytest.approx(areas[0, 1], rel=0.05)
    assert areas[0, 0] > 0.0

    # ラスタと重ならないポリゴンは 0 埋め
    far_poly = box(0.0, 0.0, 0.001, 0.001)
    areas_far = raster_code_areas([far_poly], mesh_crs, raster_path, ["10", "20"])
    assert (areas_far == 0.0).all()


def test_building_ratio_perimeter_covers_half_the_cell():
    cell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    building = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])  # 左半分を覆う建物
    buildings = gpd.GeoDataFrame({"geometry": [building]}, crs="EPSG:6670")
    areas = np.array([100.0])

    ratio, perimeter = building_ratio_perimeter([cell], areas, buildings)
    assert ratio[0] == pytest.approx(0.5)
    assert perimeter[0] == pytest.approx(30.0)  # 5+10+5+10


def test_building_ratio_perimeter_without_buildings_is_zero():
    cell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    ratio, perimeter = building_ratio_perimeter([cell], np.array([100.0]), None)
    assert ratio[0] == 0.0
    assert perimeter[0] == 0.0


def test_building_ratio_is_capped_at_ratio_max():
    cell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    building = Polygon([(-5, -5), (15, -5), (15, 15), (-5, 15)])  # セル全体を覆う
    buildings = gpd.GeoDataFrame({"geometry": [building]}, crs="EPSG:6670")
    ratio, _ = building_ratio_perimeter([cell], np.array([100.0]), buildings, ratio_max=0.95)
    assert ratio[0] == pytest.approx(0.95)
