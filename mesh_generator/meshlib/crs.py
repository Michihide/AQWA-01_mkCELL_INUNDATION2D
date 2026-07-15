"""座標系（CRS）の推定・変換ユーティリティ。"""

from __future__ import annotations

import math
from typing import Optional

import geopandas as gpd
from pyproj import CRS


def estimate_utm_epsg(lon: float, lat: float) -> int:
    """経度・緯度から適切な UTM ゾーンの EPSG コードを推定する。

    北半球: 326xx, 南半球: 327xx。
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def estimate_target_epsg_from_gdf(gdf: gpd.GeoDataFrame) -> int:
    """GeoDataFrame の重心から UTM EPSG を推定する。

    入力 CRS が地理座標系でない場合でも、一度 EPSG:4326 に変換して重心を求める。
    """
    if gdf.crs is None:
        raise ValueError("CRS が未設定の GeoDataFrame からは UTM を推定できません。")

    g4326 = gdf.to_crs(4326)
    # union_all の重心（全フィーチャの代表点）
    centroid = g4326.geometry.union_all().centroid
    return estimate_utm_epsg(centroid.x, centroid.y)


def ensure_crs(gdf: Optional[gpd.GeoDataFrame], target_epsg: int) -> Optional[gpd.GeoDataFrame]:
    """GeoDataFrame を target_epsg に再投影する（None はそのまま返す）。"""
    if gdf is None or len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        raise ValueError("CRS が未設定のため再投影できません。明示的に CRS を設定してください。")
    current = CRS.from_user_input(gdf.crs)
    target = CRS.from_epsg(target_epsg)
    if current.equals(target):
        return gdf
    return gdf.to_crs(target_epsg)


def resolve_target_epsg(area_gdf: gpd.GeoDataFrame, crs_config: dict) -> int:
    """設定に基づいて出力対象の EPSG を決定する。

    - target_epsg が指定されていればそれを使用。
    - auto_utm が True なら解析領域から UTM を推定。
    - どちらも無ければ、入力が投影座標系ならその EPSG、地理座標系なら UTM 推定。
    """
    target_epsg = crs_config.get("target_epsg")
    if target_epsg:
        return int(target_epsg)

    if crs_config.get("auto_utm", True):
        return estimate_target_epsg_from_gdf(area_gdf)

    current = CRS.from_user_input(area_gdf.crs)
    if current.is_projected and current.to_epsg():
        return int(current.to_epsg())

    # フォールバック: UTM 推定
    return estimate_target_epsg_from_gdf(area_gdf)
