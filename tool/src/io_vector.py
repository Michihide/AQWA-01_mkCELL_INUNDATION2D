"""ベクタ入出力。

拘束ブレークライン（メッシュ辺に一致させる線）と参照レイヤ（OSM 等、位置の目安
にしか使わない線）を別の型で返し、参照レイヤが幾何処理へ流れ込まないようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .config import Config
from .utils import get_logger


class VectorInputError(ValueError):
    """ベクタ入力が不正。"""


@dataclass
class ConstraintBreaklines:
    """メッシュ辺として拘束する線。`feature_preprocessing` / `gmsh_geometry` が扱う。"""

    layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return all(len(gdf) == 0 for gdf in self.layers.values()) if self.layers else True

    def total_length(self) -> float:
        return sum(float(gdf.geometry.length.sum()) for gdf in self.layers.values())


@dataclass
class ReferenceLayers:
    """位置の目安としてのみ使うレイヤ。幾何処理へは渡さない。"""

    layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return all(len(gdf) == 0 for gdf in self.layers.values()) if self.layers else True


def assert_metric_crs(crs: CRS, context: str) -> None:
    """投影座標系かつ単位が m であることを検証する。"""
    if crs is None:
        raise VectorInputError(f"{context}: CRS が未設定です")
    if crs.is_geographic:
        raise VectorInputError(f"{context}: 地理座標系 ({crs.to_string()}) では面積を評価できません")
    axis_units = {ai.unit_name for ai in crs.axis_info}
    if not axis_units <= {"metre", "meter", "m"}:
        raise VectorInputError(f"{context}: 単位が m ではありません ({sorted(axis_units)})")


def _explode_polygons(gdf: gpd.GeoDataFrame) -> list[Polygon]:
    polys: list[Polygon] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiPolygon):
            polys.extend(g for g in geom.geoms if isinstance(g, Polygon))
        elif isinstance(geom, Polygon):
            polys.append(geom)
        else:
            raise VectorInputError(
                f"解析領域に Polygon 以外のジオメトリが含まれます: {geom.geom_type}"
            )
    return polys


def domain_polygon_indices(cfg: Config, *, min_area_m2: float | None = None) -> list[int]:
    """load_domain 後の polygon_filter 用インデックス (0..n-1)。"""
    path = cfg.resolve(cfg.input.domain)
    if path is None or not path.exists():
        raise VectorInputError(f"解析領域ファイルが見つかりません: {cfg.input.domain}")

    gdf = gpd.read_file(path, layer=cfg.input.domain_layer)
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise VectorInputError(f"解析領域が空です: {path}")

    src_crs = CRS.from_user_input(gdf.crs) if gdf.crs else None
    if src_crs is None:
        raise VectorInputError(f"{path}: CRS が未設定です")

    target = CRS.from_epsg(cfg.crs.target_epsg)
    assert_metric_crs(target, "crs.target_epsg")
    if not src_crs.equals(target):
        gdf = gdf.to_crs(target)

    polys = _explode_polygons(gdf)
    threshold = cfg.input.domain_min_area if min_area_m2 is None else min_area_m2
    n_valid = sum(1 for p in polys if p.area >= threshold)
    return list(range(n_valid))


def load_domain(cfg: Config) -> tuple[list[Polygon], CRS]:
    """解析領域ポリゴンを target_epsg へ変換して返す。

    Returns:
        (polygons, crs) — polygons は domain_min_area 以上のものをファイル順で保持。
    """
    logger = get_logger()
    path = cfg.resolve(cfg.input.domain)
    if path is None or not path.exists():
        raise VectorInputError(f"解析領域ファイルが見つかりません: {cfg.input.domain}")

    gdf = gpd.read_file(path, layer=cfg.input.domain_layer)
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise VectorInputError(f"解析領域が空です: {path}")

    src_crs = CRS.from_user_input(gdf.crs) if gdf.crs else None
    if src_crs is None:
        raise VectorInputError(f"{path}: CRS が未設定です")

    target = CRS.from_epsg(cfg.crs.target_epsg)
    assert_metric_crs(target, "crs.target_epsg")

    if not src_crs.equals(target):
        logger.info("解析領域を %s -> EPSG:%d へ変換", src_crs.to_string(), cfg.crs.target_epsg)
        gdf = gdf.to_crs(target)

    polys = _explode_polygons(gdf)
    n_all = len(polys)
    polys = [p for p in polys if p.area >= cfg.input.domain_min_area]
    logger.info(
        "解析領域ポリゴン: %d 個中 %d 個が面積 >= %.0f m^2",
        n_all, len(polys), cfg.input.domain_min_area,
    )

    if cfg.input.polygon_filter is not None:
        n_before = len(polys)
        try:
            polys = [polys[i] for i in cfg.input.polygon_filter]
        except IndexError as exc:
            raise VectorInputError(
                f"input.polygon_filter={cfg.input.polygon_filter} が範囲外です"
                f" (有効ポリゴン数 {n_before})"
            ) from exc
        logger.info("polygon_filter により %d / %d ポリゴンに絞り込み", len(polys), n_before)

    if not polys:
        raise VectorInputError("有効な解析領域ポリゴンがありません")

    total_km2 = sum(p.area for p in polys) / 1e6
    logger.info("対象面積 %.2f km^2 (EPSG:%d)", total_km2, cfg.crs.target_epsg)
    return polys, target


def _load_lines(path: Path, target: CRS, clip_to: BaseGeometry | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.crs is None:
        raise VectorInputError(f"{path}: CRS が未設定です")
    if not CRS.from_user_input(gdf.crs).equals(target):
        gdf = gdf.to_crs(target)
    if clip_to is not None and not gdf.empty:
        gdf = gpd.clip(gdf, clip_to)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf.reset_index(drop=True)


def merge_constraint_breaklines(
    *parts: ConstraintBreaklines,
) -> ConstraintBreaklines:
    """同じレイヤ名の行を連結して一つの拘束集合にする。"""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for part in parts:
        for name, gdf in part.layers.items():
            if gdf is None or gdf.empty:
                continue
            if name in layers:
                layers[name] = pd.concat([layers[name], gdf], ignore_index=True)
            else:
                layers[name] = gdf.copy()
    return ConstraintBreaklines(layers=layers)


def load_constraint_breaklines(
    cfg: Config, target: CRS, clip_to: BaseGeometry | None = None
) -> ConstraintBreaklines:
    """拘束ブレークラインを読み込む。参照レイヤはここには入らない。"""
    logger = get_logger()
    layers: dict[str, gpd.GeoDataFrame] = {}
    for name, rel in cfg.input.breaklines.as_mapping().items():
        path = cfg.resolve(rel)
        if path is None or not path.exists():
            raise VectorInputError(f"拘束ブレークライン '{name}' が見つかりません: {rel}")
        gdf = _load_lines(path, target, clip_to)
        layers[name] = gdf
        logger.info("拘束ブレークライン %s: %d 本 (総延長 %.1f km)",
                    name, len(gdf), float(gdf.geometry.length.sum()) / 1000.0)

    if not layers:
        logger.info("拘束ブレークラインは未指定。地形適合の細分化のみで格子を決定します")
    return ConstraintBreaklines(layers=layers)


def load_reference_layers(
    cfg: Config, target: CRS, clip_to: BaseGeometry | None = None
) -> ReferenceLayers:
    """参照レイヤ（OSM 等）を読み込む。位置の目安としてのみ使う。"""
    logger = get_logger()
    layers: dict[str, gpd.GeoDataFrame] = {}
    for name, rel in cfg.input.reference_layers.as_mapping().items():
        path = cfg.resolve(rel)
        if path is None or not path.exists():
            logger.warning("参照レイヤ '%s' が見つかりません（スキップ）: %s", name, rel)
            continue
        try:
            gdf = _load_lines(path, target, clip_to)
        except Exception as exc:  # 参照レイヤの失敗で本処理を止めない
            logger.warning("参照レイヤ '%s' の読み込みに失敗（スキップ）: %s", name, exc)
            continue
        layers[name] = gdf
        logger.info("参照レイヤ %s: %d 件（位置の目安としてのみ使用）", name, len(gdf))
    return ReferenceLayers(layers=layers)


def write_geopackage(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        get_logger().debug("空のため GeoPackage を書き出しません: %s (%s)", path, layer)
        return
    gdf.to_file(path, layer=layer, driver="GPKG")
