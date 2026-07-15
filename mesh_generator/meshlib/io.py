"""GIS ベクタ/ラスタ データの入出力ユーティリティ。"""

from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import make_valid

# pyogrio があれば高速 I/O を利用
try:
    import pyogrio  # noqa: F401
    _ENGINE = "pyogrio"
except ImportError:  # pragma: no cover
    _ENGINE = "fiona"


def _is_none_path(path: Optional[str]) -> bool:
    if path is None:
        return True
    return str(path).strip().lower() in {"", "none", "null"}


def get_file_crs(path: str):
    """ベクタファイルの CRS を取得する（読み込まずにメタ情報のみ）。"""
    try:
        import pyogrio
        info = pyogrio.read_info(path)
        return info.get("crs")
    except Exception:
        # フォールバック: 先頭のみ読む
        head = gpd.read_file(path, rows=1)
        return head.crs


def read_vector(
    path: Optional[str],
    bbox: Optional[tuple] = None,
) -> Optional[gpd.GeoDataFrame]:
    """ベクタファイル（gpkg/shp/geojson）を読み込む。

    path が None / "none" / 存在しない場合は None を返す（任意入力対応）。
    bbox（ファイル自身の CRS での minx,miny,maxx,maxy）を指定すると、
    その範囲に交差するフィーチャのみを読み込む（巨大データ対策）。
    無効ジオメトリは make_valid で修復する。
    """
    if _is_none_path(path):
        return None
    if not os.path.exists(path):
        print(f"  警告: ファイルが見つかりません（スキップ）: {path}")
        return None

    try:
        if bbox is not None:
            gdf = gpd.read_file(path, engine=_ENGINE, bbox=tuple(bbox))
        else:
            gdf = gpd.read_file(path, engine=_ENGINE)
    except Exception:
        if bbox is not None:
            gdf = gpd.read_file(path, bbox=tuple(bbox))
        else:
            gdf = gpd.read_file(path)

    if len(gdf) == 0:
        return gdf

    # ジオメトリの妥当性を確保
    gdf = gdf[gdf.geometry.notnull()].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
    return gdf


def write_vector(gdf: gpd.GeoDataFrame, path: str, layer: Optional[str] = None) -> None:
    """GeoDataFrame をファイルに書き出す（拡張子で形式を判別）。"""
    if gdf is None or len(gdf) == 0:
        print(f"  情報: 空のためスキップ: {path}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    driver = None
    lower = path.lower()
    if lower.endswith(".gpkg"):
        driver = "GPKG"
    elif lower.endswith(".shp"):
        driver = "ESRI Shapefile"
    elif lower.endswith(".geojson") or lower.endswith(".json"):
        driver = "GeoJSON"

    kwargs = {}
    if driver:
        kwargs["driver"] = driver
    if layer and driver == "GPKG":
        kwargs["layer"] = layer
    try:
        gdf.to_file(path, engine=_ENGINE, **kwargs)
    except Exception:
        gdf.to_file(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str) -> None:
    """DataFrame を CSV に書き出す。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)


def ensure_dir(path: str) -> str:
    """ディレクトリを作成して返す。"""
    os.makedirs(path, exist_ok=True)
    return path
