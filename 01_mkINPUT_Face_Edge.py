"""
AQWA-INUNDATION2Dメッシュ作成スクリプト
TACM (Two-dimensional Analysis of Current and Wave) グリッド生成用

このスクリプトは以下の処理を行います：
1. 入力データの読み込みと前処理
2. 道路網とポリゴンの境界線の統合
3. グリッドポリゴン化と面積閾値による分類
4. ヘキサゴングリッドによる大面積ポリゴンの分割
5. 細長いメッシュのクラスタリングによる再分割
6. ダミーメッシュとの重複判定とキャリブレーションメッシュ区分
7. ラスターデータ（DEM、土地利用）からの属性付与
8. 最終出力（face.csv, side.csv）の生成

作成者: AQWA-RISK プロジェクト
"""

import os
os.environ["OMP_NUM_THREADS"] = "8"  # OpenMPの並列処理を1スレッドに制限

# 地理空間データ処理ライブラリ
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, box, mapping, MultiPolygon, shape, Point, GeometryCollection, MultiPoint
from shapely.ops import polygonize, unary_union, linemerge, snap, split, voronoi_diagram, transform as shp_transform
from shapely.validation import make_valid
from shapely.strtree import STRtree
from shapely import force_2d, prepared
from shapely.prepared import prep
from shapely.geometry import box, mapping

# ファイルI/O および空間インデックス
import shapefile
import fiona
from fiona.crs import from_epsg
from rtree import index

# 数値計算・データ処理ライブラリ
import math
from copy import deepcopy
import numpy as np
import pandas as pd

# ラスターデータ処理
import rasterio
import rasterio.features
import rasterio.mask
from rasterio import windows
from rasterio import mask as rio_mask
from rasterio.transform import rowcol
from rasterio.features import geometry_mask

# その他のユーティリティ
from datetime import datetime
from collections import defaultdict
import itertools
from tqdm import tqdm  # 進捗表示用
import ast
from math import radians, acos, degrees
from pyproj import CRS, Geod, Transformer
import csv
from rasterstats import zonal_stats
import networkx as nx
from scipy.spatial import Voronoi
import yaml
import random
from sklearn.cluster import KMeans
from pathlib import Path
import argparse
import sys

# ============================================================================
# 再現性（決定性）の確保
# 同一入力データに対して常に同一の出力（face/edge/bin）を得るため、
# 乱数シードを固定する。これにより generate_random_points 等の
# 確率的処理も実行ごとに同じ結果となる。
# ============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# 初期設定とデータ読み込み
# ============================================================================

# 作業ディレクトリの設定
working_directory = os.getcwd()

# コマンドライン引数の解析
parser = argparse.ArgumentParser(
    description='AQWA-INUNDATION2D メッシュ作成スクリプト',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
使用例:
  python 01_mkINPUT_Face_Side.py yaml/Kinu_Joso.yaml
  python 01_mkINPUT_Face_Side.py config.yaml
    """
)
parser.add_argument(
    'config_file',
    type=str,
    help='設定ファイル（YAMLファイル）のパス（例: yaml/Kinu_Joso.yaml）'
)

args = parser.parse_args()

# 設定ファイルの存在確認
config_path = Path(args.config_file)
if not config_path.exists():
    print(f"エラー: 設定ファイルが見つかりません: {args.config_file}")
    sys.exit(1)

print(f"設定ファイルを読み込み中: {args.config_file}")

# 設定ファイル（YAML）からパラメータを読み込み
with open(args.config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 基本パス設定
base_dir = config["project"]["base_dir"]
project_name = config["project"]["name"]

print(f"プロジェクト名: {project_name}")
print(f"ベースディレクトリ: {base_dir}")

# config内の{project_name}を実際のプロジェクト名に置き換える
def replace_project_name(obj, project_name):
    """再帰的に辞書・リスト内の{project_name}を置き換える"""
    if isinstance(obj, dict):
        return {k: replace_project_name(v, project_name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_project_name(item, project_name) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("{project_name}", project_name)
    else:
        return obj

config = replace_project_name(config, project_name)

# 入力ファイルパスの構築
# 道路データ（"none" の場合は道路処理をスキップしシンプル分割モード）
roads_file = config["input"]["roads"]["file"]
enable_roads = (
    roads_file is not None
    and str(roads_file).strip().lower() not in {"none", "null", ""}
)
if enable_roads:
    OSM_directory = os.path.join(base_dir, config["input"]["roads"]["dir"], roads_file)
else:
    OSM_directory = None
    print("Info: 道路データなし（roads.file=none）。シンプル分割モードを使用します。")
print(f"道路データ: {'OSM' if enable_roads else 'none（シンプル分割）'}")

# 建物データ（オプショナル）
buildings_file = config["input"]["buildings"].get("file")
if buildings_file and buildings_file.lower() not in ['none', 'null', '']:
    BIL_directory = os.path.join(base_dir, config["input"]["buildings"]["dir"], buildings_file)
else:
    BIL_directory = None
    print("Info: 建物データが指定されていません。建物処理をスキップします。")

DEM_directory = os.path.join(base_dir, config["input"]["dem"]["dir"], 
                             config["input"]["dem"]["file"])
LandUse_directory = os.path.join(base_dir, config["input"]["landuse"]["dir"], 
                                 config["input"]["landuse"]["file"])
Soil_directory = os.path.join(base_dir, config["input"]["soil"]["dir"],
                              config["input"]["soil"]["file"])
input_file = os.path.join(base_dir, config["input"]["target_area"]["dir"], 
                         config["input"]["target_area"]["file"])

# ダミーメッシュ（オプショナル）
dummy_mesh_file = config["input"]["dummy_mesh"].get("file")
if dummy_mesh_file and dummy_mesh_file.lower() not in ['none', 'null', '']:
    dummy_file = os.path.join(base_dir, config["input"]["dummy_mesh"]["dir"], dummy_mesh_file)
else:
    dummy_file = None
    print("Info: ダミーメッシュが指定されていません。ダミーメッシュ処理をスキップします。")

# EPSG設定
DEM_epsg = config["input"]["dem"]["epsg"]
LandUse_epsg = config["input"]["landuse"]["epsg"]
Soil_epsg = config["input"]["soil"]["epsg"]

# パラメータ設定
area_threshold = config["parameters"]["area_threshold"]
enable_merge_small_polygons = bool(config["parameters"].get("merge_small_polygons", False))
elevation_stat = config["parameters"].get("elevation_stat", "median")  # デフォルトは中央値
refinement = config["parameters"].get("refinement", {})
primary_refine_method = str(refinement.get("primary_method", "hex")).lower()
secondary_refine_method = str(refinement.get("secondary_method", "none")).lower()
enable_aspect_refinement = bool(refinement.get("enable_aspect", False))

def _parse_auto_or_float(value, default_value):
    """'auto' を許容し、数値なら float に変換する。"""
    if value is None:
        return float(default_value)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return float(value)

def _parse_min_samples(value, default_value):
    """'all' を許容し、数値なら int に変換する。"""
    if value is None:
        return int(default_value)
    if isinstance(value, str) and value.strip().lower() in {"all", "none"}:
        return None
    return int(value)

hex_diameter = _parse_auto_or_float(refinement.get("hex_diameter", 350.0), 350.0)
hex_area_threshold = _parse_auto_or_float(refinement.get("hex_area_threshold", 78660.0), 78660.0)
hex_area_std_multiplier = float(refinement.get("hex_area_std_multiplier", 1.0))
hex_diameter_scale = float(refinement.get("hex_diameter_scale", 1.0))
aspect_ratio_threshold = float(refinement.get("aspect_ratio_threshold", 3.0))
qt_std_threshold = float(refinement.get("qt_std_threshold", 0.5))
qt_relief_threshold = float(refinement.get("qt_relief_threshold", 2.0))
qt_max_depth = int(refinement.get("qt_max_depth", 6))
qt_min_area = float(refinement.get("qt_min_area", 500.0))
_aspect_min_raw = refinement.get("aspect_min_split_area", qt_min_area)
if isinstance(_aspect_min_raw, str) and _aspect_min_raw.strip().lower() == "auto":
    aspect_min_split_area = float(qt_min_area)
else:
    aspect_min_split_area = float(_aspect_min_raw)
aspect_merge_small = bool(refinement.get("aspect_merge_small", False))
qt_min_dem_samples = _parse_min_samples(refinement.get("qt_min_dem_samples", 16), 16)
qt_root_cell_size = _parse_auto_or_float(refinement.get("qt_root_cell_size", "auto"), None)
qt_mesh_mode = str(refinement.get("qt_mesh_mode", "clip")).strip().lower()
qt_domain = str(refinement.get("qt_domain", "target")).strip().lower()
if qt_mesh_mode not in {"clip", "pure"}:
    print(f"警告: qt_mesh_mode='{qt_mesh_mode}' は未対応です。'clip' を使用します。")
    qt_mesh_mode = "clip"
if qt_domain not in {"target", "bbox"}:
    print(f"警告: qt_domain='{qt_domain}' は未対応です。'target' を使用します。")
    qt_domain = "target"

# 分割手法の設定を検証（未知の値は安全側で既定値へフォールバック）
if primary_refine_method not in {"hex", "none", "quadtree_std"}:
    print(f"警告: primary_method='{primary_refine_method}' は未対応です。'hex' を使用します。")
    primary_refine_method = "hex"
if secondary_refine_method not in {"voronoi", "none"}:
    print(f"警告: secondary_method='{secondary_refine_method}' は未対応です。'none' を使用します。")
    secondary_refine_method = "none"
if not enable_aspect_refinement:
    secondary_refine_method = "none"
enable_voronoi_refinement = (secondary_refine_method == "voronoi") and enable_aspect_refinement
if not enable_roads:
    if enable_voronoi_refinement:
        print("Info: 道路なしのため secondary_method を none に変更します（軸平行分割のみ）。")
    enable_voronoi_refinement = False
    secondary_refine_method = "none"

# 標高統計量のバリデーション
if elevation_stat not in ["mean", "median"]:
    print(f"警告: elevation_stat '{elevation_stat}' は無効です。'median' を使用します。")
    elevation_stat = "median"

print(f"標高統計量: {elevation_stat}")
print(f"一次再分割手法: {primary_refine_method}")
print(f"二次再分割手法: {secondary_refine_method}")
print(f"enable_aspect_refinement={enable_aspect_refinement}, merge_small_polygons={enable_merge_small_polygons}")
print(f"hex_diameter={hex_diameter}, hex_area_threshold={hex_area_threshold}")
if primary_refine_method == "quadtree_std":
    print(f"quadtree: std>{qt_std_threshold}, relief>{qt_relief_threshold}, max_depth={qt_max_depth}, min_area={qt_min_area}, min_samples={qt_min_dem_samples}")
    print(f"quadtree mesh_mode={qt_mesh_mode}, domain={qt_domain}")

# 出力フォルダ設定
output_folder = os.path.join(base_dir, config["output"]["dir"])
input_dem = os.path.abspath(os.path.expanduser(DEM_directory))

# 出力フォルダの作成（存在しない場合）
os.makedirs(output_folder, exist_ok=True)
print(f"出力ディレクトリ: {output_folder}")

# 対象領域ポリゴンファイルの読み込み
input_gdf = gpd.read_file(input_file)  # GeoDataFrameとして読み込み

# ============================================================================
# フェーズ1: 基礎ライン作成と道路網の統合 (02-04)
# ============================================================================

def _count_open_lines(line_geom, tol=0.05):
    """閉じていない LineString の本数を数える。"""
    if line_geom is None or line_geom.is_empty:
        return 0
    geoms = line_geom.geoms if hasattr(line_geom, "geoms") else [line_geom]
    n = 0
    for line in geoms:
        if not isinstance(line, LineString) or line.is_empty:
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        if Point(coords[0]).distance(Point(coords[-1])) > tol:
            n += 1
    return n

def _polygon_exterior_lines(geom):
    """Polygon / MultiPolygon から外周線を取り出す。"""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom.exterior]
    if isinstance(geom, MultiPolygon):
        return [p.exterior for p in geom.geoms if not p.is_empty]
    return []

def retopologize_mesh_gdf(gdf, snap_tol=0.05, domain_geom=None, label="メッシュ再トポロジ化"):
    """結合後のポリゴン境界を統合・スナップして閉じたメッシュへ復元する。"""
    if gdf is None or len(gdf) == 0:
        return gdf

    lines = []
    for geom in gdf.geometry:
        lines.extend(_polygon_exterior_lines(geom))
    if not lines:
        return gdf

    merged = unary_union(lines)
    n_open_before = _count_open_lines(merged)
    if snap_tol > 0:
        merged = snap(merged, merged, snap_tol)
        if domain_geom is not None and not domain_geom.is_empty:
            merged = snap(merged, domain_geom.boundary, snap_tol)
    merged = linemerge(unary_union(merged))
    n_open_after = _count_open_lines(merged)

    polys = list(polygonize(merged))
    print(
        f"{label}: {len(gdf)}面 -> {len(polys)}面, "
        f"開線 before={n_open_before} after={n_open_after}",
        flush=True,
    )
    if not polys:
        print(f"警告: {label} でポリゴン化に失敗。元データを使用します。", flush=True)
        return gdf
    return gpd.GeoDataFrame(geometry=polys, crs=gdf.crs).reset_index(drop=True)

def process_line_to_polygons(line_gdf: gpd.GeoDataFrame, snap_tol=0.05) -> gpd.GeoDataFrame:
    """
    ラインセットからポリゴンメッシュを生成し、面積閾値で分類する関数

    Args:
        line_gdf: ラインのGeoDataFrame
        snap_tol: 端点スナップ許容値 [m]

    Returns:
        面積閾値で分類されたポリゴンのGeoDataFrame
    """
    line_merged = unary_union(line_gdf.geometry)
    n_open_before = _count_open_lines(line_merged)
    if snap_tol > 0:
        line_merged = snap(line_merged, line_merged, snap_tol)
        domain_geom = input_gdf.union_all()
        if domain_geom is not None and not domain_geom.is_empty:
            line_merged = snap(line_merged, domain_geom.boundary, snap_tol)
    line_merged = linemerge(unary_union(line_merged))
    n_open_after = _count_open_lines(line_merged)
    if n_open_before or n_open_after:
        print(
            f"process_line_to_polygons: 開線 before={n_open_before} after={n_open_after}",
            flush=True,
        )
    poly = list(polygonize(line_merged))
    return gpd.GeoDataFrame(geometry=poly, crs=line_gdf.crs)

# 02-04: 道路網の統合（roads.file=none の場合はスキップ）
if enable_roads:
    # 02: 対象領域の境界線を作成
    Line02 = gpd.GeoDataFrame(geometry=input_gdf.boundary, crs=input_gdf.crs)
    print('02: ', Line02.geom_type.value_counts())

    # 03: 道路データのクリップ処理
    input_OSM = gpd.read_file(OSM_directory)
    input_OSM = input_OSM.to_crs(input_gdf.crs)
    Line03 = gpd.overlay(input_OSM, input_gdf, how='intersection')
    print('03: ', Line03.geom_type.value_counts())

    # 04: 境界線と道路網の統合
    Line04 = gpd.overlay(Line02, Line03, how='union', keep_geom_type=False)
    Line04 = Line04[~Line04.geom_type.isin(['Point', 'MultiPoint'])]
    Line04 = Line04[~Line04.geom_type.isin(['GeometryCollection'])]

    invalid_geoms = Line04[~Line04.is_valid]
    print(f"無効なジオメトリの数: {len(invalid_geoms)}")
    Line04 = Line04[Line04.is_valid]

    tree = STRtree(Line04.geometry)
    tolerance = 0.1

    def snap_to_nearest(geom):
        nearby = tree.query(geom.buffer(tolerance))
        nearby_geoms = [tree.geometries[i] for i in nearby if tree.geometries[i].is_valid]
        if len(nearby_geoms) == 0:
            return geom
        unioned = unary_union(nearby_geoms)
        return snap(geom, unioned, tolerance)

    Line04['geometry'] = Line04['geometry'].apply(snap_to_nearest)
    print('04: ', Line04.geom_type.value_counts())

    Poly05 = process_line_to_polygons(Line04)
else:
    print("02-04: 道路処理をスキップ。対象領域ポリゴンをそのまま使用します。")
    Poly05 = input_gdf.copy()

print('05: ', Poly05.geom_type.value_counts())

# 05_1: ポリゴン内のホール（穴）を削除
def remove_holes(geometry):
    """ポリゴンの内部に存在するホールを削除する関数"""
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)  # 外周のみを残す
    return geometry

# 各ポリゴンのホールを削除
Poly05['geometry'] = Poly05['geometry'].apply(remove_holes)
# Poly05.to_file(f"{output_folder}/05_1_Poly.gpkg", layer='poly', driver="GPKG")
print('05_1: ',Poly05.geom_type.value_counts()) # ジオメトリタイプを確認

# 05_2: 内部ポリゴンの削除（完全に包含されている小さなポリゴンを除去）
# 欠損・空ジオメトリを先に除去
Poly05 = Poly05[Poly05.geometry.notnull() & ~Poly05.geometry.is_empty]

# 空間インデックス作成（高速な含有関係の判定のため）
tree = STRtree(Poly05.geometry)
geoms = Poly05.geometry.values

# 内部ポリゴンのインデックスを特定
inner_indices = []

for i, geom in enumerate(geoms):
    # 自分以外で重なる候補（インデックス）を取得
    possible_matches_idx = tree.query(geom)
    possible_matches_geoms = [geoms[j] for j in possible_matches_idx if j != i]

    # 自分が他のポリゴン内に完全に含まれる場合は削除対象
    for other_geom in possible_matches_geoms:
        if geom.within(other_geom):
            inner_indices.append(i)
            break

# 内部ポリゴンを削除
Poly05_cleaned = Poly05.drop(Poly05.index[inner_indices])

# 結果を出力
# Poly05_cleaned.to_file(f"{output_folder}/05_2_Poly.gpkg", layer='poly_cleaned', driver="GPKG")
print(f"削除前: {len(Poly05)}個 → 削除後: {len(Poly05_cleaned)}個")

# 05_2_2: 微小ポリゴンのマージ処理
def _fix_polygon_geom(geom):
    """ポリゴン union 後の形状を修復し、単一 Polygon を返す。"""
    if geom is None or geom.is_empty:
        return geom
    geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        parts = [
            g for g in geom.geoms
            if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty
        ]
        if not parts:
            return geom
        geom = unary_union(parts)
    if geom.geom_type == "MultiPolygon":
        rep = geom.representative_point()
        for part in geom.geoms:
            if part.contains(rep) or part.covers(rep):
                return part
        return max(geom.geoms, key=lambda p: p.area)
    return geom

def _merge_sliver_into_target(target, sliver):
    """sliver を target に union し、修復した単一 Polygon を返す。"""
    return _fix_polygon_geom(target.union(sliver))

def _shared_boundary_length(geom_a, geom_b, tol=0.05):
    """2ポリゴン間の共有境界長 [m]（僅かな隙間は tol で吸収）。"""
    if geom_a is None or geom_b is None or geom_a.is_empty or geom_b.is_empty:
        return 0.0
    shared = geom_a.boundary.intersection(geom_b.boundary)
    if shared is not None and not shared.is_empty:
        length = float(shared.length)
        if length > 0:
            return length
    if not geom_a.buffer(tol).intersects(geom_b):
        return 0.0
    shared = geom_a.boundary.intersection(geom_b.buffer(tol))
    if shared is not None and not shared.is_empty:
        return float(shared.length)
    shared = geom_a.buffer(tol).boundary.intersection(geom_b.boundary)
    if shared is not None and not shared.is_empty:
        return float(shared.length)
    return 0.0

def _pick_merge_target(i, geoms, removed, is_candidate_fn, tree):
    """隣接面のうち最適な結合先インデックスを返す（非隣接は None）。"""
    geom = geoms[i]
    touching_large = []
    touching_small = []

    for j in tree.query(geom):
        if j == i or removed[j]:
            continue
        other = geoms[j]
        if other is None or other.is_empty:
            continue
        shared_len = _shared_boundary_length(geom, other)
        if shared_len <= 0:
            continue
        if is_candidate_fn(other):
            touching_small.append((shared_len, float(other.area), j))
        else:
            touching_large.append((shared_len, float(other.area), j))

    if touching_large:
        return max(touching_large)[2]
    if touching_small:
        return max(touching_small)[2]
    return None

def _merge_into_neighbors(gdf, is_candidate_fn, label="小ポリゴン", max_passes=8):
    """条件を満たすポリゴンを、隣接面へ結合する（非隣接への結合は行わない）。"""
    gdf = gdf.reset_index(drop=True)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return gdf

    total_merged = 0
    for _pass in range(max_passes):
        n = len(geoms)
        removed = [False] * n
        merged_count = 0
        tree = STRtree(geoms)

        for i in range(n):
            if removed[i]:
                continue
            geom = geoms[i]
            if not is_candidate_fn(geom):
                continue

            best_j = _pick_merge_target(i, geoms, removed, is_candidate_fn, tree)
            if best_j is None:
                continue

            geoms[best_j] = _merge_sliver_into_target(geoms[best_j], geom)
            removed[i] = True
            merged_count += 1

        total_merged += merged_count
        geoms = [g for k, g in enumerate(geoms) if not removed[k]]
        if merged_count == 0:
            break

    n_orphan = sum(1 for g in geoms if is_candidate_fn(g))
    if total_merged:
        print(f"{label}: {total_merged} 面を隣接面へ統合", flush=True)
    if n_orphan:
        print(f"{label}: 結合先なしで残存={n_orphan} 面", flush=True)
    return gpd.GeoDataFrame(geometry=geoms, crs=gdf.crs).reset_index(drop=True)

def merge_small_polygons(gdf: gpd.GeoDataFrame, threshold=None) -> gpd.GeoDataFrame:
    """
    小さなポリゴンを最も近い大きなポリゴンにマージする関数
    これにより計算効率を向上させる
    """
    if threshold is None:
        threshold = area_threshold
    return _merge_into_neighbors(
        gdf,
        lambda geom: float(geom.area) < threshold,
        label=f"小ポリゴン(<{threshold}m²)",
    )

# 微小ポリゴンマージの実行
if enable_merge_small_polygons:
    Poly05_2_2 = merge_small_polygons(Poly05_cleaned)
else:
    Poly05_2_2 = Poly05_cleaned.reset_index(drop=True)
    print("05_2_2: merge_small_polygons=false のため小ポリゴン結合をスキップ", flush=True)
# Poly05_2_2.to_file(f"{output_folder}/05_2_2_Poly.gpkg", layer='poly', driver="GPKG")
print('05_2_2: ',Poly05_2_2.geom_type.value_counts()) # ジオメトリタイプを確認

# 05_3: 大面積ポリゴンの抽出（分割対象の特定）
# 統計的な面積閾値の算出
mean_area = Poly05_2_2.geometry.area.mean()
std_area = Poly05_2_2.geometry.area.std()
print(f"面積統計 - 平均: {mean_area:.2f}, 標準偏差: {std_area:.2f}")

# 面積閾値を自動算出（標準偏差ベース）
if hex_area_threshold is None:
    hex_area_threshold = max(float(area_threshold), float(std_area) * float(hex_area_std_multiplier))
    print(f"hex_area_threshold(auto) = std_area({std_area:.2f}) * {hex_area_std_multiplier:.2f} = {hex_area_threshold:.2f}")

# 直径を自動算出（均一な正六角形面積になるよう設定）
# 正六角形面積 A = 3*sqrt(3)/8 * D^2  => D = sqrt(8A / (3*sqrt(3)))
if hex_diameter is None:
    base_area = max(float(hex_area_threshold), float(area_threshold))
    hex_diameter = float(hex_diameter_scale) * math.sqrt((8.0 * base_area) / (3.0 * math.sqrt(3.0)))
    print(f"hex_diameter(auto) = {hex_diameter:.2f} m (base_area={base_area:.2f}, scale={hex_diameter_scale:.2f})")

# quadtree 根セル（level=0）一辺 [m]。auto 時は sqrt(qt_min_area) * 2^qt_max_depth
if primary_refine_method == "quadtree_std":
    if qt_root_cell_size is None:
        qt_root_cell_size = math.sqrt(qt_min_area) * (2 ** qt_max_depth)
        print(
            f"qt_root_cell_size(auto) = sqrt({qt_min_area}) * 2^{qt_max_depth} "
            f"= {qt_root_cell_size:.2f} m"
        )
    else:
        min_edge = qt_root_cell_size / (2 ** qt_max_depth)
        print(
            f"qt_root_cell_size={qt_root_cell_size:.2f} m "
            f"(qt_max_depth={qt_max_depth} 時の最小辺 ≈ {min_edge:.2f} m)"
        )

# 閾値以上の面積を持つポリゴンを抽出（分割対象）
Poly05_3 = Poly05_2_2[Poly05_2_2.geometry.area > hex_area_threshold]
# Poly05_3.to_file(f"{output_folder}/05_3_Poly.gpkg", layer='poly', driver="GPKG")
print('05_3: ',Poly05_3.geom_type.value_counts()) # ジオメトリタイプを確認

# ============================================================================
# フェーズ3: ヘキサゴングリッドによる大面積ポリゴンの分割 (05_4)
# ============================================================================

# 05_4: ヘキサゴン（六角形）グリッドの作成
# 大面積ポリゴンを分割するための規則的なグリッドを生成
dist = hex_diameter  # ヘキサゴンの直径（メートル）

def create_hex_grid(data, dist):
    """
    入力領域にヘキサゴングリッドを生成する関数
    
    Args:
        data: 対象領域のGeoDataFrame
        dist: ヘキサゴンの直径
        
    Returns:
        ヘキサゴンポリゴンのGeoDataFrame
    """
    # 入力領域の境界を取得
    xmin, ymin, xmax, ymax = data.total_bounds

    # グリッド座標の生成（ヘキサゴンレイアウト用）
    xcoords = np.arange(xmin, xmax, dist*1.5)
    ycoords = np.arange(ymin, ymax, dist*math.sin(math.radians(60)) *0.5)

    def create_grid(xcoords, ycoords, dist):
        """ヘキサゴン配置用のグリッドポイント生成"""
        points = []
        for count, y in enumerate(ycoords):
            if count % 2 == 0:  # 偶数行
                for x in xcoords:
                    points.append((x, y))
            else:  # 奇数行（オフセットして配置）
                for x in xcoords:
                    points.append((x + (dist*0.75), y))
        return points

    def hexagon(point, dist):
        """指定点中心のヘキサゴン座標を生成"""
        circle = 2*math.pi  # 180度をラジアン
        radius = dist/2

        # ヘキサゴンの6つの頂点座標を計算
        x = [point[0] + radius * math.cos(i * circle / 6) for i in range(6)]
        y = [point[1] + radius * math.sin(i * circle / 6) for i in range(6)]
        x.append(x[0])  # 回り込み
        y.append(y[0])

        return list(zip(x, y))

    # グリッドポイントとヘキサゴンポリゴンを生成
    grid_points = create_grid(xcoords, ycoords, dist)
    hex_points = [hexagon(i, dist) for i in grid_points]
    hex_polygons = gpd.GeoDataFrame(geometry=[Polygon(i) for i in hex_points], crs=data.crs)
    hex_polygons['id'] = [i for i in range(len(hex_polygons))]
    return hex_polygons

# ヘキサゴングリッドの作成と保存
hex_polygons = None

def extract_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    ポリゴンのGeoDataFrameから重複のない境界線を抽出
    
    Args:
        gdf: ポリゴンのGeoDataFrame
        
    Returns:
        境界線のGeoDataFrame（1つのLineStringを含む）
    """
    # すべてのポリゴンの境界線を取得
    all_boundaries = gdf.geometry.boundary

    # 重複を削除し、外周のみを抽出
    unique_boundary = unary_union(all_boundaries)

    # GeoDataFrameとして返す
    return gpd.GeoDataFrame(geometry=[unique_boundary], crs=gdf.crs)

# ヘキサゴングリッドの境界線を抽出
hex_lines = None

# ============================================================================
# フェーズ4: ポリゴン再分割 (05_5)
# ============================================================================

# 05_5: 大面積ポリゴンのヘキサゴングリッドによる再分割
def split_large_polygons_by_lines(polygon_gdf, line_gdf, area_threshold,
                                  merged_lines_path=None, polygonized_path=None):
    """
    大面積ポリゴンをヘキサゴングリッドで分割する関数
    
    処理手順:
    1. 面積が閾値以上のポリゴンからターゲット領域を求める
    2. ヘキサゴングリッドラインから、ターゲット領域内部との交差部分を抽出
    3. 元のポリゴン境界線と、抽出されたラインを統合
    4. 統合されたラインから新たなポリゴン群を生成
    5. MultiPolygonを単一ポリゴンに分解

    Args:
        polygon_gdf: 分割対象のポリゴンGeoDataFrame
        line_gdf: 分割用のラインGeoDataFrame（ヘキサゴングリッドライン）
        area_threshold: 分割対象となる面積閾値
        merged_lines_path: 中間結果のライン保存パス（デバッグ用）
        polygonized_path: 中間結果のポリゴン保存パス（デバッグ用）

    Returns:
        分割処理後のポリゴンGeoDataFrame
    """
    # 1. 面積が閾値以上のポリゴンからターゲット領域を統合
    target_gdf = polygon_gdf[polygon_gdf.geometry.area >= area_threshold]
    target_union = target_gdf.union_all()

    # 2. ラインデータから、ターゲット領域内部との交差部分を抽出
    filtered_lines = []
    for line in line_gdf.geometry:
        intersected = line.intersection(target_union)
        if not intersected.is_empty:
            filtered_lines.append(intersected)

    # 3. 元のポリゴン全体の境界線と filtered_lines を統合
    boundaries = polygon_gdf.geometry.boundary.tolist()
    all_lines = boundaries + filtered_lines
    merged_lines = unary_union(all_lines)

    # 中間結果の保存（デバッグ・確認用）
    if merged_lines_path is not None:
        merged_lines_gdf = gpd.GeoDataFrame({'geometry': [merged_lines]}, crs=polygon_gdf.crs)
        merged_lines_gdf.to_file(merged_lines_path, layer='merged_lines', driver="GPKG")
        print(f"Merged lines saved to {merged_lines_path}")

    # 4. 統合されたラインからポリゴンを生成（ポリゴン化）
    poly_list = list(polygonize(merged_lines))
    polygonized_gdf = gpd.GeoDataFrame(geometry=poly_list, crs=polygon_gdf.crs)

    # 中間結果の保存（デバッグ・確認用）
    if polygonized_path is not None:
        polygonized_gdf.to_file(polygonized_path, layer='polygonized', driver="GPKG")
        print(f"Polygonized GeoDataFrame saved to {polygonized_path}")

    # 5. explode して単一ポリゴンに分解
    final_gdf = polygonized_gdf.explode(index_parts=False)
    final_gdf.reset_index(drop=True, inplace=True)

    return final_gdf

def _iterate_polygons(geom):
    """Polygon / MultiPolygon から Polygon を順に返す。"""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if g is not None and not g.is_empty:
                yield g

def _calc_aspect_ratio_axis(geom):
    """軸平行外接矩形のアスペクト比（長辺/短辺）。"""
    if geom is None or geom.is_empty:
        return 1.0
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    if width <= 0 or height <= 0:
        return 1.0
    return max(width / height, height / width)

def _aspect_exceeds(geom, aspect_threshold):
    return _calc_aspect_ratio_axis(geom) > aspect_threshold

def _subdivide_polygon_halves(geom):
    """bbox 長辺方向に半分へ分割し、有効な部分ポリゴンを返す。"""
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    if width <= 0 or height <= 0:
        return [geom]

    if width >= height:
        midx = (minx + maxx) / 2.0
        cutters = (
            box(minx, miny, midx, maxy),
            box(midx, miny, maxx, maxy),
        )
    else:
        midy = (miny + maxy) / 2.0
        cutters = (
            box(minx, miny, maxx, midy),
            box(minx, midy, maxx, maxy),
        )

    parts = []
    for cutter in cutters:
        piece = geom.intersection(cutter)
        for p in _iterate_polygons(piece):
            if p is not None and not p.is_empty and p.area > 0:
                parts.append(p)
    return parts if len(parts) >= 2 else [geom]

def _split_one_elongated_polygon(geom, aspect_threshold, min_split_area, max_depth=16, depth=0):
    """細長1ポリゴンを軸平行2分割で再帰的に分割する。"""
    if geom is None or geom.is_empty:
        return []
    if depth >= max_depth or float(geom.area) < min_split_area:
        return [geom]
    if not _aspect_exceeds(geom, aspect_threshold):
        return [geom]

    parts = _subdivide_polygon_halves(geom)
    if len(parts) < 2:
        return [geom]

    out = []
    for part in parts:
        out.extend(
            _split_one_elongated_polygon(
                part, aspect_threshold, min_split_area, max_depth, depth + 1
            )
        )
    return out if out else [geom]

def split_elongated_polygons(polys, aspect_threshold, min_split_area, max_passes=8):
    """
    縦長・横長ポリゴンを quadtree とは独立に bbox 半分分割で細分化する。
    """
    result = [g for g in polys if g is not None and not g.is_empty]
    total_new = 0

    for pass_idx in range(max_passes):
        next_result = []
        n_changed = 0
        for geom in result:
            if _aspect_exceeds(geom, aspect_threshold) and float(geom.area) >= min_split_area:
                parts = _split_one_elongated_polygon(
                    geom, aspect_threshold, min_split_area
                )
                if len(parts) > 1:
                    n_changed += 1
                    total_new += len(parts) - 1
                next_result.extend(parts)
            else:
                next_result.append(geom)
        result = next_result
        if n_changed == 0:
            break

    n_remain = sum(
        1 for g in result
        if _aspect_exceeds(g, aspect_threshold) and float(g.area) >= min_split_area
    )
    n_skip_small = sum(
        1 for g in result
        if _aspect_exceeds(g, aspect_threshold) and float(g.area) < min_split_area
    )
    if total_new or n_remain or n_skip_small:
        print(
            f"細長メッシュ分割: 追加 {total_new} 面, "
            f"残存(>{aspect_threshold}, >={min_split_area}m²)={n_remain}, "
            f"分割スキップ(面積<{min_split_area}m²)={n_skip_small}, pass={pass_idx + 1}",
            flush=True,
        )
    return result

def split_elongated_mesh_gdf(gdf, aspect_threshold, min_split_area):
    """GeoDataFrame 内の細長ポリゴンを軸平行2分割で分割する。"""
    if gdf is None or len(gdf) == 0:
        return gdf
    polys = split_elongated_polygons(
        gdf.geometry.tolist(), aspect_threshold, min_split_area
    )
    return gpd.GeoDataFrame(geometry=polys, crs=gdf.crs).reset_index(drop=True)

def merge_elongated_slivers_gdf(gdf, aspect_threshold, max_merge_area):
    """細長かつ面積が小さいポリゴンを隣接面へ結合する。"""
    if gdf is None or len(gdf) == 0:
        return gdf

    def is_sliver(geom):
        return _aspect_exceeds(geom, aspect_threshold) and float(geom.area) < max_merge_area

    return _merge_into_neighbors(
        gdf,
        is_sliver,
        label=f"細長スリバー(aspect>{aspect_threshold}, <{max_merge_area}m²)",
    )

def maybe_merge_elongated_slivers(gdf, step_label):
    """aspect_merge_small が有効なときのみ細長スリバー結合を実行する。"""
    if not aspect_merge_small:
        return gdf
    print(
        f"{step_label}: 細長スリバー結合 "
        f"(aspect>{aspect_ratio_threshold}, 面積<{aspect_min_split_area} m²)",
        flush=True,
    )
    out = merge_elongated_slivers_gdf(
        gdf, aspect_ratio_threshold, aspect_min_split_area
    )
    print(f"{step_label}: 結合後ポリゴン数 = {len(out)}", flush=True)
    return out

def _area_m2_in_crs(geom, crs):
    """任意 CRS の Geometry 面積を m² に変換する。"""
    if geom is None or geom.is_empty:
        return 0.0
    crs_obj = CRS(crs)
    if crs_obj.is_projected:
        unit_factor = crs_obj.axis_info[0].unit_conversion_factor if crs_obj.axis_info else 1.0
        return float(geom.area) * float(unit_factor) * float(unit_factor)
    geod = crs_obj.get_geod() if hasattr(crs_obj, "get_geod") else Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(geom)
    return abs(float(area_m2))

def _dem_stats_for_polygon(src, geom, nodata_value, geom_crs):
    """ポリゴン内 DEM の標準偏差・起伏量(p95-p5)・サンプル数を返す（幾何は geom_crs、DEM は raster CRS）。"""
    if geom is None or geom.is_empty:
        return None, None, 0

    raster_crs = src.crs
    if (
        geom_crs is not None
        and raster_crs is not None
        and not CRS(geom_crs).equals(CRS(raster_crs))
    ):
        to_raster = Transformer.from_crs(geom_crs, raster_crs, always_xy=True).transform
        geom_mask = shp_transform(to_raster, geom)
    else:
        geom_mask = geom

    try:
        out_image, _ = rio_mask.mask(src, [mapping(geom_mask)], crop=True, nodata=nodata_value, all_touched=True)
    except ValueError:
        return None, None, 0

    return _band_dem_stats(out_image[0], nodata_value)

def _band_dem_stats(band, nodata_value):
    if np.issubdtype(band.dtype, np.floating):
        valid = (band != nodata_value) & (~np.isnan(band))
    else:
        valid = (band != nodata_value)

    vals = band[valid]
    if vals.size == 0:
        return None, None, 0

    std_val = float(np.std(vals))
    relief_val = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    return std_val, relief_val, int(vals.size)

def _slab_dem_stats(slab, valid_slab, domain_slab, nodata_value, need_relief=True):
    """メモリ上の DEM スライスから統計量を計算。"""
    if np.issubdtype(slab.dtype, np.floating):
        valid = valid_slab & (slab != nodata_value) & (~np.isnan(slab))
    else:
        valid = valid_slab & (slab != nodata_value)

    area_px = domain_slab.size
    dom_px = int(domain_slab.sum())
    full_cover = dom_px >= area_px if area_px else False

    vals = slab[valid]
    if vals.size == 0:
        return None, None, 0, full_cover

    std_val = float(np.std(vals))
    relief_val = None
    if need_relief:
        relief_val = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    return std_val, relief_val, int(vals.size), full_cover

class _QuadtreeDemContext:
    """領域 DEM を1回読み込み、スライス直接参照でセル統計を返す。"""

    def __init__(self, src, work_crs, nodata_value, domain_geom, bounds_xy):
        self.nodata = nodata_value if nodata_value is not None else -9999
        self.cache = {}
        self._slice_cache = {}
        raster_crs = src.crs
        self._needs_transform = (
            work_crs is not None
            and raster_crs is not None
            and not CRS(work_crs).equals(CRS(raster_crs))
        )
        if self._needs_transform:
            self._to_raster = Transformer.from_crs(work_crs, raster_crs, always_xy=True).transform
            domain_raster = shp_transform(self._to_raster, domain_geom)
        else:
            self._to_raster = None
            domain_raster = domain_geom

        xmin, ymin, xmax, ymax = bounds_xy
        pixel_size = max(abs(src.transform.a), abs(src.transform.e), 1.0)
        pad = pixel_size * 2
        if self._to_raster is not None:
            rx, ry = self._to_raster(
                [xmin - pad, xmax + pad, xmax + pad, xmin - pad],
                [ymin - pad, ymin - pad, ymax + pad, ymax + pad],
            )
            left, right = float(min(rx)), float(max(rx))
            bottom, top = float(min(ry)), float(max(ry))
        else:
            left, bottom, right, top = xmin - pad, ymin - pad, xmax + pad, ymax + pad

        win = windows.from_bounds(left, bottom, right, top, src.transform)
        win = win.round_offsets().round_lengths().intersection(
            windows.Window(0, 0, src.width, src.height)
        )
        if win.width <= 0 or win.height <= 0:
            raise ValueError("quadtree: DEM 読込 window が空です")

        self.array = src.read(1, window=win)
        self.transform = windows.transform(win, src.transform)
        self.domain_mask = geometry_mask(
            [mapping(domain_raster)],
            out_shape=self.array.shape,
            transform=self.transform,
            invert=True,
            all_touched=True,
        )
        h, w = self.array.shape
        print(
            f"quadtree DEM 読込: {w}×{h} px ({self.array.nbytes / 1e6:.1f} MB)",
            flush=True,
        )
        self._init_valid_mask()
        print("quadtree DEM 準備完了", flush=True)

    def _init_valid_mask(self):
        raw = self.array
        if np.issubdtype(raw.dtype, np.floating):
            self._valid = (raw != self.nodata) & (~np.isnan(raw)) & self.domain_mask
        else:
            self._valid = (raw != self.nodata) & self.domain_mask

    def _work_bounds_to_slice(self, minx, miny, maxx, maxy):
        if self._to_raster is not None:
            rx, ry = self._to_raster(
                [minx, maxx, maxx, minx],
                [miny, miny, maxy, maxy],
            )
        else:
            rx, ry = [minx, maxx, maxx, minx], [miny, miny, maxy, maxy]

        rows, cols = rowcol(self.transform, rx, ry)
        row0 = max(0, min(rows))
        row1 = min(self.array.shape[0], max(rows) + 1)
        col0 = max(0, min(cols))
        col1 = min(self.array.shape[1], max(cols) + 1)
        if row1 <= row0 or col1 <= col0:
            return None
        return row0, row1, col0, col1

    def slice_for_key(self, origin_x, origin_y, base_cell, level, i, j):
        cache_key = (level, i, j)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        step = base_cell / (2 ** level)
        bounds = (
            origin_x + i * step,
            origin_y + j * step,
            origin_x + (i + 1) * step,
            origin_y + (j + 1) * step,
        )
        sl = self._work_bounds_to_slice(*bounds)
        self._slice_cache[cache_key] = sl
        return sl

    def stats_for_cell(self, key, minx, miny, maxx, maxy, std_threshold, relief_threshold):
        if key in self.cache:
            return self.cache[key]

        sl = self._work_bounds_to_slice(minx, miny, maxx, maxy)
        if sl is None:
            result = (None, None, 0, False)
        else:
            r0, r1, c0, c1 = sl
            slab = self.array[r0:r1, c0:c1]
            valid_slab = self._valid[r0:r1, c0:c1]
            domain_slab = self.domain_mask[r0:r1, c0:c1]

            std_val, relief_val, n, full_cover = _slab_dem_stats(
                slab, valid_slab, domain_slab, self.nodata, need_relief=False
            )
            if (
                std_val is not None
                and std_val <= std_threshold
                and relief_threshold is not None
                and n > 0
            ):
                vals = slab[valid_slab]
                if vals.size > 0:
                    relief_val = float(np.percentile(vals, 95) - np.percentile(vals, 5))

            result = (std_val, relief_val, n, full_cover)

        self.cache[key] = result
        return result

def _square_cell_bounds(origin_x, origin_y, base_cell, level, i, j):
    """グローバル quadtree 上の正方形セル（常に等辺）。"""
    step = base_cell / (2 ** level)
    return box(
        origin_x + i * step,
        origin_y + j * step,
        origin_x + (i + 1) * step,
        origin_y + (j + 1) * step,
    )

def _square_cell_area(base_cell, level):
    step = base_cell / (2 ** level)
    return step * step

def _initial_root_cells(origin_x, origin_y, base_cell, xmin, ymin, xmax, ymax):
    """領域を覆う level=0 の正方形セル index 集合。"""
    i_min = int(math.floor((xmin - origin_x) / base_cell))
    i_max = int(math.floor((xmax - origin_x) / base_cell))
    j_min = int(math.floor((ymin - origin_y) / base_cell))
    j_max = int(math.floor((ymax - origin_y) / base_cell))
    cells = set()
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            cells.add((0, i, j))
    return cells

def _quadtree_level_index(leaves):
    index = {}
    for level, i, j in leaves:
        index.setdefault(level, {})[(i, j)] = (level, i, j)
    return index

def _cell_has_domain_pixels(dem_ctx, origin_x, origin_y, base_cell, level, i, j):
    """セル bbox が領域マスクと重なるか（shapely 不使用）。"""
    sl = dem_ctx.slice_for_key(origin_x, origin_y, base_cell, level, i, j)
    if sl is None:
        return False
    r0, r1, c0, c1 = sl
    return bool(dem_ctx.domain_mask[r0:r1, c0:c1].any())

def _find_neighbor_leaf(leaves, level_index, levels_desc, origin_x, origin_y, base_cell, level, i, j, di, dj):
    """辺 (di,dj) 方向の隣接 leaf（同レベル index 優先）。"""
    same = level_index.get(level, {}).get((i + di, j + dj))
    if same is not None and same in leaves:
        return same

    step = base_cell / (2 ** level)
    px = origin_x + (i + 0.5 + di * 0.5) * step + di * step * 1e-9
    py = origin_y + (j + 0.5 + dj * 0.5) * step + dj * step * 1e-9

    for lv in levels_desc:
        s = base_cell / (2 ** lv)
        ci = int(math.floor((px - origin_x) / s))
        cj = int(math.floor((py - origin_y) / s))
        found = level_index.get(lv, {}).get((ci, cj))
        if found is not None and found in leaves:
            return found
    return None

def _split_leaf(leaves, level_index, origin_x, origin_y, base_cell, dem_ctx, key):
    """leaf を 4 分割。追加した key のリストを返す。"""
    if key not in leaves:
        return []
    level, i, j = key
    leaves.remove(key)
    level_index.get(level, {}).pop((i, j), None)

    child_level = level + 1
    new_keys = []
    for ci, cj in ((2 * i, 2 * j), (2 * i + 1, 2 * j), (2 * i, 2 * j + 1), (2 * i + 1, 2 * j + 1)):
        if _cell_has_domain_pixels(dem_ctx, origin_x, origin_y, base_cell, child_level, ci, cj):
            child_key = (child_level, ci, cj)
            leaves.add(child_key)
            level_index.setdefault(child_level, {})[(ci, cj)] = child_key
            new_keys.append(child_key)
    return new_keys

def _enforce_2to1_balance(leaves, level_index, origin_x, origin_y, base_cell, dem_ctx, max_depth):
    """隣接 leaf のレベル差が最大 1 になるまで分割（2:1 ルール）。"""
    levels_desc = sorted(level_index.keys(), reverse=True)
    changed = True
    n_balance = 0
    while changed:
        changed = False
        for key in list(leaves):
            level, i, j = key
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nkey = _find_neighbor_leaf(
                    leaves, level_index, levels_desc,
                    origin_x, origin_y, base_cell, level, i, j, di, dj,
                )
                if nkey is None:
                    continue
                nlevel = nkey[0]
                if level - nlevel > 1 and nlevel < max_depth:
                    _split_leaf(leaves, level_index, origin_x, origin_y, base_cell, dem_ctx, nkey)
                    if nkey[0] + 1 not in levels_desc:
                        levels_desc = sorted(level_index.keys(), reverse=True)
                    changed = True
                    n_balance += 1
                elif nlevel - level > 1 and level < max_depth:
                    _split_leaf(leaves, level_index, origin_x, origin_y, base_cell, dem_ctx, key)
                    if key[0] + 1 not in levels_desc:
                        levels_desc = sorted(level_index.keys(), reverse=True)
                    changed = True
                    n_balance += 1
                    break
            if changed:
                break
    return n_balance

def _eval_quadtree_cell(
    dem_stats,
    origin_x,
    origin_y,
    base_cell,
    level,
    i,
    j,
    domain_geom,
    domain_prep,
    std_threshold,
    relief_threshold,
    min_area,
    min_dem_samples,
    max_depth,
):
    """正方形セルの分割要否と理由を返す（DEM std/relief のみ）。"""
    if level >= max_depth:
        return False, None

    cell_area = _square_cell_area(base_cell, level)
    if cell_area < min_area:
        return False, None

    step = base_cell / (2 ** level)
    minx = origin_x + i * step
    miny = origin_y + j * step
    maxx = minx + step
    maxy = miny + step

    db = domain_geom.bounds
    if maxx <= db[0] or minx >= db[2] or maxy <= db[1] or miny >= db[3]:
        return False, None

    cell = box(minx, miny, maxx, maxy)
    if not domain_prep.intersects(cell):
        return False, None

    key = (level, i, j)
    std_val, relief_val, sample_count, _full_cover = dem_stats.stats_for_cell(
        key, minx, miny, maxx, maxy, std_threshold, relief_threshold
    )

    high_std = std_val is not None and std_val > std_threshold
    high_relief = relief_val is not None and relief_val > relief_threshold
    if min_dem_samples is not None and sample_count < min_dem_samples:
        high_std = False
        high_relief = False

    if not (high_std or high_relief):
        return False, None

    if high_std:
        return True, "std"
    if high_relief:
        return True, "relief"
    return False, None

def split_polygons_by_quadtree_dem(
    polygon_gdf,
    dem_path,
    std_threshold,
    relief_threshold,
    max_depth,
    min_area,
    min_dem_samples,
    split_area_threshold,
    grid_cell_size,
    mesh_mode="clip",
    domain_gdf=None,
    domain_mode="target",
):
    """
    グローバル quadtree による DEM 適応分割（std / relief のみ）。

    mesh_mode:
      - "clip": 道路ポリゴン等と leaf セルの intersection（従来）
      - "pure": leaf セル（正方形）のみで構成（境界は階段状）
    domain_mode (pure 時):
      - "target": target_area と交差する leaf のみ（正方形のまま）
      - "bbox": target の外接 bbox 内の leaf をすべて採用
    """
    if len(polygon_gdf) == 0:
        return polygon_gdf.copy()

    mesh_mode = str(mesh_mode).strip().lower()
    domain_mode = str(domain_mode).strip().lower()

    work_crs = polygon_gdf.crs
    if work_crs is None:
        print("Warning: ポリゴン CRS が未設定のため、quadtree 分割をスキップします。")
        return polygon_gdf.copy()

    if not CRS(work_crs).is_projected:
        print(f"Warning: quadtree 分割は投影座標系を推奨します（現在: {work_crs}）。")

    work_gdf = polygon_gdf.copy()
    if not work_gdf.geometry.is_valid.all():
        work_gdf["geometry"] = work_gdf.geometry.apply(lambda g: g.buffer(0) if not g.is_valid else g)

    if domain_gdf is not None and len(domain_gdf) > 0:
        domain_source = domain_gdf.copy()
        if domain_source.crs != work_crs:
            domain_source = domain_source.to_crs(work_crs)
        if not domain_source.geometry.is_valid.all():
            domain_source["geometry"] = domain_source.geometry.apply(
                lambda g: g.buffer(0) if not g.is_valid else g
            )
        domain_geom = domain_source.union_all()
        xmin, ymin, xmax, ymax = domain_source.total_bounds
    else:
        domain_geom = work_gdf.union_all()
        xmin, ymin, xmax, ymax = work_gdf.total_bounds

    domain_prep = prep(domain_geom)
    base_cell = max(float(grid_cell_size), 1.0)
    origin_x = math.floor(xmin / base_cell) * base_cell
    origin_y = math.floor(ymin / base_cell) * base_cell
    print(
        f"quadtree グリッド: origin=({origin_x:.1f}, {origin_y:.1f}), "
        f"root_cell={base_cell:.1f} m, crs={work_crs}"
    )

    with rasterio.open(dem_path) as src:
        nodata_value = src.nodata if src.nodata is not None else -9999
        dem_stats = _QuadtreeDemContext(
            src, work_crs, nodata_value, domain_geom, (xmin, ymin, xmax, ymax)
        )

        leaves = {
            key for key in _initial_root_cells(origin_x, origin_y, base_cell, xmin, ymin, xmax, ymax)
            if _cell_has_domain_pixels(dem_stats, origin_x, origin_y, base_cell, key[0], key[1], key[2])
        }
        level_index = _quadtree_level_index(leaves)

        eval_common = (
            dem_stats, origin_x, origin_y, base_cell,
            domain_geom, domain_prep,
            std_threshold, relief_threshold,
            min_area, min_dem_samples, max_depth,
        )
        split_reason_counter = defaultdict(int)
        n_refine_pass = 0
        pending = set(leaves)

        while pending:
            min_level = min(k[0] for k in pending)
            if min_level >= max_depth:
                break

            to_split = []
            for key in pending:
                level, i, j = key
                should_split, reason = _eval_quadtree_cell(
                    dem_stats, origin_x, origin_y, base_cell, level, i, j, *eval_common[4:]
                )
                if should_split:
                    if reason:
                        split_reason_counter[reason] += 1
                    to_split.append(key)

            if not to_split:
                break

            print(
                f"quadtree refine pass {n_refine_pass + 1}: "
                f"分割 {len(to_split)} / 判定 {len(pending)} セル, leaf={len(leaves)}",
                flush=True,
            )

            next_pending = set()
            for key in to_split:
                for child_key in _split_leaf(
                    leaves, level_index, origin_x, origin_y, base_cell, dem_stats, key
                ):
                    next_pending.add(child_key)
            pending = next_pending
            n_refine_pass += 1

        print(f"quadtree 2:1 balance 開始: leaf={len(leaves)}", flush=True)
        n_balance = _enforce_2to1_balance(
            leaves, level_index, origin_x, origin_y, base_cell, dem_stats, max_depth
        )

        depth_counter = defaultdict(int)
        for level, _, _ in leaves:
            depth_counter[level] += 1

        print(
            f"quadtree 分割統計: refine_pass={n_refine_pass}, balance_splits={n_balance}, "
            f"leaf数={len(leaves)}, dem_cache={len(dem_stats.cache)}, "
            f"理由内訳={dict(split_reason_counter)}, "
            f"leaf深さヒスト={dict(sorted(depth_counter.items()))}"
        )

        leaf_cells = []
        for level, i, j in leaves:
            step = base_cell / (2 ** level)
            cx0 = origin_x + i * step
            cy0 = origin_y + j * step
            cx1 = origin_x + (i + 1) * step
            cy1 = origin_y + (j + 1) * step
            leaf_cells.append(box(cx0, cy0, cx1, cy1))

        if mesh_mode == "pure":
            if domain_mode == "bbox":
                result_polys = list(leaf_cells)
            else:
                result_polys = [
                    cell for cell in leaf_cells
                    if domain_prep.intersects(cell)
                ]
            print(
                f"quadtree pure: {len(result_polys)} 面 "
                f"(leaf={len(leaf_cells)}, domain={domain_mode}, 正方形セル)",
                flush=True,
            )
        else:
            small_polys = []
            large_polys = []
            for geom in work_gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if float(geom.area) < split_area_threshold:
                    small_polys.append(geom)
                else:
                    large_polys.append(geom)

            result_polys = list(small_polys)
            if large_polys and leaf_cells:
                print(
                    f"quadtree overlay clip: 大ポリゴン {len(large_polys)} × leaf {len(leaf_cells)}",
                    flush=True,
                )
                large_gdf = gpd.GeoDataFrame(geometry=large_polys, crs=work_crs)
                leaf_gdf = gpd.GeoDataFrame(geometry=leaf_cells, crs=work_crs)
                clipped_gdf = gpd.overlay(
                    large_gdf, leaf_gdf, how="intersection", keep_geom_type=False
                )
                if len(clipped_gdf) > 0:
                    clipped_gdf = clipped_gdf.explode(index_parts=False)
                    clipped_gdf = clipped_gdf[
                        clipped_gdf.geometry.notnull()
                        & (~clipped_gdf.geometry.is_empty)
                        & (clipped_gdf.geometry.area > 0)
                    ]
                    result_polys.extend(clipped_gdf.geometry.tolist())

    out_gdf = gpd.GeoDataFrame(geometry=result_polys, crs=work_crs)
    out_gdf = out_gdf[out_gdf.geometry.notnull() & (~out_gdf.geometry.is_empty)].reset_index(drop=True)
    if len(out_gdf) == 0:
        return polygon_gdf.copy()
    return out_gdf

# ポリゴン再分割の実行
if primary_refine_method == "none":
    print("05_4-05_5: 一次再分割をスキップします（primary_method=none）。")
    Poly05_5 = Poly05_2_2.copy()
else:
    if primary_refine_method == "quadtree_std":
        if qt_mesh_mode == "pure":
            print("05_4-05_5(quadtree): pure モード — target_area 基準で leaf セルを出力", flush=True)
            qt_input_gdf = input_gdf
        else:
            qt_input_gdf = Poly05_2_2
        Poly05_5 = split_polygons_by_quadtree_dem(
            qt_input_gdf,
            input_dem,
            std_threshold=qt_std_threshold,
            relief_threshold=qt_relief_threshold,
            max_depth=qt_max_depth,
            min_area=qt_min_area,
            min_dem_samples=qt_min_dem_samples,
            split_area_threshold=hex_area_threshold,
            grid_cell_size=qt_root_cell_size,
            mesh_mode=qt_mesh_mode,
            domain_gdf=input_gdf,
            domain_mode=qt_domain,
        )
        print(f"05_4-05_5(quadtree): 分割後ポリゴン数 = {len(Poly05_5)}")
    else:
        hex_polygons = create_hex_grid(input_gdf, dist)
        # hex_polygons.to_file(f"{output_folder}/05_4_Poly.gpkg", layer='poly', driver="GPKG")
        print('05_4: ',hex_polygons.geom_type.value_counts()) # ジオメトリタイプを確認

        hex_lines = extract_boundary(hex_polygons)
        # hex_lines.to_file(f"{output_folder}/05_4_Line.gpkg", layer='line', driver="GPKG")
        print('05_4(Line): ',hex_lines.geom_type.value_counts()) # ジオメトリタイプを確認

        area_threshold2 = hex_area_threshold
        Poly05_5 = split_large_polygons_by_lines(Poly05_2_2, hex_lines, area_threshold2,
            merged_lines_path=None,
            polygonized_path=None)

# Poly05_5.to_file(f"{output_folder}/05_5_Poly.gpkg", layer='poly', driver="GPKG")
print('05_5: ',Poly05_5.geom_type.value_counts()) # ジオメトリタイプを確認

# 05_5_1: 細長メッシュ分割（enable_aspect=false のときスキップ）
if enable_aspect_refinement:
    print(
        f"05_5_1: 細長メッシュ分割 (aspect>{aspect_ratio_threshold}, "
        f"min_split_area={aspect_min_split_area} m²)",
        flush=True,
    )
    Poly05_5 = split_elongated_mesh_gdf(Poly05_5, aspect_ratio_threshold, aspect_min_split_area)
    Poly05_5 = maybe_merge_elongated_slivers(Poly05_5, "05_5_1b")
    print(f"05_5_1: 分割後ポリゴン数 = {len(Poly05_5)}")
else:
    print("05_5_1: アスペクト比分割をスキップ", flush=True)

# ============================================================================
# フェーズ5: アスペクト比に基づく細長メッシュの再分割 (05_5_2～05_5_6)
# ============================================================================

# 05_5_2: アスペクト比の計算
def aspect_ratio(geom):
    """ポリゴンのアスペクト比（bbox 長辺/短辺）を計算する関数"""
    if geom is None or geom.is_empty:
        return None
    return _calc_aspect_ratio_axis(geom)

if enable_aspect_refinement:
    # 各ポリゴンのアスペクト比を計算
    aspect_ratios = Poly05_5.geometry.apply(aspect_ratio)

    # アスペクト比を新しい列として追加
    Poly05_5["aspect_ratio"] = aspect_ratios

    # アスペクト比付きポリゴンを保存
    # Poly05_5.to_file(f"{output_folder}/05_5_2_Poly.gpkg", layer="poly_with_aspect", driver="GPKG")

    # 05_5_3: 高アスペクト比ポリゴンにランダムポイント生成
    def generate_random_points(poly, max_points=500, density=1.0):
        """
        ポリゴン内に面積に比例した数のランダムポイントを生成する関数
    
        Args:
            poly: 対象ポリゴン（Polygon）
            max_points: 最大生成ポイント数
            density: 密度（ポイント/面積）
        
        Returns:
            ポリゴン内のランダムポイントリスト
        """
        area = poly.area
        num_points = min(int(area * density), max_points)

        points = []
        minx, miny, maxx, maxy = poly.bounds

        # ポイントポリゴン内でのランダム配置を試行
        while len(points) < num_points:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                points.append(p)

        return points

    # 細長いポリゴン（アスペクト比しきい値以上）を対象としてランダムポイント生成
    if enable_voronoi_refinement:
        target_polygons = Poly05_5[Poly05_5["aspect_ratio"] >= aspect_ratio_threshold].copy()
    else:
        target_polygons = Poly05_5.iloc[0:0].copy()
        print("05_5_2-05_5_6: 二次再分割をスキップします（secondary_method=none）。")

    # 各対象ポリゴンからランダムポイントを生成し記録
    all_points = []

    for idx, row in target_polygons.iterrows():
        geom = row.geometry
    
        # Polygon、MultiPolygon両方に対応
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        # 各ポリゴンからランダムポイントを生成
        for i, poly in enumerate(polygons):
            points = generate_random_points(poly, max_points=500, density=1.0)
            for pt in points:
                all_points.append({
                    "source_idx": idx,      # 元のポリゴンインデックス
                    "subpoly_id": i,        # MultiPolygon内のサブポリゴンID
                    "geometry": pt          # ランダムポイント座標
                })

    # ランダムポイントをGeoDataFrameに変換
    # 細長いポリゴンがない場合（構造格子など）は空のGeoDataFrameを作成
    if len(all_points) > 0:
        points_gdf = gpd.GeoDataFrame(all_points, geometry="geometry", crs=Poly05_5.crs)
    else:
        # 空のGeoDataFrameを作成
        if enable_voronoi_refinement:
            print(f"警告: アスペクト比 >= {aspect_ratio_threshold} のポリゴンが見つかりません。ランダムポイントは生成されません。")
        points_gdf = gpd.GeoDataFrame(columns=["source_idx", "subpoly_id", "geometry"], crs=Poly05_5.crs, geometry="geometry")

    # DEMによる地盤高データの追加（クラスタリング用の標高情報）
    if len(points_gdf) > 0:
        with rasterio.open(input_dem) as src:
            raster_crs = src.crs
            nodata_value = src.nodata

            # 元の座標系を保存
            original_crs = points_gdf.crs

            # DEMの座標系が取得できない場合のフォールバック処理
            if raster_crs.to_epsg() is None:
                epsg_code = DEM_epsg
                raster_crs = rasterio.crs.CRS.from_epsg(epsg_code)
                print(f"Warning: ラスターの EPSG コードが取得できなかったため、EPSG:{epsg_code} を使用します。")

            # ポイントをDEMの座標系に変換（必要に応じて）
            if points_gdf.crs != raster_crs:
                points_gdf = points_gdf.to_crs(raster_crs)

            # DEMから各ランダムポイントの標高値をサンプリング
            coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
            elevations = list(src.sample(coords))
            elevations = [val[0] if val[0] != nodata_value else np.nan for val in elevations]

            # 標高データをポイントデータに追加
            points_gdf["elevation"] = elevations

            # 元の座標系に戻す
            if not CRS(points_gdf.crs).equals(CRS(original_crs)):
                points_gdf = points_gdf.to_crs(original_crs)

        # 標高データが欠損しているポイントを除外
        points_gdf = points_gdf.dropna(subset=["elevation"])

        # 標高付きランダムポイントを保存
        # points_gdf.to_file(f"{output_folder}/05_5_3_Point.gpkg", layer="random_points", driver="GPKG")
    else:
        print("ランダムポイントが0個のため、DEM標高サンプリングをスキップします。")

    # 05_5_4: K-meansクラスタリングによる細長ポリゴンの分割ポイント生成
    clustered_points = []

    # 各細長ポリゴン（source_idx）ごとに独立してクラスタリングを実行
    for source_idx, group in points_gdf.groupby("source_idx"):
        # ポイント数が少ない場合はクラスタリングをスキップ
        if len(group) < 4:
            group["Cluster"] = -1
            clustered_points.append(group)
            continue

        # 座標データ（x, y, 標高）を配列に変換
        coords = np.array([[p.x, p.y, elev] for p, elev in zip(group.geometry, group["elevation"])])

        # 面積と標高範囲に基づいて動的にクラスタ数を決定
        area = target_polygons.loc[source_idx].geometry.area
        elev_range = group["elevation"].max() - group["elevation"].min()

        # クラスタ数決定ロジック（形状と地形の複雑さを考慮）
        if elev_range < 1:
            k = 2    # 標高差が小さい場合
        elif elev_range < 5:
            k = 3    # 中程度の標高差
        elif area < 1000:
            k = 2    # 小さな面積
        elif area < 5000:
            k = 3    # 中程度の面積
        else:
            k = 4    # 最大クラスタ数（複雑な形状）

        # K-meansクラスタリングの実行
        kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
        group["Cluster"] = kmeans.labels_

        clustered_points.append(group)

    # 全クラスタリング結果を結合してGeoDataFrameに変換
    if len(clustered_points) > 0:
        clustered_gdf = gpd.GeoDataFrame(pd.concat(clustered_points, ignore_index=True), crs=points_gdf.crs)
        # クラスタリング済みポイントを保存
        # clustered_gdf.to_file(f"{output_folder}/05_5_4_Point.gpkg", layer="random_points_clustered", driver="GPKG")
    else:
        # 空のGeoDataFrameを作成
        clustered_gdf = gpd.GeoDataFrame(columns=["source_idx", "subpoly_id", "geometry", "elevation", "Cluster"], crs=Poly05_5.crs, geometry="geometry")
        print("クラスタリング対象のポイントがないため、処理をスキップします。")

    # 05_5_5: クラスタ重心の計算（ティーセン分割用）
    # 有効なクラスタ番号を持つポイントのみを対象（-1は除外）
    valid_points = clustered_gdf[clustered_gdf["Cluster"] >= 0]

    # 各クラスタの重心を計算
    centroids = []

    # 各ポリゴン×サブポリゴン×クラスタの組み合わせごとに重心を計算
    for (src_idx, sub_id, cluster_id), group in valid_points.groupby(["source_idx", "subpoly_id", "Cluster"]):
        # 同一クラスタ内の全ポイントをMultiPointとして集約
        multipoint = MultiPoint(group.geometry.tolist())
        centroid = multipoint.centroid

        centroids.append({
            "source_idx": src_idx,    # 元のポリゴンID
            "subpoly_id": sub_id,     # MultiPolygon内のサブポリゴンID
            "Cluster": cluster_id,    # クラスタID
            "geometry": centroid      # クラスタ重心座標
        })

    # クラスタ重心をGeoDataFrameに変換
    if len(centroids) > 0:
        centroid_gdf = gpd.GeoDataFrame(centroids, geometry="geometry", crs=points_gdf.crs)
        # クラスタ重心を保存
        # centroid_gdf.to_file(f"{output_folder}/05_5_5_Point.gpkg", layer="cluster_centroids", driver="GPKG")
    else:
        # 空のGeoDataFrameを作成
        centroid_gdf = gpd.GeoDataFrame(columns=["source_idx", "subpoly_id", "Cluster", "geometry"], crs=Poly05_5.crs, geometry="geometry")
        print("クラスタ重心が0個のため、保存をスキップします。")

    # 05_5_6: ヴォロノイ（ティーセン）分割による細長ポリゴンの最終分割
    # 高アスペクト比ポリゴンを重心点を使ったヴォロノイ分割で細分割
    final_polygons = []

    # 全ポリゴンを処理（分割対象外も結果に含めるため）
    for idx, row in Poly05_5.iterrows():
        geom = row.geometry
        aspect = row["aspect_ratio"]

        if aspect < aspect_ratio_threshold or idx not in centroid_gdf["source_idx"].values:
            # 分割対象外（しきい値未満または重心なし）→そのまま保持
            final_polygons.append({
                "source_idx": idx,
                "subpoly_id": -1,
                "Cluster": -1,
                "geometry": geom
            })
            continue

        # 分割対象ポリゴン（重心ポイントが存在）
        relevant_centroids = centroid_gdf[centroid_gdf["source_idx"] == idx]
        if len(relevant_centroids) < 2:
            # 重心が1個以下の場合は分割不可→そのまま保持
            final_polygons.append({
                "source_idx": idx,
                "subpoly_id": -1,
                "Cluster": -1,
                "geometry": geom
            })
            continue

        # 重心点をヴォロノイ分割に適合するMultiPointに変換（2D化）
        points = MultiPoint([force_2d(pt) for pt in relevant_centroids.geometry])

        # ヴォロノイ（ティーセン）分割の実行
        voronoi = voronoi_diagram(points, envelope=geom, tolerance=0.0)

        # 各ポリゴンをクリップして対応付け
        for cluster_idx, centroid_row in relevant_centroids.iterrows():
            pt = centroid_row.geometry
            cluster = centroid_row["Cluster"]
            subpoly_id = centroid_row["subpoly_id"]
            src_idx = centroid_row["source_idx"]

            # centroid が含まれる Voronoi 領域を探す
            for region in voronoi.geoms:
                if region.contains(pt):
                    clipped = region.intersection(geom)
                    if clipped.is_empty:
                        continue

                    final_polygons.append({
                        "source_idx": src_idx,
                        "subpoly_id": subpoly_id,
                        "Cluster": cluster,
                        "geometry": clipped
                    })
                    break

    # ヴォロノイ分割結果をGeoDataFrameに変換して保存
    final_gdf = gpd.GeoDataFrame(final_polygons, geometry="geometry", crs=Poly05_5.crs)
    # final_gdf.to_file(f"{output_folder}/05_5_6_Poly.gpkg", layer="split_polygons", driver="GPKG")

    if enable_voronoi_refinement:
        Poly05_5 = final_gdf.copy()
        print('05_5_6: ', Poly05_5.geom_type.value_counts()) # ジオメトリタイプを確認
        print("05_5_6 後: 細長メッシュ再分割", flush=True)
        Poly05_5 = split_elongated_mesh_gdf(Poly05_5, aspect_ratio_threshold, aspect_min_split_area)
        Poly05_5 = maybe_merge_elongated_slivers(Poly05_5, "05_5_6b")

    Poly05_5["aspect_ratio"] = Poly05_5.geometry.apply(aspect_ratio)
    n_high_aspect = int((Poly05_5["aspect_ratio"] > aspect_ratio_threshold).sum())
    print(
        f"05_5 aspect 最終: 面数={len(Poly05_5)}, "
        f"aspect>{aspect_ratio_threshold}={n_high_aspect}",
        flush=True,
    )

    Poly05_5 = retopologize_mesh_gdf(
        Poly05_5,
        snap_tol=0.05,
        domain_geom=input_gdf.union_all(),
        label="05_5_7",
    )
else:
    print("05_5_2-05_5_7: アスペクト比/Voronoi 再分割をスキップ", flush=True)

# ============================================================================
# フェーズ6: ノード・エッジ変換 (05_6-05_10)
# ============================================================================

# 05_6-05_7: ポリゴンから頂点（ノード）と辺（エッジ）への変換
def polygon_to_nodes_and_edges(polygon_gdf, precision=4):
    """
    入力のポリゴンGeoDataFrameから、重複しない頂点（Node）とライン（Edge）を抽出します。
    各頂点にはNodeIDを、各ラインにはLineIDを割り当て、
    ラインの始点・終点のNodeIDはそれぞれStartNode、EndNodeとして記録します。
    また、各頂点に対して、その頂点が始点または終点となっているラインのLineIDリスト(LineList)も保持します。

    Parameters:
        polygon_gdf (GeoDataFrame): ポリゴンデータ
        precision (int): 座標の丸め精度（小数点以下桁数）

    Returns:
        points_gdf (GeoDataFrame): 頂点データ（NodeID, LineList付き）
        lines_gdf (GeoDataFrame): ラインデータ（LineID, StartNode, EndNode付き）
    """
    # ノードとエッジの管理用辞書（重複を回避するため）
    nodes = {}          # {(x, y): NodeID} 座標→ノードIDのマッピング
    edges = {}          # {(始点, 終点): LineID} エッジの一意性を保つ
    point_to_lines = {} # {NodeID: [LineID, ...]} ノードに接続するエッジの一覧

    # IDの初期化
    node_id = 1
    line_id = 1

    # ポリゴンのジオメトリを処理
    for geom in polygon_gdf.geometry:
        # Polygon, MultiPolygonに対応
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            # 外周座標取得（リストの最初と最後が同一の場合は最後を除去）
            coords = list(poly.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]

            # 各頂点の登録（丸め処理して重複回避）
            for coord in coords:
                coord_round = (round(coord[0], precision), round(coord[1], precision))
                if coord_round not in nodes:
                    nodes[coord_round] = node_id
                    point_to_lines[node_id] = []
                    node_id += 1

            n = len(coords)
            # 外周の各辺（ライン）の登録（始点と終点は元の順序を保持）
            for i in range(n):
                start_coord = (round(coords[i][0], precision), round(coords[i][1], precision))
                end_coord   = (round(coords[(i+1) % n][0], precision), round(coords[(i+1) % n][1], precision))
                # 既に登録済み（逆方向も含む）ならスキップ
                if (start_coord, end_coord) in edges or (end_coord, start_coord) in edges:
                    continue
                edges[(start_coord, end_coord)] = line_id
                # 各頂点にこのラインIDを追加
                point_to_lines[nodes[start_coord]].append(line_id)
                point_to_lines[nodes[end_coord]].append(line_id)
                line_id += 1

    # 頂点レコードの作成
    vertex_records = []
    for coord, nid in nodes.items():
        vertex_records.append({
            "NodeID": nid,
            "LineList": point_to_lines[nid],
            "geometry": Point(coord)
        })

    # ラインレコードの作成
    line_records = []
    for (start, end), lid in edges.items():
        line_records.append({
            "LineID": lid,
            "StartNode": nodes[start],
            "EndNode": nodes[end],
            "geometry": LineString([start, end])
        })

    points_gdf = gpd.GeoDataFrame(vertex_records, geometry="geometry", crs=polygon_gdf.crs)
    lines_gdf  = gpd.GeoDataFrame(line_records,  geometry="geometry", crs=polygon_gdf.crs)

    return points_gdf, lines_gdf

# 05_5の分割結果からノード・エッジを抽出
Point05_5, Line05_5 = polygon_to_nodes_and_edges(Poly05_5, precision=4)
# Point05_5.to_file(f"{output_folder}/05_6_Point.gpkg", layer='point', driver="GPKG")
# Line05_5.to_file(f"{output_folder}/05_7_Line.gpkg", layer='line', driver="GPKG")
print('05_6: ',Point05_5.geom_type.value_counts()) # ジオメトリタイプを確認
print('05_7: ',Line05_5.geom_type.value_counts()) # ジオメトリタイプを確認

# 05_8: トラフィック（接続エッジ数）の計算
def add_traffic_column(points_gdf):
    """
    points_gdf に Traffic 列（接続するLineID数）を追加する。

    Parameters:
        points_gdf (GeoDataFrame): LineList 列が存在するポイントデータ

    Returns:
        GeoDataFrame: Traffic 列が追加されたポイントデータ
    """
    # LineListの要素数をカウントしてトラフィック値とする
    points_gdf['Traffic'] = points_gdf['LineList'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return points_gdf

# トラフィック列を追加してエッジ抽出処理を完了
Point05_5 = add_traffic_column(Point05_5)
# Point05_5.to_file(f"{output_folder}/05_8_Point.gpkg", layer='point', driver="GPKG")
print('05_8: ',Point05_5.geom_type.value_counts()) # ジオメトリタイプを確認

print(Point05_5[['NodeID', 'LineList', 'Traffic']].head())

# ============================================================================
# フェーズ7: 角度フィルタリングによる不要エッジの削除 (05_9-05_10)
# ============================================================================

# 05_9-05_10: 小さな角度で交わるエッジの自動統合処理
def calculate_angle(line1, line2):
    """
    2つの線分のなす角度を計算する関数
    
    Args:
        line1, line2: 角度計算対象の線分（LineString）
        
    Returns:
        角度（度数、0-180度）
    """
    # 線分の端点座標を取得
    x1, y1 = line1.coords[0]
    x2, y2 = line1.coords[1]
    x3, y3 = line2.coords[0]
    x4, y4 = line2.coords[1]

    # 線分の方向ベクトルを計算
    vector1 = np.array([x2 - x1, y2 - y1])
    vector2 = np.array([x4 - x3, y4 - y3])

    # 内積による余弦計算
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # 内積の値を[-1, 1]に制限（数値誤差対策）
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # 関数arccosで角度をラジアンから度に変換
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def angle_filtering(points_gdf, lines_gdf, angle_threshold=8.0):
    # 出力用のラインデータを準備
    new_lines_gdf = lines_gdf.copy()

    # 出力用のポイントデータを準備
    new_points_gdf = points_gdf.copy()

    # NodeIDとangleを格納するリストを初期化
    angle_data = []
    line_check = []

    # tqdmの進捗表示
    tqdm.pandas(desc="Processing points")

    # Trafficが2の場合のフィルタリングを事前に行う
    filtered_points_gdf = points_gdf[points_gdf['Traffic'] == 2].copy()

    # 'LineList'をリスト化（文字列 -> リスト）
    filtered_points_gdf['LineList'] = filtered_points_gdf['LineList'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # PointNodeIDごとにラインの情報を取得
    filtered_points_gdf, new_lines_gdf, line_list_dict = update_lines_and_points(filtered_points_gdf, lines_gdf)

    # ライン処理を行う
    for _, point in tqdm(filtered_points_gdf.iterrows(), total=len(filtered_points_gdf), desc="Filtering points"):
        #line_ids = line_list_dict.get(str(point['NodeID']), [])
        start_point = new_points_gdf.loc[new_points_gdf['NodeID'] == point['NodeID']]
        if isinstance(start_point['LineList'].iloc[0], str):  # 文字列の場合
            line_ids = eval(start_point['LineList'].iloc[0])
        else:  # すでにリストの場合
            line_ids = start_point['LineList'].iloc[0]

        if len(line_ids) == 2:
            lines_to_check = new_lines_gdf[new_lines_gdf['LineID'].isin(line_ids)]

            # lines_to_checkをCSV出力用に保存
            for _, line in lines_to_check.iterrows():
                line_check.append({
                    'NodeID': point['NodeID'],
                    'LineID': line['LineID'],
                    'LineGeometry': line['geometry']
                })

            if len(lines_to_check) == 2:
                line1, line2 = lines_to_check.iloc[0], lines_to_check.iloc[1]
                angle = calculate_angle(line1['geometry'], line2['geometry'])
                angle_data.append({'NodeID': point['NodeID'], 'Angle': angle})

                if angle <= angle_threshold:
                    # 結合するノードを特定（StartNode, EndNodeと一致するかを確認）
                    point1, NodeID1 = find_connected_point(line1, point, new_points_gdf)
                    point2, NodeID2 = find_connected_point(line2, point, new_points_gdf)

                    if not point1.empty and not point2.empty:
                        coord1 = point1.geometry.iloc[0]
                        coord2 = point2.geometry.iloc[0]

                        # 新しいLineStringを作成
                        new_line = create_new_line(NodeID1, NodeID2, coord1, coord2, new_lines_gdf)

                        # 新しいラインをリストに追加
                        new_lines_gdf = pd.concat([new_lines_gdf, new_line], ignore_index=True)

                        # 結合前のラインとポイントを削除
                        new_lines_gdf = new_lines_gdf[~new_lines_gdf['LineID'].isin([line1['LineID'], line2['LineID']])].copy()
                        new_points_gdf = new_points_gdf[~new_points_gdf['NodeID'].isin([point['NodeID']])].copy()

                        # LineListを再設定（point1, point2に対してLineListを更新）
                        new_points_gdf = update_line_list_for_points(new_points_gdf, point1['NodeID'], line1['LineID'], new_line['LineID'])
                        new_points_gdf = update_line_list_for_points(new_points_gdf, point2['NodeID'], line2['LineID'], new_line['LineID'])

                        # どこかでNodeIDが文字列に変わるため、整数に変換
                        new_points_gdf['NodeID'] = new_points_gdf['NodeID'].astype(int).copy()

    # 重複削除
    new_points_gdf = new_points_gdf.drop_duplicates(subset='geometry')
    new_lines_gdf = new_lines_gdf.drop_duplicates(subset='geometry')

    # NodeIDとangleの情報をCSVに出力（デバッグ用：コメントアウト）
    # angle_df = pd.DataFrame(angle_data)
    # angle_df.to_csv('node_id_and_angle.csv', index=False)

    # lines_to_checkの情報をCSVに出力（デバッグ用：コメントアウト）
    # line_check_df = pd.DataFrame(line_check)
    # line_check_df.to_csv('lines_to_check.csv', index=False)

    # NodeIDとLineIDの振り直し
    new_points_gdf.loc[:, 'NodeID'] = range(0, len(new_points_gdf))
    new_lines_gdf.loc[:, 'LineID'] = range(0, len(new_lines_gdf))

    # StartNode, EndNode再設定
    new_points_gdf, new_lines_gdf = process_geodata_optimized(new_points_gdf, new_lines_gdf)

    # LineList再設定
    new_points_gdf, new_lines_gdf, _ = update_lines_and_points(new_points_gdf, new_lines_gdf)

    # 出力用のラインデータとポイントデータをコピー（エラー回避用）
    new_points_gdf = new_points_gdf.copy()
    new_lines_gdf = new_lines_gdf.copy()

    # CRSの設定
    new_points_gdf.set_crs(points_gdf.crs, allow_override=True, inplace=True)
    new_lines_gdf.set_crs(lines_gdf.crs, allow_override=True, inplace=True)

    return new_points_gdf, new_lines_gdf

def update_lines_and_points(points_gdf, lines_gdf):
    # 'LineList'属性が存在しない場合は初期化する
    if 'LineList' not in points_gdf.columns:
        points_gdf['LineList'] = None

    # StartNode と EndNode
    points_gdf.loc[:, 'NodeID'] = points_gdf['NodeID']
    lines_gdf.loc[:, 'StartNode'] = lines_gdf['StartNode']
    lines_gdf.loc[:, 'EndNode'] = lines_gdf['EndNode']

    # NodeID と一致する StartNode または EndNode の LineID を取得
    # LineID がStartNode または EndNode に含まれているラインを抽出する
    start_lines = lines_gdf[['LineID', 'StartNode']]
    end_lines = lines_gdf[['LineID', 'EndNode']]

    # ポイントデータの NodeID と一致する LineID をリスト化
    start_match = pd.merge(points_gdf[['NodeID']], start_lines, how='left', left_on='NodeID', right_on='StartNode')
    end_match = pd.merge(points_gdf[['NodeID']], end_lines, how='left', left_on='NodeID', right_on='EndNode')

    # マッチした結果の LineID をリストとして格納
    line_list_dict = {}

    # start_match と end_match の結果を結合し、LineID をリスト化
    for idx, row in start_match.iterrows():
        node_id = row['NodeID']
        line_id = row['LineID']
        if node_id not in line_list_dict:
            line_list_dict[node_id] = []
        if pd.notna(line_id) and line_id not in line_list_dict[node_id]:
            line_list_dict[node_id].append(int(line_id))  # 明示的に整数に変換

    for idx, row in end_match.iterrows():
        node_id = row['NodeID']
        line_id = row['LineID']
        if node_id not in line_list_dict:
            line_list_dict[node_id] = []
        if pd.notna(line_id) and line_id not in line_list_dict[node_id]:
            line_list_dict[node_id].append(int(line_id))  # 明示的に整数に変換

    # 'LineList' カラムにリスト化された LineID をセット
    points_gdf['LineList'] = points_gdf['NodeID'].map(line_list_dict)

    # 整数に戻す
    points_gdf.loc[:, 'NodeID'] = points_gdf['NodeID'].astype(int)
    lines_gdf.loc[:, 'StartNode'] = lines_gdf['StartNode'].astype(int)
    lines_gdf.loc[:, 'EndNode'] = lines_gdf['EndNode'].astype(int)

    return points_gdf, lines_gdf, line_list_dict

def process_geodata_optimized(new_points_gdf, new_lines_gdf):
    # NodeID辞書の作成
    node_id_dict = {tuple(point.geometry.coords[0]): point['NodeID'] for _, point in new_points_gdf.iterrows()}

    # new_lines_gdfの端点座標を抽出 -> StartNode、EndNodeをNodeIDに基づいて上書き
    for idx, line in new_lines_gdf.iterrows():
        # ラインの始点と終点の座標を取得
        start_coords = tuple(line.geometry.coords[0])
        end_coords = tuple(line.geometry.coords[-1])

        # 始点・終点の座標を NodeID 辞書で検索して、それぞれの NodeID を取得
        start_node_id = node_id_dict.get(start_coords, None)
        end_node_id = node_id_dict.get(end_coords, None)

        # StartNode と EndNode を new_lines_gdf に上書き
        new_lines_gdf.loc[idx, 'StartNode'] = start_node_id
        new_lines_gdf.loc[idx, 'EndNode'] = end_node_id

    new_lines_gdf.loc[:, 'StartNode'] = new_lines_gdf['StartNode'].astype(int)
    new_lines_gdf.loc[:, 'EndNode'] = new_lines_gdf['EndNode'].astype(int)

    return new_points_gdf, new_lines_gdf

def update_line_list_for_points(points_gdf, node_id, old_line_id, new_line_id):
    # 指定されたnode_idのポイントのLineListを更新する
    points_gdf = points_gdf.copy()  # データフレームの変更を避けるためにコピーを作成

    # NodeID列を文字列型に変換しておく
    node_id = str(node_id)  # node_idが文字列であることを確認
    points_gdf['NodeID'] = points_gdf['NodeID'].astype(str).copy()

    matching_points = points_gdf[points_gdf['NodeID'] == node_id].copy()

    # 各ポイントに対してLineListを更新
    for idx, point in matching_points.iterrows():
        line_list = point['LineList']
        if old_line_id in line_list:
            line_list.remove(old_line_id)  # old_line_idを削除
        line_list.append(new_line_id)  # new_line_idを追加
        points_gdf.at[idx, 'LineList'] = line_list  # 更新されたLineListをセット

    return points_gdf

# 新しいラインの作成
def create_new_line(NodeID1, NodeID2, coord1, coord2, new_lines_gdf):
    line = LineString([coord1, coord2])
    new_line = gpd.GeoDataFrame({
        'LineID': [max(new_lines_gdf['LineID']) + 1],
        'StartNode': [NodeID1],
        'EndNode': [NodeID2],
        'geometry': [line]
    }, crs=new_lines_gdf.crs)
    return new_line

# 接続する点を取得
def find_connected_point(line, point, new_points_gdf):
    if line['StartNode'] == point['NodeID']:
        connected_point = new_points_gdf[new_points_gdf['NodeID'] == line['EndNode']]
        NodeID = line['EndNode']
    else:
        connected_point = new_points_gdf[new_points_gdf['NodeID'] == line['StartNode']]
        NodeID = line['StartNode']
    return connected_point, NodeID

Point05_6, Line05_6 = angle_filtering(Point05_5, Line05_5, angle_threshold=8.0)

# 繰り返し回数のカウンタ
iteration_count = 1

# 初期のデータ数を取得
while True:
    iteration_count += 1

    Point05_6_2, Line05_6_2 = angle_filtering(Point05_6, Line05_6, angle_threshold=8.0) # 上書きスタイル

    # データ数が一致するか確認
    if len(Point05_6) == len(Point05_6_2):
        print(f"Processing completed successfully after {iteration_count} iterations.")
        break
    else:
        print(f"Iteration {iteration_count}: point_inputfile count = {len(Point05_6)}, point_outputfile count = {len(Point05_6_2)}")
        Point05_6 = Point05_6_2
        Line05_6  = Line05_6_2

# Point05_6_2.to_file(f"{output_folder}/05_9_Point.gpkg", layer='point', driver="GPKG")
# Line05_6_2.to_file(f"{output_folder}/05_10_Line.gpkg", layer='line', driver="GPKG")

# ============================================================================
# フェーズ8: 最終ポリゴン生成とダミーメッシュ処置 (06-06_2)
# ============================================================================

# 06: 角度フィルタリング後のエッジから最終ポリゴン生成
Poly06 = process_line_to_polygons(Line05_6_2)

if enable_merge_small_polygons:
    Poly06 = merge_small_polygons(Poly06)
else:
    print("06: merge_small_polygons=false のため小ポリゴン結合をスキップ", flush=True)

# ===== ユーティリティ =====
# 形状の妥当化（Shapely 2.x があれば make_valid、なければ buffer(0)）
try:
    from shapely.validation import make_valid
    def _fix_geom(geom):
        return make_valid(geom)
except Exception:
    def _fix_geom(geom):
        try:
            return geom.buffer(0)
        except Exception:
            return geom  # 最後の保険

def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    return gdf

def exterior_as_lines(geom):
    # Polygon → LineString、MultiPolygon → MultiLineString（外周のみ）
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return LineString(geom.exterior.coords)  # ★ LinearRing → LineString
    if geom.geom_type == "MultiPolygon":
        return MultiLineString([LineString(p.exterior.coords) for p in geom.geoms if not p.is_empty])
    # それ以外は境界線。必要に応じて holes を含む点に注意
    b = geom.boundary
    return b

# ===== 0) 前処理：CRS整合 & ジオメトリ妥当化 =====
Poly06 = fix_geometries(Poly06)

if Poly06.crs is None:
    raise ValueError("Poly06 の CRS が未設定です。CRS を設定してください。")

# ダミーメッシュの処理（オプショナル）
if dummy_file is not None:
    # ダミーメッシュ読み込み（キャリブレーション対象メッシュ）
    print("Info: ダミーメッシュを読み込んでいます...")
    dummy_mesh_gdf = gpd.read_file(dummy_file)
    dummy_mesh_gdf = fix_geometries(dummy_mesh_gdf)
    
    if dummy_mesh_gdf.crs != Poly06.crs:
        dummy_mesh_gdf = dummy_mesh_gdf.to_crs(Poly06.crs)
    
    # ===== ① Poly06 の外周線を作成 =====
    # 複数フィーチャを1つにまとめ、その外周線のみを抽出
    poly06_union = Poly06.geometry.union_all()                # (Multi)Polygon
    poly06_outline = exterior_as_lines(poly06_union)          # (Multi)LineString or LineString
    poly06_outline_gdf = gpd.GeoDataFrame(geometry=[poly06_outline], crs=Poly06.crs)
    
    # 外周線の保存
    # poly06_outline_gdf.to_file(f"{output_folder}/06_1_outline.gpkg", layer="line", driver="GPKG")
    
    # ===== ② 外周線（の内側＝Poly06 領域）で dummy_mesh_gdf をカット =====
    # gpd.clip はポリゴン領域でクリップするので、外周線そのものではなくポリゴン本体（union）を使用
    mesh_in_poly06 = gpd.clip(dummy_mesh_gdf, poly06_union)
    
    # ===== ③ カット後の dummy_mesh_gdf で Poly06 を分割 =====
    use_dummy_mesh = True
else:
    # ダミーメッシュが指定されていない場合
    print("Info: ダミーメッシュを使用しません。全てのセルをCalMesh=0として処理します。")
    mesh_in_poly06 = None
    use_dummy_mesh = False

# ===== ユーティリティ関数定義（共通） =====
def explode_singleparts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """MultiPolygon を行分割して必ず Polygon 単位にする（index も整理）"""
    try:
        # GeoPandas 0.12+ 推奨
        gdf_single = gdf.explode(index_parts=True, ignore_index=True)
    except TypeError:
        # 旧版フォールバック
        gdf_single = gdf.explode(index_parts=True)
        gdf_single = gdf_single.reset_index(drop=True)
    # 念のためポリゴン以外（万一の線/点）は除外
    if "geom_type" in gdf_single.columns:
        mask = gdf_single["geom_type"].isin(["Polygon"])
    else:
        mask = gdf_single.geometry.geom_type.isin(["Polygon"])
    return gdf_single[mask].copy()

# ===== ③ ダミーメッシュによる分割処理（オプショナル） =====
if use_dummy_mesh:
    # ===== ③' 分割前に Poly06 をシングルパート化（推奨） =====
    Poly06_single = explode_singleparts(Poly06)
    
    # ===== ③ 分割（右辺はカット済みメッシュの geometry のみ） =====
    Poly06 = gpd.overlay(
        Poly06_single,
        mesh_in_poly06[["geometry"]],
        how="identity",
        keep_geom_type=True,     # ポリゴン以外の余剰（交線など）を除外
    )
    
    # ===== ③'' 念押しで分割結果もシングルパート化 =====
    Poly06 = explode_singleparts(Poly06)
else:
    # ダミーメッシュがない場合はシングルパート化のみ
    Poly06 = explode_singleparts(Poly06)

# ===== ④ 分割後の Poly06 を GeoPackage 出力 =====
out_path = Path(f"{output_folder}/06_1_Poly.gpkg")
# 同名レイヤを上書きしたい場合はファイルを消す（GeoPackageは1ファイル多レイヤのため）
# 既存ファイルを残して同名レイヤを更新する確実なAPIがないため、最も単純な方法を採用
if out_path.exists():
    out_path.unlink()

# Poly06.to_file(out_path, layer="poly", driver="GPKG")

# 念のため Polygon のみ残す（マルチや線が紛れても弾く）
Poly06 = Poly06[Poly06.geometry.geom_type == "Polygon"].copy()

# sjoinで「メッシュ内に含まれる」かを判定（within）- ダミーメッシュがある場合のみ
if use_dummy_mesh:
    left = Poly06[["geometry"]].copy()
    right = mesh_in_poly06[["geometry"]].copy()

def _auto_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    """地理座標系のとき、重心の経度から UTM EPSG を推定（WGS84系）。"""
    c = gdf.union_all().centroid
    lon = float(c.x)
    lat = float(c.y)
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone  # 326=北半球, 327=南半球

def add_calmesh_by_coverage(poly06_split: gpd.GeoDataFrame,
                            mesh_in_poly06: gpd.GeoDataFrame,
                            threshold: float = 0.9,
                            shrink_tol: float = 0.0) -> gpd.GeoDataFrame:
    """
    ダミーメッシュとの面積重複率でキャリブレーションメッシュを判定する関数
    
    判定基準: 交差面積 / ポリゴン面積 >= threshold でCalMesh=1（キャリブレーションメッシュ）
    
    Args:
        poly06_split: 対象ポリゴンGeoDataFrame
        mesh_in_poly06: ダミーメッシュGeoDataFrame  
        threshold: キャリブレーション判定の面積閾値（デフォルト0.9=90%）
        shrink_tol: メッシュ収縮許容値（バッファ誤差対策、0.0=縮約なし）
        
    Returns:
        キャリブレーション判定フラグ付きポリゴンGeoDataFrame
    """
    # 前提: CRS整合
    if poly06_split.crs is None or mesh_in_poly06.crs is None:
        raise ValueError("CRS が未設定です。両方に CRS を設定してください。")
    if mesh_in_poly06.crs != poly06_split.crs:
        mesh_in_poly06 = mesh_in_poly06.to_crs(poly06_split.crs)

    # 面積計算は投影座標で（地理座標なら UTM に一時投影）
    work_crs = poly06_split.crs
    need_project = getattr(work_crs, "is_geographic", False)
    if need_project:
        utm_epsg = _auto_utm_epsg(poly06_split)
        poly_w = poly06_split.to_crs(utm_epsg)
        mesh_w = mesh_in_poly06.to_crs(utm_epsg)
    else:
        poly_w = poly06_split
        mesh_w = mesh_in_poly06

    # メッシュ全体を union（重なりが無い前提でも、これが最も頑健）
    mesh_union = unary_union(mesh_w.geometry)
    if shrink_tol and shrink_tol > 0:
        try:
            mesh_union = mesh_union.buffer(-shrink_tol)
        except Exception:
            # 収縮で消滅した場合などはそのまま使う
            pass

    prepared = prep(mesh_union)

    # 交差面積の割合（coverage）を算出
    def _coverage_ratio(g):
        if g.is_empty:
            return 0.0
        a = g.area
        if a == 0:
            return 0.0
        # 交差しなければ 0（高速化のため prepared.intersects）
        if not prepared.intersects(g):
            return 0.0
        inter = g.intersection(mesh_union)
        return (inter.area / a) if not inter.is_empty else 0.0

    coverage = poly_w.geometry.apply(_coverage_ratio)

    # 元の GeoDataFrame に属性付与（geometry/CRS はそのまま）
    out = poly06_split.copy()
    out["coverage"] = coverage.values
    out["CalMesh"] = (out["coverage"] >= float(threshold)).astype("int8")
    return out

# CalMesh判定の実行（ダミーメッシュがある場合のみ）
if use_dummy_mesh:
    # 90%面積重複率でキャリブレーションメッシュ判定
    Poly06 = add_calmesh_by_coverage(Poly06, mesh_in_poly06,
                                        threshold=0.90,  # 90%以上の重複率ならキャリブレーション対象
                                        shrink_tol=0.0)  # 収縮許容値0（ノイズ対策なし）
else:
    # ダミーメッシュがない場合は全てCalMesh=0に設定
    Poly06["CalMesh"] = np.int8(0)

# CalMesh判定済みポリゴンを保存
out_path = Path(f"{output_folder}/06_1_1_Poly.gpkg")
if out_path.exists():
    out_path.unlink()
# Poly06.to_file(out_path, layer="poly", driver="GPKG")

def _remove_holes(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms if not p.is_empty])
    return geom

def dissolve_calmesh1(poly06_split: gpd.GeoDataFrame,
                    merge_tol: float = 0.0,
                    drop_holes: bool = False,
                    keep_largest: bool = False):
    """CalMesh=1 を 1 ジオメトリへ溶解（必要に応じて穴除去/最大島のみ残す）"""
    g = poly06_split[poly06_split["CalMesh"] == 1]
    if g.empty:
        return None

    geoms = g.geometry
    if merge_tol and merge_tol > 0:
        geoms = geoms.buffer(0)  # 軽い修復
        dissolved = geoms.buffer(merge_tol).unary_union.buffer(-merge_tol)
    else:
        dissolved = unary_union(geoms)

    if drop_holes:
        dissolved = _remove_holes(dissolved)

    if keep_largest and dissolved.geom_type == "MultiPolygon":
        dissolved = max(list(dissolved.geoms), key=lambda p: p.area)
        dissolved = Polygon(dissolved.exterior)

    return dissolved

# CalMesh=1の溶解処理（ダミーメッシュがある場合のみ）
if use_dummy_mesh:
    # 使い方（基本：そのまま溶解）
    zeros = Poly06[Poly06["CalMesh"] == 0].copy()
    
    dissolved_geom = dissolve_calmesh1(Poly06,
                                    merge_tol=0.0,     # 近接スリバー吸収したい場合は >0 に
                                    drop_holes=False,  # 穴を消す場合 True
                                    keep_largest=False # 単一Polygonにしたい場合 True
                                    )
    
    # 出力用に、CalMesh=0 をそのまま + CalMesh=1 の溶解 1 レコードを結合
    if dissolved_geom is not None:
        # ★ 最小スキーマのみ（CalMesh と geometry）。他列は持たせない（= 警告回避の肝）
        poly_one = gpd.GeoDataFrame(
            {"CalMesh": [np.int8(1)]},
            geometry=[dissolved_geom],
            crs=Poly06.crs
        )
    else:
        poly_one = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=Poly06.crs), crs=Poly06.crs)
else:
    # ダミーメッシュがない場合は溶解処理不要（全てCalMesh=0）
    zeros = Poly06.copy()
    poly_one = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=Poly06.crs), crs=Poly06.crs)

def _drop_all_na_columns(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """geometry を除く『全行NA』の列を落とす（concat前の保険）。"""
    if df is None or df.empty:
        return df
    non_geom = [c for c in df.columns if c != "geometry"]
    to_drop = [c for c in non_geom if df[c].isna().all()]
    return df.drop(columns=to_drop) if to_drop else df

# ★ 空のDFを除外してから concat（警告回避）
dfs = []
for df in (zeros, poly_one):
    if df is not None and not df.empty:
        dfs.append(_drop_all_na_columns(df))

if not dfs:
    Poly06 = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=Poly06.crs), crs=Poly06.crs)
elif len(dfs) == 1:
    Poly06 = dfs[0].copy()
else:
    # ※ ここで union スキーマになりますが、poly_one は最小列のみなので警告は出ません
    Poly06 = gpd.GeoDataFrame(
        pd.concat(dfs, ignore_index=True, sort=False),
        geometry="geometry",
        crs=Poly06.crs
    )

# 型調整
if "CalMesh" in Poly06.columns:
    Poly06["CalMesh"] = Poly06["CalMesh"].astype("int8")

# GeoPackage に保存（別ファイルに出力）
out_path = Path(f"{output_folder}/06_1_2_Poly.gpkg")
if out_path.exists():
    out_path.unlink()
# Poly06.to_file(out_path, layer="Poly", driver="GPKG")
print(f"出力先: {out_path.resolve()}")

# 06_2: 面積, CNフィールドを追加
Poly06['area'] = Poly06.geometry.area
Poly06['CN'] = range(1, len(Poly06) + 1)

# Poly06.to_file(f"{output_folder}/06_2_Poly.gpkg", layer='poly', driver="GPKG")
print('06_2 (Poly): ',Poly06.geom_type.value_counts()) # ジオメトリタイプを確認

def polygon_to_nodes_and_edges_with_attributes(polygon_gdf, precision=4):
    """
    ポリゴンデータから、属性を維持しつつ、頂点(Point)と線分(Line)データを作成する。
    同じ座標の頂点には同じNodeIDを、同じ辺（向きに関係なく）には同じLineIDを割り当てる。
    同じセル（CN）内での重複を防ぐ。

    Parameters:
        polygon_gdf (GeoDataFrame): ポリゴンデータ
        precision (int): 座標丸め精度（小数点以下桁数）

    Returns:
        points_gdf (GeoDataFrame): 頂点データ（NodeID、ポリゴン属性付き）
        lines_gdf (GeoDataFrame): 線分データ（LineID、StartNode、EndNode、ポリゴン属性付き）
    """

    # グローバル管理用の辞書
    coord_to_node_id = {}  # {(x, y): NodeID} 座標→ノードIDのマッピング
    edge_to_line_id = {}   # {(coord1, coord2): LineID} 正規化された辺→ラインIDのマッピング
    
    # ノード・ライン情報を格納する辞書
    # {NodeID: {CN: ノードデータ}} - CNごとに1つのみ保持
    node_info = {}  
    # {LineID: {CN: ラインデータ}} - CNごとに1つのみ保持
    line_info = {}  

    node_id_counter = 1
    line_id_counter = 1

    def normalize_edge(coord1, coord2):
        """
        辺を正規化（座標の小さい方を最初にする）
        これにより、(A→B)と(B→A)を同じ辺として扱える
        """
        return tuple(sorted([coord1, coord2]))

    for idx, row in polygon_gdf.iterrows():
        geom = row.geometry
        cn = row.get('CN', idx)  # セル番号を取得（存在しない場合はインデックス）

        if isinstance(geom, (Polygon, MultiPolygon)):
            polygons = [geom] if isinstance(geom, Polygon) else geom.geoms

            # このセルで既に処理したノードと辺を追跡（セルレベルで管理）
            processed_nodes_in_cell = set()
            processed_edges_in_cell = set()

            for poly in polygons:
                # 外周座標を取得
                exterior_coords = list(poly.exterior.coords)
                
                # 平面直角座標系では反時計回り（CCW）が標準
                # Shapelyで向きを確認し、時計回りの場合は反転
                from shapely.geometry import LinearRing
                ring = LinearRing(exterior_coords)
                if not ring.is_ccw:
                    # 時計回り → 反時計回りに反転
                    # 最後の座標（閉じた座標）を除去してから反転
                    if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                        exterior_coords = exterior_coords[:-1]
                    exterior_coords = exterior_coords[::-1]
                    # 辺の処理のために閉じた座標を再追加
                    exterior_coords = exterior_coords + [exterior_coords[0]]
                
                # ノード・ライン生成
                for i in range(len(exterior_coords) - 1):
                    # 座標丸め
                    start_coords = (round(exterior_coords[i][0], precision), round(exterior_coords[i][1], precision))
                    end_coords = (round(exterior_coords[i + 1][0], precision), round(exterior_coords[i + 1][1], precision))

                    # 退化エッジ（座標丸めで始点=終点に潰れたゼロ長辺）をスキップ。
                    # quadtree clip 等で生じる重複頂点・極小スリバー由来。
                    # これを残すと StartNode==EndNode の不正辺となり 02 で警告/欠落する。
                    if start_coords == end_coords:
                        continue

                    # === ノード処理 ===
                    # 始点のノードID取得または新規作成
                    if start_coords not in coord_to_node_id:
                        coord_to_node_id[start_coords] = node_id_counter
                        node_info[node_id_counter] = {}
                        node_id_counter += 1
                    start_node_id = coord_to_node_id[start_coords]

                    # 終点のノードID取得または新規作成
                    if end_coords not in coord_to_node_id:
                        coord_to_node_id[end_coords] = node_id_counter
                        node_info[node_id_counter] = {}
                        node_id_counter += 1
                    end_node_id = coord_to_node_id[end_coords]

                    # ノードデータを記録（同じCNで未処理の場合のみ）
                    if start_node_id not in processed_nodes_in_cell:
                        start_point_data = row.to_dict()
                        start_point_data['NodeID'] = start_node_id
                        start_point_data['geometry'] = Point(start_coords)
                        node_info[start_node_id][cn] = start_point_data
                        processed_nodes_in_cell.add(start_node_id)

                    if end_node_id not in processed_nodes_in_cell:
                        end_point_data = row.to_dict()
                        end_point_data['NodeID'] = end_node_id
                        end_point_data['geometry'] = Point(end_coords)
                        node_info[end_node_id][cn] = end_point_data
                        processed_nodes_in_cell.add(end_node_id)

                    # === エッジ処理 ===
                    # 辺を正規化（向きに関係なく同じ辺として扱う）
                    normalized_edge = normalize_edge(start_coords, end_coords)
                    
                    # 辺のラインID取得または新規作成
                    if normalized_edge not in edge_to_line_id:
                        edge_to_line_id[normalized_edge] = line_id_counter
                        line_info[line_id_counter] = {}
                        line_id_counter += 1
                    line_id = edge_to_line_id[normalized_edge]

                    # ラインデータを記録（同じCNで未処理の場合のみ）
                    if line_id not in processed_edges_in_cell:
                        line_data = row.to_dict()
                        line_data['LineID'] = line_id
                        line_data['StartNode'] = start_node_id
                        line_data['EndNode'] = end_node_id
                        line_data['geometry'] = LineString([start_coords, end_coords])
                        line_info[line_id][cn] = line_data
                        processed_edges_in_cell.add(line_id)

    # 全てのデータを展開（CNごとに1つずつ）
    point_records = []
    for node_id, cn_dict in node_info.items():
        for cn, point_data in cn_dict.items():
            point_records.append(point_data)
    
    line_records = []
    for line_id, cn_dict in line_info.items():
        for cn, line_data in cn_dict.items():
            line_records.append(line_data)

    # GeoDataFrame作成
    points_gdf = gpd.GeoDataFrame(point_records, geometry='geometry', crs=polygon_gdf.crs)
    lines_gdf = gpd.GeoDataFrame(line_records, geometry='geometry', crs=polygon_gdf.crs)

    # 統計情報の表示
    unique_nodes = len(coord_to_node_id)
    unique_lines = len(edge_to_line_id)
    total_node_records = sum(len(cn_dict) for cn_dict in node_info.values())
    total_line_records = sum(len(cn_dict) for cn_dict in line_info.values())
    
    print(f"ユニークなノード数: {unique_nodes}")
    print(f"ユニークなライン数: {unique_lines}")
    print(f"総ノードレコード数: {total_node_records}（各セルごとに1つ）")
    print(f"総ラインレコード数: {total_line_records}（各セルごとに1つ）")

    return points_gdf, lines_gdf

Point06, Line06 = polygon_to_nodes_and_edges_with_attributes(Poly06, precision=4) # ここのPointデータは不使用

# LN追加（LineIDと同じ値を使用：重複除去済み）
Line06['LN'] = Line06['LineID']

# Line06.to_file(f"{output_folder}/06_2_Line.gpkg", layer='line', driver="GPKG")
print('06_2 (Line): ',Line06.geom_type.value_counts()) # ジオメトリタイプを確認

# 06_3: 外周かの判断（外周 ID=1, 内部 ID=0）
def _auto_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    c = gdf.unary_union.centroid
    lon, lat = float(c.x), float(c.y)
    zone = int(math.floor((lon + 180) / 6) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone

def _exteriors_as_lines(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return LineString(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon":
        return MultiLineString([LineString(p.exterior.coords) for p in geom.geoms if not p.is_empty])
    return geom.boundary

def _line_overlap_ratio_with_band(line, band_geom):
    if line.is_empty or line.length == 0:
        return 0.0
    inter = line.intersection(band_geom)
    if inter.is_empty:
        return 0.0
    return inter.length / line.length

def label_lines_robust(Line06: gpd.GeoDataFrame,
                    polys: gpd.GeoDataFrame,
                    tol_outer: float = 0.20,
                    tol_int: float = 0.20,
                    outer_ratio: float = 0.90,
                    interface_ratio: float = 0.70) -> gpd.GeoDataFrame:
    """
    ラインに ID を付与（1:外枠, 2:0/1境界, 0:その他）。
    tol_* はバッファ帯の半幅（CRS単位: m 推奨）。
    """
    # --- 前処理: CRS ---
    if Line06.crs is None or polys.crs is None:
        raise ValueError("Line06 / polys の CRS が未設定です。")
    if Line06.crs != polys.crs:
        polys = polys.to_crs(Line06.crs)

    work_crs = Line06.crs
    is_geo = getattr(work_crs, "is_geographic", False)

    if is_geo:
        utm = _auto_utm_epsg(polys)
        lines_w = Line06.to_crs(utm)
        polys_w = polys.to_crs(utm)
    else:
        lines_w = Line06.copy()
        polys_w = polys.copy()

    # --- union 作成 ---
    U = unary_union(polys_w.geometry)  # 全体
    cal1 = unary_union(polys_w.loc[polys_w["CalMesh"] == 1, "geometry"]) if (polys_w["CalMesh"] == 1).any() else None
    cal0 = unary_union(polys_w.loc[polys_w["CalMesh"] == 0, "geometry"]) if (polys_w["CalMesh"] == 0).any() else None

    # --- 外枠帯（外側リングのみを太らせる）---
    outer_lines = _exteriors_as_lines(U)
    band_outer = outer_lines.buffer(tol_outer)  # 線→帯

    # --- 0/1 境界帯 ---
    if (cal0 is not None) and (cal1 is not None) and (not cal0.is_empty) and (not cal1.is_empty):
        band_interface = cal1.buffer(tol_int).intersection(cal0.buffer(tol_int))
    else:
        band_interface = None  # 片方しか無ければ ID=2 は存在しない

    # --- 判定 ---
    g = lines_w.copy()
    g["ID"] = 0

    # 1) 外枠
    g["__r_outer"] = g.geometry.apply(lambda ln: _line_overlap_ratio_with_band(ln, band_outer))
    g.loc[g["__r_outer"] >= float(outer_ratio), "ID"] = 1

    # 2) 0/1境界（外枠以外のみ）
    if band_interface is not None and (not band_interface.is_empty):
        mask = g["ID"] != 1
        g.loc[mask, "__r_int"] = g.loc[mask, "geometry"].apply(lambda ln: _line_overlap_ratio_with_band(ln, band_interface))
        g.loc[mask & (g["__r_int"] >= float(interface_ratio)), "ID"] = 2
        g.drop(columns=["__r_int"], inplace=True, errors="ignore")

    g.drop(columns=["__r_outer"], inplace=True, errors="ignore")
    g["ID"] = g["ID"].astype("int8")

    # 元CRSに戻す
    if is_geo:
        g = g.to_crs(work_crs)

    return g

Line06 = label_lines_robust(Line06, Poly06, tol_outer=0.20, tol_int=0.20, outer_ratio=0.90, interface_ratio=0.70)

# 重複チェック：同じCN+LNの組み合わせが複数ある場合は1つに統合
print(f"\n=== Line06の重複チェック ===")
print(f"処理前のレコード数: {len(Line06)}")
cn_ln_counts_before = Line06.groupby(['CN', 'LN']).size()
duplicates_before = cn_ln_counts_before[cn_ln_counts_before > 1]
if len(duplicates_before) > 0:
    print(f"⚠️ 重複しているCN+LN: {len(duplicates_before)}件")
    print(f"   最大重複数: {duplicates_before.max()}")
    # 各CN+LNの組み合わせごとに最初のレコードのみを保持
    Line06 = Line06.drop_duplicates(subset=['CN', 'LN'], keep='first')
    print(f"処理後のレコード数: {len(Line06)}")
else:
    print("✓ 重複なし")

# Line06.to_file(f"{output_folder}/06_3_Line.gpkg", layer='line', driver="GPKG")

# 06_4: ラインデータの端点からポイント作成
def extract_endpoints_from_lines(lines_gdf):
    """
    ラインデータから端点（始点・終点）をポイントデータとして抽出する。
    Parameters:
        lines_gdf (GeoDataFrame): ラインデータ
    Returns:
        GeoDataFrame: 端点データ（Point型、StartEndフラグ付き）
    """

    endpoints = []

    for idx, line in lines_gdf.iterrows():
        if line.geometry.is_empty or line.geometry is None:
            continue

        # 始点・終点を取得
        start_point = Point(line.geometry.coords[0])
        end_point = Point(line.geometry.coords[-1])

        # 属性を持たせたい場合（オプション）
        start_record = line.copy()
        start_record.geometry = start_point
        start_record['StartEnd'] = 'start'

        end_record = line.copy()
        end_record.geometry = end_point
        end_record['StartEnd'] = 'end'

        endpoints.append(start_record)
        endpoints.append(end_record)

    # GeoDataFrame化
    endpoints_gdf = gpd.GeoDataFrame(endpoints, crs=lines_gdf.crs)

    return endpoints_gdf

Point06 = extract_endpoints_from_lines(Line06) # Lineの端点に変更（LineID、外枠IDが欲しい）
# Point06.to_file(f"{output_folder}/06_4_Point.gpkg", layer='point', driver="GPKG")

# 06_5: ラスタサンプリング
def raster_sampling(gdf, raster_file, column_prefix):
    # 元の CRS を記録
    original_crs = gdf.crs
    original_crs_obj = CRS(original_crs)

    # ラスタファイルを読み込む
    with rasterio.open(raster_file) as src:
        raster_crs = src.crs
        raster_crs_obj = CRS(raster_crs)

        # EPSGコードが取得できない場合、強制的にEPSG:2451に設定
        if raster_crs.to_epsg() is None:
            epsg_code = DEM_epsg
            raster_crs = rasterio.crs.CRS.from_epsg(epsg_code)
            print(f"Warning: ラスターの EPSG コードが取得できなかったため、EPSG:{epsg_code} を使用します。")

        print(f"入力データの CRS: {gdf.crs}")
        print(f"DEM の CRS (EPSG): {raster_crs}")

        # geodataframe の CRS がラスタと異なる場合は変換する
        if not original_crs_obj.equals(raster_crs_obj):
            gdf_temp = gdf.to_crs(raster_crs)
        else:
            gdf_temp = gdf.copy()

        # サンプリング処理
        for idx, row in gdf_temp.iterrows():
            geom = row.geometry
            
            try:
                # Pointジオメトリの場合はsampleメソッドを使う
                if geom.geom_type == 'Point':
                    # ポイント座標でサンプリング
                    coords = [(geom.x, geom.y)]
                    values = list(src.sample(coords))
                    if len(values) > 0:
                        value = values[0][0]
                        if value != src.nodata:
                            gdf_temp.at[idx, f'{column_prefix}_mean'] = value
                        else:
                            gdf_temp.at[idx, f'{column_prefix}_mean'] = np.nan
                    else:
                        gdf_temp.at[idx, f'{column_prefix}_mean'] = np.nan
                else:
                    # Polygon/LineStringの場合はmaskメソッドを使う
                    geom_interface = [geom.__geo_interface__]
                    out_image, out_transform = rasterio.mask.mask(src, geom_interface, crop=True)
                    out_image = out_image[0]  # 1バンド目を取得

                    # 有効なピクセル値を取得
                    valid_pixels = out_image[out_image != src.nodata]

                    if valid_pixels.size > 0:
                        mean_value = np.mean(valid_pixels)
                        gdf_temp.at[idx, f'{column_prefix}_mean'] = mean_value
                    else:
                        gdf_temp.at[idx, f'{column_prefix}_mean'] = np.nan
            except (ValueError, rasterio.errors.WindowError) as e:
                # ジオメトリがラスタ範囲外の場合はスキップ
                print(f"Warning: ジオメトリ {idx} がラスタ範囲外です。NaNを設定します。")
                gdf_temp.at[idx, f'{column_prefix}_mean'] = np.nan

    # カラム名を短縮 (10文字以内) に変更
    gdf_temp = gdf_temp.rename(columns={f'{column_prefix}_mean': f'{column_prefix}_mn'})

    # サンプリング後、一時的に変換した場合は元の CRS に戻す
    if not original_crs_obj.equals(raster_crs_obj):
        gdf_result = gdf_temp.to_crs(original_crs)
    else:
        gdf_result = gdf_temp

    return gdf_result

Point06 = raster_sampling(Point06, input_dem, 'merge1')
# Point06.to_file(f"{output_folder}/06_5_Point.gpkg", layer='point', driver="GPKG")

# 07: 辺の重心の地盤高（ラスタサンプリング）
Point07 = Line06.copy()
Point07['geometry'] = Line06.centroid

Point07 = raster_sampling(Point07, input_dem, 'merge2')

# Point07.to_file(f"{output_folder}/07_Point.gpkg", layer='point', driver="GPKG")
print('07: ',Point07.geom_type.value_counts()) # ジオメトリタイプを確認

# edge.gpkgとして保存（06_3_Line.gpkg + 07_Point.gpkgのmerge2_mn）
# デバッグ: Line06の構造を確認
print(f"\n=== Line06の構造確認 ===")
print(f"Line06のカラム: {list(Line06.columns)}")
print(f"StartNode in Line06: {'StartNode' in Line06.columns}")
print(f"EndNode in Line06: {'EndNode' in Line06.columns}")

# Line06に07_Point.gpkgのmerge2_mnを追加
Line06_with_merge2 = Line06.merge(
    Point07[['LN', 'merge2_mn']], 
    on='LN', 
    how='left'
)

print(f"Line06_with_merge2のカラム: {list(Line06_with_merge2.columns)}")

# CN+LNの重複を除去（最初のレコードを保持）
print(f"\n=== edge.gpkg作成時の重複チェック ===")
print(f"処理前のレコード数: {len(Line06_with_merge2)}")
cn_ln_counts = Line06_with_merge2.groupby(['CN', 'LN']).size()
duplicates = cn_ln_counts[cn_ln_counts > 1]
if len(duplicates) > 0:
    print(f"⚠️ 重複しているCN+LN: {len(duplicates)}件")
    print(f"   最大重複数: {duplicates.max()}")
    Line06_with_merge2 = Line06_with_merge2.drop_duplicates(subset=['CN', 'LN'], keep='first')
    print(f"処理後のレコード数: {len(Line06_with_merge2)}")
else:
    print("✓ 重複なし")

edge_gpkg_path = f"{output_folder}/edge.gpkg"
Line06_with_merge2.to_file(edge_gpkg_path, layer='line', driver="GPKG")
print(f"✅ edge.gpkg を保存しました: {edge_gpkg_path}")

# 08: 属性の結合(key: LN)
print(f"\n=== Point08作成前の行数確認 ===")
print(f"Point06の行数: {len(Point06)}")
print(f"Point07の行数: {len(Point07)}")
print(f"Point06のユニークなLN数: {Point06['LN'].nunique()}")
print(f"Point07のユニークなLN数: {Point07['LN'].nunique()}")

# Point07にLNの重複がある場合、各LNごとに1レコードのみを保持
# （理論的にはLine06の重複除去により不要だが、念のため）
point07_counts = Point07.groupby('LN').size()
if (point07_counts > 1).any():
    print(f"⚠️ Point07にLNの重複あり: {(point07_counts > 1).sum()}件")
    # 各LNごとに最初のレコードを保持
    Point07_unique = Point07.drop_duplicates(subset='LN', keep='first')
    print(f"   重複除去後のPoint07行数: {len(Point07_unique)}")
else:
    Point07_unique = Point07
    print("✓ Point07にLNの重複なし")

# マージ実行（左結合：Point06の全行を保持）
Point08 = Point06.merge(Point07_unique[['LN', 'merge2_mn']], on='LN', how='left')

print(f"\n=== Point08作成後の行数確認 ===")
print(f"Point08の行数: {len(Point08)}")
if len(Point08) == len(Point06):
    print("✅ 行数が保持されました")
else:
    print(f"❌ 警告: 行数が変化しました（{len(Point06)} → {len(Point08)}）")
    print(f"   差分: {len(Point08) - len(Point06)}行")

# node_idカラムの作成（StartNode/EndNodeとStartEndから決定）
if 'node_id' not in Point08.columns:
    if 'StartNode' in Point08.columns and 'EndNode' in Point08.columns and 'StartEnd' in Point08.columns:
        # StartEndが'start'の場合はStartNode、'end'の場合はEndNodeを使用
        Point08['node_id'] = Point08.apply(
            lambda row: row['StartNode'] if row['StartEnd'] == 'start' else row['EndNode'],
            axis=1
        )
        print("✓ StartNode/EndNodeとStartEndからnode_idを作成しました")
        print(f"  StartEnd='start'の数: {(Point08['StartEnd'] == 'start').sum()}")
        print(f"  StartEnd='end'の数: {(Point08['StartEnd'] == 'end').sum()}")
    elif 'NodeID' in Point08.columns:
        # NodeIDが存在する場合はそれを使用
        Point08['node_id'] = Point08['NodeID']
        print("✓ NodeIDからnode_idを作成しました")
    else:
        # どちらも存在しない場合はエラー
        print("❌ エラー: StartNode/EndNodeまたはNodeIDが存在しません")
        raise ValueError("node_idを作成するための情報が不足しています")

print(f"ユニークなnode_id数: {Point08['node_id'].nunique()}")

# CNの昇順で並び替え
print(f"\n=== Point08のソート ===")
print(f"ソート前の最初の5行のCN: {Point08['CN'].head().tolist()}")
Point08 = Point08.sort_values('CN').reset_index(drop=True)
print(f"ソート後の最初の5行のCN: {Point08['CN'].head().tolist()}")
print(f"✅ CNの昇順で並び替えました")

# Point08.to_file(f"{output_folder}/08_Point.gpkg", layer='point', driver="GPKG")
print('08: ',Point08.geom_type.value_counts()) # ジオメトリタイプを確認

# ============================================================================
# フェーズ9: ラスターデータからの属性付与 (09-12)
# ============================================================================

# 09: 建物形状の抽出（ターゲットポリゴンとの交差処理）
def intersection(gdf_input, gdf_overlay):
    """建物データとターゲットポリゴンの交差部分を抽出する関数"""
    gdf_input = gdf_input.to_crs(gdf_overlay.crs)  # CRSを統一
    gdf_intersection = gpd.overlay(gdf_input, gdf_overlay, how='intersection', keep_geom_type=False)

    # ポリゴンジオメトリのみを抽出（ラインやポイントは除外）
    gdf_intersection = gdf_intersection[gdf_intersection.geometry.type == 'Polygon']

    return gdf_intersection

# 建物データの読み込みと交差処理（オプショナル）
if BIL_directory is not None:
    input_BIL = gpd.read_file(BIL_directory)
    
    Poly09 = intersection(input_BIL, Poly06)  # 建物データとターゲット領域の交差
    # Poly09.to_file(f"{output_folder}/09_Poly.gpkg", layer='poly', driver="GPKG")
    
    # セルごとの建物面積統合と面積計算
    Poly09 = Poly09.dissolve(by='CN', aggfunc='first').reset_index()  # CN（セル番号）で統合
    Poly09['bill_area'] = Poly09.geometry.area  # 建物面積を計算
    Poly09['bill_perimeter'] = Poly09.geometry.length  # 建物外周長さを計算
    # Poly09.to_file(f"{output_folder}/09_2_Poly.gpkg", layer='poly', driver="GPKG")
    print('09: ',Poly09.geom_type.value_counts()) # ジオメトリタイプを確認
    
    # 10: 建物占有率の算定とマージ処理
    Poly10 = Poly06.merge(Poly09[['CN', 'bill_area', 'bill_perimeter']], on='CN', how='left')
    Poly10['bill_area'] = Poly10['bill_area'].fillna(0)  # 欠損を0に置換
    Poly10['bill_perimeter'] = Poly10['bill_perimeter'].fillna(0)  # 建物外周長さの欠損を0に置換
    Poly10['ratio'] = Poly10.apply(lambda row: min(row['bill_area'] / row['area'], 0.95) if row['area'] > 0 else 0, axis=1)
else:
    # 建物データがない場合は建物面積=0として処理
    print("Info: 建物データがないため、bill_area=0, bill_perimeter=0, ratio=0として処理します。")
    Poly10 = Poly06.copy()
    Poly10['bill_area'] = 0.0
    Poly10['bill_perimeter'] = 0.0
    Poly10['ratio'] = 0.0
# Poly10.to_file(f"{output_folder}/10_Poly.gpkg", layer='poly', driver="GPKG")
print('10: ',Poly10.geom_type.value_counts()) # ジオメトリタイプを確認

# 10_2: セル地盤高の算定
def calculate_elevation(gdf, dem_tif, stat_type="median"):
    """
    Shapefile の各ポリゴンに対して、DEM から標高統計量を計算する関数。
    
    Parameters:
        gdf (GeoDataFrame): 入力 GeoDataFrame
        dem_tif (str): DEM の GeoTIFF ファイルのパス
        stat_type (str): 統計量のタイプ ('mean' or 'median')
    
    Returns:
        GeoDataFrame: 標高統計量が追加された GeoDataFrame
    """
    # CRSの保存
    original_crs = gdf.crs

    # CRS が None の場合、エラーを出す
    if gdf.crs is None:
        raise ValueError(f"Error: {gdf} の CRS が設定されていません。")

    # DEM データを開く
    with rasterio.open(dem_tif) as src:
        raster_crs = src.crs  # DEM の座標系
        nodata_value = src.nodata  # NoData 値を取得

    # EPSGコードが取得できない場合、強制的にEPSG:2451に設定
    if raster_crs.to_epsg() is None:
        epsg_code = DEM_epsg
        raster_crs = rasterio.crs.CRS.from_epsg(epsg_code)
        print(f"Warning: ラスターの EPSG コードが取得できなかったため、EPSG:{epsg_code} を使用します。")

    print(f"入力データの CRS: {gdf.crs}")
    print(f"DEM の CRS (EPSG): {raster_crs}")

    # CRS の変換（異なる場合のみ）
    if not CRS(gdf.crs).equals(CRS(raster_crs)):
        gdf = gdf.to_crs(raster_crs)

    # 平均値と中央値を計算（小さいポリゴンも含めるため all_touched=True）
    stats = zonal_stats(
        gdf, dem_tif,
        stats=['mean', 'median'],  # 平均値と中央値を取得
        nodata=nodata_value,  # NoData を考慮
        all_touched=True  # 小さいポリゴンもピクセルを取得
    )

    # 指定された統計量を使用（Null の場合はもう一方、それもなければ0）
    if stat_type == "mean":
        gdf['_median'] = [
            s['mean'] if s['mean'] is not None else s['median'] if s['median'] is not None else 0
            for s in stats
        ]
        stat_label = "平均値"
    else:  # median
        gdf['_median'] = [
            s['median'] if s['median'] is not None else s['mean'] if s['mean'] is not None else 0
            for s in stats
        ]
        stat_label = "中央値"

    print(f"\n=== 標高データの統計（{stat_label}使用） ===")
    print(f"最小標高: {gdf['_median'].min():.2f} m")
    print(f"最大標高: {gdf['_median'].max():.2f} m")
    print(f"平均標高: {gdf['_median'].mean():.2f} m")
    print(f"標高0のセル数: {(gdf['_median'] == 0).sum()}")

    # 元の座標系に戻す
    if not CRS(gdf.crs).equals(CRS(original_crs)):
        gdf = gdf.to_crs(original_crs)

    return gdf

Poly10 = calculate_elevation(Poly10, input_dem, elevation_stat)
# Poly10.to_file(f"{output_folder}/10_2_Poly.gpkg", layer='poly', driver="GPKG")
print('10_2: ',Poly10.geom_type.value_counts()) # ジオメトリタイプを確認

# 11: 土地利用
def area_in_square_meters(geom, crs):
    """Geometry area を CRS に応じて m² へ正規化する。"""
    if geom is None or geom.is_empty:
        return 0.0
    crs_obj = CRS(crs)
    if crs_obj.is_projected:
        unit_factor = crs_obj.axis_info[0].unit_conversion_factor if crs_obj.axis_info else 1.0
        return float(geom.area) * float(unit_factor) * float(unit_factor)
    geod = crs_obj.get_geod() if hasattr(crs_obj, "get_geod") else Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(geom)
    return abs(float(area_m2))


def calculate_raster_overlap_area_with_majority(gdf, landuse_tif, output_csv, fallback_epsg=None):
    """
    各ポリゴンに対し、ラスターのピクセルごとの重なり面積を合計し、
    面積が最大となるピクセル値を _majority として付与（元コードと同一ロジック）。
    ラスタは一度だけ開き、各ポリゴンは mask() のウィンドウ切り出しで効率化。
    """

    with rasterio.open(landuse_tif) as src:
        raster_crs = src.crs
        nodata_value = src.nodata if src.nodata is not None else 255
        # pixel_size は正値、実際のアフィンでは y 解像度は負になるが res は正で返る
        pixel_width, pixel_height = src.res
        half_w, half_h = pixel_width / 2.0, pixel_height / 2.0

        if raster_crs is None or raster_crs.to_epsg() is None:
            if fallback_epsg is not None:
                raster_crs = rasterio.crs.CRS.from_epsg(fallback_epsg)
                print(f"Warning: ラスタの EPSG が不明のため、EPSG:{fallback_epsg} を使用します。")
            else:
                print("Warning: ラスタの EPSG が不明です（fallback_epsg を指定可能）。")

        print(f"Nodata 値: {nodata_value}, 1ピクセル面積: {abs(pixel_width * pixel_height)}")
        print(f"ベクタCRS: {gdf.crs} / ラスタCRS: {raster_crs}")

        # ベクタをラスタCRSへ
        if not CRS(gdf.crs).equals(CRS(raster_crs)):
            gdf_proc = gdf.to_crs(raster_crs)
        else:
            gdf_proc = gdf

        results = []

        for _, row in gdf_proc.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                record = row.to_dict()
                record["_majority"] = int(nodata_value)
                results.append(record)
                continue

            # 形状が壊れている場合に備え軽く修正（面積は不変）
            if not geom.is_valid:
                geom = geom.buffer(0)

            try:
                # ここで必要最小のウィンドウだけ読み込まれる（reopenしない）
                out_image, out_transform = rio_mask.mask(
                    src, [mapping(geom)], crop=True,
                    nodata=nodata_value, all_touched=True
                )
            except ValueError:
                # 「Input shapes do not overlap raster.」等はここに来る
                record = row.to_dict()
                record["_majority"] = int(nodata_value)
                results.append(record)
                continue

            band = out_image[0]  # 1バンド想定
            # dtype が整数のときに np.isnan は使わない
            if np.issubdtype(band.dtype, np.floating):
                valid = (band != nodata_value) & (~np.isnan(band))
            else:
                valid = (band != nodata_value)

            rows, cols = np.where(valid)
            if rows.size == 0:
                record = row.to_dict()
                record["_majority"] = int(nodata_value)
                results.append(record)
                continue

            # ピクセル中心座標
            xs, ys = rasterio.transform.xy(out_transform, rows, cols)

            # 交差判定の高速化
            prep_poly = prepared.prep(geom)

            from collections import defaultdict
            pixel_areas = defaultdict(float)

            # 値とセル形状の交差面積を集計
            vals = band[rows, cols]
            # Pythonのdictキーは int に揃える（列名化のため）
            if np.issubdtype(vals.dtype, np.floating):
                vals = np.rint(vals).astype(int)
            else:
                vals = vals.astype(int)

            for v, x, y in zip(vals.tolist(), xs, ys):
                cell = box(x - half_w, y - half_h, x + half_w, y + half_h)
                if not prep_poly.intersects(cell):
                    continue
                inter_geom = geom.intersection(cell)
                inter_area = area_in_square_meters(inter_geom, raster_crs)
                if inter_area > 0.0:
                    pixel_areas[v] += inter_area

            if pixel_areas:
                max_pixel_value = max(pixel_areas, key=pixel_areas.get)
            else:
                max_pixel_value = int(nodata_value)

            record = row.to_dict()
            record.update(pixel_areas)     # 値ごとの重なり面積を列として追加
            record["_majority"] = int(max_pixel_value)
            results.append(record)

    # 出力
    df = pd.DataFrame(results)
    
    # 数値列のカラム名を文字列に変換（土地利用コード等）
    df.columns = df.columns.astype(str)
    
    # null値を0に置き換え（geometry列以外）
    # まずPandas DataFrameとして処理
    non_geom_cols = [col for col in df.columns if col != 'geometry']
    for col in non_geom_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[col] = df[col].fillna(0)
    
    # geometryカラムが存在する場合はそれを使用、なければgdf_procから復元
    if 'geometry' in df.columns:
        gdf_result = gpd.GeoDataFrame(df, crs=gdf_proc.crs)
    else:
        # resultsにgeometryが含まれていない場合
        df['geometry'] = gdf_proc.geometry.values
        gdf_result = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf_proc.crs)
    
    # 固定の土地利用コードリストに対応（存在しない場合も0で埋める）
    required_landuse_codes = ['10', '20', '50', '60', '70', '91', '92', '100', '110', '140', '150', '160', '255']
    for code in required_landuse_codes:
        if code not in gdf_result.columns:
            gdf_result[code] = 0.0
    
    print(f"\n=== 土地利用面積データの統計 ===")
    print(f"総カラム数: {len(gdf_result.columns)}")
    # 土地利用コードのカラム（数値のみのカラム名）を抽出
    landuse_cols = [col for col in gdf_result.columns if col.isdigit()]
    if landuse_cols:
        print(f"土地利用コード数: {len(landuse_cols)}")
        print(f"土地利用コード例: {sorted([int(c) for c in landuse_cols[:10]])}")
        # 各土地利用の総面積を表示
        total_areas = {col: gdf_result[col].sum() for col in landuse_cols[:5]}
        print(f"主要土地利用の総面積（m²）: {total_areas}")
        # 固定リストのコードでデータが存在するものを表示
        existing_required = [c for c in required_landuse_codes if gdf_result[c].sum() > 0]
        if existing_required:
            print(f"固定リスト内で実際にデータが存在するコード: {', '.join(existing_required)}")
    else:
        print("⚠️ 警告: 土地利用コードのカラムが見つかりません")
    
    if output_csv is not None:
        # CSV出力
        gdf_result.to_csv(output_csv, index=False)
        print(f"CSVを出力しました: {output_csv}")

    return gdf_result

input_LandUse  = os.path.abspath(os.path.expanduser(LandUse_directory))
# output_csv = f"{output_folder}/polygon_raster_coverage.csv"  # 出力するCSV（不要のためコメントアウト）

Poly11 = calculate_raster_overlap_area_with_majority(Poly10, input_LandUse, output_csv=None)

def calculate_soil_area(gdf, soil_tif, fallback_epsg=None):
    """
    各ポリゴンに対して土壌ラスターの重なり面積を集計し、
    `soil_<code>_area`（コード別の面積[m²]）を付与する。
    """
    with rasterio.open(soil_tif) as src:
        raster_crs = src.crs
        nodata_value = src.nodata if src.nodata is not None else -9999
        pixel_width, pixel_height = src.res
        half_w, half_h = pixel_width / 2.0, pixel_height / 2.0

        if raster_crs is None or raster_crs.to_epsg() is None:
            if fallback_epsg is not None:
                raster_crs = rasterio.crs.CRS.from_epsg(fallback_epsg)
                print(f"Warning: 土壌ラスタの EPSG が不明のため、EPSG:{fallback_epsg} を使用します。")
            else:
                print("Warning: 土壌ラスタの EPSG が不明です（fallback_epsg を指定可能）。")

        if not CRS(gdf.crs).equals(CRS(raster_crs)):
            gdf_proc = gdf.to_crs(raster_crs)
        else:
            gdf_proc = gdf

        required_soil_codes = [str(code) for code in range(1, 18)]
        results = []

        for _, row in gdf_proc.iterrows():
            geom = row.geometry
            record = row.to_dict()

            for code in required_soil_codes:
                record[f"soil_{code}_area"] = 0.0

            if geom is None or geom.is_empty:
                results.append(record)
                continue

            if not geom.is_valid:
                geom = geom.buffer(0)

            try:
                out_image, out_transform = rio_mask.mask(
                    src, [mapping(geom)], crop=True,
                    nodata=nodata_value, all_touched=True
                )
            except ValueError:
                results.append(record)
                continue

            band = out_image[0]
            if np.issubdtype(band.dtype, np.floating):
                valid = (band != nodata_value) & (~np.isnan(band))
            else:
                valid = (band != nodata_value)

            rows, cols = np.where(valid)
            if rows.size == 0:
                results.append(record)
                continue

            xs, ys = rasterio.transform.xy(out_transform, rows, cols)
            prep_poly = prepared.prep(geom)

            vals = band[rows, cols]
            if np.issubdtype(vals.dtype, np.floating):
                vals = np.rint(vals).astype(int)
            else:
                vals = vals.astype(int)

            soil_area_by_code = defaultdict(float)
            for v, x, y in zip(vals.tolist(), xs, ys):
                if v < 1 or v > 17:
                    continue
                cell = box(x - half_w, y - half_h, x + half_w, y + half_h)
                if not prep_poly.intersects(cell):
                    continue
                inter_geom = geom.intersection(cell)
                inter_area = area_in_square_meters(inter_geom, raster_crs)
                if inter_area > 0.0:
                    soil_area_by_code[str(v)] += inter_area

            for code, area_value in soil_area_by_code.items():
                record[f"soil_{code}_area"] = area_value

            results.append(record)

    df = pd.DataFrame(results)
    df.columns = df.columns.astype(str)

    non_geom_cols = [col for col in df.columns if col != 'geometry']
    for col in non_geom_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[col] = df[col].fillna(0)

    if 'geometry' in df.columns:
        gdf_result = gpd.GeoDataFrame(df, crs=gdf_proc.crs)
    else:
        df['geometry'] = gdf_proc.geometry.values
        gdf_result = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf_proc.crs)

    if not CRS(gdf_result.crs).equals(CRS(gdf.crs)):
        gdf_result = gdf_result.to_crs(gdf.crs)

    soil_area_cols = [f"soil_{code}_area" for code in required_soil_codes]
    print("\n=== 土壌面積データの統計 ===")
    print(f"土壌面積カラム数: {len(soil_area_cols)}")
    print(f"土壌面積カラム: {', '.join(soil_area_cols)}")

    return gdf_result

input_Soil = os.path.abspath(os.path.expanduser(Soil_directory))
Poly11 = calculate_soil_area(Poly11, input_Soil, fallback_epsg=Soil_epsg)

# 保存前のカラム確認
print(f"\n=== 11_Poly.gpkg 保存前の確認 ===")
print(f"保存前のカラム数: {len(Poly11.columns)}")
landuse_cols_before = [col for col in Poly11.columns if col.isdigit()]
print(f"土地利用コード数（保存前）: {len(landuse_cols_before)}")

# GeoPackageに保存
# Poly11.to_file(f"{output_folder}/11_Poly.gpkg", layer='poly', driver="GPKG")
print('11: ',Poly11.geom_type.value_counts()) # ジオメトリタイプを確認

# face.gpkgとして保存（11_Poly.gpkgのコピー）
# 平面直角座標系では反時計回り（CCW）が標準なので、ポリゴンを修正
def ensure_ccw_polygon(geom):
    """ポリゴンを反時計回り（CCW）に修正"""
    from shapely.geometry import Polygon, MultiPolygon, LinearRing
    
    if geom.is_empty:
        return geom
    
    if isinstance(geom, Polygon):
        ring = LinearRing(geom.exterior.coords)
        if not ring.is_ccw:
            # 時計回り → 反時計回りに反転
            coords = list(geom.exterior.coords)
            # 最後の座標（閉じた座標）を除去してから反転
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            # 反転（最初から最後まで逆順）
            coords_ccw = coords[::-1]
            return Polygon(coords_ccw)
        return geom
    
    elif isinstance(geom, MultiPolygon):
        # 各ポリゴンを個別に処理
        return MultiPolygon([ensure_ccw_polygon(p) for p in geom.geoms])
    
    return geom

Poly11['geometry'] = Poly11['geometry'].apply(ensure_ccw_polygon)
face_gpkg_path = f"{output_folder}/face.gpkg"
Poly11.to_file(face_gpkg_path, layer='poly', driver="GPKG")
print(f"✅ face.gpkg を保存しました（反時計回りに修正済み）: {face_gpkg_path}")

# # 保存後の確認（読み込んで検証）
# Poly11_verify = gpd.read_file(f"{output_folder}/11_Poly.gpkg", layer='poly')
# print(f"\n=== 11_Poly.gpkg 保存後の確認 ===")
# print(f"保存後のカラム数: {len(Poly11_verify.columns)}")
# landuse_cols_after = [col for col in Poly11_verify.columns if col.isdigit()]
# print(f"土地利用コード数（保存後）: {len(landuse_cols_after)}")
# 
# if len(landuse_cols_before) == len(landuse_cols_after):
#     print("✅ 全ての土地利用データが正しく保存されました")
# else:
#     print(f"⚠️ 警告: 土地利用データが減少しました（{len(landuse_cols_before)} → {len(landuse_cols_after)}）")
#     missing = set(landuse_cols_before) - set(landuse_cols_after)
#     if missing:
#         print(f"   失われたカラム: {sorted([int(c) for c in missing])[:20]}")

def extract_raster_values_to_csv(gdf, landuse_tif, output_csv):
    """
    各ポリゴンに対して、土地利用ラスターデータのピクセル値を取得し、CSV形式で出力する。

    Parameters:
        gdf (GeoDataFrame): ポリゴンデータ（属性を持つ）
        landuse_tif (str): ラスターデータ（GeoTIFF）
        output_csv (str): 出力するCSVファイルのパス
    """

    # ラスターデータの座標系取得
    with rasterio.open(landuse_tif) as src:
        raster_crs = src.crs
        nodata_value = src.nodata if src.nodata is not None else 255  # NoData 値の処理

    # CRSをラスターデータに揃える
    gdf = gdf.to_crs(raster_crs)

    # 各ポリゴンごとにラスターデータのピクセル値を取得（リスト）
    stats = zonal_stats(
        gdf, landuse_tif,
        stats=['count'],  # ピクセル数だけ取得（値は `band` に格納される）
        raster_out=True,  # ラスターデータの値を取得
        nodata=nodata_value,
        all_touched=True
    )

    # 各ポリゴンごとにピクセル値を格納
    raster_values = []
    for stat in stats:
        raster_array = stat['mini_raster_array'].compressed()  # NoData除外
        raster_values.append(raster_array.tolist())  # リスト化

    # GeoDataFrameにラスターデータの値を追加（カラム名: "raster_values"）
    gdf['raster_values'] = raster_values

    # CSV形式で保存
    gdf.drop(columns='geometry').to_csv(output_csv, index=False)

    print(f"CSVファイルを出力しました: {output_csv}")

# output_csv = f"{output_folder}/polygon_raster_values.csv"  # 出力するCSV（不要のためコメントアウト）
# extract_raster_values_to_csv(Poly10, input_LandUse, output_csv)

# ============================================================================
# フェーズ10: 最終出力ファイル生成 (12)
# ============================================================================

# 12: セル中心座標の作成
Point12 = Poly11.copy()
Point12['geometry'] = Poly10.centroid  # ポリゴンの重心点をセット
Point12['xcoord'] = Point12.geometry.x  # X座標
Point12['ycoord'] = Point12.geometry.y  # Y座標
# Point12.to_file(f"{output_folder}/12_Point.gpkg", layer='point', driver="GPKG")
print('12: ',Point12.geom_type.value_counts()) # ジオメトリタイプを確認

# ============================================================================
# TACM出力ファイル生成（edge.csv, face.csv）
# ============================================================================

# edge.csv生成：Point08をベースに作成（08.gpkgと完全一致、CN昇順）
print(f"\n=== edge.csv生成 ===")

# 座標カラムの追加（まだない場合）
if 'xcoord' not in Point08.columns or 'ycoord' not in Point08.columns:
    Point08['xcoord'] = Point08.geometry.x
    Point08['ycoord'] = Point08.geometry.y

# Point08の統計情報を表示
print(f"Point08の行数: {len(Point08)}")
print(f"ユニークなLN数: {Point08['LN'].nunique()}")
print(f"ユニークなCN数: {Point08['CN'].nunique()}")
print(f"ユニークな座標数: {Point08.groupby(['xcoord', 'ycoord']).ngroups}")
print(f"ユニークなnode_id数: {Point08['node_id'].nunique()}")
print(f"CNの範囲: {Point08['CN'].min()} ～ {Point08['CN'].max()}")
print(f"✓ Point08は既にCN昇順にソート済み")

# CN+LNの組み合わせチェック
cn_ln_counts = Point08.groupby(['CN', 'LN']).size()
print(f"CN+LNの組み合わせ数: {len(cn_ln_counts)}")
print(f"CN+LNの出現回数（最小/最大/平均）: {cn_ln_counts.min()}/{cn_ln_counts.max()}/{cn_ln_counts.mean():.2f}")

# Point08には既にStartNode/EndNodeが含まれているので、そのまま使用
print(f"\n=== StartNode/EndNode情報を確認 ===")
print(f"Point08のカラム: {list(Point08.columns)}")
print(f"StartNode in Point08: {'StartNode' in Point08.columns}")
print(f"EndNode in Point08: {'EndNode' in Point08.columns}")

# 必要なカラムの順序を定義
required_columns = ['CN', 'ID', 'LN', 'StartNode', 'EndNode', 'xcoord', 'ycoord', 'node_id', 'merge1_mn', 'merge2_mn']

# 存在するカラムのみを選択
existing_columns = [col for col in required_columns if col in Point08.columns]
missing_columns = [col for col in required_columns if col not in Point08.columns]

if missing_columns:
    print(f"❌ エラー: 以下のカラムが存在しません: {missing_columns}")
else:
    print(f"✅ 全ての必須カラムが存在します")

# edge.csvを出力（Point08の内容をそのまま使用）
df_edge = Point08[existing_columns].copy()

print(f"\n=== edge.csv出力情報 ===")
print(f"出力カラム: {list(df_edge.columns)}")
print(f"出力行数: {len(df_edge)}")

# UTF-8で保存
edge_csv_path = os.path.join(output_folder, 'edge.csv')
df_edge.to_csv(edge_csv_path, encoding='utf-8', index=False)
print(f"✅ edge.csvを出力しました: {edge_csv_path}")

# # 08.gpkgと比較検証
# print(f"\n=== 08.gpkgとの整合性確認 ===")
# point08_verify = gpd.read_file(f"{output_folder}/08_Point.gpkg", layer='point')
# if 'xcoord' not in point08_verify.columns:
#     point08_verify['xcoord'] = point08_verify.geometry.x
#     point08_verify['ycoord'] = point08_verify.geometry.y
# 
# # カラムの比較
# common_cols = [col for col in existing_columns if col in point08_verify.columns and col != 'geometry']
# if len(point08_verify) == len(df_side):
#     print(f"✅ 行数一致: {len(df_side)}")
# else:
#     print(f"❌ 行数不一致: side.csv={len(df_side)}, 08.gpkg={len(point08_verify)}")
# 
# # ソート順の確認
# print(f"08.gpkgの最初の5行のCN: {point08_verify['CN'].head().tolist()}")
# print(f"side.csvの最初の5行のCN: {df_side['CN'].head().tolist()}")
# 
# # 各カラムの一致確認
# for col in common_cols:
#     if col in ['xcoord', 'ycoord', 'merge1_mn', 'merge2_mn']:
#         # 浮動小数点の比較（小数点以下の誤差を考慮）
#         match = np.allclose(df_side[col].fillna(0), point08_verify[col].fillna(0), rtol=1e-9, atol=1e-9)
#     else:
#         # 整数の比較
#         match = (df_side[col] == point08_verify[col]).all()
#     
#     if match:
#         print(f"  ✅ {col}: 一致")
#     else:
#         print(f"  ❌ {col}: 不一致")
#         if not match and col not in ['xcoord', 'ycoord', 'merge1_mn', 'merge2_mn']:
#             diff_count = (df_side[col] != point08_verify[col]).sum()
#             print(f"     差異のある行数: {diff_count}")

# face.csv生成：セル中心データ（Point12）をCSV形式で出力
# face_latin_path = os.path.join(output_folder, 'face_latin.csv')  # 不要のためコメントアウト
# Point12.to_csv(face_latin_path, index=False)

# 列の順序を設定（_majorityを除外し、土地利用面積データを全て含める）
# df_face = pd.read_csv(face_latin_path, encoding='latin1')  # 直接DataFrameを使用
df_face = Point12.drop(columns='geometry') if 'geometry' in Point12.columns else Point12.copy()

# 基本カラム（_majorityは除外、_medianを使用）
base_columns = ['CN', 'bill_area', 'area', 'ratio', '_median', 'xcoord', 'ycoord', 'CalMesh', 'bill_perimeter']

# 固定の土地利用コードリスト（存在しない場合も0で埋める）
required_landuse_codes = ['10', '20', '50', '60', '70', '91', '92', '100', '110', '140', '150', '160', '255']

# 実際に存在する土地利用コードを確認
existing_landuse_cols = [col for col in df_face.columns if col.isdigit()]

# 存在しない土地利用コードを0で追加
for code in required_landuse_codes:
    if code not in df_face.columns:
        df_face[code] = 0.0

# カラムの順序：基本カラム + 固定の土地利用コードリスト（昇順）
landuse_cols = sorted(required_landuse_codes, key=int)
soil_area_cols = [f"soil_{code}_area" for code in range(1, 18)]
new_column_order = base_columns + landuse_cols + soil_area_cols

# 存在するカラムのみを選択
existing_columns = [col for col in new_column_order if col in df_face.columns]
df_face = df_face[existing_columns]

print(f"\n=== face.csv 出力情報 ===")
print(f"総カラム数: {len(df_face.columns)}")
print(f"土地利用面積カラム数: {len(landuse_cols)}")
print(f"土地利用コード: {', '.join(landuse_cols)}")
print(f"土壌面積カラム数: {len(soil_area_cols)}")
print(f"実際にデータが存在するコード数: {len(existing_landuse_cols)}")
if existing_landuse_cols:
    print(f"  データが存在するコード: {', '.join(sorted(existing_landuse_cols, key=int))}")
print(f"カラム順序: {df_face.columns.tolist()}")

# face.csv を UTF-8 で再保存
face_csv_path = os.path.join(output_folder, 'face.csv')
df_face.to_csv(face_csv_path, encoding='utf-8', index=False)
print(f"✅ face.csvを出力しました: {face_csv_path}")
