# AQWA-INUNDATION2D メッシュ作成ツール

AQWA-INUNDATION2D（2次元氾濫解析モデル）用のメッシュデータを生成するPythonスクリプトです。地理空間データ（GeoPackage, Shapefile, GeoTIFF）から、氾濫解析に必要な構造化メッシュファイルを自動生成します。

## 📋 目次

- [概要](#概要)
- [機能](#機能)
- [必要な環境](#必要な環境)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [ファイル構成](#ファイル構成)
- [設定ファイル（YAML）](#設定ファイルyaml)
- [出力ファイル](#出力ファイル)
- [ワークフロー](#ワークフロー)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

このツールは、以下の入力データから氾濫解析用のメッシュファイルを生成します：

**入力データ**:
- DEM（数値標高モデル）- GeoTIFF形式（必須）
- 土地利用データ - GeoTIFF形式（必須）
- OpenStreetMap道路データ - Shapefile形式（必須）
- 対象領域ポリゴン - Shapefile形式（必須）
- OpenStreetMap建物データ - Shapefile形式（**オプション**）
- ダミーメッシュ - Shapefile形式（**オプション**）

**出力データ**:
- `{project_name}.txt` - AQWA-INUNDATION2D用メッシュファイル
- `face.csv` / `face.gpkg` - セル中心データ（標高、土地利用、座標など）
- `edge.csv` / `edge.gpkg` - セル辺データ（節点座標、標高など）
- 各種GeoPackageファイル（中間データ）

---

## 機能

### 1. メッシュ生成（`01_mkINPUT_Face_Side.py`）

このスクリプトは、地理空間データからAQWA-INUNDATION2D用のセルメッシュを生成します。

**主な処理内容**:

1. **入力データの読み込みと前処理**
   - YAML設定ファイルの解析と`{project_name}`プレースホルダーの展開
   - 対象領域ポリゴンの読み込みとバッファリング
   - OpenStreetMap道路データと建物データの読み込み
   - DEM（数値標高モデル）と土地利用ラスタの読み込み

2. **セル生成点の抽出とVoronoi分割**
   - OSM道路ラインから等間隔でサンプル点を抽出
   - 「長くて細い」ポリゴン（アスペクト比≥3.0）を検出し、クラスタリングで追加の点を配置
   - ダミーメッシュ（オプション）との統合：ダミーメッシュがない場合は道路点のみを使用
   - Voronoi分割によるセルポリゴンの生成と対象領域でのクリッピング

3. **ポリゴンの整合性チェックと修正**
   - `make_valid()`による無効なジオメトリの自動修正
   - `explode()`によるMultiPolygonの分解
   - 小さなポリゴン（閾値以下）の統合

4. **標高統計の計算**
   - 各セルポリゴンに対してDEMから標高統計量を計算
   - YAMLパラメータで指定された統計量（`mean`または`median`）を使用
   - ラスタ範囲外のジオメトリに対するエラーハンドリング

5. **土地利用データの集計**
   - 各セルポリゴンに対して土地利用ラスタから画素値ごとの面積を計算
   - 固定カテゴリ（10, 20, 50, 60, 70, 91, 92, 100, 110, 140, 150, 160, 255）すべてを出力
   - 存在しないカテゴリは0.0で埋める

6. **建物ポロシティの計算**（オプション）
   - 各セルポリゴンと建物ポリゴンのオーバーラップ面積を計算
   - 建物の周長（perimeter）を計算
   - ポロシティ（空隙率）`ratio = 1 - (建物面積 / セル面積)`を算出
   - 建物データがない場合は`bill_area=0`, `bill_perimeter=0`, `ratio=0`として処理

7. **セル辺の抽出とユニーク番号の割り当て**
   - 各セルポリゴンの境界線を抽出し、座標ごとにユニークな`node_id`を割り当て
   - 辺（edge）ごとにユニークな`LN`（ライン番号）を割り当て
   - 同じ座標・同じ辺は異なるセル間で同じIDを共有（座標の丸め誤差を回避）
   - セル内での辺の重複を除去（MultiPolygon対応）

8. **境界フラグの設定**
   - 対象領域の外縁と接する辺を`ID=1`（outer）としてマーク
   - 内部の辺を`ID=0`（interface）としてマーク

9. **標高のサンプリング**
   - Point geometryに対しては`rasterio.sample`を使用
   - Polygon/LineString geometryに対しては`rasterio.mask.mask`を使用
   - 辺の節点（`merge1_mn`）と辺の中心点（`merge2_mn`）の標高を取得

10. **出力ファイルの生成**
    - `face.csv` / `face.gpkg`: セル中心データ（標高、土地利用、建物、座標など）
    - `edge.csv` / `edge.gpkg`: セル辺データ（節点座標、標高、LN、node_id、StartNode、EndNodeなど）
    - 各種`.gpkg`: 中間データ（02_Line, 06_2_Poly, 08_Point, 10_2_Polyなど）

**特徴**:
- ✅ 座標の丸め誤差を完全に回避（node_idベースの管理）
- ✅ MultiPolygonの適切な処理（各パートで辺の重複を防止）
- ✅ ラスタ範囲外のジオメトリに対する堅牢なエラーハンドリング
- ✅ オプション入力（建物、ダミーメッシュ）の柔軟な対応

---

### 2. セルデータ構築（`02_mkCELL.py`）

このスクリプトは、`01_mkINPUT_Face_Edge.py`が生成した`face.csv`と`edge.csv`から、AQWA-INUNDATION2D用の最終メッシュファイル（`{project_name}.txt`）を生成します。

**主な処理内容**:

1. **入力データの読み込み**
   - YAML設定ファイルの解析と`{project_name}`プレースホルダーの展開
   - `face.csv`（セル中心データ）の読み込み
   - `edge.csv`（セル辺データ）の読み込み

2. **セル辺のLNベースのペアリング**
   - 各セル内で、同じ`LN`を持つ2つの辺（始点と終点）をペアリング
   - 辺のペアからラインオブジェクトを生成
   - `peri`（境界フラグ）と`node_id`を保持

3. **隣接セルの自動探索**
   - node_idペアの正規化（小さい方を先にソート）により、方向を問わずマッチング
   - 同じnode_idペアを持つ別のセルのラインを探索
   - `CN_end`（隣接セル番号）を設定（境界の場合は0のまま）

4. **セル重心間距離（DL）の計算**
   - 内部辺: CN_bgnとCN_endのセル重心間の距離
   - 境界辺: CN_bgnのセル重心から辺の中心点（cnt）までの距離

5. **方向コサインの計算**
   - 内部辺: `(x_cell_end - x_cell_bgn)/DL`, `(y_cell_end - y_cell_bgn)/DL`（CN_bgn → CN_end）
   - 境界辺: `(x_cnt - x_cell_bgn)/DL`, `(y_cnt - y_cell_bgn)/DL`（CN_bgn → 辺の中点）
   - セル重心から隣接セル重心（または辺の中点）への単位ベクトル成分

6. **逆距離加重平均の重みの計算**
   - **weight_bgn**: 辺の中心点の値を補間する際のCN_bgnの重み
   - **weight_end**: 辺の中心点の値を補間する際のCN_endの重み
   - **weight_cnt**: セルの値を補間する際の辺の中心点の重み（境界辺でも計算）
   - セル重心と辺の中心点の距離に基づく逆距離加重

7. **辺長の計算**
   - 辺の始点（X_bgn, Y_bgn）と終点（X_end, Y_end）の直線距離
   - `edge_length = sqrt((X_end - X_bgn)² + (Y_end - Y_bgn)²)`

8. **拡張フォーマットでの出力**
   - 1行目: face.gpkgの絶対パス
   - 2行目: edge.gpkgの絶対パス
   - 3行目: 総セル数
   - セル情報（1行）: CN, LN数, X, Y, 面積, 標高, ratio, 建物外周長, 土地利用面積（全13カテゴリ）, CalMesh
   - ライン情報（各ライン1行、反時計回り順序、23列）: LN, peri, CN_bgn, CN_end, 合計面積, DL, dx/DL, dy/DL, weight_bgn, weight_end, weight_cnt, node_bgn, node_end, X_bgn, Y_bgn, Z_bgn, X_end, Y_end, Z_end, X_cnt, Y_cnt, Z_cnt, edge_length

**特徴**:
- ✅ LNベースの自動ペアリング（セル内で同じLNの辺を自動検出）
- ✅ node_idペアの正規化による高精度マッチング
- ✅ 境界辺での適切なDL、方向コサイン、weight_cntの計算
- ✅ 反時計回り順序での辺出力（平面直角座標系に準拠）
- ✅ 拡張フォーマットによる詳細な幾何情報の出力
- ✅ face.gpkgとedge.gpkgパスの明示的な記録

---

## 必要な環境

### Pythonバージョン
- Python 3.8 以上

### 必須ライブラリ
```
geopandas >= 0.10.0
pandas >= 1.3.0
numpy >= 1.21.0
shapely >= 1.8.0
rasterio >= 1.2.0
rasterstats >= 0.15.0
pyogrio >= 0.5.0
pyyaml >= 5.4.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
```

---

## インストール

### 1. Conda環境の作成（推奨）

```bash
# 新しいConda環境を作成
conda create -n AQWA-RISK python=3.10

# 環境をアクティベート
conda activate AQWA-RISK
```

### 2. 必要なライブラリのインストール

```bash
# GeoPandas（主要な地理空間ライブラリ）
conda install -c conda-forge geopandas

# Rasterio（ラスタデータ処理）
conda install -c conda-forge rasterio

# Rasterstats（ゾーン統計）
conda install -c conda-forge rasterstats

# その他のライブラリ
conda install -c conda-forge pyyaml scikit-learn scipy pyogrio
```

### 3. リポジトリのクローン（またはダウンロード）

```bash
cd /path/to/your/workspace
git clone <repository-url>
cd 01_mkCELL_INUNDATION2D
```

---

## 使用方法

### 基本的なワークフロー

#### **ステップ1: 設定ファイルの準備**

`yaml/`ディレクトリに、プロジェクト用のYAMLファイルを作成します。

```bash
cp yaml/Kinu_Joso.yaml yaml/Your_Project.yaml
```

YAMLファイルを編集し、プロジェクト名と入力データのパスを設定します。

```yaml
project:
  name: "Your_Project"
  base_dir: "/path/to/your/workspace/01_mkCELL_INUNDATION2D"

input:
  dem:
    dir: "Cell/{project_name}/input_gpkg_tif"
    file: "DEM5m2_intprt.tif"
    epsg: 2451
  # ... (その他の設定)
```

#### **ステップ2: メッシュデータの生成**

```bash
conda activate AQWA-RISK
python 01_mkINPUT_Face_Edge.py yaml/Your_Project.yaml
```

**出力**: `Cell/Your_Project/output_gpkg_csv/`に以下のファイルが生成されます
- `face.csv` / `face.gpkg` - セル中心データ
- `edge.csv` / `edge.gpkg` - セル辺データ
- 各種`.gpkg`ファイル（中間データ）

#### **ステップ3: cell.txtの生成**

```bash
python 02_mkCELL.py yaml/Your_Project.yaml
```

**出力**: `Cell/Your_Project/Your_Project.txt`（AQWA-INUNDATION2D用メッシュファイル）

#### **ステップ4: 一括実行（推奨）**

両方のスクリプトを連続実行する場合:

```bash
conda activate AQWA-RISK
bash run_all.sh yaml/Your_Project.yaml
```

---

## ファイル構成

```
01_mkCELL_INUNDATION2D/
├── 01_mkINPUT_Face_Edge.py     # メッシュ生成スクリプト
├── 02_mkCELL.py                # cell.txt生成スクリプト
├── run_all.sh                  # 一括実行スクリプト
├── README.md                   # このファイル
│
├── yaml/                       # 設定ファイル
│   ├── Kinu_Joso.yaml         # 鬼怒川常総プロジェクト設定例
│   ├── Test_inun_riv_coupling.yaml  # 氾濫・河川連携検証用（50m 構造化格子）
│   └── ...
│
├── Cell/                       # プロジェクトデータ
│   └── {project_name}/
│       ├── {project_name}.txt              # 最終出力（AQWA用メッシュ）
│       ├── input_gpkg_tif/                 # 入力データ
│       │   ├── DEM5m2_intprt.tif
│       │   ├── LandUse_2451.tif
│       │   ├── 01.shp                      # 対象領域
│       │   └── DummyMesh2.shp              # ダミーメッシュ（オプション）
│       └── output_gpkg_csv/                # 中間・出力データ
│           ├── face.csv / face.gpkg
│           ├── edge.csv / edge.gpkg
│           ├── 02_Line.gpkg
│           ├── 06_2_Poly.gpkg
│           ├── 08_Point.gpkg
│           └── 10_2_Poly.gpkg
│
└── OpenStreetMap/              # OSMデータ
    ├── kanto-latest-free.shp/
    │   ├── gis_osm_roads_free_1.shp
    │   ├── gis_osm_buildings_a_free_1.shp
    │   └── ...
    └── ...
```

---

## 設定ファイル（YAML）

### YAMLファイルの構造

```yaml
# プロジェクト基本設定
project:
  name: "Project_Name"           # プロジェクト名（{project_name}として使用される）
  base_dir: "/path/to/workspace" # ワークスペースのベースディレクトリ

# 入力データ設定
input:
  # 道路データ（OpenStreetMap）
  roads:
    dir: "OpenStreetMap/kanto-latest-free.shp"
    file: "gis_osm_roads_free_1.shp"
  
  # 建物データ（OpenStreetMap、オプション）
  buildings:
    dir: "OpenStreetMap/kanto-latest-free.shp"
    file: "gis_osm_buildings_a_free_1.shp"  # 建物を使用しない場合は "none" を指定
  
  # DEMデータ（標高データ）
  dem:
    dir: "Cell/{project_name}/input_gpkg_tif"
    file: "DEM5m2_intprt.tif"
    epsg: 2451
  
  # 土地利用データ
  landuse:
    dir: "Cell/{project_name}/input_gpkg_tif"
    file: "LandUse_2451.tif"
    epsg: 2451
  
  # 対象領域ポリゴン
  target_area:
    dir: "Cell/{project_name}/input_gpkg_tif"
    file: "01.shp"
  
  # ダミーメッシュ（キャリブレーション用、オプション）
  dummy_mesh:
    dir: "Cell/{project_name}/input_gpkg_tif"
    file: "DummyMesh2.shp"  # ダミーメッシュを使用しない場合は "none" を指定

# 出力データ設定
output:
  dir: "Cell/{project_name}/output_gpkg_csv"  # CSVとGeoPackageの出力先
  cell_txt_dir: "Cell/{project_name}"         # cell.txtの出力先
  cell_txt: "{project_name}.txt"              # cell.txtのファイル名

# パラメータ設定
parameters:
  area_threshold: 10.0  # 面積閾値（小さなポリゴンの統合用）[m²]
  elevation_stat: "median"  # 標高統計量: "mean"（平均値）または "median"（中央値）
```

### `{project_name}`の使用

`{project_name}`はYAMLファイル内で自動的に`project.name`の値に置き換えられます。

**例**: `project.name: "Kinu_Joso"`の場合
- `Cell/{project_name}/input_gpkg_tif` → `Cell/Kinu_Joso/input_gpkg_tif`
- `{project_name}.txt` → `Kinu_Joso.txt`

### 標高統計量の選択

`elevation_stat`パラメータで、セル標高の統計量を選択できます：

- **`"median"`（中央値）**: デフォルト設定。外れ値の影響を受けにくい。
- **`"mean"`（平均値）**: 全ピクセルの平均値。滑らかな地形に適している。

```yaml
parameters:
  elevation_stat: "mean"  # 平均値を使用する場合
```

---

## 出力ファイル

### 1. `{project_name}.txt` - AQWA-INUNDATION2D用メッシュファイル

**フォーマット**:
```
[face.gpkgの絶対パス]
[edge.gpkgの絶対パス]
[総セル数]
[セル1の情報]
  CN LN数 X Y 面積 標高 ratio 建物外周長 土地利用面積(全コード) CalMesh
  [ライン1] LN peri CN_bgn CN_end 合計面積 DL dx/DL dy/DL 
            weight_bgn weight_end weight_cnt node_bgn node_end
            X_bgn Y_bgn Z_bgn X_end Y_end Z_end X_cnt Y_cnt Z_cnt edge_length
  [ライン2] ...
  ...
[セル2の情報]
  ...
```

**セル情報の説明**:
- `CN`: セル番号
- `LN数`: セル内のライン数
- `X, Y`: セル重心座標 [m]
- `面積`: セル面積 [m²]
- `標高`: セル標高（中央値）[m]
- `ratio`: 建物ポロシティ（空隙率）
- `建物外周長`: 建物の周長 [m]
- `土地利用面積`: 各土地利用コード（10,20,50,...）の面積 [m²]
- `CalMesh`: 計算メッシュフラグ（0=計算対象、1=ダミー）

**ライン情報の説明**（23列）:
- `LN`: ライン番号（共有辺で統一）
- `peri`: 境界フラグ（1=外縁、0=内部）
- `CN_bgn`: 始点セル番号
- `CN_end`: 終点セル番号（0=境界）
- `合計面積`: CN_bgnとCN_endの面積の合計 [m²]（境界の場合はCN_bgnのみ）
- `DL`: セル重心間距離 [m]（境界の場合はCN_bgnの重心から辺の中心点までの距離）
- `dx/DL, dy/DL`: 方向コサイン（単位ベクトル成分）
  - 内部辺: (x_end - x_bgn)/DL, (y_end - y_bgn)/DL（CN_bgn → CN_end）
  - 境界辺: (x_cnt - x_bgn)/DL, (y_cnt - y_bgn)/DL（CN_bgn → 辺の中点）
- `weight_bgn`: 辺の中心点の値を補間する際のCN_bgnの重み（逆距離加重平均）
- `weight_end`: 辺の中心点の値を補間する際のCN_endの重み（逆距離加重平均）
- `weight_cnt`: セルの値を補間する際の辺の中心点の重み（逆距離加重平均、境界辺でも計算）
- `node_bgn, node_end`: 辺の始点・終点の節点番号（ユニーク）
- `X_bgn, Y_bgn, Z_bgn`: 辺の始点座標・標高 [m]
- `X_end, Y_end, Z_end`: 辺の終点座標・標高 [m]
- `X_cnt, Y_cnt, Z_cnt`: 辺の中心座標・標高 [m]
- `edge_length`: 辺長 [m]（始点と終点の距離）

### 2. `face.csv` - セル中心データ

**カラム**:
- `CN`: セル番号
- `bill_area`: 建物面積 [m²]
- `area`: セル面積 [m²]
- `ratio`: 建物ポロシティ（空隙率）
- `_median`: 標高中央値 [m]
- `xcoord`, `ycoord`: セル中心座標 [m]
- `CalMesh`: 計算メッシュフラグ（0=計算対象、1=ダミー）
- `bill_perimeter`: 建物周長 [m]
- `10`, `20`, `50`, ... : 土地利用コード別面積 [m²]

### 3. `edge.csv` / `edge.gpkg` - セル辺データ

**カラム**:
- `CN`: セル番号
- `ID`: 境界フラグ（1=外縁、0=内部）
- `LN`: ライン番号
- `xcoord`, `ycoord`: 節点座標 [m]
- `node_id`: 節点番号（ユニーク）
- `StartNode`, `EndNode`: 辺の始点・終点のnode_id（反時計回り順序）
- `merge1_mn`: 節点標高 [m]
- `merge2_mn`: 辺の中心標高 [m]
- `peri`: 周辺フラグ（1=境界、0=内部）

### 4. 中間GeoPackageファイル

- `02_Line.gpkg`: セル境界線
- `06_2_Poly.gpkg`: Voronoiセル（標高統計付き）
- `08_Point.gpkg`: 節点データ（node_id, StartNode, EndNode付き）
- `10_2_Poly.gpkg`: セルポリゴン（標高統計付き）

**注**: `face.gpkg`と`edge.gpkg`はそれぞれ最終的なセル中心データとセル辺データです。

---

## ワークフロー

### 処理の流れ

```
┌─────────────────────────────────────────────────────────────────┐
│                  [1] 01_mkINPUT_Face_Edge.py                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │          入力データ               │
                │  - DEM (標高ラスタ)              │
                │  - 土地利用 (ラスタ)             │
                │  - OSM道路 (Shapefile)           │
                │  - OSM建物 (Shapefile, optional) │
                │  - 対象領域 (Shapefile)          │
                │  - ダミーメッシュ (optional)     │
                └─────────────────┬─────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 1: データ読み込みと前処理                  │
        │  - YAML設定ファイルの解析                        │
        │  - 対象領域のバッファリング（50m）               │
        │  - 座標系の統一                                  │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 2: セル生成点の抽出                        │
        │  - OSM道路から等間隔サンプリング                 │
        │  - 「長細いポリゴン」検出とクラスタリング       │
        │  - ダミーメッシュとの統合                        │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 3: Voronoi分割とクリッピング               │
        │  - scipy.spatial.Voronoiによる分割               │
        │  - 対象領域でのクリッピング                      │
        │  - ポリゴンの整合性チェック（make_valid）        │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 4: 標高統計の計算                          │
        │  - rasterstatsでゾーン統計                       │
        │  - YAMLパラメータで平均値/中央値を選択           │
        │  - 選択された統計量 → _median                    │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 5: 土地利用面積の集計                      │
        │  - 画素値ごとの面積計算                          │
        │  - 固定カテゴリ13種すべてを出力                  │
        │  - 存在しないカテゴリは0.0                       │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 6: 建物ポロシティの計算                    │
        │  - セルと建物のオーバーラップ面積                │
        │  - 建物周長の計算                                │
        │  - ratio = 1 - (建物面積 / セル面積)            │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 7: セル辺の抽出と番号割り当て              │
        │  - 各ポリゴンの境界線を抽出                      │
        │  - 座標ごとにユニークなnode_idを割り当て         │
        │  - 辺ごとにユニークなLN（ライン番号）を割り当て  │
        │  - セル間で同じ座標・同じ辺は同じIDを共有        │
        │  - MultiPolygon内での辺の重複除去                │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 8: 境界フラグの設定                        │
        │  - 外縁辺: ID=1 (peri=1)                         │
        │  - 内部辺: ID=0 (peri=0)                         │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 9: 標高のサンプリング                      │
        │  - 節点標高: merge1_mn (Point geometry)          │
        │  - 辺中心標高: merge2_mn (LineString geometry)   │
        └─────────────────────────┬─────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │          出力ファイル             │
                │  - face.csv / face.gpkg          │
                │  - edge.csv / edge.gpkg          │
                │  - 02_Line.gpkg                  │
                │  - 06_2_Poly.gpkg                │
                │  - 08_Point.gpkg                 │
                │  - 10_2_Poly.gpkg                │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼

┌─────────────────────────────────────────────────────────────────┐
│                     [2] 02_mkCELL.py                            │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │          入力ファイル             │
                │  - face.csv / face.gpkg          │
                │  - edge.csv / edge.gpkg          │
                └─────────────────┬─────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 1: データ読み込み                          │
        │  - YAML設定ファイルの解析                        │
        │  - face.csvの読み込み（セル中心データ）          │
        │  - edge.csvの読み込み（セル辺データ）            │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 2: セル辺のLNベースのペアリング            │
        │  - 各セル内でCN+LNでグループ化                   │
        │  - 同じLNを持つ2つの辺をペアリング               │
        │  - ラインオブジェクトの生成                      │
        │    (X_bgn, Y_bgn, node_bgn) ←→ (X_end, Y_end, node_end) │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 3: 隣接セルの探索                          │
        │  - node_idペアの正規化（min, max）               │
        │  - 同じnode_idペアを持つ別セルのラインを探索     │
        │  - CN_endを設定（境界の場合は0）                 │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 4: セル重心間距離（DL）の計算              │
        │  - 内部辺: セル重心間距離                        │
        │  - 境界辺: 重心から辺の中心点までの距離          │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 5: 方向コサインの計算                      │
        │  - dx/DL = (x_cell_bgn - x_cell_end) / DL        │
        │  - dy/DL = (y_cell_bgn - y_cell_end) / DL        │
        │  - 境界辺: 0.0, 0.0                              │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 6: 逆距離加重平均の重みの計算              │
        │  - weight_bgn: 辺中心への補間時のCN_bgnの重み   │
        │  - weight_end: 辺中心への補間時のCN_endの重み   │
        │  - weight_cnt: セルへの補間時の辺中心の重み     │
        │    （境界辺でも計算）                            │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 7: 辺長の計算                              │
        │  - edge_length = sqrt((X_end-X_bgn)²+(Y_end-Y_bgn)²) │
        └─────────────────────────┬─────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Step 8: 拡張フォーマットでの出力                │
        │  - 1行目: 総セル数                               │
        │  - セル情報（1行/セル）: 22列                    │
        │  - ライン情報（1行/ライン）: 23列                │
        └─────────────────────────┬─────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │          出力ファイル             │
                │  - {project_name}.txt            │
                │    (AQWA-INUNDATION2D用)         │
                └──────────────────────────────────┘
```

---

## トラブルシューティング

### よくあるエラー

#### 1. `ModuleNotFoundError: No module named 'geopandas'`

**原因**: 必要なライブラリがインストールされていません。

**解決方法**:
```bash
conda activate AQWA-RISK
conda install -c conda-forge geopandas rasterio rasterstats
```

#### 2. `エラー: 設定ファイルが見つかりません`

**原因**: YAMLファイルのパスが正しくありません。

**解決方法**:
```bash
# YAMLファイルの存在確認
ls yaml/

# 正しいパスでスクリプトを実行
python 01_mkINPUT_Face_Side.py yaml/Kinu_Joso.yaml
```

#### 3. `OSError: Cannot save file into a non-existent directory`

**原因**: 出力ディレクトリが存在しません。

**解決方法**: スクリプトが自動的に作成しますが、手動で作成する場合:
```bash
mkdir -p Cell/Your_Project/output_gpkg_csv
```

#### 4. `隣接セルのマッチング失敗`

**原因**: 座標の丸め誤差により、隣接セルが見つかりません。

**解決方法**: `02_mkCELL.py`は`node_id`ベースのマッチングを使用するため、座標の丸め誤差を回避できます。

#### 5. オプショナル入力（建物データ、ダミーメッシュ）の扱い

**ダミーメッシュを使用しない場合**:

YAMLファイルで`file: "none"`と指定してください。

```yaml
input:
  dummy_mesh:
    dir: "Cell/{project_name}/input_gpkg_tif"
    file: "none"  # ダミーメッシュを使用しない
```

この場合、道路データから抽出した点のみでVoronoi分割を行います。

**建物データを使用しない場合**:

YAMLファイルで`file: "none"`と指定してください。

```yaml
input:
  buildings:
    dir: "OpenStreetMap/kanto-latest-free.shp"
    file: "none"  # 建物データを使用しない
```

この場合、全てのセルで`bill_area=0`、`bill_perimeter=0`、`ratio=0`として処理されます。

---

## 注意事項

### データ品質
- **DEM**: 解像度5m以下を推奨
- **土地利用**: 国土数値情報（国土地理院）の画素値コードに対応
- **OSM**: 最新のデータを使用（Geofabrik等からダウンロード）

### 座標系
- **入力データ**: すべて同じ座標系（EPSG）に統一する必要があります
- **推奨座標系**: 平面直角座標系（EPSG: 2443-2461）

### メッシュサイズ
- セル数が多すぎる場合（10,000以上）、処理時間が長くなります
- `parameters.area_threshold`で小さなセルを統合できます

---

## ライセンス

このプロジェクトは研究・教育目的で使用できます。

---

## 参考文献

- AQWA-INUNDATION2D: 2次元氾濫解析モデル
- GeoPandas: https://geopandas.org/
- Shapely: https://shapely.readthedocs.io/
- Rasterio: https://rasterio.readthedocs.io/

---

## 更新履歴

詳細な修正履歴は`README_MODIFICATION.md`を参照してください。

### 主な変更点
- **2025/10**: 標高統計量の選択機能追加（平均値/中央値をYAMLで選択可能）
- **2025/10**: YAMLファイル構造の改善（roads, buildingsを分離）
- **2025/10**: オプショナル入力（建物データ、ダミーメッシュ）の柔軟な対応（`"none"`指定可能）
- **2025/10**: 境界辺でも方向コサイン（dx/DL, dy/DL）を計算（セル重心→辺中点）
- **2025/10**: セル辺のLNベースのペアリング（同一LNで2つの辺を自動マッチング）
- **2025/10**: 境界LNでのDLと逆距離加重平均の重み計算を改善
- **2025/10**: ライン情報に辺長（edge_length）を追加（23列目）
- **2025/10**: 隣接セル探索のnode_idペア正規化（順序不問のマッチング）
- **2025/10**: ポリゴン外縁の反時計回り（CCW）順序を保証（平面直角座標系に準拠）
- **2025/10**: face.gpkgとedge.gpkgの出力を追加（side→edgeにリネーム）
- **2024/XX**: YAMLファイルで`{project_name}`プレースホルダーをサポート
- **2024/XX**: `node_id`ベースの高精度マッチングを実装
- **2024/XX**: cell.txtの出力先をプロジェクトディレクトリ直下に変更
- **2024/XX**: 標高計算を平均値から中央値に変更
- **2024/XX**: 土地利用面積データの完全保持を実装
- **2024/XX**: セル辺の循環順整合性チェックを強化

---

## お問い合わせ

不具合の報告や機能リクエストは、Issues にてご連絡ください。

---

**Happy Mesh Generation! 🌊🗺️**

