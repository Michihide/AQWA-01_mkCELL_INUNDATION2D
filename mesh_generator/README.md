# 地形適合 三角形メッシュ 生成プログラム

QGIS/GIS データ（解析対象領域ポリゴン・DEM・建物ポリゴンなど）から、
氾濫解析モデルに入力できる **三角形メッシュ**（node / cell / edge テーブル）を
生成する Python プログラムです。

`scipy.spatial.Delaunay` をベースに実装しており、将来的に
Triangle / Gmsh による制約付き Delaunay へ差し替えやすいモジュール構成です。

> 本プログラムは既存の四角形/Voronoi 系メッシュ生成（`01_mkINPUT_Face_Edge.py`）
> とは独立した、別系統の三角形メッシュ生成ツールです。

---

## 目的

- 任意の解析境界ポリゴン内に **三角形のみ** のメッシュを生成する。
- 最小メッシュサイズ（既定 5 m）を守る。
- DEM 標高分布（z_range / z_std）が大きい領域を細分化し、セル内標高ばらつきを抑える。
- 建物の周囲にメッシュを配置し、建物を「粗度として扱う」か「穴として除外」かを選べる。

---

## 入力データ

| 種別 | 必須 | 形式 | 説明 |
|------|:----:|------|------|
| 解析対象領域ポリゴン | ✅ | gpkg/shp/geojson | メッシュを張る範囲 |
| DEM | ✅ | GeoTIFF | 標高ラスタ |
| 建物ポリゴン | 任意 | gpkg/shp/geojson | 建物境界・周囲点・属性に使用 |
| 河川ライン | 任意 | gpkg/shp/geojson | ライン沿いに点配置 |
| 道路ライン | 任意 | gpkg/shp/geojson | ライン沿いに点配置 |
| 重要施設ポイント | 任意 | gpkg/shp/geojson | 周囲を細かくメッシュ化 |

緯度経度（地理座標系）の入力は、解析領域重心から推定した UTM など
**メートル単位の投影座標系へ自動変換** されます。

---

## インストール

```bash
pip install -r requirements.txt
# conda 環境（例: AQWA-RISK）なら主要ライブラリは導入済みのことが多い
```

---

## 実行方法

### 設定ファイルで実行

```bash
python create_tri_mesh.py --config configs/sample_config.yaml
```

### 引数のみで簡易実行

```bash
python create_tri_mesh.py \
  --area data/analysis_area.gpkg \
  --dem data/dem.tif \
  --buildings data/buildings.gpkg \
  --out output_mesh \
  --min-distance 5 \
  --interior-spacing 20
```

主な引数:

- `--config`: YAML/JSON 設定ファイル
- `--area / --dem / --buildings / --rivers / --roads / --facilities`: 入力
- `--out`: 出力ディレクトリ
- `--min-distance`: 最小点間距離 [m]
- `--interior-spacing`: 内部点間隔 [m]
- `--building-mode`: `building_as_roughness` | `building_as_hole`
- `--no-refine`: 標高ベースの反復細分化を無効化

CLI 引数は設定ファイルの値を上書きします。

---

## config.yaml の説明

`configs/sample_config.yaml` を参照してください。主なセクション:

- `input`: 入力ファイルパス（任意入力は `none` で無効化）
- `output`: 出力ディレクトリ
- `crs`: `auto_utm`（UTM 自動推定）/ `target_epsg`（明示指定）
- `mesh`: 各種点間隔・最小距離・建物モード
- `elevation`: 勾配点・標高細分化のしきい値・反復回数
- `quality`: 品質警告のしきい値

---

## 出力ファイル

```
output_mesh/
  gis/
    triangles.gpkg            # 三角形ポリゴン＋全属性
    points.gpkg               # 生成点群（type / priority）
    flagged_triangles.gpkg    # 品質または標高分布に問題のある三角形
  csv/
    nodes.csv                 # node_id, x, y
    cells.csv                 # cell_id, node1..3, 重心, 面積, 標高統計, 建物率, 品質
    edges.csv                 # edge_id, node_a/b, left/right_cell, 辺長, 法線, 境界種別
    mesh_quality_summary.csv  # 品質サマリ
  preview/
    mesh_preview.png
    elevation_range_preview.png
    building_fraction_preview.png
    quality_flags_preview.png
```

---

## QGIS での確認方法

1. QGIS で `gis/triangles.gpkg` を開く。
2. プロパティ → シンボロジで `z_range` や `building_fraction` を段階表示にすると、
   標高ばらつきや建物占有率を可視化できる。
3. `gis/flagged_triangles.gpkg` を重ねると、品質や標高分布に問題のある三角形を確認できる。
4. `gis/points.gpkg` で点群の種類（type）と優先度（priority）を確認できる。

---

## 建物の扱い

### building_as_roughness（推奨・既定）

- 建物内部にも三角形を作る。
- 各三角形に `building_fraction`（建物重複面積割合）・`building_count`・`building_intersect` を付与。
- 計算モデル側で建物占有率を粗度・遮蔽率として利用できる。
- 三角形のみを保ちやすい。

### building_as_hole

- 重心が建物内にある三角形を削除する（Difference は使わず重心判定）。
- 建物境界をまたぐ三角形（交差するが重心は外）には `building_warning = True` を付与。
- 内部規則点は建物内部を除外して生成する。

---

## メッシュ品質の見方

各三角形に以下を付与:

- `area`, `perimeter`, `edge_min/max/mean`
- `aspect_ratio`（最長辺/最短辺）, `compactness`（4πA/P²）
- `min_angle`, `max_angle`
- 品質フラグ: `flag_short_edge`, `flag_small_area`, `flag_small_angle`, `flag_high_aspect`, `quality_flag`

既定の警告基準: 辺長 < 5 m / 面積 < 12.5 m² / 最小角 < 20° / アスペクト比 > 8。

---

## 標高分布に基づく再分割の考え方

1. 初期点群で Delaunay → 各三角形の標高統計（z_range, z_std）を計算。
2. `z_range > 閾値` または `z_std > 閾値`、もしくは長辺かつ標高差ありの三角形を抽出。
3. その重心に追加点を入れ、最小距離を守って間引き、再 Delaunay。
4. `max_refine_iter` 回まで繰り返す（1 反復の追加点数に上限あり）。

これにより、起伏の大きい領域だけ局所的に細かいメッシュになります。

---

## 注意事項

- **QGIS 標準 Delaunay との差**: 本実装は点群の Delaunay 後に「重心が領域内」の
  三角形のみ残す方式です。境界線を厳密にメッシュ辺として保持しません。
- **厳密な制約付き Delaunay が必要な場合**は `triangle` または `Gmsh` 連携を
  追加してください（モジュール `triangulation.py` を差し替え可能な設計です）。
- 三角形のみを保つため、Clip / Difference による切断は行いません
  （境界外・建物内の除去は重心判定）。
- 建物が非常に多い場合に備え、建物属性計算では STRtree（空間インデックス）を使用します。
- 乱数を使わないため、同一入力・同一環境では出力が再現します。

---

## プログラム構成

```
mesh_generator/
  create_tri_mesh.py        # CLI エントリポイント
  meshlib/
    config.py               # 設定読み込み・既定値
    io.py                   # GIS 入出力
    crs.py                  # 座標系推定・変換
    points.py               # 点列化・規則点・間引き
    buildings.py            # 建物点・建物属性
    elevation.py            # 標高統計・勾配点・細分化判定
    triangulation.py        # Delaunay・境界外除去・再採番
    quality.py              # 品質指標・フラグ
    adjacency.py            # node/cell/edge テーブル
    plotting.py             # プレビュー画像
  configs/
    sample_config.yaml
  requirements.txt
  README.md
```
