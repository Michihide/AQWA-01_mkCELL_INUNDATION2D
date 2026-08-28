# 地形適合型ハイブリッド非構造格子生成ツール

DEM・解析領域ポリゴン・拘束ブレークライン（堤防・道路など）から、氾濫解析（2D
浸水想定）向けの非構造格子を自動生成する。既定は三角形のみ。外周四角形帯は
`mesh.boundary_quad_band.enabled: true` のときだけ付ける。Gmsh の Python API を
用い、生成 → 検査 → 局所細分化 → 修復を収束するまで反復する。

## 実行方法

作業ディレクトリは **`tool/`** とする。Python 3.11+ と Gmsh が必要。
`flake.nix` があるため `nix develop` が最も手早い。

```bash
cd tool
nix develop          # 初回のみ。Gmsh 等の依存を用意
```

### メッシュ生成

```bash
# 開発用設定（tool/config/。AQWA 絶対パス・開発向け出力先）
python run.py --config config/kuma_hitoyoshi.yaml
python run.py --config config/hii.yaml

# データセット設定（各データセットの yaml/ から実行）
python run.py --config ../Mesh/Kuma_Hitoyoshi/yaml/mesh_config.yaml
python run.py --config ../Mesh/Hii/yaml/mesh_config.yaml
```

### 氾濫ブロック並列（Kuma 等）

解析領域を氾濫ブロック（ポリゴン）単位に分割し、ブロックごとに Gmsh を回して
face/edge/cell.bin をマージする。YAML の `parallel.enabled: true` でも有効だが、
明示的には `--blocks` を付ける。

```bash
# 全域（30 ブロック、8 並列）
python run.py --config ../Mesh/Kuma_Hitoyoshi/yaml/mesh_config.yaml --blocks --workers 8

# 地形緩和版（セル数削減。現在ソルバー投入済み — Kuma_Hitoyoshi/MESH_ACTIVE.md 参照）
python run.py --config ../Mesh/Kuma_Hitoyoshi/yaml/mesh_config_relaxed_terrain.yaml --blocks --workers 8

# 単ブロック試験（設定変更の短時間確認）
python run.py --config ../Mesh/Kuma_Hitoyoshi/yaml/mesh_config_relaxed_boundary_test.yaml \
  --blocks --block-indices 0 --workers 1
```

### 既存メッシュからソルバー出力のみ再生成

```bash
python run.py --config config/hii.yaml --export-only
python run.py --config ../Mesh/Kuma_Hitoyoshi/yaml/mesh_config.yaml --export-only \
  --mesh ../Mesh/Kuma_Hitoyoshi/output_gmsh_tool/kuma_mesh.msh
```

### 補助スクリプト

```bash
# DEM から盛り土天端線を検出（input.breaklines.embankments 用 GPKG）
python scripts/detect_embankments.py --help

# OSM 道路を breakline GPKG 化（幹線のみ、または --all-classes で全道路）
python scripts/make_osm_road_breaklines.py --help

# 生成済みメッシュの隣接要素サイズ比を事後検査
python scripts/check_level_balance.py --help
```

### テスト

```bash
cd tool
pytest
```

nix を使わない場合は `requirements.txt` を任意の Python 3.11+ 環境に
インストールする。

---

## 目次

- [リポジトリ構成](#リポジトリ構成)
- [設定ファイル](#設定ファイル)
- [パイプラインの流れ](#パイプラインの流れ)
- [メッシュ作成で気をつけている点](#メッシュ作成で気をつけている点)
- [ディレクトリ構成（tool/）](#ディレクトリ構成tool)
- [出力](#出力)
- [既知の制限](#既知の制限)

特殊辺ラインの入力と手順は [docs/README.md](../docs/README.md)。
mesh.bin のバイト列は [docs/mesh_bin_spec.md](../docs/mesh_bin_spec.md)。

## リポジトリ構成

```
gmsh/
├── tool/                          本ツール（ここで run.py を実行）
│   ├── config/                    開発・検証用 YAML（AQWA 絶対パス等）
│   └── data/                      検出済み盛り土・OSM 道路などの GPKG
├── Kuma_Hitoyoshi/
│   ├── yaml/                      本番相当のメッシュ設定
│   ├── output_gpkg_csv/           標準版 face/edge 出力
│   ├── output_relaxed/            地形緩和版出力（ソルバー投入済み）
│   └── MESH_ACTIVE.md             現在使用中の cell.bin の記録
├── Hii/
│   ├── yaml/                      斐伊川のメッシュ設定
│   └── input_gpkg_tif/            入力 DEM・領域（同梱）
├── docs/mesh_requirements.md      外注向け要件定義
└── mesh_bundle/                   外注配布用バンドル（legacy + experimental + sample_data）
```

## 設定ファイル

設定 YAML は **2 か所** に置ける。いずれも `run.py --config` で指定する。
パスは **YAML ファイルの位置**（`base_dir`）からの相対パスで解決される。
データセット用 YAML は `yaml/` に置き、入力・出力は `..` でデータセット直下を
参照するのが標準。

| 場所 | 用途 |
| --- | --- |
| `tool/config/*.yaml` | 開発・単体検証。`tool/output/` 等へ出力 |
| `{Dataset}/yaml/*.yaml` | 本番データセット向け。`output_gpkg_csv/` 等へ出力 |

### Kuma_Hitoyoshi の設定一覧

| ファイル | 内容 |
| --- | --- |
| `yaml/mesh_config.yaml` | 標準（地形適応・反復 12 回） |
| `yaml/mesh_config_relaxed_terrain.yaml` | 地形適応を弱めセル数削減（**現行ソルバー投入**） |
| `yaml/mesh_config_osm_roads.yaml` | OSM 全道路を拘束 breakline 化（重い） |
| `yaml/mesh_config_relaxed_boundary_test.yaml` | 境界帯改善の単ブロック試験用 |

### 主要な設定項目

既定値は `src/config.py` 参照。未知のキーはロード時にエラーになる。

| キー | 役割 |
| --- | --- |
| `input.domain` / `input.dem` | 解析領域・DEM のパス |
| `input.special_edges.file` | 特殊辺ライン 1 本（kind 列で区別。厳密なメッシュ拘束） |
| `input.breaklines.*` | 幾何だけの拘束ブレークライン（堤防天端など） |
| `input.reference_layers.*` | 参照レイヤ（OSM 等、診断・サイズ場の目安） |
| `crs.target_epsg` | メートル単位の投影座標系（Web Mercator 等は禁止） |
| `mesh.min_element_area` | 面積下限 [m²]。設計思想の起点 |
| `mesh.boundary_quad_band.*` | 外周四角形帯の幅・サイズ・自己交差時の縮小 |
| `mesh.breakline_processing.*` | 拘束線の間引き・簡略化・最小離隔 |
| `mesh.max_neighbor_element_ratio` | 隣接要素の代表辺長比上限（レベル差 1） |
| `mesh.max_iterations` / `refinement_factor` | 反復回数と細分化率 |
| `mesh.max_refine_attempts` | 同一地点の細分化回数上限 |
| `terrain.plane_fit_rmse_max` / `slope_direction_spread_max_deg` | 地形適合基準 |
| `quality.*` | 三角形・四角形の幾何品質 |
| `mesh.interior_quads.*` | 内部四角形化（既定で無効） |
| `parallel.*` | 氾濫ブロック並列（`enabled`, `workers` 等） |
| `output.gpkg_csv.*` | ソルバー向け `face.gpkg` / `edge.gpkg` |
| `output.cell_bin.*` | AQWA 互換 `cell.bin` |

### 設定例（`Hii/yaml/mesh_config.yaml`）

```yaml
input:
  domain: ../input_gpkg_tif/input_gpkg.gpkg
  dem: ../input_gpkg_tif/DEM1M.tif
  special_edges:
    file: ../input_gpkg_tif/special_edges.gpkg
    default_kind: ROAD
  breaklines:
    embankments: ../../tool/data/hii_embankments.gpkg

output:
  directory: ../output_gmsh_tool
  basename: hii_mesh
  gpkg_csv:
    directory: ../output_gpkg_csv
    face_gpkg: face.gpkg
    edge_gpkg: edge.gpkg
  cell_bin:
    enabled: true
    directory: ..
    filename: Hii.bin
```

## パイプラインの流れ

```
解析領域の読込・修復
  -> 特殊辺ライン / 拘束ブレークライン / 参照レイヤの読込
  -> 外周四角形帯の構築（Γ1 を決める）
  -> 拘束線の前処理（間引き・ノーディング・帯との干渉回避）
  -> DEM の読込
  -> 初期サイズ場の構築
  -> [ Gmsh でメッシュ生成 -> 修復 -> 品質・地形評価 -> サイズ場更新 ] を反復
  -> 成果物の出力（msh/VTU/XDMF/GeoPackage/CSV/mesh.bin/診断図）
```

`--blocks` 指定時は、上記を氾濫ブロックごとに実行し、`block_merge.py` で
face/edge/cell.bin を統合する。

反復は `src/main.py` の `_iterate()` が担う。各反復で全基準を満たすか
`mesh.max_iterations` に達すると終了する。**面積下限に到達したために解消
できない違反は「解消不能」として記録するだけで、反復を止める理由にはしない**
（無限に細分化しないため）。

## メッシュ作成で気をつけている点

### 1. 面積下限はハード制約、ただし境界は例外

`mesh.min_element_area`（既定 625 m²）を下回る要素は作らないことを最優先の
制約にしている。サイズ場・拘束線の前処理・修復のすべてがこの下限を守るように
設計されている（`config.py` の `triangle_size_floor` / `quad_size_floor`、
`size_field.py` のフロア機構など）。

一方で、**解析領域の外周（堤防・道路などの地形境界そのもの）は面積下限の対象
外**にしている。外周の形状再現性を最優先にするためで、`mesh_parser.py` の
`boundary_touching_element_mask()` で「外周四角形帯の要素」および「区間
スキップにより外形線まで下りた内側三角形」を判定し、面積評価
（`quality_metrics.py`）・修復（`mesh_repair.py` の統合・collapse）・
細分化判定（`adaptive_refinement.py`）のすべてで除外する。ただし退化に近い
面積（`_DEGENERATE_AREA_EPS`）は別途排除しており、無制限に小さい要素を許す
わけではない。**内部の拘束ブレークライン（堤防・道路等が領域内部にある場合）
には、この例外は適用されない**（面積下限が優先され、下限を守れない区間は
拘束を諦める。「問題解決」参照）。

### 2. 境界形状の再現性（既定は三角形のみ）

既定では外周四角形帯は無効（`mesh.boundary_quad_band.enabled: false`）で、
領域全体を三角形で埋める。境界は `target_size`（未指定なら `global_min_size`）
間隔でサンプルする。四角形帯を使うときだけ `enabled: true` にする。

有効にすると外周に 1 層の四角形帯（`boundary_quad_band.py`）を Transfinite で
敷き、内向きオフセットで作った内側境界（Γ1）とを対にして各区間を四角形にする。

- 帯の幅・接線方向サイズ（`width` / `target_size`）は、面積下限に対応する
  正方形サイズ（`sqrt(min_element_area)`）に合わせるのが既定の考え方。
  帯を細かくするほど堤防・道路の折れ線形状を忠実に再現できる。
- 内向きオフセットが自己交差する頂点では帯幅を段階的に縮める
  （`min_width_ratio` / `max_shrink_passes`）。
- 四角形を置けない区間（自己交差が解消しない等）は、**境界の頂点を動かさず**
  に三角形へ譲る（区間スキップ）。スキップした区間の両端には、内側要素との
  段差を滑らかにする閉じ三角形を挿入する。
- 頂点を切り落として面積を稼ぐ `max_corner_cut` は既定で無効（0）。
  境界形状を一切変えない方を優先している。
- 解析領域からくびれ・スパイクを除去する `narrow_feature_radius`
  （モルフォロジー開処理）は、帯の**縮小後の最小幅**
  （`width * min_width_ratio`）を基準に自動決定する。

### 3. 地形適合性（DEM 基準）

各要素について DEM をラスタライズし、`terrain_metrics.py` で次を評価する。

- **平面フィット RMSE**（`terrain.plane_fit_rmse_max`）
- **勾配方向のばらつき**（`terrain.slope_direction_spread_deg_max`）
- 代表標高は面積加重平均など複数の方式から選べる
  （`representative_elevation_method`）
- DEM サンプル数が `minimum_dem_samples_per_element` に満たない要素は細分化対象

地形基準は細分化しても解消しない場合があるため、`mesh.max_refine_attempts` で
同じ場所の細分化回数に上限を設けている。

### 4. 要素品質基準

`quality_metrics.py` / `config.py` の `QualityConfig` で、三角形・四角形
それぞれに幾何品質の基準を課す。内部の四角形は既定で無効
（`mesh.interior_quads.enabled: false`）。

### 5. 隣接要素サイズの「レベル差 1」

`mesh.max_neighbor_element_ratio`（既定 2.0）で、辺を共有する要素どうしの
代表辺長比の上限を課している。実要素側の保証はこのキーの役割で、違反する
粗い三角形は最長辺二分割で修復する。

### 6. 特殊辺ラインと拘束ブレークライン

手順は **計算範囲と特殊辺ラインを先に定め、そこからメッシュを生成する**。
特殊辺はソルバーの `kind`（`ROAD` / `CULVERT` / `Q` / `W` / `WALL`）を
持つ線で、Gmsh の拘束条件になる。生成後、両端がライン上にある辺を
`special_edges.csv`（mesh.bin の strc）へ書く。

```yaml
input:
  special_edges:
    file: special_edges.gpkg  # ROAD / CULVERT / Q などを 1 ファイルに
    default_kind: ROAD        # kind 列が無い線。道路 / culvrt なども可
    z_mode: absolute          # zc が絶対標高か、地盤からの相対高さ
    match_tol: 2.0            # 辺対応付けの許容 [m]。0 なら自動
```

属性列の既定は `kind`, `zc`, `H`, `z_road`, `B`, `cover`, `z_mode`。`qgroup` は Q / W のときだけ。
`zc` / `z_road` が無ければ辺の `z_crest`。`z_mode: relative` なら DEM に足す。
`B` が 0 ならメッシュ辺長（カルバート開口幅はここに書く）。
カルバートの `cover` は開口上の厚さ（常に m。`z_mode` の対象外）。天端は `zc + H + cover`。
折れ線は折点で区間に分け、属性は区間ごとに載せる。
特殊辺は近接・短尺を理由に拘束から落とさない。
交差せずに下限より近い、または重なるときは距離の傾向を出して止める。

幾何だけの拘束（堤防天端など、ソルバー kind を付けない線）は従来どおり
`input.breaklines`。`breaklines.py` が間引き・ノーディング・外周帯との干渉回避
を行い、特殊辺も同じ前処理を通る。

参照レイヤ（OSM 道路等、`reference_layers`）は既定では診断表示専用。
`features.reference_layers_for_size_field: true` にするとサイズ場だけを細かくする。

DEM から検出した盛り土天端線（`embankment_detection.py`）と OSM は型で分離。
5 m DEM 区域では `scripts/make_osm_road_breaklines.py` で OSM 道路を GPKG 化し、
`input.breaklines.roads` に指定できる。幹線のみが既定で、`--all-classes` で
全道路クラスを含められる（Kuma 全道路版は Gmsh が非常に重い）。
ソルバーの道路堰・カルバートは `input.special_edges.file` にまとめる。

### 7. DEM からの盛り土検出

1m DEM が理想だが 5m でも実用（`scripts/detect_embankments.py` /
`src/embankment_detection.py`）。OSM は検出結果の事後照合にのみ使う。

### 8. 座標系（EPSG）

`crs.target_epsg` はメートル単位の投影座標系を必須とする。Web Mercator
（EPSG:3857）や地理座標系は禁止（`config.py` の `_FORBIDDEN_EPSG`）。

### 9. 反復・収束・打ち切り

- 各反復で品質・地形の違反箇所周辺のサイズ場を下げ、次のメッシュを再生成
- 面積下限の壁に当たっている違反（`blocked_by_area_floor`）は収束判定から除外
- `max_iterations` に達した場合は警告を出して打ち切り

## ディレクトリ構成（tool/）

```
tool/
├── run.py                      エントリポイント
├── config/                     開発用 YAML
├── data/                       補助 GeoPackage（盛り土・OSM 道路等）
├── src/
│   ├── config.py               設定の読込・検証
│   ├── main.py                 パイプライン全体（反復・出力）
│   ├── block_runner.py         氾濫ブロック並列実行
│   ├── block_merge.py          ブロック成果物のマージ
│   ├── block_helpers.py        ブロック分割ユーティリティ
│   ├── io_vector.py            領域・拘束線・参照レイヤの読込
│   ├── special_edges.py        特殊辺ラインの拘束化と辺対応付け
│   ├── io_raster.py            DEM 読込
│   ├── geometry_cleaning.py    領域ポリゴンの修復
│   ├── boundary_quad_band.py   外周四角形帯
│   ├── breaklines.py           拘束ブレークライン前処理
│   ├── embankment_detection.py 盛り土天端線検出
│   ├── size_field.py           背景サイズ場
│   ├── reference_size_field.py 参照レイヤに基づくサイズ場
│   ├── mesh_generator.py       Gmsh メッシュ生成
│   ├── gmsh_geometry.py        Gmsh ジオメトリ構築
│   ├── mesh_parser.py          メッシュデータ構造・境界判定
│   ├── mesh_repair.py          面積下限・レベル差違反の修復
│   ├── quality_metrics.py      要素品質評価
│   ├── terrain_metrics.py      地形指標
│   ├── adaptive_refinement.py  サイズ場更新・収束判定
│   ├── solver_gpkg_export.py   face/edge GPKG 出力
│   ├── cell_bin_export.py      cell.bin 出力
│   ├── zonal_stats.py          ラスタ重なり集計
│   ├── diagnostics.py          診断図
│   ├── exporters.py            msh/VTU 等
│   └── utils.py
├── scripts/                    盛り土検出・OSM 道路 GPKG 化・事後検査等
└── tests/                      pytest
```

## 出力

`output.directory` 以下（データセット YAML では `../output_gmsh_tool` 等）に
生成される。

- `mesh.msh` / `mesh.vtu` / `mesh.xdmf` — メッシュ本体
- `mesh_elements.gpkg` / `mesh_elements.csv` — 要素属性（診断用）
- `mesh_nodes.csv` / `mesh_connectivity.csv` — 節点・接続
- `iteration_log.csv` / `iterations/` — 反復統計・中間メッシュ
- `summary.json` — 要素数・違反件数等のサマリ
- `diagnostics/` — 品質・地形・帯・レベル差の診断図（`save_diagnostics: true`）

`output.gpkg_csv.directory` 以下に 01_mkMESH_INUN2DH 互換のソルバー入力:

- `face.gpkg` / `face.csv` — 面要素
- `edge.gpkg` / `edge.csv` — 辺要素
- `{project}.bin`（`output.cell_bin`）— AQWA が読む格子（中身は AQWAMESH）

`--blocks` 実行時は各ブロックの中間出力が `blocks/block_NNN/` に残り
（`parallel.keep_block_outputs: true`）、マージ後に統合 face/edge/cell.bin が
データセット直下等に出力される。

### ソルバー入力（mesh.bin）

`output.cell_bin.enabled: true` で `02_Solver` が読む AQWAMESH を出力する。
面の幾何・代表標高はメッシュから計算し、土地利用・土壌は指定ラスタとの重なりを
ピクセル単位で集計する。建物ポリゴン（`buildings`）から `chi_bld` / `bld_peri` を
求める。植生ポリゴン（`vegetation`）から `chi_veg` と属性 `Cd` / `a` / `Hv`
（列名は `veg_field_*` で変更可）を交差面積で面へ載せる。入力未指定時は 0 埋め。

## 既知の制限

- 面積下限の境界例外は**解析領域の外形線**に限る。内部の拘束ブレークラインには
  適用されない。
- 地形基準は細分化しても解消しない地形が存在し得る。`max_refine_attempts` 到達時は
  「解消不能」として記録されるだけでエラーにはならない。
- DEM 解像度は 1–2 m が望ましいが、5 m でも実用（球磨川での検証実績）。
- `cell.bin` の土地利用・土壌統計は、ラスタが要素より粗い場合、絶対面積としては
  過大評価され得る。ソルバー側は面内の相対比率としてしか使わない。
- OSM 全道路を breakline 化（`mesh_config_osm_roads.yaml`）は Gmsh が非常に重く、
  全域実行前に `--block-indices 0` での試験を推奨。
- 外周は `target_size` 間隔で再サンプリングするため、解析領域ポリゴンと
  `mesh_elements.gpkg` の外形が 1 頂点単位では一致しない（意図した動作）。
  範囲が分断される場合は `narrow_feature_radius` の設定ミスを疑う。
