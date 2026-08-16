# メッシュ作成プログラム 外注要件書

## 1. 目的・スコープ

### 1.1 目的

DEM・解析領域・堤防/道路線から、2D 氾濫解析ソルバー **AQWA-INUNDATION2D** 向けの非構造メッシュを自動生成する。

### 1.2 メッシュの形

- **外周**：細かい四角形 1 層（境界形状優先）
- **内部**：三角形主体
- **出力**：`face.gpkg` / `edge.gpkg` / `cell.bin`（既存 `01_mkMESH_INUN2DH` 互換）

### 1.3 対象データセット

| データセット | 規模 | 用途 |
|---|---|---|
| 球磨川・人吉（Kuma_Hitoyoshi） | ~215 km²、30 氾濫ブロック | 本番相当 |
| 日置（Hii） | 小規模 | 回帰テスト |

### 1.4 外注の位置づけ

既存 Python/Gmsh パイプラインの引き継ぎ・拡張、性能改善、保守。CLI + YAML 実行。

---

## 2. 入力

| 項目 | 形式 | 内容 |
|---|---|---|
| 解析領域 | GeoPackage（ポリゴン） | 氾濫解析対象域 |
| DEM | GeoTIFF | 5 m 以上、1–2 m 推奨 |
| 拘束線 | GeoPackage（LineString） | 堤防・道路等。メッシュ辺に一致させる |
| 参照レイヤ | GeoPackage | OSM 等。診断・サイズ場ガイド用（拘束とは別） |
| 属性ラスタ | GeoTIFF / Shapefile | 土地利用・土壌・建物（cell.bin 用、任意） |
| 設定 | YAML | パス・閾値・優先順位・出力先をすべて記述 |

**座標系**：メートル単位の投影座標（EPSG:6670 等）。Web Mercator / 緯度経度は不可。

---

## 3. 形状・サイズ

| 要件 | 内容 |
|---|---|
| 面積下限 | **任意設定**（例：625 m²）。`0` または `enforce_min_element_area: false` で無効化可 |
| 外周帯 | 幅 ~25 m の四角形帯（YAML で変更可）。境界折れ線を再現 |
| 外周例外 | 解析領域**外形線**上の要素は面積下限の対象外可 |
| 内部拘束線 | 領域内の堤防・道路は外周例外の対象外 |
| 反復 | 違反箇所を細かくして再生成。回数上限は YAML で設定 |

### 3.1 拘束線 vs 面積下限の優先（YAML で選択）

両立できない区間で、どちらを優先するかを設定する。

| モード | 挙動 |
|---|---|
| `area_floor`（現行既定） | 面積下限を優先。守れない拘束は見送り（sacrifice） |
| `breaklines` | 拘束線を優先。面積下限を下回る要素を許容（ログ・summary に記録） |
| `balanced`（任意） | 拘束カバレッジ下限・面積割れ許容率などを YAML で指定 |

```yaml
mesh:
  min_element_area: 625.0           # 任意。0 で無効
  enforce_min_element_area: true

features:
  constraint_priority: area_floor   # area_floor | breaklines | balanced
  min_breakline_coverage: 0.95      # balanced / breaklines 時
  max_area_violation_ratio: 0.01    # balanced / breaklines 時
```

---

## 4. 標高代表性

**前提**：ソルバーは各要素を **頂点標高の線形補間（区間線形勾配）** で地形を表現する。

| 要件 | 内容 |
|---|---|
| ノード標高 | DEM から各格子点に標高を与える |
| 代表標高 | 要素重心付近の代表値（面積加重平均等、YAML で方式選択） |
| 判定 | 要素内 DEM と頂点平面近似の RMSE ≤ 閾値（例：1.5 m） |
| 対応 | 超える要素は細分化 |

---

## 5. 斜面代表性

**評価方法**：**ノード間の勾配**（隣接ノードの標高差 ÷ 水平距離）から要素内の斜面を評価する。

| 要件 | 内容 |
|---|---|
| 勾配の向き | 要素内各辺の勾配方向のばらつき ≤ 閾値（例：60°） |
| 勾配の大きさ | 要素内 \|∇z\| がおおむね均一（平坦地は判定除外） |
| 対応 | 違反要素は細分化 |

---

## 6. レベル差（粗密の段差）

メッシュを **辺長の段（レベル）** に分けたとき、**辺を共有する隣接要素** のレベル差は **1 以内** とする。

| 項目 | 内容 |
|---|---|
| 1 レベル | 代表辺長が 2 倍（面積は約 1/4）— `level_step_ratio` で変更可 |
| 代表辺長 `L` | 三角形：`L = √(4×面積/√3)`、四角形：`L = √面積` |
| 隣接 | **辺を共有**する要素同士（頂点のみの接触は含めない） |
| 制約 | `max(L_i, L_j) / min(L_i, L_j) ≤ 2.0` ⇔ レベル差 ≤ 1 |
| 理由 | 要素ごとの時間刻みを変える陽解法で、段差 2 以上は計算不安定 |

---

## 7. 隣接要素の重心間距離

ソルバー `cell.bin` の **`dl`** と同一定義。

### 7.1 記号の定義

| 記号 | 意味 |
|---|---|
| `L_i` | 要素 i の代表辺長 [m] |
| `L_j` | 要素 j の代表辺長 [m] |
| `(L_i + L_j) / 2` | 2 要素の平均的なスケール |
| `dl` | 要素 i と j の **重心間の直線距離** [m]（内部共有辺の場合） |

```
    要素 i                    要素 j
  ┌─────────┐              ┌─────────┐
  │    ● Ci │              │ Cj ●    │
  │         │── 共有辺 ────│         │
  └─────────┘              └─────────┘
              ←── dl ──→
```

- **内部の辺**：`dl` = 2 要素の重心間距離
- **外周の辺**：`dl` = 重心から辺中点までの距離（別扱い）

### 7.2 判定基準（YAML で設定）

| 比 | 意味 | 例 |
|---|---|---|
| `dl / ((L_i + L_j) / 2) ≤ 上限` | 重心距離が平均辺長に対して大きすぎない | 上限 2.0 |
| `dl / min(L_i, L_j) ≥ 下限` | 重心が共有辺に近すぎない（スリバー防止） | 下限 0.5 |

**例**（一辺 10 m の正方形 2 枚が接している場合）

- `L_i = L_j = 10 m`、`dl = 10 m`
- `dl / 平均 = 10 / 10 = 1.0` → OK

---

## 8. パラメータ設定（YAML）

流域・DEM・道路密度ごとに調整が必要。**判定・修復・優先順位に使う数値はすべて YAML で変更可能**（ハードコード禁止）。

```yaml
mesh:
  min_element_area: 625.0              # 任意
  enforce_min_element_area: true
  global_min_size: 45.0
  global_max_size: 150.0
  max_neighbor_element_ratio: 2.0      # レベル差（辺長比）
  level_step_ratio: 2.0
  centroid_distance:
    enabled: true
    dl_over_mean_size_max: 2.0
    dl_over_min_size_min: 0.5
  max_iterations: 12
  max_refine_attempts: 2
  refinement_factor: 0.7
  boundary_quad_band:
    width: 25.0
    target_size: 25.0
  breakline_processing:
    simplify_tolerance: 5.0
    min_length: 45.0

terrain:
  plane_fit_rmse_max: 1.5
  slope_direction_spread_max_deg: 60.0
  slope_magnitude_spread_max: 0.0      # 0 = 無効
  representative_elevation_method: area_weighted_mean
  minimum_dem_samples_per_element: 4
  flat_slope_threshold: 0.01

quality:
  triangle_min_angle_deg: 25.0
  triangle_aspect_ratio_max: 4.0

features:
  constraint_priority: area_floor
  breaklines_as_mesh_edges: true

input:
  domain: ...
  dem: ...
  breaklines:
    embankments: ...
    roads: ...

output:
  gpkg_csv:
    face_gpkg: face.gpkg
    edge_gpkg: edge.gpkg
  cell_bin:
    enabled: true
    filename: Kuma_Hitoyoshi.bin

parallel:
  enabled: true
  workers: 8
```

**運用ルール**

- 各基準は個別に有効/無効可能
- 未知キーは起動時エラー
- 実行後 `summary.json` に実測値・違反件数・trade-off 統計を記録

---

## 9. 出力

| 種別 | ファイル | 必須 |
|---|---|---|
| ソルバー入力 | `face.gpkg`, `edge.gpkg`, `cell.bin` | ○ |
| メッシュ本体 | `mesh.msh`, `mesh.vtu` | 推奨 |
| 診断 | `summary.json`, 診断図, `iteration_log.csv` | 推奨 |
| 拘束確認 | `mesh_breaklines.gpkg` | 推奨 |

**並列**：氾濫ブロック単位で分割生成 → 統合 → domain 全体の `cell.bin` を 1 本出力。

**再出力**：`python run.py --config ... --export-only` で既存 `.msh` から face/edge/bin のみ再生成。

---

## 10. 受入基準

- [ ] 球磨川・人吉で全ブロック生成完了
- [ ] AQWA ソルバーが `cell.bin` をエラーなく読込
- [ ] 設定した面積下限を満たす、または `breaklines`/`balanced` 時は許容率内
- [ ] 標高・斜面・レベル差・重心距離の違反が YAML 閾値以内
- [ ] 拘束カバレッジ ≥ 設定値
- [ ] `constraint_priority` 切替が再実行のみで反映される
- [ ] 主要パラメータが YAML + README に文書化
- [ ] pytest 等のテスト pass

---

## 11. 技術・納品

| 項目 | 内容 |
|---|---|
| 言語 | Python 3.11+ |
| メッシュエンジン | Gmsh（または同等、出力互換必須） |
| 実行 | CLI + YAML |
| 環境 | Nix flake 推奨、または requirements.txt |
| 納品 | ソース、README、設定例、テスト、Kuma 実行結果・summary |

---

## 12. 処理フロー

```
入力（領域・DEM・拘束線）＋ YAML 設定
        ↓
解析領域修復 → 外周四角形帯 → 拘束線前処理
        ↓
        constraint_priority（面積下限 vs 拘束線）
        ↓
Gmsh メッシュ生成
        ↓
┌──────────────────────────────────────────┐
│ 品質チェック（すべて YAML 閾値）           │
│  ・面積下限（任意・優先順位あり）          │
│  ・標高 RMSE                               │
│  ・斜面 方向/大きさ                        │
│  ・隣接 辺長比 → レベル差 ≤ 1             │
│  ・隣接 重心間距離 dl                     │
│  ・幾何品質                                │
└──────────────────────────────────────────┘
        ↓ 違反 → 細分化 → 再生成（反復）
face.gpkg / edge.gpkg / cell.bin 出力
        ↓
summary.json
```

---

## 13. 既知課題・未実装

| 項目 | 内容 |
|---|---|
| OSM 全道路 | 拘束曲線が多く Gmsh が極端に遅い |
| 5 m DEM | 盛り土検出が弱い → OSM 道路を次善策 |
| 地形違反残存 | 反復上限後も一定割合残り得る |
| 未実装 | `constraint_priority` 切替 |
| 未実装 | ノード勾配の斜面大きさ代表性 |
| 未実装 | 重心間距離の品質チェック・修復 |

---

## 14. 発注側で決めること

1. 外注形態（新規 / 引き継ぎ / 性能改善）
2. 面積下限の値（625 m² は例）
3. 拘束 vs 面積のデフォルト優先
4. 道路：幹線のみ vs 全 OSM
5. 標高・斜面・レベル差・重心距離の閾値
6. 地形違反・面積割れの許容率
7. 納期・ソルバー検証環境の有無

---

## 参考

- 現行ツール：`tool/README.md`
- 設定例：`Kuma_Hitoyoshi/yaml/mesh_config.yaml`
- ソルバー I/O：`02_Solver/compile/src/inun2dh/inun2dh_io.f90`
