# mesh.bin 仕様（版 1）

mkMESH が作業ファイルを書き、`pack_mesh` が `mesh.bin` に固める。
ソルバー（`feature/mesh-bin-reader`）は **`mesh.bin` だけ** を読む。旧 `cell.bin` は読まない。

この文書は次の 2 箇所に同文で置く。

- `01_mkMESH_INUN2DH/docs/mesh_bin_spec.md`
- `02_Solver/docs/solver/mesh_bin_spec.md`

`face.gpkg` / `edge.gpkg` は GIS 確認用。ソルバーは見ない。
特殊辺ラインの用意と YAML は [README.md](README.md)。

## データの流れ

作業ファイル（node / edge / face / landuse / soil / building / veg / special_edges / couple）
→ `pack_mesh` → `mesh.bin` → inun2dh / Liu extras

## 識別子と向き

- 点・辺・面の id は **1 始まりの密な通し番号**（欠番なし）。`nnode` / `nedge` / `nface` と一致する。
- 辺 id は edge.gpkg の LineID / 旧 `cell.bin` の `en` と同じ規則（共有辺は 1 本）。
- 面 id は三角形 → 四角形の順（旧 cell.bin の `cn` と同じ）。
- 面の辺列は **反時計回り（CCW）、閉じない**。
- `edge.bin` の `(v1, v2)` は **左面から見て CCW**。
  - `face_left`: その向きで辺を CCW に辿る面
  - `face_right`: 反対面。外辺は `0`
- 旧 LIE の `bc_flag`（隣接なし）は幾何の `face_right = 0` で表す。構造物・流量・水位・壁の区別は `special_edges.csv` の `kind` に書く。変換スクリプトは外周辺を自動で `WALL` にしない。

## 作業ファイル（mkMESH が書く）

エンディアンはリトルエンディアン。整数は 2 の補数。実数は IEEE-754 binary64。
幾何・属性の `.bin` 作業ファイルの中身は、下の対応する `mesh.bin` セクションペイロードと **同じバイト列**。

### 幾何（必須）

**`node.bin`** — `nnode` レコード、各 28 B

| フィールド | 型 | 意味 |
|---|---|---|
| id | i32 | 点 id |
| x, y, z | f64 | 座標 [m]。`z` は点標高 |

**`edge.bin`** — `nedge` レコード、各 36 B（旧ファイルは 28 B。読込はセクション長で判別）

| フィールド | 型 | 意味 |
|---|---|---|
| id | i32 | 辺 id |
| v1, v2 | i32 | 端点 id（左面 CCW） |
| face_left, face_right | i32 | 左右の面 id。外辺の right は 0 |
| z_crest | f64 | 天端標高 [m]（辺中点の DEM。欠損時は端点 `z` の平均） |
| fr | f64 | 未指定外辺の流出フルード数。0 なら付けない。内部辺は 0 |

**`face.bin`** — 可変長。各面:

| フィールド | 型 | 意味 |
|---|---|---|
| id | i32 | 面 id |
| dummy | i32 | 1 = ダミー面（計算しない）、0 = 計算面 |
| x, y | f64 | 重心 [m] |
| A | f64 | 面積 [m²] |
| z_bed | f64 | 地盤高 [m] |
| n_sides | i32 | 辺数（3 または 4） |
| edge_id[n_sides] | i32 | CCW の辺 id（閉じない） |

### 属性（欠けていた分。別ファイル）

ゼロの行は書かない（疎配列）。空ならレコード数 0。

**`landuse.bin`** — `nentry` (u32) のあと各 16 B: `face_id` (i32), `code` (i32), `area` (f64)。
`code` は国土数値情報の画素値（10, 20, 50, 60, 70, 91, 92, 100, 110, 140, 150, 160, 255）。

**`soil.bin`** — 同じ形。`code` は Green-Ampt 用 1–17。

**`building.bin`** — `nentry` (u32) のあと各 20 B: `face_id` (i32), `chi_bld` (f64), `bld_peri` (f64)。
`chi_bld` は旧 `bld_ratio`。建物連続式の侵入口高さ `height_entry_bld` は YAML のまま。`bld_peri` はここから読む。

**`veg.bin`** — `nentry` (u32) のあと各 36 B: `face_id` (i32), `chi_veg` (f64), `a` (f64), `H_v` (f64), `C_D` (f64)。
`chi_veg` は建物と同じくポリゴン交差の被覆率。`a` / `H_v` / `C_D` は交差した植生ポリゴン属性の面積重み平均。
mkMESH は `output.cell_bin.vegetation` でポリゴンを指定する。列名の既定は `Cd`, `a`, `Hv`（`C_D` / `H_v` などの別名も可）。未指定なら `nentry = 0`。

**`special_edges.csv`**（UTF-8、先頭行がヘッダ）

```
edge_id,kind,zc,H,z_road,qgroup,B,fr
```

`kind` は `NONE` / `ROAD` / `CULVERT` / `Q` / `W` / `WALL` / `FROUDE`（大文字小文字は無視）。
`fr` は `FROUDE` 辺、または `output.default_fr` で塗る外周辺のフルード数。空ファイルはヘッダのみでよい。

mkMESH では **計算範囲と特殊辺ラインを先に定め**、ラインを Gmsh の拘束条件にしてから
メッシュを切る。生成後、両端がライン上にある辺を `special_edges.csv` に書く。
YAML は `input.special_edges.file`（1 ファイル。`kind` 列で ROAD / CULVERT 等を区別。無ければ `default_kind`）。
特殊辺は近接を理由に拘束から落とさない。交差せずに下限より近い／重なる入力は
距離の傾向を出して止める。
属性列の既定は `kind`, `zc`, `H`, `z_road`, `qgroup`, `B`, `cover`, `fr`。`zc` / `z_road` が無ければ辺の `z_crest`。`B` が 0 なら辺長。`fr` は `FROUDE` または `output.default_fr`。
カルバートに `cover` があれば天端は `zc + H + cover`（`cover` は常に厚さ m）。
幾何だけの拘束は従来どおり `input.breaklines`。旧 cell.bin からの変換は空のまま。

**`couple.csv`**

```
edge_id,river_link,kp,bank
```

`bank` は `1` / `L` / `left`（左岸）または `2` / `R` / `right`（右岸）。LFP / r2ri 用。空ならヘッダのみ。

## mesh.bin に入れないもの

マニング \(n\)、Green-Ampt パラメータ、\(Q(t),w(t)\)、降雨、初期水位。
辺長・法線・`A_CV`・逆距離重みは **読込時** に点座標から計算する（ファイルには載せない）。

## `mesh.bin` コンテナ

リトルエンディアン stream。先頭 320 B が固定ヘッダ。

### 固定ヘッダ（32 B）

| オフセット | 型 | 内容 |
|---|---|---|
| 0 | 8 B | 魔法数 `AQWAMESH` |
| 8 | u16 | 版。この文書は **1** |
| 10 | u16 | フラグ（予約、0） |
| 12 | i32 | EPSG。不明なら 0 |
| 16 | u32 | nnode |
| 20 | u32 | nedge |
| 24 | u32 | nface |
| 28 | u32 | 予約（0） |

### 入力ハッシュ（144 B）

作業ファイル 9 個 × 16 B。順序:

0. `node.bin`
1. `edge.bin`
2. `face.bin`
3. `landuse.bin`
4. `soil.bin`
5. `building.bin`
6. `veg.bin`
7. `special_edges.csv`
8. `couple.csv`

各値はファイルバイト列の SHA-256 の先頭 16 B。欠ける／空の入力は 16 B の 0。
ソルバーは版 1 では検証しなくてよい（読み飛ばしてよい）。

### セクションオフセット表（144 B）

9 セクション × (`offset` u64, `length` u64)。オフセットはファイル先頭から。
長さ 0 は空セクション。順序はハッシュと同じ（node … couple）。
`special_edges.csv` はセクション名 **strc**、`couple.csv` は **couple**。

### セクションペイロード

- **node / edge / face**: 上の作業ファイルと同じレコード列。
- **landuse / soil / building / veg**: 上と同じ（先頭に `nentry` u32）。
- **strc**: `nentry` (u32) のあと各 44 B（旧 36 B は `B` なし。読込時は長さで判別）

  | フィールド | 型 |
  |---|---|
  | edge_id | i32 |
  | kind | i32 |
  | zc | f64 |
  | H | f64 |
  | z_road | f64 |
  | qgroup | i32 |
  | B | f64 |

  `kind`: 0=NONE, 1=ROAD, 2=CULVERT, 4=Q, 5=W, 6=WALL, 7=FROUDE（3 は欠番）

- **couple**: `nentry` (u32) のあと各 20 B: `edge_id` (i32), `river_link` (i32), `kp` (f64), `bank` (i32)

各エンティティは 1 回だけ。空セクションは `length = 0`、または疎配列なら `nentry = 0`。

### 氾濫ブロックトレーラー（任意）

版は **1 のまま**。ヘッダのセクション数は 9。旧ファイル（トレーラー無し）は全面 `block_id = 1`。
新規の `pack_mesh` は常にトレーラーを付ける。

最終セクションの末尾の直後:

| フィールド | 型 | 内容 |
|---|---|---|
| magic | 8 B | `AQWABLK1` |
| nface | u32 | ヘッダの `nface` と一致 |
| block_id[nface] | i32 | 面ごとの氾濫ブロック番号（**1 始まり**） |

`--blocks` ではディレクトリ `block_NNN` の `NNN+1`。単一ポリゴンなら 1。
GIS の `face.csv` / `face.gpkg` にも同じ `block` 列を書く。

## ソルバーが計算する派生量

点座標と左右面から、旧 cell.bin と同じ定義で次を求める。

- 辺長、中点、端点座標
- `fn_self` = `face_left`、`fn_adj` = `face_right`
- 外辺は `bc_flag = 1`
- 辺の `fr` > 0 かつ qin 未指定の外辺 → フルード流出（`INUN_BC_FROUDE`）
- `A_CV`（両側面積の和。外辺は自面のみ）
- 重心間距離 `dl` と単位方向 `dx/dl`, `dy/dl`
- 辺中点への逆距離重み（自面・隣面・面側）

土地利用の疎配列は固定 13 スロット（画素値 10…255）へ展開する。土壌は 1–17。
`gamma_nu = 1 - chi_bld`、`A_flat = gamma_nu * A`。

Liu extras（論文 Ex1–6 の内蔵メッシュは対象外）:

- strc の ROAD / CULVERT → `e_strc`, `e_zc`, `e_Hstr`, `e_zc_road`, `e_Bstr`（`B`>0 ならその値、なければ辺長）
- Q → `e_qgroup`
- W → `e_w_spec` は時系列側。ここでは辺を水位境界として印すだけ
- couple → `e_is_interface`
- veg → `chi_veg`, `a_veg`, `H_veg`, `Cd_veg`
- ファイルに strc / couple / veg があるときだけ extras 配列を確保する。内蔵メッシュのファイル名タグは、`mesh.bin` から塗った場合は上書きしない

## 旧 cell.bin からの変換

`cell_bin_to_mesh.py` は重複辺を `en` で 1 本に畳む。点座標は node セクションだけに置く。
外周辺の `bc_flag` は `face_right = 0` にする。`special_edges.csv` は空（ヘッダのみ）。
土地利用・土壌のゼロ面積は落とす。`dummy` は面レコードの CalMesh 列。
