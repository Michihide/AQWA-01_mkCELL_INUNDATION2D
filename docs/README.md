# 特殊辺ラインからメッシュを作る

氾濫格子は、**計算範囲と特殊辺ラインを先に定めてから**切る。
特殊辺はソルバーが読む辺（道路堰・カルバート・流量境界など）であり、
同時に Gmsh の拘束条件になる。生成後に辺を選んで kind を塗る手順は使わない。

バイナリのバイト列は [mesh_bin_spec.md](mesh_bin_spec.md)。
ツール全体の実行方法は [tool/README.md](../tool/README.md)。

## 用意するもの

| 入力 | 形 | 必須 |
| --- | --- | --- |
| 計算範囲 | ポリゴン（GeoPackage 等） | 必須 |
| DEM | GeoTIFF | 必須 |
| 特殊辺 | ライン | ソルバー特殊辺があるとき |
| ブレークライン | ライン | 任意。幾何だけ揃えたい線 |

特殊辺とブレークラインはどちらもメッシュ辺に一致させる。違いは kind の有無。

| | `input.special_edges` | `input.breaklines` |
| --- | --- | --- |
| 役割 | ソルバーが読む辺 | 幾何だけの拘束 |
| 例 | 道路堰、カルバート、流入、水位、壁 | 堤防天端、盛土、河道法線 |
| 出力 | `special_edges.csv`（mesh.bin の strc） | 辺の形だけ。kind は付かない |

道路の**形**だけ揃えたいときは `breaklines.roads`。
道路堰・カルバートなどは 1 本の `special_edges.file` にまとめ、`kind` 列で区別する。

## 手順

```
1. 計算範囲ポリゴンを決める
2. 特殊辺ラインを決める（折点で区間に分かれ、区間ごとに kind と属性を載せる）
3. 必要なら幾何だけのブレークラインを足す
4. YAML に domain / special_edges / breaklines を書く
5. mkMESH でメッシュを生成する
6. 両端がライン上にある辺が special_edges.csv に書かれる
```

ラインが拘束なので、格子の辺は特殊辺の折れ線に沿う。
折点（頂点）で線は区間に分かれ、各区間が 1 本の特殊辺になる。区間ごとに
`kind` / `zc` / `H` などを変えられる。後から辺 ID を拾って CSV を手書きする必要はない。

## YAML

パスは YAML ファイルの位置からの相対。未知のキーはエラーになる。

```yaml
input:
  domain: ../input/area.gpkg
  dem: ../input/dem.tif
  special_edges:
    file: ../input/special_edges.gpkg   # ROAD / CULVERT / Q などを 1 ファイルに
    default_kind: ROAD                  # kind 列が無い線に使う
    z_mode: absolute                    # zc / z_road が絶対標高か相対高さ
    match_tol: 2.0                      # 辺対応付けの許容 [m]。0 なら自動
    # field_kind: kind
    # field_zc: zc
    # field_H: H
    # field_z_road: z_road
    # field_B: B
    # field_cover: cover
    # field_z_mode: z_mode
    # field_fr: fr
    default_fr: 0.0                 # ラインに fr が無い FROUDE 用。外周全体は output.default_fr
  breaklines:
    embankments: ../input/embankments.gpkg

crs:
  target_epsg: 6670          # メートル単位の投影座標系。4326 / 3857 は不可

output:
  default_fr: 0.35           # 特殊辺でない外周辺。0 なら塗らない
  cell_bin:
    enabled: true
    filename: mesh.bin
```

特殊辺は **近接や短尺を理由に拘束から落とさない**。交点での交差はよい。
交差せずに下限（`2 * min_element_area / global_min_size`、Kinu 既定は 25 m）より
近い、または線分が重なるときは、距離の傾向（交差 / 重なり / 隙間の階級）を
出して止める。幾何だけのブレークラインは、面積下限のために近接区間を見送ることがある。

`match_tol` が 0 のときは `max(1 m, global_min_size × 0.05)`。
拘束が効いていれば端点は線上にあるので、数メートルあれば足りる。

## kind

| kind | 意味 | 使う属性 | ソルバー |
| --- | --- | --- | --- |
| `ROAD` | 道路堰 | `zc`, `z_road` | `e_strc`, `e_zc`, `e_zc_road`。幅は辺長 |
| `CULVERT` | カルバート | `zc`, `H`, `B`, `cover` | 管＋天端（`zc+H+cover`）越流 |
| `Q` | 流量境界 | `qgroup` | `e_qgroup`。時系列はソルバー YAML |
| `W` | 水位境界 | `qgroup`（任意） | 辺を水位境界として印す。水位時系列は YAML |
| `WALL` | 壁 | なし | 辺を印すだけ。外周を自動では付けない |
| `FROUDE` | フルード流出 | `fr` | 辺の `fr` に書く。qin 未指定ならソルバーが使う |
| `NONE` | 無視 | — | 書かない |

大文字小文字は問わない。`道路` / `堰` / `カルバート` / `暗渠` などの別名と、
近い綴り（`culvrt` → `CULVERT`）も通す。`NONE` と空欄はスキップする。

同じ辺に複数ラインが重なったときは、
`WALL` > `Q` / `W` > `FROUDE` > `CULVERT` > `ROAD` の順で残す。

## ラインの属性

ジオメトリは LineString（MultiLineString は分解する）。
3 点以上の折れ線は折点で 2 点の線分に分ける。属性を区間で変えたいときは、
QGIS で折点を打つか、あらかじめ線を切って各行に値を書く。
CRS は `crs.target_epsg` と同じメートル投影。計算範囲の外は切り捨てる。

| 列（既定） | 別名 | 型 | 無いとき |
| --- | --- | --- | --- |
| `kind` | `type`, `special_kind` | 上表の名前 | `other` では必須。他キーではファイル名で決まる |
| `zc` | `Zc`, `z_c`, `e_zc` | m | 絶対なら標高。相対なら地盤からの高さ。無ければ辺の `z_crest` |
| `H` | `Hstr`, `e_H` | m | 0 |
| `z_road` | `zroad`, `e_zc_road` | m | `zc` と同じ規則 |
| `B` | `Bstr`, `e_Bstr`, `width` | m | 0 ならメッシュ辺長。カルバート開口幅 |
| `cover` | `T`, `thickness`, `土被り` | m | 常に厚さ m。カルバートでは天端を `zc+H+cover` にする。無ければ `z_road` |
| `z_mode` | `z_ref`, `datum` | `absolute` / `relative` | YAML の `z_mode`。行ごと上書き可 |
| `qgroup` | `q_group`, `e_qgroup` | 整数 | **Q / W のときだけ**。ROAD / CULVERT には不要 |
| `fr` | `Fr`, `froude` | 0–1 | `FROUDE` の指定フルード数。無ければ `output.default_fr` |

`z_mode: absolute`（絶対標高）が既定。`relative` なら `zc` / `z_road` を辺中点 DEM に足す。
`絶対` / `相対` でもよい。

列名は `field_kind` などでも変えられる。

QGIS では計算範囲と同じ投影でラインを引き、上の列を付ける。
ポリゴンや点は読まない。

## メッシュ側で起きること

1. 計算範囲を読む。
2. 特殊辺ラインを範囲で切り、折点で区間に分ける。近接の傾向を出し、下限未満なら止める。
3. ブレークラインと同じ拘束集合に入れ、交点分割と頂点間引きを行う。
4. 外周帯との干渉回避（`breaklines.py`）を通す。特殊辺は近接で落とさない。
5. その拘束の上で三角形メッシュを切る。
6. 各メッシュ辺について、中点と両端がラインから `match_tol` 以内なら採用する。
7. `special_edges.csv` と mesh.bin の **strc** セクションに書く。

```
edge_id,kind,zc,H,z_road,qgroup,B,fr
12,ROAD,12.40,0.0,12.40,0,0.0,0.0
48,CULVERT,10.00,1.5,12.00,0,3.0,0.0
80,FROUDE,0.0,0.0,0.0,0,0.0,0.35
```

`edge_id` は 1 始まり。`face.gpkg` / `edge.gpkg` の辺番号と同じ規則。
河道連結（`couple.csv`）は別ファイルで、ここでは書かない。

未指定なら CSV はヘッダのみ、strc は空。旧 cell.bin からの変換も空のまま。

## 実行

```bash
cd 01_mkMESH_INUN2DH/tool
nix develop
python run.py --config path/to/mesh_config.yaml
```

既存 `.msh` から mesh.bin だけ作り直すときは `--export-only`。
特殊辺ラインは設定から再読込し、そのメッシュの辺へ対応付ける。
ラインを拘束せずに切ったメッシュへ後から載せる用途ではない
（辺がラインから外れると対応付かない）。

確認:

```bash
cd tool
nix develop -c python -m pytest tests/test_special_edges.py
```

実装は `tool/src/special_edges.py`。設定は `input.special_edges`（`tool/src/config.py`）。
