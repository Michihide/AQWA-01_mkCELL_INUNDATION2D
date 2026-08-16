# 01_mkMESH_INUN2DH — 氾濫2次元メッシュ生成（Gmsh 版）

AQWA 氾濫解析（INUN2DH）向けの **地形適合型ハイブリッド非構造格子** を生成するモジュールです。
2025-08 以降、メインの実装は **Gmsh Python API** ベースの `tool/` です。

旧来の Python / ネイティブパイプライン（`01_mkINPUT_Face_Edge.py` 等）は `legacy/` に退避してあり、
Git ブランチ `legacy/native-pre-gmsh` からいつでも復元できます。

---

## ディレクトリ構成

```text
01_mkMESH_INUN2DH/
├── tool/              Gmsh メッシュ生成（メイン）
├── docs/              要件・設計メモ
├── mesh_bundle/       参考用バンドル（巨大。Git 未追跡）
├── Cell/              事例データ・出力（Git 未追跡）
│   ├── Hii/
│   ├── Kinu/
│   └── Kuma_Hitoyoshi/
├── Landuse_100m.tif   全国土地利用ラスタ（ローカル配置）
├── japan_soil.tif
├── OpenStreetMap/
└── legacy/            旧ネイティブメッシュパイプライン
```

AQWA 他モジュールは次のパスを参照します（**`Cell/` 配下は変更なし**）。

```text
01_mkMESH_INUN2DH/Cell/${river_name}/output_gpkg_csv/face.gpkg
01_mkMESH_INUN2DH/Cell/${river_name}/output_gpkg_csv/edge.gpkg
```

---

## 使い方（Gmsh）

作業ディレクトリは **`tool/`**。

```bash
cd tool
nix develop    # 推奨。Gmsh 等を用意

# 開発用設定（tool/config/）
python run.py --config config/hii.yaml
python run.py --config config/kuma_hitoyoshi.yaml

# 事例 yaml（Cell/ 配下）
python run.py --config ../Cell/Hii/yaml/mesh_config.yaml
python run.py --config ../Cell/Kinu/yaml/mesh_config.yaml

# 球磨川：氾濫ブロック並列
python run.py --config config/kuma_hitoyoshi.yaml --blocks --workers 8
```

詳細は [`tool/README.md`](tool/README.md) を参照。

---

## 旧パイプライン（legacy/）

```bash
cd legacy
./run_all.sh    # 旧手順（01_mkINPUT_Face_Edge.py → 02_mkCELL.py）
```

説明は [`legacy/README.md`](legacy/README.md)（旧版 README）を参照。

---

## 旧版へのロールバック

統合前の状態は Git で固定済みです。

| 方法 | コマンド |
|------|----------|
| ブランチ | `git checkout legacy/native-pre-gmsh` |
| タグ | `git checkout legacy-native-20250817` |
| 現在の Gmsh 版に戻す | `git checkout main` |

親リポジトリ（AQWA）では submodule のコミット SHA も合わせて記録してください。

```bash
cd /path/to/AQWA
git add 01_mkMESH_INUN2DH
git commit -m "Update 01_mkMESH_INUN2DH submodule (gmsh integration)"
```

---

## 統合の経緯

- 旧：`Desktop/AQWA/01_mkMESH_INUN2DH`（ネイティブ / mesh_generator）
- 新：`Desktop/gmsh` の `tool/` + 事例を本ディレクトリに統合（2025-08-17）
- 事例出力は AQWA 互換のため `Cell/` 配下に配置

`Desktop/gmsh/` は統合後もローカルに残してあります。削除してよい場合は、
本ディレクトリと `Cell/` の内容を確認してから行ってください。
