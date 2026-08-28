# 01_mkMESH_INUN2DH — 氾濫2次元メッシュ生成（Gmsh 版）

AQWA 氾濫解析（INUN2DH）向けの **地形適合型ハイブリッド非構造格子** を生成するモジュールです。
メインの実装は **Gmsh Python API** ベースの `tool/` です。

---

## ディレクトリ構成

```text
01_mkMESH_INUN2DH/
├── tool/              Gmsh メッシュ生成（メイン）
├── docs/              要件・設計メモ
├── mesh_bundle/       参考用バンドル（巨大。Git 未追跡）
├── Mesh/              事例データ・出力（Git 未追跡）
│   ├── Hii/
│   ├── Kinu/
│   └── Kuma_Hitoyoshi/
├── Landuse_100m.tif   全国土地利用ラスタ（ローカル配置）
├── japan_soil.tif
└── OpenStreetMap/
```

AQWA 他モジュールは次のパスを参照します（**`Mesh/` 配下**）。

```text
01_mkMESH_INUN2DH/Mesh/${river_name}/output_gpkg_csv/face.gpkg
01_mkMESH_INUN2DH/Mesh/${river_name}/output_gpkg_csv/edge.gpkg
```

---

## 使い方

作業ディレクトリは **`tool/`**。

```bash
cd tool
nix develop    # 推奨。Gmsh 等を用意

# 開発用設定（tool/config/）
python run.py --config config/hii.yaml
python run.py --config config/kuma_hitoyoshi.yaml

# 事例 yaml（Mesh/ 配下）
python run.py --config ../Mesh/Hii/yaml/mesh_config.yaml
python run.py --config ../Mesh/Kinu/yaml/mesh_config.yaml

# 球磨川：氾濫ブロック並列
python run.py --config config/kuma_hitoyoshi.yaml --blocks --workers 8
```

詳細は [`tool/README.md`](tool/README.md) を参照。

旧ネイティブパイプラインは Git 履歴（ブランチ `legacy/native-pre-gmsh`）に残してある。
