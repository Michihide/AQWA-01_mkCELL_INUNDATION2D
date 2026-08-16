"""地形適合三角形メッシュ生成ライブラリ (meshlib)。

QGIS/GIS データ（解析領域ポリゴン・DEM・建物など）から、氾濫解析向けの
三角形メッシュ（node / cell / edge テーブル）を生成するモジュール群。
"""

__all__ = [
    "config",
    "crs",
    "io",
    "points",
    "buildings",
    "elevation",
    "triangulation",
    "quality",
    "adjacency",
    "plotting",
]

__version__ = "0.1.0"
