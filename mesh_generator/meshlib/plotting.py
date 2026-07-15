"""確認用の可視化（PNG）出力。matplotlib を使用。"""

from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # GUI 不要のバックエンド
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
import numpy as np  # noqa: E402


def _triangle_patches(triangles_gdf: gpd.GeoDataFrame):
    patches = []
    for geom in triangles_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        xy = np.array(geom.exterior.coords)[:-1]
        patches.append(MplPolygon(xy, closed=True))
    return patches


def plot_mesh(
    triangles_gdf: gpd.GeoDataFrame,
    out_path: str,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    title: str = "Triangular Mesh",
) -> None:
    """メッシュ（三角形ワイヤフレーム）を描画する。"""
    fig, ax = plt.subplots(figsize=(10, 10))
    patches = _triangle_patches(triangles_gdf)
    pc = PatchCollection(patches, facecolor="none", edgecolor="#1f77b4", linewidths=0.3)
    ax.add_collection(pc)
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.boundary.plot(ax=ax, color="red", linewidth=1.0)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(f"{title} (n={len(triangles_gdf)})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    _save(fig, out_path)


def plot_field(
    triangles_gdf: gpd.GeoDataFrame,
    column: str,
    out_path: str,
    title: str,
    cmap: str = "viridis",
) -> None:
    """三角形を指定列の値で塗り分けて描画する。"""
    if column not in triangles_gdf or len(triangles_gdf) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    patches = _triangle_patches(triangles_gdf)
    values = triangles_gdf[column].to_numpy(dtype=float)
    pc = PatchCollection(patches, cmap=cmap, edgecolor="none")
    pc.set_array(values)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    cbar = fig.colorbar(pc, ax=ax, shrink=0.7)
    cbar.set_label(column)
    _save(fig, out_path)


def plot_quality_flags(
    triangles_gdf: gpd.GeoDataFrame,
    out_path: str,
    title: str = "Quality Flags",
) -> None:
    """品質フラグの有無で三角形を色分けする。"""
    if "quality_flag" not in triangles_gdf or len(triangles_gdf) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    patches = _triangle_patches(triangles_gdf)
    flags = triangles_gdf["quality_flag"].to_numpy()
    colors = ["#d62728" if f else "#2ca02c" for f in flags]
    pc = PatchCollection(patches, facecolor=colors, edgecolor="white", linewidths=0.1)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    n_flag = int(flags.sum())
    ax.set_title(f"{title} (flagged={n_flag}/{len(triangles_gdf)})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    _save(fig, out_path)


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
