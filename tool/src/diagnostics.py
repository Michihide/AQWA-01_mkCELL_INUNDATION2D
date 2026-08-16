"""診断図の出力。

参照レイヤ（OSM 等）はここでのみ使う。格子の分割には一切関与しない。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from .adaptive_refinement import RefinementStep  # noqa: E402
from .boundary_quad_band import PolygonBand  # noqa: E402
from .config import Config  # noqa: E402
from .font_utils import setup_japanese_font  # noqa: E402
from .io_vector import ReferenceLayers  # noqa: E402
from .mesh_parser import Mesh  # noqa: E402
from .quality_metrics import QualityReport, quality_violations  # noqa: E402
from .terrain_metrics import TerrainReport, terrain_violations  # noqa: E402
from .utils import get_logger  # noqa: E402

DPI = 150


def _new_axes(title: str):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    return fig, ax


def _polygons(mesh: Mesh) -> list[np.ndarray]:
    return mesh.element_polygons()


def _draw_mesh(ax, mesh: Mesh, values=None, cmap="viridis", vmin=None, vmax=None,
               edge_lw: float = 0.1, label: str = ""):
    coll = PolyCollection(
        _polygons(mesh), array=values, cmap=cmap,
        edgecolors="black", linewidths=edge_lw,
    )
    if values is not None:
        coll.set_clim(vmin, vmax)
    else:
        coll.set_facecolor("none")
    ax.add_collection(coll)
    ax.autoscale_view()
    if values is not None:
        cb = ax.figure.colorbar(coll, ax=ax, shrink=0.8)
        if label:
            cb.set_label(label)
    return coll


def _overlay_reference(ax, references: ReferenceLayers) -> None:
    """参照レイヤを薄く重ねる。位置の目安を目視確認するためだけの表示。"""
    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:pink"]
    for i, (name, gdf) in enumerate(references.layers.items()):
        if gdf.empty:
            continue
        gdf.plot(ax=ax, color=colors[i % len(colors)], linewidth=0.6, alpha=0.6,
                 zorder=5, label=f"参照: {name}")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_diagnostics(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport | None,
    bands: list[PolygonBand],
    references: ReferenceLayers,
    steps: list[RefinementStep],
    out_dir: Path,
) -> list[Path]:
    logger = get_logger()
    setup_japanese_font()
    d = out_dir / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(name: str, fig) -> None:
        p = d / name
        _save(fig, p)
        written.append(p)

    # 1. メッシュ全体
    fig, ax = _new_axes("メッシュ")
    _draw_mesh(ax, mesh, edge_lw=0.15)
    emit("01_mesh.png", fig)

    # 2. 要素種別
    fig, ax = _new_axes("要素種別（黄=四角形、紫=三角形）")
    _draw_mesh(ax, mesh, values=(quality.kind == 4).astype(float), cmap="viridis",
               vmin=0, vmax=1, label="四角形=1")
    emit("02_element_kind.png", fig)

    # 3. 要素面積
    fig, ax = _new_axes(f"要素面積 [m^2]（下限 {cfg.mesh.min_element_area:.0f}）")
    _draw_mesh(ax, mesh, values=quality.area, cmap="magma", label="面積 [m^2]")
    emit("03_area.png", fig)

    # 4. 面積下限違反（境界に接する要素は形状再現性を優先するため対象外）
    below = (quality.area < cfg.mesh.min_element_area) & ~mesh.boundary_touching_element_mask()
    fig, ax = _new_axes(f"面積下限を割る要素: {int(below.sum())} 個")
    _draw_mesh(ax, mesh, values=below.astype(float), cmap="Reds", vmin=0, vmax=1)
    emit("04_area_violation.png", fig)

    # 5. 最小内角
    fig, ax = _new_axes("最小内角 [deg]")
    _draw_mesh(ax, mesh, values=quality.min_angle_deg, cmap="plasma", label="最小内角 [deg]")
    emit("05_min_angle.png", fig)

    # 6. アスペクト比
    fig, ax = _new_axes("アスペクト比")
    _draw_mesh(ax, mesh, values=np.clip(np.nan_to_num(quality.aspect_ratio, posinf=10.0), 1, 6),
               cmap="cividis", label="アスペクト比")
    emit("06_aspect_ratio.png", fig)

    # 7. 四角形の scaled Jacobian
    if (quality.kind == 4).any():
        fig, ax = _new_axes("四角形の scaled Jacobian（負なら凹・反転）")
        _draw_mesh(ax, mesh, values=np.nan_to_num(quality.scaled_jacobian, nan=1.0),
                   cmap="coolwarm", vmin=-1, vmax=1, label="scaled Jacobian")
        emit("07_scaled_jacobian.png", fig)

    # 8. 品質違反の合成
    viol = quality_violations(quality, cfg, mesh)
    any_viol = np.logical_or.reduce(list(viol.values()))
    fig, ax = _new_axes(f"品質・面積の違反要素: {int(any_viol.sum())} 個")
    _draw_mesh(ax, mesh, values=any_viol.astype(float), cmap="Reds", vmin=0, vmax=1)
    emit("08_quality_violation.png", fig)

    # 9-12. 地形指標
    if terrain is not None:
        fig, ax = _new_axes("代表標高 [m]")
        _draw_mesh(ax, mesh, values=terrain.elevation, cmap="terrain", label="標高 [m]")
        emit("09_elevation.png", fig)

        fig, ax = _new_axes(f"平面フィット RMSE [m]（基準 {cfg.terrain.plane_fit_rmse_max}）")
        _draw_mesh(ax, mesh, values=terrain.plane_fit_rmse, cmap="inferno",
                   vmax=float(np.percentile(terrain.plane_fit_rmse, 99)), label="RMSE [m]")
        emit("10_plane_fit_rmse.png", fig)

        fig, ax = _new_axes(
            f"勾配方向のばらつき [deg]（基準 {cfg.terrain.slope_direction_spread_max_deg}）"
        )
        _draw_mesh(ax, mesh, values=terrain.slope_direction_spread_deg, cmap="twilight",
                   label="ばらつき [deg]")
        emit("11_slope_spread.png", fig)

        t_viol = terrain_violations(terrain, cfg)
        any_t = np.logical_or.reduce(list(t_viol.values()))
        fig, ax = _new_axes(f"地形基準の違反要素: {int(any_t.sum())} 個")
        _draw_mesh(ax, mesh, values=any_t.astype(float), cmap="Oranges", vmin=0, vmax=1)
        emit("12_terrain_violation.png", fig)

    # 13. 外周四角形帯と参照レイヤ
    fig, ax = _new_axes("外周四角形帯と参照レイヤ（参照は目安表示のみ）")
    for band in bands:
        for ring in band.rings():
            closed = np.vstack([ring.outer, ring.outer[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="tab:blue", lw=0.8)
            iface = ring.interior_ring()
            closed = np.vstack([iface, iface[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="tab:green", lw=0.6)
    if cfg.features.reference_layers_in_diagnostics:
        _overlay_reference(ax, references)
        if references.layers:
            ax.legend(loc="upper right", fontsize=8)
    ax.autoscale_view()
    emit("13_band_and_reference.png", fig)

    # 14. 反復履歴
    if len(steps) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        it = [s.iteration for s in steps]
        axes[0].plot(it, [s.n_elements for s in steps], "o-")
        axes[0].set_ylabel("要素数")
        axes[0].grid(alpha=0.3)
        axes[1].plot(it, [s.violations.get("terrain_viol_plane_fit_rmse", 0) for s in steps],
                     "o-", label="平面フィット RMSE")
        axes[1].plot(it, [s.violations.get("terrain_viol_slope_direction_spread", 0) for s in steps],
                     "s-", label="勾配方向ばらつき")
        axes[1].plot(it, [s.violations.get("viol_area_below_min", 0) for s in steps],
                     "^-", label="面積下限")
        axes[1].set_xlabel("反復")
        axes[1].set_ylabel("違反要素数")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        emit("14_iteration_history.png", fig)

    logger.info("診断図を %d 枚出力: %s", len(written), d)
    return written
