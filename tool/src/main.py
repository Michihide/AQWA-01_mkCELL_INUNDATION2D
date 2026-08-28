"""パイプライン本体。

  領域読込 -> 特殊辺・拘束線読込 -> 修復 -> 外周四角形帯 -> DEM 読込 -> 初期サイズ場
    -> [メッシュ生成 -> 修復 -> 品質・地形評価 -> サイズ場更新] を反復
    -> 出力

反復は、全基準を満たすか max_iterations に達すると終了する。面積下限 625 m^2
に達したために解消できない違反は「解消不能」として記録し、反復を止める理由に
しない（無限に細分化しないため）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from .adaptive_refinement import (
    RefinementStep,
    has_converged,
    log_step,
    refine_size_field,
    summarize_unresolvable,
)
from .boundary_quad_band import (
    PolygonBand,
    band_report,
    build_polygon_band,
    triangle_only_polygon_band,
)
from .breaklines import (
    PreparedBreaklines,
    breakline_coverage,
    breaklines_to_geodataframe,
    prepare_breaklines,
)
from .cell_bin_export import compute_cell_attributes
from .mesh_bin_export import build_and_write_mesh_bin
from .config import Config, load_config
from .diagnostics import write_diagnostics
from .exporters import (
    write_connectivity_csv,
    write_element_csv,
    write_element_gpkg,
    write_iteration_log,
    write_mesh_files,
    write_node_csv,
    write_summary_json,
)
from .geometry_cleaning import clean_polygons
from .io_raster import DemGrid, load_dem
from .io_vector import (
    ReferenceLayers,
    VectorInputError,
    load_constraint_breaklines,
    load_domain,
    load_reference_layers,
    merge_constraint_breaklines,
)
from .special_edges import (
    assert_special_edge_clearance,
    load_special_edge_features,
    special_features_to_breaklines,
)
from .mesh_generator import generate_mesh, gmsh_session
from .mesh_parser import Mesh, load_mesh_from_msh
from .mesh_repair import repair_mesh
from .quality_metrics import (
    QualityReport,
    evaluate_quality,
    neighbor_ratio_stats,
    summarize,
)
from .reference_size_field import apply_reference_layers_to_size_field
from .size_field import SizeField
from .solver_gpkg_export import write_solver_gpkg_csv
from .terrain_metrics import TerrainReport, compute_terrain_metrics, summarize_terrain
from .utils import get_logger, setup_logging, stage


def build_bands(cfg: Config, polys: list[Polygon]) -> list[PolygonBand]:
    """全ポリゴンの外周帯を作る。四角形帯が無効なら全域を三角形にする。"""
    logger = get_logger()
    b = cfg.mesh.boundary_quad_band
    if not b.enabled:
        spacing = b.target_size if b.target_size > 0.0 else cfg.mesh.global_min_size
        logger.info(
            "外周四角形帯は無効。境界を %.1f m 間隔でサンプルし、全域を三角形で埋めます",
            spacing,
        )
        bands = [triangle_only_polygon_band(poly, spacing) for poly in polys]
        if not bands:
            raise RuntimeError("解析領域ポリゴンがありません")
        return bands

    bands: list[PolygonBand] = []
    failed: list[tuple[float, str]] = []

    for poly in polys:
        band = build_polygon_band(
            poly,
            width=b.width,
            spacing=b.target_size,
            min_width_ratio=b.min_width_ratio,
            max_passes=b.max_shrink_passes,
            min_element_area=cfg.mesh.min_element_area,
            max_corner_cut=b.resolved_max_corner_cut(),
            corner_angle_deg=b.corner_angle_deg,
        )
        if band.ok:
            bands.append(band)
        else:
            failed.append((poly.area, band.reason))

    if failed:
        logger.warning(
            "外周四角形帯を構築できなかったポリゴンが %d 個あります（合計 %.3f km^2）",
            len(failed), sum(a for a, _ in failed) / 1e6,
        )
        for area, reason in failed:
            logger.warning("  %.3f km^2: %s", area / 1e6, reason)
    if not bands:
        raise RuntimeError("外周四角形帯を構築できたポリゴンがありません")

    stats = band_report_all(bands, cfg.mesh.min_element_area)
    logger.info(
        "外周四角形帯: %d 個, 面積 %.0f - %.0f m^2 (中央値 %.0f), "
        "下限未満 %d（参考。境界要素は面積下限の対象外）",
        stats["count"], stats["area_min"], stats["area_max"],
        stats["area_median"], stats["below_min"],
    )
    if stats["skipped"]:
        logger.info(
            "外周のうち %d 区間（%.1f%%）は四角形を置けないため三角形に譲りました"
            "（境界形状は変えていません）",
            stats["skipped"],
            100.0 * stats["skipped"] / max(stats["skipped"] + stats["count"], 1),
        )
    if stats["corner_cuts"]:
        logger.warning(
            "境界の頂点を %d 個落としました（最大 %.1f m）。max_corner_cut を 0 に"
            "すれば境界形状を保てます", stats["corner_cuts"], stats["corner_cut_max_m"],
        )
    return bands


def band_report_all(bands: list[PolygonBand], min_area: float) -> dict[str, float]:
    reports = [band_report(b, min_area) for b in bands]
    live = [r for r in reports if r["count"]]
    return {
        "count": sum(r["count"] for r in reports),
        "skipped": sum(r["skipped"] for r in reports),
        "below_min": sum(r.get("below_min", 0) for r in reports),
        "area_min": min((r["area_min"] for r in live), default=float("nan")),
        "area_max": max((r["area_max"] for r in live), default=float("nan")),
        "area_median": float(np.median([r["area_median"] for r in live])) if live
        else float("nan"),
        "corner_cuts": sum(b.corner_cuts for b in bands),
        "corner_cut_max_m": max((b.max_corner_deviation for b in bands), default=0.0),
    }


def initial_size_field(
    cfg: Config,
    bands: list[PolygonBand],
    breaklines: PreparedBreaklines | None = None,
    references: ReferenceLayers | None = None,
) -> SizeField:
    """一様な最大サイズから始め、内側領域の境界と拘束線の付近のサイズを固定する。

    内側領域の境界（Γ1、および帯から外した区間では Γ0）の節点間隔は帯の接線方向
    サイズで決まっており動かせない。反復細分化でその近傍だけ過度に細かくすると
    細長い三角形ができるため、interface_size を下限として狭い範囲だけ掛ける。
    全域を interface_size に引き下げない（内部は global_max_size から開始する）。

    拘束線も同じ理由で床が要る。拘束線に 1 辺を載せた三角形は、線に沿う辺長を
    サイズ場から与えられるので、そこを細かくすると面積下限を割る。
    """
    b = cfg.mesh.boundary_quad_band
    bounds = _bands_bounds(bands)
    sf = SizeField.uniform(
        bounds,
        spacing=cfg.mesh.size_field_grid,
        value=cfg.mesh.global_max_size,
        floor=cfg.mesh.global_min_size,
        ceiling=cfg.mesh.global_max_size,
        pad=4.0 * cfg.mesh.size_field_grid,
    )
    if b.enabled:
        iface = b.resolved_interface_size()
        floor_radius = b.resolved_interface_floor_radius(cfg.mesh.size_field_grid)
        for band in bands:
            for ring in band.rings():
                # 帯から外した区間では Γ0 が内側領域の境界になる。そこも節点間隔が
                # 固定されている点は Γ1 と同じなので、同じ床を掛ける。
                pts = ring.interior_ring()
                if not len(pts):
                    continue
                sf.raise_floor(pts[:, 0], pts[:, 1], iface, radius=floor_radius)

    if breaklines is not None and not breaklines.is_empty():
        floor = cfg.mesh.breakline_processing.resolved_size_floor(
            cfg.mesh.min_element_area, cfg.mesh.global_min_size
        )
        for line in breaklines.lines:
            xy = _densify(line.coords, cfg.mesh.size_field_grid)
            sf.raise_floor(xy[:, 0], xy[:, 1], floor, radius=floor)

    if references is not None:
        apply_reference_layers_to_size_field(cfg, sf, bands, references)

    sf.attempts[:] = 0
    sf.smooth(cfg.mesh.max_neighbor_size_ratio)
    return sf


def _densify(coords: np.ndarray, step: float) -> np.ndarray:
    """折れ線上に step 間隔で点を打つ。サイズ場は格子なので頂点だけでは穴が開く。"""
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    out = [coords[:1]]
    for i, length in enumerate(seg):
        n = max(1, int(np.ceil(length / step)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(coords[i] + t * (coords[i + 1] - coords[i]))
    return np.vstack(out)


def _bands_bounds(bands: list[PolygonBand]) -> tuple[float, float, float, float]:
    bb = np.array([b.polygon.bounds for b in bands])
    return (bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max())


def run_pipeline(cfg: Config) -> dict:
    """設定に従ってメッシュを生成し、成果物を出力する。"""
    logger = get_logger()
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with stage("解析領域の読み込み"):
        polys, crs = load_domain(cfg)
        polys = clean_polygons(
            polys,
            min_area=cfg.input.domain_min_area,
            narrow_feature_radius=cfg.mesh.resolved_narrow_feature_radius(),
        )
        if not polys:
            raise VectorInputError(
                "解析領域ポリゴンが cleaning 後に空になりました。"
                " domain_min_area または narrow_feature_radius を確認してください。"
            )

    domain = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    with stage("線データの読み込み"):
        breaklines = load_constraint_breaklines(cfg, crs, domain)
        special_feats = load_special_edge_features(cfg, crs, domain)
        if special_feats:
            bp = cfg.mesh.breakline_processing
            spacing = bp.resolved_min_vertex_spacing(cfg.mesh.global_min_size)
            clearance = bp.resolved_min_clearance(cfg.mesh.min_element_area, spacing)
            assert_special_edge_clearance(special_feats, clearance)
            breaklines = merge_constraint_breaklines(
                breaklines,
                special_features_to_breaklines(special_feats, crs),
            )
        references = load_reference_layers(cfg, crs, domain)

    with stage("外周四角形帯の構築"):
        bands = build_bands(cfg, polys)

    with stage("拘束ブレークラインの前処理"):
        prepared = prepare_breaklines(breaklines, bands, cfg)

    with stage("DEM の読み込み"):
        dem = load_dem(cfg, crs, _bands_bounds(bands))

    size_field = initial_size_field(cfg, bands, prepared, references)
    logger.info("初期サイズ場: %s", size_field.stats())

    mesh, quality, terrain, steps = _iterate(
        cfg, bands, size_field, dem, out_dir, prepared, references
    )

    with stage("成果物の出力"):
        summary = _write_outputs(
            cfg, mesh, quality, terrain, crs, out_dir, steps, bands, references, prepared, dem,
        )
    return summary


def _iterate(
    cfg: Config,
    bands: list[PolygonBand],
    size_field: SizeField,
    dem: DemGrid,
    out_dir: Path,
    breaklines: PreparedBreaklines | None = None,
    references: ReferenceLayers | None = None,
) -> tuple[Mesh, QualityReport, TerrainReport, list[RefinementStep]]:
    logger = get_logger()
    steps: list[RefinementStep] = []
    mesh: Mesh | None = None
    quality: QualityReport | None = None
    terrain: TerrainReport | None = None

    for iteration in range(1, cfg.mesh.max_iterations + 1):
        with gmsh_session(cfg, f"iter{iteration}"):
            mesh, _, _ = generate_mesh(cfg, bands, size_field, breaklines)
            mesh, _ = repair_mesh(mesh, cfg)
            quality = evaluate_quality(mesh)

        terrain = compute_terrain_metrics(mesh, dem, cfg)
        converged, counts = has_converged(mesh, quality, terrain, cfg)
        unresolved = sum(
            v for k, v in counts.items() if k != "blocked_by_area_floor"
        ) - counts["blocked_by_area_floor"]
        counts.update(
            summarize_unresolvable(
                mesh, quality, terrain, size_field, cfg, mesh.element_centroids()
            )
        )

        is_last = iteration >= cfg.mesh.max_iterations
        n_refined, n_coarsened = 0, 0
        if not converged and not is_last:
            n_refined, n_coarsened, _ = refine_size_field(
                size_field, mesh, quality, terrain, cfg
            )
            if references is not None:
                apply_reference_layers_to_size_field(
                    cfg, size_field, bands, references, count_attempts=False,
                )

        counts["plane_fit_rmse_area_weighted"] = float(
            (terrain.plane_fit_rmse * quality.area).sum() / quality.area.sum()
        )
        counts["slope_spread_area_weighted"] = float(
            (terrain.slope_direction_spread_deg * quality.area).sum() / quality.area.sum()
        )

        step = RefinementStep(
            iteration=iteration,
            n_elements=mesh.n_elements,
            n_triangles=len(mesh.triangles),
            n_quads=len(mesh.quads),
            area_min=float(quality.area.min()),
            n_refined=n_refined,
            n_coarsened=n_coarsened,
            n_at_area_floor=int((quality.area <= cfg.mesh.min_element_area * 1.05).sum()),
            violations=counts,
        )
        steps.append(step)
        log_step(step)

        if cfg.output.save_iteration_meshes:
            write_mesh_files(
                mesh, quality, out_dir / "iterations", f"iter{iteration:02d}",
                write_msh=False, write_xdmf=False,
            )

        if converged:
            logger.info("全基準を満たしたため反復を終了します（反復 %d）", iteration)
            break
        if is_last:
            logger.warning(
                "max_iterations=%d に達しました。未解消の違反 %d 件が残っています",
                cfg.mesh.max_iterations, unresolved,
            )
            break
        if n_refined == 0 and n_coarsened == 0:
            logger.info(
                "細分化できる要素が無くなったため反復を終了します（反復 %d）。"
                " 残る違反は面積下限 625 m^2 が律速です", iteration,
            )
            break

    assert mesh is not None and quality is not None
    return mesh, quality, terrain, steps


def _write_outputs(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport | None,
    crs: CRS,
    out_dir: Path,
    steps: list[RefinementStep],
    bands: list[PolygonBand],
    references: ReferenceLayers,
    breaklines: PreparedBreaklines | None = None,
    dem: DemGrid | None = None,
) -> dict:
    base = cfg.output.basename

    if terrain is not None:
        mesh.node_z = _node_elevations(mesh, terrain)

    write_mesh_files(
        mesh, quality, out_dir, base,
        write_msh=cfg.output.write_msh,
        write_vtu=cfg.output.write_vtu,
        write_xdmf=cfg.output.write_xdmf,
    )
    elements_gpkg_path = write_element_gpkg(mesh, quality, crs, out_dir / f"{base}_elements.gpkg")
    write_element_csv(mesh, quality, out_dir / f"{base}_elements.csv")
    write_node_csv(mesh, out_dir / f"{base}_nodes.csv")
    write_connectivity_csv(mesh, out_dir / f"{base}_connectivity.csv")
    write_iteration_log([s.as_row() for s in steps], out_dir / "iteration_log.csv")

    face_gpkg_path = edge_gpkg_path = elements_gpkg_path
    cell_attrs = None
    if cfg.output.gpkg_csv.enabled or cfg.output.cell_bin.enabled:
        if terrain is None:
            get_logger().warning(
                "地形評価結果が無いため face/edge GeoPackage を出力できません"
            )
        else:
            with stage("face/edge GeoPackage の出力"):
                face_gpkg_path, edge_gpkg_path = write_solver_gpkg_csv(
                    cfg, mesh, quality, terrain, crs, dem,
                )
                if cfg.output.cell_bin.enabled:
                    cell_attrs = compute_cell_attributes(cfg, mesh, quality, crs)

    summary = summarize(quality, cfg, mesh)
    summary.update(
        neighbor_ratio_stats(mesh, quality, cfg.mesh.max_neighbor_element_ratio)
    )
    if terrain is not None:
        summary.update(summarize_terrain(terrain, cfg, quality.area))
    summary["n_iterations"] = len(steps)
    summary["crs"] = crs.to_string()
    summary["band_quads"] = band_report_all(bands, cfg.mesh.min_element_area)
    if cfg.output.gpkg_csv.enabled:
        summary["face_gpkg"] = str(face_gpkg_path.resolve())
        summary["edge_gpkg"] = str(edge_gpkg_path.resolve())

    if breaklines is not None and not breaklines.is_empty():
        summary.update(breaklines.stats())
        summary.update(breakline_coverage(mesh, breaklines))
        breaklines_to_geodataframe(breaklines, crs).to_file(
            out_dir / f"{base}_breaklines.gpkg", layer="constrained", driver="GPKG"
        )
        get_logger().info(
            "拘束ブレークラインの被覆率: %.4f（未被覆 %.1f m）",
            summary["breakline_coverage"], summary["breakline_uncovered_m"],
        )

    write_summary_json(summary, out_dir / "summary.json")

    if cfg.output.save_diagnostics:
        write_diagnostics(cfg, mesh, quality, terrain, bands, references, steps, out_dir)

    if cfg.output.cell_bin.enabled:
        if terrain is None or dem is None:
            get_logger().warning(
                "output.cell_bin.enabled=true ですが地形情報が無いため mesh.bin を出力できません"
            )
        else:
            with stage("mesh.bin の出力"):
                bin_path = build_and_write_mesh_bin(
                    cfg, mesh, quality, terrain, crs, dem, attrs=cell_attrs,
                )
                if bin_path is not None:
                    summary["mesh_bin"] = str(bin_path)

    return summary


def _node_elevations(mesh: Mesh, terrain: TerrainReport) -> np.ndarray:
    """要素代表標高を節点へ面積按分で配分する。可視化用。"""
    z_sum = np.zeros(mesh.n_nodes)
    w_sum = np.zeros(mesh.n_nodes)
    idx = 0
    for conn in (mesh.triangles, mesh.quads):
        for elem in conn:
            z = terrain.elevation[idx]
            if np.isfinite(z):
                z_sum[elem] += z
                w_sum[elem] += 1.0
            idx += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w_sum > 0, z_sum / w_sum, 0.0)


def export_solver_outputs(cfg: Config, mesh_path: Path | None = None) -> dict:
    """既存 .msh から face/edge GeoPackage・CSV（および mesh.bin）を書き出す。"""
    logger = get_logger()
    out_dir = cfg.output_dir
    base = cfg.output.basename
    msh = mesh_path or (out_dir / f"{base}.msh")
    if not msh.is_file():
        raise FileNotFoundError(f"メッシュファイルが見つかりません: {msh}")

    with stage(f"メッシュの読み込み ({msh.name})"):
        mesh = load_mesh_from_msh(msh)
        logger.info(
            "節点 %d, 三角形 %d, 四角形 %d",
            mesh.n_nodes, len(mesh.triangles), len(mesh.quads),
        )

    crs = CRS.from_epsg(cfg.crs.target_epsg)
    xs, ys = mesh.nodes[:, 0], mesh.nodes[:, 1]
    bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    with stage("DEM の読み込み"):
        dem = load_dem(cfg, crs, bounds)

    quality = evaluate_quality(mesh)
    terrain = compute_terrain_metrics(mesh, dem, cfg)

    if not cfg.output.gpkg_csv.enabled:
        raise RuntimeError("output.gpkg_csv.enabled が false です")

    with stage("face/edge GeoPackage の出力"):
        face_gpkg_path, edge_gpkg_path = write_solver_gpkg_csv(
            cfg, mesh, quality, terrain, crs, dem,
        )

    summary = {
        "n_elements": mesh.n_elements,
        "n_triangles": len(mesh.triangles),
        "n_quads": len(mesh.quads),
        "face_gpkg": str(face_gpkg_path.resolve()),
        "edge_gpkg": str(edge_gpkg_path.resolve()),
    }

    if cfg.output.cell_bin.enabled:
        with stage("mesh.bin の出力"):
            attrs = compute_cell_attributes(cfg, mesh, quality, crs)
            bin_path = build_and_write_mesh_bin(
                cfg, mesh, quality, terrain, crs, dem, attrs=attrs,
            )
            if bin_path is not None:
                summary["mesh_bin"] = str(bin_path.resolve())

    return summary


def main(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    setup_logging(cfg.output_dir)
    logger = get_logger()
    logger.info("設定: %s", Path(config_path).resolve())
    logger.info(
        "面積下限 %.0f m^2（正三角形換算 %.1f m / 正方形換算 %.1f m）",
        cfg.mesh.min_element_area, cfg.mesh.triangle_size_floor, cfg.mesh.quad_size_floor,
    )
    if cfg.parallel.enabled:
        from .block_runner import run_blocks_parallel
        return run_blocks_parallel(cfg, Path(config_path))
    return run_pipeline(cfg)
