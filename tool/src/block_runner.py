"""氾濫ブロック単位の並列実行と統合後の mesh.bin 出力。"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pyproj import CRS

from .block_helpers import block_mesh_path, config_for_block, list_block_indices
from .block_merge import merge_block_outputs
from .mesh_bin_export import build_and_write_mesh_bin
from .config import Config, load_config
from .io_raster import load_dem
from .main import run_pipeline
from .mesh_parser import Mesh, load_mesh_from_msh
from .quality_metrics import evaluate_quality
from .run_summary import aggregate_block_summaries, build_quality_summary
from .terrain_metrics import compute_terrain_metrics
from .utils import get_logger, setup_logging, stage


def _default_workers(requested: int) -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(requested, max(1, cpu - 2), 8))


def _run_block_worker(config_path: str, block_index: int) -> tuple[int, float]:
    """ProcessPool 用。1 ブロックのメッシュ生成のみ（face/edge はブロック dir へ）。"""
    cfg = load_config(config_path)
    cfg = config_for_block(cfg, block_index)
    setup_logging(cfg.output_dir)
    t0 = time.perf_counter()
    run_pipeline(cfg)
    return block_index, time.perf_counter() - t0


def merge_block_meshes(cfg: Config, block_indices: list[int]) -> Mesh:
    """ブロック .msh を連結（氾濫ブロックは離散なので節点 ID は単純オフセット）。"""
    nodes_parts: list[np.ndarray] = []
    tris_parts: list[np.ndarray] = []
    quads_parts: list[np.ndarray] = []
    tri_surf: list[np.ndarray] = []
    quad_surf: list[np.ndarray] = []
    node_offset = 0
    tri_blocks: list[np.ndarray] = []
    quad_blocks: list[np.ndarray] = []

    for idx in block_indices:
        path = block_mesh_path(cfg, idx)
        if not path.is_file():
            raise FileNotFoundError(f"ブロックメッシュがありません: {path}")
        m = load_mesh_from_msh(path)
        nodes_parts.append(m.nodes)
        bid = np.int32(idx + 1)
        if len(m.triangles):
            tris_parts.append(m.triangles + node_offset)
            tri_surf.append(m.tri_surface)
            tri_blocks.append(np.full(len(m.triangles), bid, dtype=np.int32))
        if len(m.quads):
            quads_parts.append(m.quads + node_offset)
            quad_surf.append(m.quad_surface)
            quad_blocks.append(np.full(len(m.quads), bid, dtype=np.int32))
        node_offset += m.n_nodes

    block_parts = tri_blocks + quad_blocks
    block_id = (
        np.concatenate(block_parts)
        if block_parts
        else np.empty(0, dtype=np.int32)
    )
    return Mesh(
        nodes=np.vstack(nodes_parts),
        triangles=np.vstack(tris_parts) if tris_parts else np.empty((0, 3), dtype=np.int64),
        quads=np.vstack(quads_parts) if quads_parts else np.empty((0, 4), dtype=np.int64),
        tri_surface=np.concatenate(tri_surf) if tri_surf else np.empty(0, dtype=np.int64),
        quad_surface=np.concatenate(quad_surf) if quad_surf else np.empty(0, dtype=np.int64),
        surface_roles={1: "interior"},
        block_id=block_id,
    )


def finalize_merged_quality(cfg: Config, block_indices: list[int]) -> dict:
    """統合メッシュの品質・地形指標を評価する。"""
    mesh = merge_block_meshes(cfg, block_indices)
    crs = CRS.from_epsg(cfg.crs.target_epsg)
    xs, ys = mesh.nodes[:, 0], mesh.nodes[:, 1]
    bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    dem = load_dem(cfg, crs, bounds)
    quality = evaluate_quality(mesh)
    terrain = compute_terrain_metrics(mesh, dem, cfg)
    return build_quality_summary(cfg, mesh, quality, terrain), mesh, quality, terrain, dem, crs


def finalize_merged_cell_bin(
    cfg: Config, block_indices: list[int],
) -> tuple[Path | None, dict]:
    """統合メッシュから mesh.bin を書き出し、品質サマリーを返す。"""
    with stage("統合メッシュから mesh.bin を出力"):
        quality_summary, mesh, quality, terrain, dem, crs = finalize_merged_quality(
            cfg, block_indices,
        )
        if not cfg.output.cell_bin.enabled:
            return None, quality_summary

        bin_path = build_and_write_mesh_bin(
            cfg, mesh, quality, terrain, crs, dem,
        )
        return bin_path, quality_summary


def run_blocks_parallel(
    cfg: Config,
    config_path: Path,
    *,
    workers: int | None = None,
    block_indices: list[int] | None = None,
) -> dict:
    """全ブロックを並列生成し、face/edge と mesh.bin を統合出力する。"""
    logger = get_logger()
    indices = block_indices if block_indices is not None else list_block_indices(cfg)
    if not indices:
        raise RuntimeError("有効な氾濫ブロックがありません")

    n_workers = workers if workers is not None else _default_workers(cfg.parallel.workers)

    logger.info(
        "並列メッシュ生成: %d ブロック, workers=%d",
        len(indices), n_workers,
    )

    failures: list[int] = []
    completed = 0
    started = time.perf_counter()
    config_str = str(config_path.resolve())

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_block_worker, config_str, idx): idx for idx in indices
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _, elapsed = fut.result()
                completed += 1
                logger.info(
                    "block %03d 完了 (%.1fs) %d/%d 経過 %.0fs",
                    idx, elapsed, completed, len(indices), time.perf_counter() - started,
                )
            except Exception as exc:
                logger.error("block %03d 失敗: %s", idx, exc)
                failures.append(idx)

    if failures:
        raise RuntimeError(f"失敗したブロック: {failures}")

    crs = CRS.from_epsg(cfg.crs.target_epsg)
    with stage("ブロック face/edge の統合"):
        merge_block_outputs(
            cfg.blocks_dir,
            cfg.gpkg_csv_dir,
            crs,
            block_indices=indices,
            keep_blocks=cfg.parallel.keep_block_outputs,
        )

    bin_path: Path | None = None
    merged_quality: dict = {}
    if cfg.output.cell_bin.enabled:
        bin_path, merged_quality = finalize_merged_cell_bin(cfg, indices)
    else:
        with stage("統合メッシュの品質評価"):
            merged_quality, *_ = finalize_merged_quality(cfg, indices)

    block_summaries: list[dict] = []
    for idx in indices:
        summary_path = cfg.blocks_dir / f"block_{idx:03d}" / "summary.json"
        if summary_path.is_file():
            block_summaries.append(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )

    meta = aggregate_block_summaries(block_summaries)
    if merged_quality:
        quality_summary = merged_quality
    else:
        quality_summary = meta

    if meta.get("n_iterations"):
        quality_summary["n_iterations"] = meta["n_iterations"]
    if meta.get("band_quad_count") is not None:
        quality_summary["band_quad_count"] = meta["band_quad_count"]

    summary = {
        "n_blocks": len(indices),
        "workers": n_workers,
        "face_gpkg": str((cfg.gpkg_csv_dir / cfg.output.gpkg_csv.face_gpkg).resolve()),
        "edge_gpkg": str((cfg.gpkg_csv_dir / cfg.output.gpkg_csv.edge_gpkg).resolve()),
        **quality_summary,
    }
    if bin_path is not None:
        summary["mesh_bin"] = str(bin_path.resolve())
    return summary
