"""パイプライン終端。正規化テーブルと属性を作業ファイルと mesh.bin に書く。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import CRS

from .cell_bin_export import LANDUSE_CODES, SOIL_CODES, CellAttributes, compute_cell_attributes
from .config import Config
from .io_raster import DemGrid
from .lts_levels import BINNING_TO_CODE, compute_lts_levels
from .mesh_bin_io import (
    WORK_FILES,
    MeshAttributes,
    apply_boundary_froude,
    building_from_dense,
    default_fr_from_config,
    dense_areas_to_sparse,
    hash_file,
    veg_from_dense,
    write_mesh_bin,
    write_work_files,
)
from .mesh_cache_order import align_solver_tables, apply_cache_order
from .mesh_parser import Mesh
from .mesh_tables import build_normalized_tables
from .quality_metrics import QualityReport
from .special_edges import special_edges_for_mesh
from .terrain_metrics import TerrainReport
from .solver_gpkg_export import face_block_ids
from .utils import get_logger


def attributes_to_mesh_attrs(attrs: CellAttributes) -> MeshAttributes:
    lu_codes = [int(c) for c in LANDUSE_CODES]
    soil_codes = [int(c) for c in SOIL_CODES]
    return MeshAttributes(
        landuse=dense_areas_to_sparse(attrs.landuse_areas, lu_codes),
        soil=dense_areas_to_sparse(attrs.soil_areas, soil_codes),
        building=building_from_dense(attrs.bld_ratio, attrs.bld_peri),
        veg=veg_from_dense(attrs.chi_veg, attrs.a_veg, attrs.H_veg, attrs.Cd_veg),
    )


def build_and_write_mesh_bin(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    crs: CRS,
    dem: DemGrid,
    attrs: CellAttributes | None = None,
) -> Path | None:
    """土地利用・土壌・建物を疎形式で載せ、作業ファイルと mesh.bin を書く。

    vegetation 未指定なら veg.bin は nentry=0。
    special_edges は input.special_edges のラインを拘束した辺へ対応付ける。
    couple.csv は空ヘッダのみ。
    """
    logger = get_logger()
    cb = cfg.output.cell_bin
    if not cb.enabled:
        return None

    if attrs is None:
        attrs = compute_cell_attributes(cfg, mesh, quality, crs)

    dummy = np.zeros(mesh.n_elements, dtype=np.int32)
    tables = build_normalized_tables(mesh, quality, terrain, dem, dummy=dummy)
    tables.faces.block = face_block_ids(
        len(tables.faces), mesh, cfg.input.polygon_filter,
    )
    mesh_attrs = attributes_to_mesh_attrs(attrs)
    mesh_attrs.special_edges = special_edges_for_mesh(cfg, tables)
    apply_boundary_froude(
        tables.edges,
        mesh_attrs.special_edges,
        mesh_attrs.couple,
        default_fr=default_fr_from_config(cfg),
    )
    if cb.index_order == "rcm":
        tables, mesh_attrs, perm = apply_cache_order(tables, mesh_attrs)
        logger.info(
            "面・辺・頂点番号を詰めました: 内部辺 |L-R| 平均 %.1f -> %.1f、"
            "辺走査 |ΔL| 平均 %.1f -> %.1f",
            perm.span_before, perm.span_after, perm.step_before, perm.step_after,
        )
        _align_exported_tables(cfg, tables.faces.x, tables.faces.y, perm.face_new_to_old)
        face_perm = perm.face_new_to_old
    else:
        face_perm = None

    filename = cb.filename or "mesh.bin"
    out_dir = cfg.cell_bin_dir
    work_dir = out_dir / "mesh_work"
    write_work_files(work_dir, tables, mesh_attrs)
    output_path = out_dir / filename
    hashes = [hash_file(work_dir / name) for name in WORK_FILES]
    lts_levels = None
    lts_lmax = 0
    lts_code = 2
    if cb.lts.enabled:
        lts_levels, lts_lmax = compute_lts_levels(
            tables,
            max_level=int(cb.lts.max_level),
            binning=cb.lts.binning,  # type: ignore[arg-type]
        )
        lts_code = BINNING_TO_CODE[cb.lts.binning]
        logger.info(
            "LTS レベルを mesh.bin に付与: lmax=%d binning=%s",
            lts_lmax,
            cb.lts.binning,
        )
    write_mesh_bin(
        output_path, tables, mesh_attrs,
        epsg=int(crs.to_epsg() or cfg.crs.target_epsg),
        hashes=hashes,
        lts_levels=lts_levels,
        lts_lmax=lts_lmax,
        lts_binning_code=lts_code,
    )
    n_block = int(tables.faces.block.max()) if len(tables.faces) else 1
    logger.info(
        "mesh.bin を出力: %s（点 %d, 辺 %d, 面 %d, ブロック %d）",
        output_path.name, len(tables.nodes), len(tables.edges),
        len(tables.faces), n_block,
    )
    if face_perm is not None:
        perm_path = output_path.with_name(output_path.name + ".face_perm.npy")
        np.save(perm_path, face_perm)
        logger.info(
            "面番号の対応を出力: %s（新しい面 → 古い面、0 始まり）。"
            "降雨 CSV の列はこの順に並べ替えてください",
            perm_path.name,
        )
    return output_path


def _align_exported_tables(cfg: Config, face_x, face_y, face_perm: np.ndarray) -> None:
    """rcm のとき、既存の face/edge 出力の CN を新しい面番号に合わせる。"""
    if not cfg.output.gpkg_csv.enabled:
        return
    logger = get_logger()
    gc = cfg.output.gpkg_csv
    directory = cfg.gpkg_csv_dir
    try:
        updated = align_solver_tables(
            directory / gc.face_gpkg,
            directory / gc.edge_gpkg,
            face_x,
            face_y,
            face_perm,
            face_csv=directory / gc.face_csv,
            edge_csv=directory / gc.edge_csv,
        )
    except Exception as exc:
        logger.warning("face/edge の CN は元の番号のままです: %s", exc)
        return
    if updated:
        logger.info("face/edge の CN を詰めた面番号に合わせました")
    elif (directory / gc.face_gpkg).is_file():
        logger.warning(
            "face.gpkg の CN は詰めた面番号と別のままです。"
            "この mesh.bin の描画には CN を合わせた gpkg が必要です"
        )
