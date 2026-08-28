"""パイプライン終端。正規化テーブルと属性を作業ファイルと mesh.bin に書く。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import CRS

from .cell_bin_export import LANDUSE_CODES, SOIL_CODES, CellAttributes, compute_cell_attributes
from .config import Config
from .io_raster import DemGrid
from .mesh_bin_io import (
    MeshAttributes,
    building_from_dense,
    dense_areas_to_sparse,
    pack_work_directory,
    veg_from_dense,
    write_work_files,
)
from .mesh_parser import Mesh
from .mesh_tables import build_normalized_tables
from .quality_metrics import QualityReport
from .special_edges import special_edges_for_mesh
from .terrain_metrics import TerrainReport
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
    mesh_attrs = attributes_to_mesh_attrs(attrs)
    mesh_attrs.special_edges = special_edges_for_mesh(cfg, tables)

    filename = cb.filename or "mesh.bin"
    out_dir = cfg.cell_bin_dir
    work_dir = out_dir / "mesh_work"
    write_work_files(work_dir, tables, mesh_attrs)
    output_path = out_dir / filename
    pack_work_directory(work_dir, output_path, epsg=int(crs.to_epsg() or cfg.crs.target_epsg))
    logger.info(
        "mesh.bin を出力: %s（点 %d, 辺 %d, 面 %d）",
        output_path.name, len(tables.nodes), len(tables.edges), len(tables.faces),
    )
    return output_path
