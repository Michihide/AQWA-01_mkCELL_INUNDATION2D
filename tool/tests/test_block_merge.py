from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS

from src.block_merge import merge_block_csvs
from src.solver_gpkg_export import write_edge_csv, write_face_csv
from tests.test_solver_gpkg_export import _attrs
from tests.test_cell_bin_export import _dem, _mesh, _quality, _terrain


def _write_block(block_dir: Path, x_shift: float = 0.0) -> None:
    block_dir.mkdir(parents=True, exist_ok=True)
    mesh = _mesh()
    if x_shift:
        mesh.nodes = mesh.nodes.copy()
        mesh.nodes[:, 0] += x_shift
    quality = _quality()
    terrain = _terrain()
    attrs = _attrs()
    write_face_csv(mesh, quality, terrain, attrs, block_dir / "face.csv")
    write_edge_csv(mesh, quality, _dem(), block_dir / "edge.csv")


def test_merge_two_blocks(tmp_path: Path):
    blocks = tmp_path / "blocks"
    _write_block(blocks / "block_000")
    _write_block(blocks / "block_001", x_shift=1000.0)

    out = tmp_path / "merged"
    crs = CRS.from_epsg(6670)
    merge_block_csvs(blocks, out, crs)

    face = gpd.read_file(out / "face.gpkg", layer="poly")
    edge = gpd.read_file(out / "edge.gpkg", layer="line")
    assert len(face) == 4
    assert list(face["CN"]) == [1, 2, 3, 4]
    assert list(face["block"]) == [1, 1, 2, 2]
    assert (edge["LN"] == edge["LineID"]).all()


def test_merge_single_block_ignores_stale_neighbors(tmp_path: Path):
    blocks = tmp_path / "blocks"
    _write_block(blocks / "block_000")
    _write_block(blocks / "block_001", x_shift=1000.0)

    out = tmp_path / "merged"
    crs = CRS.from_epsg(6670)
    merge_block_csvs(blocks, out, crs, block_indices=[0])

    face = gpd.read_file(out / "face.gpkg", layer="poly")
    assert len(face) == 2
    assert list(face["CN"]) == [1, 2]
