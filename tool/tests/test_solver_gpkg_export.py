from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS

from src.cell_bin_export import CellAttributes
from src.io_raster import DemGrid
from src.solver_gpkg_export import write_edge_gpkg, write_face_gpkg
from tests.test_cell_bin_export import _mesh, _quality, _terrain


def _attrs(n: int = 2) -> CellAttributes:
    return CellAttributes(
        landuse_areas=np.zeros((n, 13)),
        soil_areas=np.zeros((n, 17)),
        bld_ratio=np.zeros(n),
        bld_peri=np.zeros(n),
    )


def test_face_edge_gpkg_schema(tmp_path: Path):
    mesh = _mesh()
    quality = _quality()
    terrain = _terrain()
    dem = DemGrid(
        array=np.zeros((100, 100), dtype=np.float32),
        x_min=-50.0,
        y_max=50.0,
        px=1.0,
        crs=CRS.from_epsg(6670),
    )
    crs = CRS.from_epsg(6670)
    attrs = _attrs()

    face_path = tmp_path / "face.gpkg"
    edge_path = tmp_path / "edge.gpkg"
    write_face_gpkg(mesh, quality, terrain, attrs, crs, face_path)
    write_edge_gpkg(mesh, quality, dem, crs, edge_path)

    face = gpd.read_file(face_path)
    edge = gpd.read_file(edge_path)
    assert "CN" in face.columns
    assert "area" in face.columns
    assert "soil_1_area" in face.columns
    assert list(face["CN"]) == [1, 2]
    assert len(edge) == 8
    assert {"CN", "LineID", "StartNode", "EndNode", "LN", "ID"}.issubset(edge.columns)
    assert (edge["LN"] == edge["LineID"]).all()
    # 共有辺は両セルで同じ LN
    edge["key"] = edge.apply(
        lambda r: tuple(sorted([int(r.StartNode), int(r.EndNode)])), axis=1,
    )
    shared = edge.groupby("key")["LN"].nunique()
    assert (shared <= 1).all()
