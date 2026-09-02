from __future__ import annotations

import numpy as np
import pytest
from pyproj import CRS

from src.cell_bin_export import LANDUSE_CODES, SOIL_CODES, write_cell_bin
from src.io_raster import DemGrid
from src.mesh_bin_io import (
    BLK_MAGIC,
    KIND_FROUDE,
    KIND_Q,
    KIND_ROAD,
    MeshAttributes,
    SpecialEdgeTable,
    CoupleTable,
    VegTable,
    apply_boundary_froude,
    building_from_dense,
    decode_edges,
    dense_areas_to_sparse,
    encode_block_trailer,
    encode_edges,
    pack_mesh,
    pack_work_directory,
    read_work_files,
    unpack_mesh,
    veg_from_dense,
    write_mesh_bin,
    write_work_files,
)
from src.zonal_stats import vegetation_chi_and_params
from src.mesh_parser import Mesh, orient_ccw
from src.mesh_tables import build_normalized_tables
from src.quality_metrics import QualityReport
from src.terrain_metrics import TerrainReport

from scripts.cell_bin_to_mesh import convert, read_cell_bin

# 2 枚の正方形（一辺 10 m）が x=10 の辺を共有する。左が要素 0、右が要素 1。
NODES = [
    [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0], [0.0, 10.0],
]
QUADS = [[0, 1, 4, 5], [1, 2, 3, 4]]


def _mesh() -> Mesh:
    mesh = Mesh(
        nodes=np.asarray(NODES, dtype=float),
        triangles=np.empty((0, 3), dtype=np.int64),
        quads=np.asarray(QUADS, dtype=np.int64),
        tri_surface=np.empty(0, dtype=np.int64),
        quad_surface=np.ones(2, dtype=np.int64),
    )
    orient_ccw(mesh)
    mesh.node_z = np.zeros(len(NODES))
    return mesh


def _quality() -> QualityReport:
    n = 2
    nan = np.full(n, np.nan)
    return QualityReport(
        area=np.array([100.0, 100.0]),
        kind=np.full(n, 4, dtype=np.int8),
        min_angle_deg=np.full(n, 90.0),
        max_angle_deg=np.full(n, 90.0),
        aspect_ratio=np.ones(n),
        radius_ratio=nan,
        scaled_jacobian=np.ones(n),
        is_convex=np.ones(n, dtype=bool),
    )


def _terrain() -> TerrainReport:
    return TerrainReport(
        n_samples=np.full(2, 10, dtype=np.int64),
        elevation=np.array([1.0, 2.0]),
        plane_fit_rmse=np.zeros(2),
        slope_magnitude=np.zeros(2),
        slope_direction_spread_deg=np.zeros(2),
    )


def _dem() -> DemGrid:
    return DemGrid(
        array=np.zeros((100, 100), dtype=np.float32),
        x_min=-50.0, y_max=50.0, px=1.0, crs=CRS.from_epsg(6670),
    )


def test_shared_edge_is_written_once():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    assert len(tables.nodes) == 6
    assert len(tables.faces) == 2
    # 外周 6 + 共有 1
    assert len(tables.edges) == 7
    assert list(tables.nodes.id) == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(tables.nodes.x, [0, 10, 20, 20, 10, 0])
    np.testing.assert_allclose(tables.nodes.y, [0, 0, 0, 10, 10, 10])

    internals = [
        i for i in range(len(tables.edges))
        if tables.edges.face_left[i] != 0 and tables.edges.face_right[i] != 0
    ]
    assert len(internals) == 1
    ie = internals[0]
    assert {int(tables.edges.face_left[ie]), int(tables.edges.face_right[ie])} == {1, 2}
    v1, v2 = int(tables.edges.v1[ie]), int(tables.edges.v2[ie])
    xs = {float(tables.nodes.x[v1 - 1]), float(tables.nodes.x[v2 - 1])}
    assert xs == {10.0}

    used = []
    for eids in tables.faces.edge_ids:
        used.extend(int(e) for e in eids)
    assert used.count(int(tables.edges.id[ie])) == 2
    assert tables.faces.dummy.tolist() == [0, 0]
    assert tables.faces.z_bed.tolist() == pytest.approx([1.0, 2.0])
    assert tables.faces.A.tolist() == pytest.approx([100.0, 100.0])


def test_pack_roundtrip_and_coords_only_on_nodes(tmp_path):
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    path = write_mesh_bin(tmp_path / "mesh.bin", tables, epsg=6670)
    packed = unpack_mesh(path.read_bytes())
    assert packed.epsg == 6670
    assert packed.version == 1
    assert len(packed.mesh.nodes) == 6
    assert len(packed.mesh.edges) == 7
    assert len(packed.mesh.faces) == 2
    np.testing.assert_allclose(packed.mesh.nodes.x, tables.nodes.x)
    np.testing.assert_allclose(packed.mesh.nodes.y, tables.nodes.y)
    # 辺レコードに座標フィールドは無い（36 B = id + 4*i32 + z_crest + fr）
    assert path.stat().st_size > 320
    raw = path.read_bytes()
    assert raw[:8] == b"AQWAMESH"
    assert BLK_MAGIC in raw
    np.testing.assert_array_equal(packed.mesh.faces.block, [1, 1])


def test_block_trailer_roundtrip_and_legacy_ones():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    tables.faces.block = np.array([1, 2], dtype=np.int32)
    packed = unpack_mesh(pack_mesh(tables, epsg=6670))
    np.testing.assert_array_equal(packed.mesh.faces.block, [1, 2])

    raw = pack_mesh(tables, epsg=6670)
    trailer = encode_block_trailer(tables.faces.block)
    legacy = raw[: -len(trailer)]
    assert BLK_MAGIC not in legacy
    old = unpack_mesh(legacy)
    np.testing.assert_array_equal(old.mesh.faces.block, [1, 1])


def test_attributes_landuse_soil_building_dummy(tmp_path):
    tables = build_normalized_tables(
        _mesh(), _quality(), _terrain(), _dem(), dummy=np.array([0, 1], dtype=np.int32),
    )
    landuse = np.zeros((2, len(LANDUSE_CODES)))
    landuse[0, 0] = 40.0  # code 10
    soil = np.zeros((2, len(SOIL_CODES)))
    soil[1, 2] = 12.5  # code 3
    chi = np.array([0.2, 0.0])
    peri = np.array([8.0, 0.0])
    attrs = MeshAttributes(
        landuse=dense_areas_to_sparse(landuse, [int(c) for c in LANDUSE_CODES]),
        soil=dense_areas_to_sparse(soil, [int(c) for c in SOIL_CODES]),
        building=building_from_dense(chi, peri),
        veg=VegTable(),
    )
    assert len(attrs.veg) == 0
    assert list(attrs.landuse.code) == [10]
    assert list(attrs.soil.code) == [3]
    assert list(attrs.building.face_id) == [1]
    assert attrs.building.chi_bld[0] == pytest.approx(0.2)

    write_work_files(tmp_path, tables, attrs)
    assert (tmp_path / "veg.bin").stat().st_size == 4  # nentry=0
    packed = unpack_mesh(pack_mesh(tables, attrs, epsg=6670))
    assert packed.mesh.faces.dummy.tolist() == [0, 1]
    assert packed.attrs.landuse.area[0] == pytest.approx(40.0)
    assert packed.attrs.soil.area[0] == pytest.approx(12.5)
    assert packed.attrs.building.bld_peri[0] == pytest.approx(8.0)
    assert len(packed.attrs.veg) == 0
    veg = veg_from_dense(
        np.array([0.0, 0.4]),
        np.array([0.0, 0.8]),
        np.array([0.0, 2.5]),
        np.array([0.0, 1.1]),
    )
    attrs.veg = veg
    packed_veg = unpack_mesh(pack_mesh(tables, attrs, epsg=6670))
    assert list(packed_veg.attrs.veg.face_id) == [2]
    assert packed_veg.attrs.veg.chi_veg[0] == pytest.approx(0.4)
    assert packed_veg.attrs.veg.a[0] == pytest.approx(0.8)
    assert packed_veg.attrs.veg.H_v[0] == pytest.approx(2.5)
    assert packed_veg.attrs.veg.C_D[0] == pytest.approx(1.1)
    assert len(packed.attrs.special_edges) == 0
    assert len(packed.attrs.couple) == 0


def test_special_edges_and_couple_roundtrip(tmp_path):
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    shared = int(np.where(
        (tables.edges.face_left != 0) & (tables.edges.face_right != 0)
    )[0][0])
    eid = int(tables.edges.id[shared])
    attrs = MeshAttributes(
        special_edges=SpecialEdgeTable(
            edge_id=np.array([eid], dtype=np.int32),
            kind=np.array([KIND_ROAD], dtype=np.int32),
            zc=np.array([1.02]),
            H=np.array([0.0]),
            z_road=np.array([1.02]),
            qgroup=np.array([0], dtype=np.int32),
        ),
        couple=CoupleTable(
            edge_id=np.array([eid], dtype=np.int32),
            river_link=np.array([3], dtype=np.int32),
            kp=np.array([12.5]),
            bank=np.array([1], dtype=np.int32),
        ),
    )
    write_work_files(tmp_path, tables, attrs)
    mesh2, attrs2, _hashes = read_work_files(tmp_path)
    assert attrs2.special_edges.kind[0] == KIND_ROAD
    assert attrs2.couple.river_link[0] == 3
    out = pack_work_directory(tmp_path, tmp_path / "mesh.bin", epsg=6670)
    packed = unpack_mesh(out.read_bytes())
    assert packed.attrs.special_edges.zc[0] == pytest.approx(1.02)
    assert packed.attrs.couple.kp[0] == pytest.approx(12.5)
    assert packed.attrs.special_edges.kind[0] != KIND_Q


def test_cell_bin_to_mesh_dedups_edges(tmp_path):
    mesh = _mesh()
    out_cell = write_cell_bin(
        mesh, _quality(), _terrain(), _dem(),
        np.zeros((2, 13)), np.zeros((2, 17)),
        np.array([0.1, 0.0]), np.array([4.0, 0.0]),
        face_gpkg_path="face.gpkg", edge_gpkg_path="edge.gpkg",
        output_path=tmp_path / "old.bin",
    )
    dest = tmp_path / "converted"
    mesh_path = convert(out_cell, dest, epsg=6670)
    packed = unpack_mesh(mesh_path.read_bytes())
    assert len(packed.mesh.edges) == 7
    assert len(packed.mesh.nodes) == 6
    internals = [
        i for i in range(len(packed.mesh.edges))
        if packed.mesh.edges.face_right[i] != 0
    ]
    assert len(internals) == 1
    np.testing.assert_allclose(packed.mesh.faces.A, [100.0, 100.0])
    assert packed.attrs.building.chi_bld[0] == pytest.approx(0.1)
    assert packed.attrs.building.bld_peri[0] == pytest.approx(4.0)
    # 点座標は node のみ。辺セクションは 7 * 36 B（+fr）または旧 7 * 28 B
    raw = unpack_mesh(mesh_path.read_bytes())
    assert raw.mesh.nodes.z.shape == (6,)
    tables, _ = read_cell_bin(out_cell)
    assert len(tables.edges) == 7


def test_vegetation_polygon_params():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    faces = [box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)]
    areas = np.array([100.0, 100.0])
    gdf = geopandas.GeoDataFrame(
        {"C_D": [1.2], "a": [0.5], "H_v": [3.0]},
        geometry=[box(0.0, 0.0, 5.0, 10.0)],
        crs="EPSG:6670",
    )
    chi, a, hv, cd = vegetation_chi_and_params(faces, areas, gdf)
    assert chi[0] == pytest.approx(0.5)
    assert a[0] == pytest.approx(0.5)
    assert hv[0] == pytest.approx(3.0)
    assert cd[0] == pytest.approx(1.2)
    assert chi[1] == pytest.approx(0.0)
    assert a[1] == pytest.approx(0.0)


def test_edge_fr_roundtrip_and_legacy_28():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    fr = np.zeros(len(tables.edges))
    outer = tables.edges.face_right == 0
    fr[outer] = 0.5
    tables.edges.fr = fr
    packed = unpack_mesh(pack_mesh(tables, epsg=6670))
    np.testing.assert_allclose(packed.mesh.edges.fr[outer], 0.5)
    np.testing.assert_allclose(packed.mesh.edges.fr[~outer], 0.0)

    raw36 = encode_edges(tables.edges)
    # 新形式は 36 B。旧 28 B も読める。
    assert len(raw36) == len(tables.edges) * 36
    legacy = b"".join(
        raw36[i * 36:i * 36 + 28] for i in range(len(tables.edges))
    )
    old = decode_edges(legacy, nedge=len(tables.edges))
    np.testing.assert_allclose(old.fr, 0.0)
    np.testing.assert_array_equal(old.v1, tables.edges.v1)


def test_apply_boundary_froude_skips_wall_and_inlet():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    outer = np.where(tables.edges.face_right == 0)[0]
    wall_eid = int(tables.edges.id[outer[0]])
    open_eid = int(tables.edges.id[outer[1]])
    apply_boundary_froude(
        tables.edges,
        SpecialEdgeTable(
            edge_id=np.array([wall_eid], dtype=np.int32),
            kind=np.array([KIND_ROAD], dtype=np.int32),
            zc=np.array([0.0]),
            H=np.array([0.0]),
            z_road=np.array([0.0]),
            qgroup=np.array([0], dtype=np.int32),
        ),
        default_fr=0.3,
    )
    assert tables.edges.fr[wall_eid - 1] == pytest.approx(0.0)
    assert tables.edges.fr[open_eid - 1] == pytest.approx(0.3)
    apply_boundary_froude(
        tables.edges,
        SpecialEdgeTable(
            edge_id=np.array([open_eid], dtype=np.int32),
            kind=np.array([KIND_FROUDE], dtype=np.int32),
            zc=np.array([0.0]),
            H=np.array([0.0]),
            z_road=np.array([0.0]),
            qgroup=np.array([0], dtype=np.int32),
            fr=np.array([0.5]),
        ),
        default_fr=0.3,
    )
    assert tables.edges.fr[open_eid - 1] == pytest.approx(0.5)
