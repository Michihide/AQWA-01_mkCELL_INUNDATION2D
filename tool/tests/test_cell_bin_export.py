from __future__ import annotations

import struct

import numpy as np
import pytest
from pyproj import CRS

from src.cell_bin_export import _EDGE_STRUCT, _FACE_STRUCT, _PATH_FIELD_LEN, write_cell_bin
from src.io_raster import DemGrid
from src.mesh_parser import Mesh, orient_ccw
from src.quality_metrics import QualityReport
from src.terrain_metrics import TerrainReport

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


def _read_back(path):
    with open(path, "rb") as f:
        face_gpkg = f.read(_PATH_FIELD_LEN)
        edge_gpkg = f.read(_PATH_FIELD_LEN)
        total_face, total_edge, total_face_index = struct.unpack("<iii", f.read(12))

        faces = []
        edges = []
        for _ in range(total_face):
            raw = f.read(_FACE_STRUCT.size)
            fields = _FACE_STRUCT.unpack(raw)
            cn, n_edge = fields[0], fields[1]
            x, y, area, bl, bld_ratio, bld_peri = fields[2:8]
            lu = fields[8:21]
            soil = fields[21:38]
            calmesh = fields[38]
            faces.append(dict(
                cn=cn, n_edge=n_edge, x=x, y=y, area=area, bl=bl,
                bld_ratio=bld_ratio, bld_peri=bld_peri, lu=lu, soil=soil, calmesh=calmesh,
            ))
            for _e in range(n_edge):
                raw = f.read(_EDGE_STRUCT.size)
                ef = _EDGE_STRUCT.unpack(raw)
                edges.append(dict(
                    en=ef[0], bc_flag=ef[1], cn_self=ef[2], cn_adj=ef[3],
                    a_cv=ef[4], dl=ef[5], cos_x=ef[6], cos_y=ef[7],
                    w_self=ef[8], w_adj=ef[9], w_face=ef[10],
                    v_st=ef[11], v_en=ef[12],
                    x_st=ef[13], y_st=ef[14], bl_st=ef[15],
                    x_en=ef[16], y_en=ef[17], bl_en=ef[18],
                    x_cnt=ef[19], y_cnt=ef[20], bl_cnt=ef[21],
                    length=ef[22],
                ))
        assert f.read() == b""
    return face_gpkg, edge_gpkg, total_face, total_edge, total_face_index, faces, edges


def test_header_and_face_records(tmp_path):
    mesh = _mesh()
    quality = _quality()
    terrain = _terrain()
    dem = _dem()
    landuse = np.zeros((2, 13))
    soil = np.zeros((2, 17))
    bld_ratio = np.zeros(2)
    bld_peri = np.zeros(2)

    out = write_cell_bin(
        mesh, quality, terrain, dem, landuse, soil, bld_ratio, bld_peri,
        face_gpkg_path="/tmp/face.gpkg", edge_gpkg_path="/tmp/edge.gpkg",
        output_path=tmp_path / "test.bin",
    )

    face_gpkg, edge_gpkg, total_face, total_edge, total_face_index, faces, edges = _read_back(out)

    assert face_gpkg.decode("utf-8").strip() == "/tmp/face.gpkg"
    assert edge_gpkg.decode("utf-8").strip() == "/tmp/edge.gpkg"
    assert total_face == 2
    # 各要素 4 辺 x 2 要素 = 8 レコード（内部辺は両側から 1 本ずつ）
    assert total_edge == 8
    assert total_face_index == 2  # ダミーメッシュ非対応のため全セルが計算対象

    assert [f["cn"] for f in faces] == [1, 2]
    assert [f["calmesh"] for f in faces] == [0, 0]
    assert faces[0]["area"] == pytest.approx(100.0)
    assert faces[0]["bl"] == pytest.approx(1.0)
    assert faces[1]["bl"] == pytest.approx(2.0)
    assert faces[0]["x"] == pytest.approx(5.0)
    assert faces[1]["x"] == pytest.approx(15.0)
    assert all(v == pytest.approx(0.0) for v in faces[0]["lu"])
    assert all(v == pytest.approx(0.0) for v in faces[0]["soil"])


def test_internal_edge_adjacency_and_boundary_flags(tmp_path):
    mesh = _mesh()
    quality = _quality()
    terrain = _terrain()
    dem = _dem()
    landuse = np.zeros((2, 13))
    soil = np.zeros((2, 17))
    bld_ratio = np.zeros(2)
    bld_peri = np.zeros(2)

    out = write_cell_bin(
        mesh, quality, terrain, dem, landuse, soil, bld_ratio, bld_peri,
        face_gpkg_path="face.gpkg", edge_gpkg_path="edge.gpkg",
        output_path=tmp_path / "test.bin",
    )
    *_rest, edges = _read_back(out)

    # en は 1..total_edge の一意な通し番号
    ens = sorted(e["en"] for e in edges)
    assert ens == list(range(1, len(edges) + 1))

    boundary = [e for e in edges if e["bc_flag"] == 1]
    internal = [e for e in edges if e["bc_flag"] == 0]
    assert len(internal) == 2  # 共有辺は両面から 1 本ずつ
    assert len(boundary) == 6  # 残り 6 辺は外周

    for e in boundary:
        assert e["cn_adj"] == 0
    assert {e["cn_self"] for e in internal} == {1, 2}
    for e in internal:
        assert e["cn_adj"] != 0
        # A_CV_edge は両側の面積の和
        assert e["a_cv"] == pytest.approx(200.0)
        assert e["dl"] == pytest.approx(10.0)  # 重心間距離 (5,5)-(15,5)
        # 自面から隣接面への向き（左→右は +x、右→左は -x）
        assert abs(e["cos_x"]) == pytest.approx(1.0)
        assert e["cos_y"] == pytest.approx(0.0)
        assert e["w_self"] == pytest.approx(0.5)
        assert e["w_adj"] == pytest.approx(0.5)

    # 共有辺の頂点は (10,0)-(10,10) で長さ 10
    shared = [e for e in edges if e["bc_flag"] == 0]
    for e in shared:
        assert e["length"] == pytest.approx(10.0)
        assert {round(e["x_st"]), round(e["x_en"])} == {10}
