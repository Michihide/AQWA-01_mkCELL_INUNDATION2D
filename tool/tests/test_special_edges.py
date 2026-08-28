from __future__ import annotations

import textwrap

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, box

from src.config import Config, ConfigError, load_config
from src.breaklines import PreparedLine, drop_crowded_segments
from src.io_vector import VectorInputError, merge_constraint_breaklines
from src.mesh_bin_export import build_and_write_mesh_bin
from src.mesh_bin_io import KIND_CULVERT, KIND_ROAD, KIND_WALL, parse_kind, parse_z_mode, unpack_mesh
from src.mesh_tables import build_normalized_tables
from src.special_edges import (
    SpecialEdgeFeature,
    assert_special_edge_clearance,
    load_special_edge_features,
    match_special_edges,
    pick_clear_special_features,
    special_edge_proximity_report,
    special_features_to_breaklines,
    special_edges_for_mesh,
    split_line_at_vertices,
)

from test_mesh_bin import _dem, _mesh, _quality, _terrain


def _write_yaml(tmp_path, body: str):
    path = tmp_path / "c.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _shared_edge_index(tables) -> int:
    internals = [
        i for i in range(len(tables.edges))
        if tables.edges.face_left[i] != 0 and tables.edges.face_right[i] != 0
    ]
    assert len(internals) == 1
    return internals[0]


def _road_line() -> LineString:
    return LineString([(10.0, 0.0), (10.0, 10.0)])


def test_special_edges_config_is_accepted(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        file: special_edges.gpkg
        default_kind: CULVERT
        match_tol: 2.0
    """))
    assert cfg.input.special_edges.file == "special_edges.gpkg"
    assert cfg.input.special_edges.default_kind == "CULVERT"
    assert cfg.input.special_edges.resolved_match_tol(40.0) == pytest.approx(2.0)


def test_kind_aliases_and_fuzzy():
    assert parse_kind("道路") == KIND_ROAD
    assert parse_kind("カルバート") == KIND_CULVERT
    assert parse_kind("culvrt") == KIND_CULVERT
    assert parse_kind("weir") == KIND_ROAD
    assert parse_z_mode("相対") == "relative"
    assert parse_z_mode("絶対標高") == "absolute"


def test_culvert_cover_sets_crest_above_opening():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    ie = _shared_edge_index(tables)
    feat = SpecialEdgeFeature(
        _road_line(), "CULVERT", 0.0, 1.5, None, 0, 3.0, "relative", 0.5,
    )
    table = match_special_edges(tables, [feat], match_tol=1.0)
    crest = float(tables.edges.z_crest[ie])
    assert table.zc[0] == pytest.approx(crest)
    assert table.H[0] == pytest.approx(1.5)
    assert table.z_road[0] == pytest.approx(crest + 1.5 + 0.5)
    assert table.B[0] == pytest.approx(3.0)


def test_relative_zc_adds_z_crest():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    ie = _shared_edge_index(tables)
    feat = SpecialEdgeFeature(_road_line(), "ROAD", 0.5, 0.0, 0.4, 0, None, "relative")
    table = match_special_edges(tables, [feat], match_tol=1.0)
    crest = float(tables.edges.z_crest[ie])
    assert table.zc[0] == pytest.approx(crest + 0.5)
    assert table.z_road[0] == pytest.approx(crest + 0.4)


def test_both_kind_is_rejected(tmp_path):
    path = _write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        file: special_edges.gpkg
        default_kind: BOTH
    """)
    with pytest.raises(ConfigError, match="BOTH"):
        load_config(path)
    from src.mesh_bin_io import parse_kind
    with pytest.raises(ValueError, match="BOTH"):
        parse_kind("BOTH")
    with pytest.raises(ValueError, match="BOTH"):
        parse_kind("3")


def test_special_edges_unknown_key_is_rejected(tmp_path):
    path = _write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        road: roads.gpkg
    """)
    with pytest.raises(ConfigError, match=r"input.special_edges: 未知のキー \['road'\]"):
        load_config(path)


def test_match_tol_defaults_from_min_size():
    se = Config().input.special_edges
    assert se.resolved_match_tol(40.0) == pytest.approx(2.0)
    assert se.resolved_match_tol(10.0) == pytest.approx(1.0)


def test_match_road_line_on_shared_edge():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    ie = _shared_edge_index(tables)
    table = match_special_edges(
        tables,
        [SpecialEdgeFeature(_road_line(), "ROAD", 1.02, 0.0, 1.02, 0)],
        match_tol=1.0,
    )
    assert list(table.edge_id) == [int(tables.edges.id[ie])]
    assert list(table.kind) == [KIND_ROAD]
    assert table.zc[0] == pytest.approx(1.02)
    assert table.z_road[0] == pytest.approx(1.02)


def test_match_skips_unrelated_line():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    table = match_special_edges(
        tables,
        [SpecialEdgeFeature(LineString([(100.0, 0.0), (100.0, 10.0)]), "ROAD", 0.0, 0.0, 0.0, 0)],
        match_tol=1.0,
    )
    assert len(table) == 0


def test_zc_falls_back_to_z_crest():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    ie = _shared_edge_index(tables)
    table = match_special_edges(
        tables,
        [SpecialEdgeFeature(_road_line(), "ROAD", None, 0.5, None, 0)],
        match_tol=1.0,
    )
    assert table.H[0] == pytest.approx(0.5)
    assert table.zc[0] == pytest.approx(float(tables.edges.z_crest[ie]))
    assert table.z_road[0] == pytest.approx(float(tables.edges.z_crest[ie]))


def test_wall_outranks_road_on_same_edge():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    ie = _shared_edge_index(tables)
    table = match_special_edges(
        tables,
        [
            SpecialEdgeFeature(_road_line(), "ROAD", 1.0, 0.0, 1.0, 0),
            SpecialEdgeFeature(_road_line(), "WALL", 2.0, 0.0, 2.0, 0),
        ],
        match_tol=1.0,
    )
    assert list(table.edge_id) == [int(tables.edges.id[ie])]
    assert list(table.kind) == [KIND_WALL]
    assert table.zc[0] == pytest.approx(2.0)


def _feat(line: LineString) -> SpecialEdgeFeature:
    return SpecialEdgeFeature(line, "ROAD", None, 0.0, None, 0)


def test_crossing_special_edges_are_allowed():
    feats = [
        _feat(LineString([(0.0, 0.0), (100.0, 0.0)])),
        _feat(LineString([(50.0, -50.0), (50.0, 50.0)])),
    ]
    assert_special_edge_clearance(feats, 25.0)
    report = special_edge_proximity_report(feats, 25.0)
    assert report["bins"]["cross"] == 1
    assert report["conflicts"] == []


def test_far_special_edges_are_allowed():
    feats = [
        _feat(LineString([(0.0, 0.0), (100.0, 0.0)])),
        _feat(LineString([(0.0, 40.0), (100.0, 40.0)])),
    ]
    assert_special_edge_clearance(feats, 25.0)


def test_close_parallel_special_edges_stop():
    feats = [
        _feat(LineString([(0.0, 0.0), (100.0, 0.0)])),
        _feat(LineString([(0.0, 8.0), (100.0, 8.0)])),
    ]
    with pytest.raises(VectorInputError, match="近すぎます"):
        assert_special_edge_clearance(feats, 25.0)
    report = special_edge_proximity_report(feats, 25.0)
    assert report["bins"]["gap_lt_10"] == 1
    assert len(report["conflicts"]) == 1
    assert report["shortest"] == pytest.approx(8.0)


def test_overlapping_special_edges_stop():
    feats = [
        _feat(LineString([(0.0, 0.0), (100.0, 0.0)])),
        _feat(LineString([(50.0, 0.0), (150.0, 0.0)])),
    ]
    with pytest.raises(VectorInputError, match="近すぎます"):
        assert_special_edge_clearance(feats, 25.0)
    report = special_edge_proximity_report(feats, 25.0)
    assert report["bins"]["overlap"] == 1


def test_pick_clear_keeps_longest_and_crossing():
    long = _feat(LineString([(0.0, 0.0), (200.0, 0.0)]))
    close = _feat(LineString([(0.0, 10.0), (80.0, 10.0)]))
    cross = _feat(LineString([(100.0, -50.0), (100.0, 50.0)]))
    kept = pick_clear_special_features([close, long, cross], 25.0)
    assert {id(f) for f in kept} == {id(long), id(cross)}
    assert_special_edge_clearance(kept, 25.0)


def test_strict_special_edges_survive_crowding():
    a = PreparedLine(
        coords=np.array([[0.0, 0.0], [100.0, 0.0]]),
        layer="special_edges", band_index=0, strict=True,
    )
    b = PreparedLine(
        coords=np.array([[0.0, 10.0], [100.0, 10.0]]),
        layer="special_edges", band_index=0, strict=True,
    )
    kept, dropped = drop_crowded_segments([a, b], bands=[], min_clearance=25.0)
    assert dropped == 0.0
    assert len(kept) == 2
    assert all(ln.strict for ln in kept)


def test_special_features_merge_into_breaklines():
    feats = [SpecialEdgeFeature(_road_line(), "ROAD", None, 0.0, None, 0)]
    merged = merge_constraint_breaklines(
        special_features_to_breaklines(feats, CRS.from_epsg(6670)),
    )
    assert "special_edges" in merged.layers
    assert len(merged.layers["special_edges"]) == 1


def test_polyline_splits_at_vertices():
    segs = split_line_at_vertices(LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 8.0)]))
    assert len(segs) == 2
    assert segs[0].coords[:] == [(0.0, 0.0), (10.0, 0.0)]
    assert segs[1].coords[:] == [(10.0, 0.0), (10.0, 8.0)]
    assert_special_edge_clearance(
        [SpecialEdgeFeature(s, "ROAD", None, 0.0, None, 0) for s in segs],
        25.0,
    )


def test_load_splits_polyline_and_keeps_per_segment_attrs(tmp_path):
    gpkg = tmp_path / "poly.gpkg"
    gpd.GeoDataFrame(
        {"kind": ["ROAD", "CULVERT"], "zc": [1.0, 2.0], "H": [0.0, 1.5], "B": [0.0, 3.0]},
        geometry=[
            LineString([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]),
            LineString([(0.0, 5.0), (10.0, 5.0)]),
        ],
        crs="EPSG:6670",
    ).to_file(gpkg, driver="GPKG")
    cfg = load_config(_write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        file: poly.gpkg
        match_tol: 1.0
    """))
    feats = load_special_edge_features(cfg, CRS.from_epsg(6670), box(-1, -1, 21, 6))
    assert len(feats) == 3
    roads = [f for f in feats if f.kind == "ROAD"]
    culv = [f for f in feats if f.kind == "CULVERT"]
    assert len(roads) == 2
    assert all(f.zc == pytest.approx(1.0) for f in roads)
    assert len(culv) == 1
    assert culv[0].H == pytest.approx(1.5)
    assert culv[0].zc == pytest.approx(2.0)
    assert culv[0].B == pytest.approx(3.0)


def test_load_road_gpkg_and_match(tmp_path):
    gpkg = tmp_path / "road.gpkg"
    gpd.GeoDataFrame(
        {"zc": [12.5], "H": [1.0], "z_road": [12.5]},
        geometry=[_road_line()],
        crs="EPSG:6670",
    ).to_file(gpkg, driver="GPKG")
    cfg = load_config(_write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        file: road.gpkg
        match_tol: 1.0
    """))
    feats = load_special_edge_features(cfg, CRS.from_epsg(6670), box(0, 0, 20, 10))
    assert len(feats) == 1
    assert feats[0].kind == "ROAD"
    assert feats[0].zc == pytest.approx(12.5)
    assert feats[0].B is None

    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    table = special_edges_for_mesh(cfg, tables, clip_to=box(0, 0, 20, 10))
    assert len(table) == 1
    assert table.kind[0] == KIND_ROAD
    assert table.zc[0] == pytest.approx(12.5)


def test_missing_kind_column_uses_default(tmp_path):
    gpkg = tmp_path / "mixed.gpkg"
    gpd.GeoDataFrame(
        {"name": ["a"]},
        geometry=[_road_line()],
        crs="EPSG:6670",
    ).to_file(gpkg, driver="GPKG")
    cfg = load_config(_write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        file: mixed.gpkg
        default_kind: WALL
    """))
    feats = load_special_edge_features(cfg, CRS.from_epsg(6670), box(0, 0, 20, 10))
    assert len(feats) == 1
    assert feats[0].kind == "WALL"


def test_other_uses_kind_column(tmp_path):
    gpkg = tmp_path / "mixed.gpkg"
    gpd.GeoDataFrame(
        {"kind": ["Q"], "qgroup": [3]},
        geometry=[_road_line()],
        crs="EPSG:6670",
    ).to_file(gpkg, driver="GPKG")
    cfg = load_config(_write_yaml(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      special_edges:
        file: mixed.gpkg
        match_tol: 1.0
    """))
    feats = load_special_edge_features(cfg, CRS.from_epsg(6670), box(0, 0, 20, 10))
    assert len(feats) == 1
    assert feats[0].kind == "Q"
    assert feats[0].qgroup == 3


def test_build_and_write_mesh_bin_writes_strc(tmp_path):
    gpkg = tmp_path / "road.gpkg"
    gpd.GeoDataFrame(
        {"zc": [3.5]},
        geometry=[_road_line()],
        crs="EPSG:6670",
    ).to_file(gpkg, driver="GPKG")
    cfg = Config()
    cfg.base_dir = tmp_path
    cfg.input.special_edges.file = "road.gpkg"
    cfg.input.special_edges.match_tol = 1.0
    cfg.output.cell_bin.enabled = True
    cfg.output.cell_bin.directory = str(tmp_path)
    cfg.output.cell_bin.filename = "mesh.bin"

    path = build_and_write_mesh_bin(
        cfg, _mesh(), _quality(), _terrain(), CRS.from_epsg(6670), _dem(),
    )
    packed = unpack_mesh(path.read_bytes())
    assert len(packed.attrs.special_edges) == 1
    assert packed.attrs.special_edges.kind[0] == KIND_ROAD
    assert packed.attrs.special_edges.zc[0] == pytest.approx(3.5)
