from __future__ import annotations

import textwrap

import pytest

from src.config import ConfigError, load_config


def _write(tmp_path, *bodies: str):
    path = tmp_path / "c.yaml"
    path.write_text("\n".join(textwrap.dedent(b) for b in bodies), encoding="utf-8")
    return path


MINIMAL = """
    input:
      domain: d.gpkg
      dem: d.tif
    """


def test_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.mesh.min_element_area == 625.0
    assert cfg.crs.target_epsg == 6670
    assert cfg.features.reference_layers_for_size_field is False


def test_size_floors_follow_min_area(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.mesh.quad_size_floor == pytest.approx(25.0)
    assert cfg.mesh.triangle_size_floor == pytest.approx(38.0, abs=0.1)


def test_unknown_key_is_rejected(tmp_path):
    path = _write(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      breaklnies: x.gpkg
    """)
    with pytest.raises(ConfigError, match=r"input: 未知のキー \['breaklnies'\]"):
        load_config(path)


def test_reference_layer_cannot_be_declared_as_breakline(tmp_path):
    """OSM 用のキーは reference_layers 側にしか存在しない。"""
    path = _write(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      breaklines:
        osm_roads: r.gpkg
    """)
    with pytest.raises(ConfigError, match=r"input.breaklines: 未知のキー \['osm_roads'\]"):
        load_config(path)


def test_reference_layer_key_is_accepted_in_reference_layers(tmp_path):
    cfg = load_config(_write(tmp_path, """
    input:
      domain: d.gpkg
      dem: d.tif
      reference_layers:
        osm_roads: r.gpkg
    """))
    assert cfg.input.reference_layers.as_mapping() == {"osm_roads": "r.gpkg"}
    assert cfg.input.breaklines.as_mapping() == {}


def test_web_mercator_is_rejected(tmp_path):
    path = _write(tmp_path, MINIMAL, """
    crs:
      target_epsg: 3857
    """)
    with pytest.raises(ConfigError, match="面積評価に使えません"):
        load_config(path)


def test_min_size_below_area_floor_is_rejected(tmp_path):
    path = _write(tmp_path, MINIMAL, """
    mesh:
      global_min_size: 20.0
    """)
    with pytest.raises(ConfigError, match="正三角形の辺長"):
        load_config(path)


def test_band_smaller_than_area_floor_is_rejected(tmp_path):
    path = _write(tmp_path, MINIMAL, """
    mesh:
      boundary_quad_band:
        enabled: true
        width: 10.0
        target_size: 20.0
    """)
    with pytest.raises(ConfigError, match="面積下限"):
        load_config(path)


def test_interface_size_defaults_to_1_5x(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    band = cfg.mesh.boundary_quad_band
    assert band.resolved_interface_size() == pytest.approx(band.target_size * 1.5)


def test_interface_floor_radius_defaults_to_size_field_grid(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    band = cfg.mesh.boundary_quad_band
    assert band.resolved_interface_floor_radius(cfg.mesh.size_field_grid) == pytest.approx(
        cfg.mesh.size_field_grid
    )


def test_gpkg_csv_defaults_and_paths(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL, """
    output:
      gpkg_csv:
        directory: out_solver
        face_gpkg: face.gpkg
        edge_gpkg: edge.gpkg
    """))
    gc = cfg.output.gpkg_csv
    assert gc.enabled is True
    assert gc.face_gpkg == "face.gpkg"
    assert cfg.gpkg_csv_dir == (tmp_path / "out_solver").resolve()
