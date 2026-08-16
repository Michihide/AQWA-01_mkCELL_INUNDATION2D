"""合成データでパイプライン全体を通す。実データは重いので別途 run.py で確認する。"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from scripts.make_synthetic_data import RIDGE_Y
from src.boundary_quad_band import build_polygon_band
from src.config import load_config
from src.geometry_cleaning import clean_polygons
from src.gmsh_geometry import PG_DOMAIN_BOUNDARY, PG_QUAD_BOUNDARY, PG_TRI_INTERIOR
from src.io_raster import load_dem
from src.io_vector import load_domain
from src.main import initial_size_field, run_pipeline
from src.mesh_generator import generate_mesh, gmsh_session
from src.mesh_repair import repair_mesh
from src.quality_metrics import evaluate_quality
from src.terrain_metrics import compute_terrain_metrics


def _config(tmp_path, data_dir, **overrides) -> str:
    body = f"""
    input:
      domain: {data_dir / 'synthetic_domain.gpkg'}
      domain_min_area: 1000.0
      polygon_filter: [0]
      dem: {data_dir / 'synthetic_dem.tif'}
    mesh:
      max_iterations: {overrides.get('max_iterations', 1)}
    terrain:
      plane_fit_rmse_max: {overrides.get('plane_fit_rmse_max', 100.0)}
      slope_direction_spread_max_deg: {overrides.get('spread', 180.0)}
    output:
      directory: {tmp_path / 'out'}
      basename: test_mesh
      save_diagnostics: false
      save_iteration_meshes: false
    """
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


@pytest.fixture
def rect_setup(tmp_path, synthetic_data):
    cfg = load_config(_config(tmp_path, synthetic_data))
    polys, crs = load_domain(cfg)
    polys = clean_polygons(polys, cfg.input.domain_min_area,
                           narrow_feature_radius=cfg.mesh.resolved_narrow_feature_radius())
    b = cfg.mesh.boundary_quad_band
    bands = [
        build_polygon_band(p, b.width, b.target_size,
                           min_element_area=cfg.mesh.min_element_area)
        for p in polys
    ]
    return cfg, crs, polys, bands


def test_domain_is_reprojected_to_metric_crs(rect_setup):
    _, crs, polys, _ = rect_setup
    assert crs.to_epsg() == 6670
    assert not crs.is_geographic
    assert polys[0].area == pytest.approx(1200.0 * 800.0, rel=0.01)


def test_initial_size_field_stays_coarse_away_from_boundary(rect_setup):
    """Γ1 近傍だけ下限を掛け、内部は global_max_size のまま始める。"""
    cfg, _, polys, bands = rect_setup
    sf = initial_size_field(cfg, bands)
    centroid = np.array([polys[0].centroid.coords[0]])
    assert sf.sample(centroid[:, 0], centroid[:, 1])[0] == pytest.approx(
        cfg.mesh.global_max_size, rel=1e-6
    )


def test_the_boundary_is_mostly_quads(rect_setup):
    """境界は四角形が原則。置けない区間だけ三角形に譲る。"""
    cfg, _, polys, bands = rect_setup
    sf = initial_size_field(cfg, bands)
    with gmsh_session(cfg):
        mesh, tags, groups = generate_mesh(cfg, bands, sf)
        assert PG_QUAD_BOUNDARY in groups
        assert PG_TRI_INTERIOR in groups
        assert PG_DOMAIN_BOUNDARY in groups
        band_surfaces = set(tags.band_surfaces)

    ring = bands[0].exterior
    tri_segments = set(ring.skipped) | set(ring.closing_indices())
    assert len(tri_segments) / ring.n < 0.2  # 譲るのは一部だけ

    boundary = polys[0].exterior
    on_boundary = np.array([
        boundary.distance(Point(*p)) < 1e-6 for p in mesh.nodes
    ])
    # 境界に接する三角形は、四角形を置けなかった区間の上にしかない
    tri_edges = MultiLineString([
        [ring.outer[i], ring.outer[(i + 1) % ring.n]] for i in tri_segments
    ])
    for tri in mesh.triangles:
        touching = mesh.nodes[tri][on_boundary[tri]]
        for p in touching:
            assert tri_edges.distance(Point(*p)) < 1e-6

    assert set(mesh.quad_surface.tolist()) <= band_surfaces


def test_the_boundary_shape_is_not_altered(rect_setup):
    """四角形を置くために境界の頂点を落とさない。長方形の角は角のまま。"""
    _, _, polys, bands = rect_setup
    band = bands[0]

    assert band.corner_cuts == 0
    outline = LineString(np.vstack([band.exterior.outer, band.exterior.outer[:1]]))
    assert outline.hausdorff_distance(polys[0].exterior) < 1e-6
    for corner in polys[0].exterior.coords:
        assert outline.distance(Point(*corner)) < 1e-6


def test_all_elements_meet_area_floor(rect_setup):
    cfg, _, _, bands = rect_setup
    sf = initial_size_field(cfg, bands)
    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf)
        mesh, _ = repair_mesh(mesh, cfg)
        report = evaluate_quality(mesh)
    assert report.area.min() >= cfg.mesh.min_element_area


def test_all_quads_are_convex(rect_setup):
    cfg, _, _, bands = rect_setup
    sf = initial_size_field(cfg, bands)
    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf)
        mesh, _ = repair_mesh(mesh, cfg)
        report = evaluate_quality(mesh)
    assert report.is_convex.all()


def test_mesh_covers_the_domain(rect_setup):
    cfg, _, polys, bands = rect_setup
    sf = initial_size_field(cfg, bands)
    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf)
        report = evaluate_quality(mesh)
    assert report.area.sum() == pytest.approx(polys[0].area, rel=0.01)


def test_terrain_metrics_detect_the_synthetic_ridge(rect_setup):
    cfg, crs, polys, bands = rect_setup
    dem = load_dem(cfg, crs, polys[0].bounds)
    sf = initial_size_field(cfg, bands)
    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf)
    terrain = compute_terrain_metrics(mesh, dem, cfg)

    assert (terrain.n_samples > 0).all()
    # 堤防をまたぐ要素は、平面で近似できないため RMSE が大きい
    spans_ridge = np.array([
        p[:, 1].min() <= RIDGE_Y <= p[:, 1].max() for p in mesh.element_polygons()
    ])
    assert spans_ridge.any()
    assert (terrain.plane_fit_rmse[spans_ridge].mean()
            > 5.0 * terrain.plane_fit_rmse[~spans_ridge].mean())


def test_pipeline_writes_expected_outputs(tmp_path, synthetic_data):
    cfg = load_config(_config(tmp_path, synthetic_data))
    summary = run_pipeline(cfg)
    out = tmp_path / "out"
    for name in ("test_mesh.vtu", "test_mesh.msh", "test_mesh_elements.gpkg",
                 "test_mesh_elements.csv", "test_mesh_nodes.csv", "summary.json"):
        assert (out / name).exists(), name
    assert summary["viol_area_below_min"] == 0
    assert summary["n_quads"] > 0
    assert summary["n_triangles"] > 0
