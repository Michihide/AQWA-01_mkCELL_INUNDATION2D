"""拘束ブレークラインが実際にメッシュ辺として再現されるかを確かめる。"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from scripts.make_synthetic_data import ORIGIN_X, RIDGE_Y
from src.boundary_quad_band import build_polygon_band
from src.breaklines import (
    PreparedLine,
    _resample,
    _tidy,
    breakline_coverage,
    drop_crowded_segments,
    prepare_breaklines,
    trim_close_endpoints,
)
from src.config import BreaklineProcessingConfig, load_config
from src.geometry_cleaning import clean_polygons
from src.io_vector import load_constraint_breaklines, load_domain
from src.main import initial_size_field, run_pipeline
from src.mesh_generator import generate_mesh, gmsh_session
from src.mesh_parser import Mesh
from src.mesh_repair import repair_mesh
from src.quality_metrics import evaluate_quality


def _config(tmp_path, data_dir, **overrides) -> str:
    body = f"""
    input:
      domain: {data_dir / 'synthetic_domain.gpkg'}
      domain_min_area: 1000.0
      polygon_filter: [0]
      dem: {data_dir / 'synthetic_dem.tif'}
      breaklines:
        embankments: {data_dir / 'synthetic_breaklines.gpkg'}
    mesh:
      max_iterations: {overrides.get('max_iterations', 1)}
      boundary_quad_band:
        enabled: true
    terrain:
      plane_fit_rmse_max: 100.0
      slope_direction_spread_max_deg: 180.0
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
def setup(tmp_path, synthetic_data):
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
    lines = load_constraint_breaklines(cfg, crs, polys[0])
    return cfg, crs, polys, bands, lines


def test_resample_drops_dense_vertices_but_keeps_ends():
    coords = np.stack([np.linspace(0.0, 100.0, 101), np.zeros(101)], axis=1)
    out = _resample(coords, min_spacing=45.0)
    assert out[0] == pytest.approx(coords[0])
    assert out[-1] == pytest.approx(coords[-1])
    gaps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert (gaps >= 45.0).all()


def test_resample_never_leaves_a_short_final_segment():
    # 終点が直前の頂点のすぐ手前にある。終点は落とせないので手前を落とすしかない
    coords = np.array([[0.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
    out = _resample(coords, min_spacing=45.0)
    gaps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert (gaps >= 45.0).all()
    assert out[-1] == pytest.approx(coords[-1])


def _line(coords, layer="embankments") -> PreparedLine:
    return PreparedLine(coords=np.asarray(coords, dtype=float), layer=layer, band_index=0)


def test_parallel_lines_that_are_too_close_lose_the_crowded_stretch():
    """面積下限を優先する。間に薄い要素を強制する区間は拘束をやめる。"""
    a = _line([[0.0, 0.0], [600.0, 0.0]])
    b = _line([[0.0, 15.0], [600.0, 15.0]])
    kept, dropped = drop_crowded_segments([a, b], bands=[], min_clearance=28.0)

    assert dropped == pytest.approx(1200.0, rel=0.05)  # 2 本とも全長が落ちる
    assert sum(ln.length for ln in kept) == pytest.approx(0.0, abs=1.0)


def test_lines_far_enough_apart_are_kept_whole():
    a = _line([[0.0, 0.0], [600.0, 0.0]])
    b = _line([[0.0, 60.0], [600.0, 60.0]])
    kept, dropped = drop_crowded_segments([a, b], bands=[], min_clearance=28.0)

    assert dropped == 0.0
    assert len(kept) == 2


def test_a_right_angle_junction_survives():
    """繋がっている線は接続点で接するが、それは近接ではない。

    noding 後の形、つまり T 字の交点で 3 本に分かれた状態で与える。
    """
    left = _line([[0.0, 0.0], [300.0, 0.0]])
    right = _line([[300.0, 0.0], [600.0, 0.0]])
    stem = _line([[300.0, 0.0], [300.0, 400.0]])
    kept, dropped = drop_crowded_segments(
        [left, right, stem], bands=[], min_clearance=28.0
    )

    assert dropped == 0.0
    assert len(kept) == 3


def test_a_shallow_junction_loses_the_stretch_where_the_two_stay_close():
    """浅い角度で分かれる 2 本は、接続点の先も近いままで薄い要素を作る。"""
    a = _line([[0.0, 0.0], [600.0, 0.0]])
    b = _line([[0.0, 0.0], [600.0, 60.0]])  # 約 5.7 度
    kept, dropped = drop_crowded_segments([a, b], bands=[], min_clearance=28.0)

    # 間隔 28 m に開くのは x = 280 m あたり。それより手前が落ちる
    assert dropped > 400.0
    for ln in kept:
        assert np.linalg.norm(ln.coords[0]) > 100.0


def test_an_endpoint_stopping_short_of_another_line_is_pulled_back():
    a = _line([[0.0, 0.0], [600.0, 0.0]])
    b = _line([[300.0, 17.0], [300.0, 400.0]])  # a の 17 m 手前で終わっている
    kept, trimmed = trim_close_endpoints([a, b], min_clearance=28.0)

    assert trimmed > 0.0
    moved = [ln for ln in kept if ln.coords[0][1] > 17.5]
    assert len(moved) == 1
    assert moved[0].coords[0][1] >= 28.0


def test_endpoints_shared_at_a_junction_are_left_alone():
    """noding 済みの接続点まで後退させてしまうと、網の形が崩れる。"""
    a = _line([[0.0, 0.0], [300.0, 0.0]])
    b = _line([[300.0, 0.0], [300.0, 400.0]])
    kept, trimmed = trim_close_endpoints([a, b], min_clearance=28.0)

    assert trimmed == 0.0
    assert len(kept) == 2


def test_cutting_a_line_never_leaves_a_short_segment():
    """近接区間を切り落とした跡には、元の頂点との間に短い線分が残る。

    そのまま gmsh へ渡すと、そこだけ辺が短くなって面積下限を割る。
    """
    cut = _line([[0.0, 0.0], [3.0, 0.0], [48.0, 0.0], [93.0, 0.0], [200.0, 0.0]])
    tidied = _tidy([cut], min_spacing=45.0, min_length=90.0)

    assert len(tidied) == 1
    gaps = np.linalg.norm(np.diff(tidied[0].coords, axis=0), axis=1)
    assert (gaps >= 45.0 - 1e-9).all()


def test_a_line_too_short_after_cutting_is_dropped():
    tidied = _tidy([_line([[0.0, 0.0], [40.0, 0.0]])], min_spacing=45.0, min_length=90.0)
    assert tidied == []


def test_size_floor_is_derived_from_the_area_lower_bound():
    bp = BreaklineProcessingConfig()
    floor = bp.resolved_size_floor(min_element_area=625.0, global_min_size=45.0)
    # 最悪 0.29*s^2 の三角形ができるので、それが 625 m^2 を上回ること
    assert 0.29 * floor**2 > 625.0
    assert floor > 45.0


def test_size_floor_never_goes_below_the_global_minimum():
    bp = BreaklineProcessingConfig()
    assert bp.resolved_size_floor(100.0, 45.0) == 45.0


def test_size_field_floor_is_raised_along_the_breakline(setup):
    cfg, _, _, bands, lines = setup
    prepared = prepare_breaklines(lines, bands, cfg)
    expected = cfg.mesh.breakline_processing.resolved_size_floor(
        cfg.mesh.min_element_area, cfg.mesh.global_min_size
    )

    plain = initial_size_field(cfg, bands)
    with_lines = initial_size_field(cfg, bands, prepared)

    xy = prepared.lines[0].coords
    iy, ix = with_lines._indices(xy[:, 0], xy[:, 1])
    assert (with_lines.floor_grid[iy, ix] >= expected - 1e-6).all()
    assert with_lines.floor_grid.max() >= plain.floor_grid.max()


def test_prepared_lines_stay_inside_the_interior_region(setup):
    cfg, _, _, bands, lines = setup
    prepared = prepare_breaklines(lines, bands, cfg)
    assert not prepared.is_empty()

    margin = cfg.mesh.breakline_processing.resolved_interior_margin(cfg.mesh.global_min_size)
    region = bands[0].interior.buffer(-margin + 1e-6)
    for ln in prepared.lines:
        assert region.contains(LineString(ln.coords))


def test_prepared_vertices_respect_the_minimum_spacing(setup):
    cfg, _, _, bands, lines = setup
    prepared = prepare_breaklines(lines, bands, cfg)
    for ln in prepared.lines:
        gaps = np.linalg.norm(np.diff(ln.coords, axis=0), axis=1)
        assert (gaps >= cfg.mesh.global_min_size - 1e-6).all()


def test_breakline_becomes_mesh_edges(setup):
    cfg, _, _, bands, lines = setup
    prepared = prepare_breaklines(lines, bands, cfg)
    sf = initial_size_field(cfg, bands)

    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf, prepared)

    assert len(mesh.constrained_edges) > 0
    coverage = breakline_coverage(mesh, prepared)
    assert coverage["breakline_coverage"] == pytest.approx(1.0, abs=1e-3)


def test_constrained_nodes_lie_on_the_breakline(setup):
    cfg, _, _, bands, lines = setup
    prepared = prepare_breaklines(lines, bands, cfg)
    sf = initial_size_field(cfg, bands)

    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf, prepared)

    line = LineString(prepared.lines[0].coords)
    for idx in np.flatnonzero(mesh.constrained_node_mask()):
        assert line.distance(Point(*mesh.nodes[idx])) < 1e-6


def test_repair_preserves_the_breakline(setup):
    cfg, _, _, bands, lines = setup
    prepared = prepare_breaklines(lines, bands, cfg)
    sf = initial_size_field(cfg, bands)

    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf, prepared)
        before = mesh.constrained_edge_set()
        mesh, _ = repair_mesh(mesh, cfg)

    # 修復後も拘束辺はすべて実在する要素辺のままであること
    edges = set()
    for elems in (mesh.triangles, mesh.quads):
        for elem in elems:
            m = len(elem)
            for k in range(m):
                a, b = int(elem[k]), int(elem[(k + 1) % m])
                edges.add((a, b) if a < b else (b, a))
    assert before <= edges

    assert evaluate_quality(mesh).area.min() >= cfg.mesh.min_element_area
    assert breakline_coverage(mesh, prepared)["breakline_coverage"] == pytest.approx(1.0, abs=1e-3)


def test_a_trapped_small_triangle_gives_up_its_constraint():
    """統合も縮約も拘束辺に塞がれたとき、面積下限を優先して拘束を外す。

    細い三角形の 3 辺すべてが拘束辺という状況を、解析領域の内部に作る
    （外形線には接しない）。境界に接する要素は面積下限の対象外になったため、
    この違反が本当に拘束辺（内部ブレークライン）由来であることをはっきり
    させるために、外側をもう一巡り三角形で囲んで外形線から切り離す。
    拘束を守る限りどう動かしても面積下限を満たせない。
    """
    nodes = np.array([
        [0.0, 0.0], [40.0, 0.0], [80.0, 0.0], [120.0, 0.0],
        [0.0, 40.0], [40.0, 8.0], [80.0, 40.0], [120.0, 40.0],
        [60.0, -500.0],  # 外側を一巡りする扇の頂点。外形線をここへ追いやる。
    ])
    inner_tris = [[0, 1, 5], [1, 2, 5], [0, 5, 4], [4, 5, 6], [2, 6, 5],
                  [2, 3, 6], [3, 7, 6]]
    # 内側パッチの外周（0-1-2-3-7-6-4-0）を、外側の 1 点 8 との扇三角形で覆う。
    # こうすると外周の各辺が扇三角形と共有され、パッチのどの三角形も外形線に
    # 接しなくなる（外形線は扇の外側の辺だけになる）。
    outer_loop = [0, 1, 2, 3, 7, 6, 4]
    fan_tris = [
        [outer_loop[i], outer_loop[(i + 1) % len(outer_loop)], 8]
        for i in range(len(outer_loop))
    ]
    tris = np.array(inner_tris + fan_tris)
    mesh = Mesh(
        nodes=nodes,
        triangles=tris,
        quads=np.zeros((0, 4), dtype=np.int64),
        tri_surface=np.zeros(len(tris), dtype=np.int64),
        quad_surface=np.zeros(0, dtype=np.int64),
        constrained_edges=np.array([[0, 5], [1, 5], [2, 5], [5, 6]]),
    )
    assert not mesh.boundary_touching_element_mask()[: len(inner_tris)].any()
    cfg = _minimal_cfg()

    repaired, stats = repair_mesh(mesh, cfg)

    assert evaluate_quality(repaired).area.min() >= cfg.mesh.min_element_area
    assert stats.sacrificed_constraints > 0


def test_a_small_triangle_boxed_in_by_quads_is_freed():
    """四角形だけに囲まれた小三角形は、隣を三角形へ戻せば組み直せる。

    ただし戻した先の 2 枚とも面積下限を満たせる対角線があることが前提。
    """
    nodes = np.array([
        [0.0, 0.0], [60.0, 0.0], [120.0, 0.0], [260.0, 0.0],
        [70.0, 20.0], [110.0, 20.0],
        [0.0, 80.0], [60.0, 80.0], [120.0, 80.0], [260.0, 80.0],
    ])
    # 中央の細い三角形 (1,2,5)-(1,5,4) ではなく 1 枚だけ小さくする
    tris = np.array([[1, 2, 5]])
    quads = np.array([
        [0, 1, 4, 6], [6, 4, 5, 7], [7, 5, 8, 9], [2, 3, 8, 5],
    ])
    mesh = Mesh(
        nodes=nodes,
        triangles=tris,
        quads=quads,
        tri_surface=np.zeros(1, dtype=np.int64),
        quad_surface=np.zeros(4, dtype=np.int64),
    )
    from src.mesh_repair import split_quads_around_small_triangles

    u, v = nodes[2] - nodes[1], nodes[5] - nodes[1]
    assert abs(u[0] * v[1] - u[1] * v[0]) / 2 < 625.0

    out, split = split_quads_around_small_triangles(mesh, min_area=625.0)
    assert split == 1
    assert len(out.quads) == 3
    assert len(out.triangles) == 3


def test_a_quad_split_that_would_create_a_new_area_violation_is_refused():
    """割った先のどちらの対角線でも半分が面積下限を割るなら、割らない。

    割っても違反が別の場所に移るだけなら意味が無い。元の小三角形は残るが、
    新しい違反は作らない。
    """
    nodes = np.array([
        [0.0, 0.0], [60.0, 0.0], [120.0, 0.0], [180.0, 0.0],
        [70.0, 20.0], [110.0, 20.0],
        [0.0, 80.0], [60.0, 80.0], [120.0, 80.0], [180.0, 80.0],
    ])
    tris = np.array([[1, 2, 5]])
    quads = np.array([
        [0, 1, 4, 6], [6, 4, 5, 7], [7, 5, 8, 9], [2, 3, 8, 5],
    ])
    mesh = Mesh(
        nodes=nodes,
        triangles=tris,
        quads=quads,
        tri_surface=np.zeros(1, dtype=np.int64),
        quad_surface=np.zeros(4, dtype=np.int64),
    )
    from src.mesh_repair import split_quads_around_small_triangles

    out, split = split_quads_around_small_triangles(mesh, min_area=625.0)
    assert split == 0
    assert len(out.quads) == 4
    assert len(out.triangles) == 1


def test_the_boundary_quad_band_is_never_split():
    """外周帯は Transfinite で構造が決まっているので割ってはいけない。"""
    nodes = np.array([
        [0.0, 0.0], [60.0, 0.0], [120.0, 0.0],
        [70.0, 20.0], [0.0, 80.0], [120.0, 80.0],
    ])
    mesh = Mesh(
        nodes=nodes,
        triangles=np.array([[0, 1, 3]]),
        quads=np.array([[1, 2, 5, 3]]),
        tri_surface=np.zeros(1, dtype=np.int64),
        quad_surface=np.array([7]),
        surface_roles={7: "quad_boundary"},
    )
    from src.mesh_repair import split_quads_around_small_triangles

    out, split = split_quads_around_small_triangles(mesh, min_area=625.0)
    assert split == 0
    assert len(out.quads) == 1


def _minimal_cfg():
    from src.config import Config

    cfg = Config()
    cfg.mesh.min_element_area = 625.0
    return cfg


def test_pipeline_reports_breakline_coverage(tmp_path, synthetic_data):
    cfg = load_config(_config(tmp_path, synthetic_data))
    summary = run_pipeline(cfg)

    assert summary["breakline_count"] >= 1
    assert summary["breakline_coverage"] == pytest.approx(1.0, abs=1e-3)
    assert summary["viol_area_below_min"] == 0
    assert (tmp_path / "out" / "test_mesh_breaklines.gpkg").exists()


def test_breakline_forces_nodes_along_the_ridge(tmp_path, synthetic_data):
    """堤防天端の線上に節点が並ぶこと。拘束が効いていることの直接の確認。"""
    cfg = load_config(_config(tmp_path, synthetic_data))
    polys, crs = load_domain(cfg)
    polys = clean_polygons(polys, cfg.input.domain_min_area,
                           narrow_feature_radius=cfg.mesh.resolved_narrow_feature_radius())
    b = cfg.mesh.boundary_quad_band
    bands = [build_polygon_band(p, b.width, b.target_size,
                                min_element_area=cfg.mesh.min_element_area) for p in polys]
    lines = load_constraint_breaklines(cfg, crs, polys[0])
    prepared = prepare_breaklines(lines, bands, cfg)
    sf = initial_size_field(cfg, bands)

    with gmsh_session(cfg):
        mesh, _, _ = generate_mesh(cfg, bands, sf, prepared)

    on_ridge = np.abs(mesh.nodes[:, 1] - RIDGE_Y) < 1e-6
    assert on_ridge.sum() >= 5
    xs = np.sort(mesh.nodes[on_ridge, 0])
    assert xs.min() < ORIGIN_X + 200.0
    assert xs.max() > ORIGIN_X + 900.0
