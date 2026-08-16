from __future__ import annotations

import numpy as np
import pytest

from src.mesh_parser import Mesh, orient_ccw
from src.quality_metrics import polygon_areas, quad_quality, triangle_quality
from src.size_field import SizeField


def test_polygon_area_sign_follows_orientation():
    ccw = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=float)
    assert polygon_areas(ccw)[0] == pytest.approx(100.0)
    assert polygon_areas(ccw[:, ::-1])[0] == pytest.approx(-100.0)


def test_equilateral_triangle_is_ideal():
    s = 50.0
    nodes = np.array([[0, 0], [s, 0], [s / 2, s * np.sqrt(3) / 2]])
    q = triangle_quality(nodes, np.array([[0, 1, 2]]))
    assert q["area"][0] == pytest.approx(np.sqrt(3) / 4 * s**2)
    assert q["min_angle"][0] == pytest.approx(60.0)
    assert q["radius_ratio"][0] == pytest.approx(1.0)
    assert q["aspect_ratio"][0] == pytest.approx(1.0)


def test_square_is_ideal_quad():
    nodes = np.array([[0, 0], [25, 0], [25, 25], [0, 25]], dtype=float)
    q = quad_quality(nodes, np.array([[0, 1, 2, 3]]))
    assert q["area"][0] == pytest.approx(625.0)
    assert q["min_angle"][0] == pytest.approx(90.0)
    assert q["scaled_jacobian"][0] == pytest.approx(1.0)
    assert bool(q["is_convex"][0])


def test_concave_quad_is_detected():
    nodes = np.array([[0, 0], [100, 0], [50, 40], [100, 100]], dtype=float)
    q = quad_quality(nodes, np.array([[0, 1, 2, 3]]))
    assert not bool(q["is_convex"][0])
    assert q["scaled_jacobian"][0] < 0.0
    assert q["max_angle"][0] > 180.0


def test_orient_ccw_flips_clockwise_elements():
    nodes = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    mesh = Mesh(
        nodes=nodes,
        triangles=np.array([[0, 2, 1]]),
        quads=np.array([[0, 3, 2, 1]]),
        tri_surface=np.array([1]),
        quad_surface=np.array([1]),
    )
    orient_ccw(mesh)
    assert polygon_areas(nodes[mesh.triangles])[0] > 0
    assert polygon_areas(nodes[mesh.quads])[0] > 0


def test_size_field_respects_floor():
    sf = SizeField.uniform((0, 0, 1000, 1000), spacing=50.0, value=150.0,
                           floor=45.0, ceiling=150.0)
    sf.apply_min(np.array([500.0]), np.array([500.0]), np.array([5.0]))
    assert sf.values.min() == pytest.approx(45.0)


def test_size_field_smoothing_enforces_lipschitz():
    sf = SizeField.uniform((0, 0, 1000, 1000), spacing=50.0, value=150.0,
                           floor=45.0, ceiling=150.0)
    sf.apply_min(np.array([500.0]), np.array([500.0]), np.array([45.0]))
    sf.smooth(max_ratio=1.5)
    v = sf.values
    slope = 0.5 * 50.0
    assert np.all(np.abs(np.diff(v, axis=0)) <= slope + 1e-6)
    assert np.all(np.abs(np.diff(v, axis=1)) <= slope + 1e-6)


def test_size_field_sampling_round_trips():
    sf = SizeField.uniform((0, 0, 500, 500), spacing=25.0, value=100.0,
                           floor=45.0, ceiling=150.0)
    sf.apply_min(np.array([250.0]), np.array([250.0]), np.array([60.0]))
    assert sf.sample(np.array([250.0]), np.array([250.0]))[0] == pytest.approx(60.0)
    assert sf.sample(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(100.0)
