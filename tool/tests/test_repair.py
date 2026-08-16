from __future__ import annotations

import numpy as np
import pytest

from src.mesh_parser import Mesh, orient_ccw
from src.mesh_repair import collapse_small_triangles, merge_small_triangles
from src.quality_metrics import polygon_areas

MIN_AREA = 625.0

# 幅 4 m の細い三角形 [6,7,4] が四角形 3 枚と三角形 1 枚に囲まれた配置。
# 統合相手の三角形 [6,4,5] と組んでも凹四角形になるため、辺の縮約でしか消せない。
PINCHED_NODES = [
    [0, 0], [100, 0], [200, 0], [200, 100], [100, 100], [0, 100],
    [98, 48], [102, 48],
]
PINCHED_QUADS = [[0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7]]
PINCHED_TRIS = [[6, 7, 4], [6, 4, 5]]


def _mesh(nodes, triangles, quads=None) -> Mesh:
    quads = [] if quads is None else quads
    mesh = Mesh(
        nodes=np.asarray(nodes, dtype=float),
        triangles=np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
        quads=np.asarray(quads, dtype=np.int64).reshape(-1, 4),
        tri_surface=np.ones(len(triangles), dtype=np.int64),
        quad_surface=np.ones(len(quads), dtype=np.int64),
    )
    orient_ccw(mesh)
    return mesh


def _pinched_mesh() -> Mesh:
    return _mesh(PINCHED_NODES, PINCHED_TRIS, PINCHED_QUADS)


def _ignore_boundary_exemption(monkeypatch) -> None:
    """境界に接する要素の面積下限除外を無効化する。

    ここでのテストは統合・縮約そのものの幾何ロジックを検証するためのもので、
    外形線への接触とは無関係。テスト用の小さな単独メッシュはノードがすべて
    外形線上に乗ってしまうため、除外ロジックを無効化して従来どおり検証する。
    """
    monkeypatch.setattr(
        Mesh, "boundary_touching_element_mask",
        lambda self: np.zeros(self.n_elements, dtype=bool),
    )


def test_two_small_triangles_merge_into_one_quad(monkeypatch):
    # 面積 450 m2 の直角三角形 2 枚。単独では下限割れだが、合わせれば正方形になる
    _ignore_boundary_exemption(monkeypatch)
    nodes = [[0, 0], [30, 0], [30, 30], [0, 30]]
    mesh = _mesh(nodes, [[0, 1, 2], [0, 2, 3]])
    merged, stats = merge_small_triangles(
        mesh, min_area=MIN_AREA, max_aspect_ratio=4.0,
        min_interior_angle_deg=45.0, max_interior_angle_deg=135.0,
        min_scaled_jacobian=0.3,
    )
    assert stats.merged_pairs == 1
    assert len(merged.triangles) == 0
    assert polygon_areas(merged.nodes[merged.quads])[0] == pytest.approx(900.0)


def test_merge_is_rejected_when_result_would_be_concave(monkeypatch):
    # 共有辺の向かい合う頂点が内側に入り込み、統合すると凹四角形になる配置
    _ignore_boundary_exemption(monkeypatch)
    nodes = [[0, 0], [100, 0], [50, 40], [50, 10]]
    mesh = _mesh(nodes, [[0, 1, 2], [0, 3, 1]])
    _, stats = merge_small_triangles(
        mesh, min_area=1e9, max_aspect_ratio=4.0,
        min_interior_angle_deg=45.0, max_interior_angle_deg=135.0,
        min_scaled_jacobian=0.3,
    )
    assert stats.merged_pairs == 0
    assert stats.rejected_nonconvex > 0


def test_collapse_removes_a_triangle_that_cannot_be_merged():
    mesh = _pinched_mesh()
    assert (polygon_areas(mesh.nodes[mesh.triangles]) < MIN_AREA).sum() == 1

    repaired, stats = collapse_small_triangles(mesh, min_area=MIN_AREA)
    assert stats.collapsed_edges == 1
    assert stats.remaining_undersized == 0
    assert repaired.n_elements < mesh.n_elements


def test_collapse_preserves_total_area():
    mesh = _pinched_mesh()
    before = polygon_areas(mesh.nodes[mesh.triangles]).sum() + polygon_areas(mesh.nodes[mesh.quads]).sum()

    repaired, _ = collapse_small_triangles(mesh, min_area=MIN_AREA)
    after = polygon_areas(repaired.nodes[repaired.triangles]).sum()
    if len(repaired.quads):
        after += polygon_areas(repaired.nodes[repaired.quads]).sum()
    assert after == pytest.approx(before)


def test_collapse_keeps_the_outer_boundary_fixed():
    mesh = _pinched_mesh()
    repaired, _ = collapse_small_triangles(mesh, min_area=MIN_AREA)

    used = np.unique(np.concatenate([repaired.triangles.ravel(), repaired.quads.ravel()]))
    kept = repaired.nodes[used]
    for corner in ([0, 0], [200, 0], [200, 100], [0, 100]):
        assert np.isclose(kept, corner).all(axis=1).any()


def test_collapse_never_inverts_elements():
    repaired, _ = collapse_small_triangles(_pinched_mesh(), min_area=MIN_AREA)
    for elems in (repaired.triangles, repaired.quads):
        if len(elems):
            assert (polygon_areas(repaired.nodes[elems]) > 0).all()
