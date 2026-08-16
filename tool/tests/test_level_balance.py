"""隣り合う要素の大きさの段差。

要素ごとに時間刻みを変える解法では、この段差がそのまま dt の段差になる。
辺長比を 2 以下に抑えれば、「1 段で辺長が半分」のレベル分けで隣どうしの差が
必ず 1 以内になる。
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest

from src.adaptive_refinement import has_converged, refine_size_field
from src.config import ConfigError, load_config
from src.mesh_parser import Mesh
from src.mesh_repair import _bisect_pass, split_band_quads_for_ratio, split_coarse_triangles
from src.quality_metrics import (
    element_adjacency,
    element_size_estimate,
    evaluate_quality,
    neighbor_ratio_stats,
    neighbor_ratio_violations,
)
from src.size_field import SizeField


def _grid_mesh(sizes: list[float], width: float = 40.0) -> Mesh:
    """指定の代表辺長をもつ四角形を縦に積んだメッシュ。

    辺を共有する以上、共有辺の長さは共通になる。奥行きで大きさを変えるので、
    代表辺長 s の要素は幅 width・奥行き s^2/width になる。
    """
    nodes = [[0.0, 0.0], [width, 0.0]]
    left, right = 0, 1
    quads = []
    y = 0.0
    for s in sizes:
        y += s * s / width
        nodes += [[width, y], [0.0, y]]
        new_right, new_left = len(nodes) - 2, len(nodes) - 1
        quads.append([left, right, new_right, new_left])
        left, right = new_left, new_right
    return Mesh(
        nodes=np.array(nodes),
        triangles=np.zeros((0, 3), dtype=np.int64),
        quads=np.array(quads),
        tri_surface=np.zeros(0, dtype=np.int64),
        quad_surface=np.zeros(len(quads), dtype=np.int64),
    )


def test_only_edge_sharing_elements_count_as_neighbours():
    """頂点だけを共有する要素は隣ではない。フラックスをやり取りしないため。"""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    mesh = Mesh(
        nodes=nodes,
        triangles=np.array([[0, 1, 2], [1, 3, 4]]),
        quads=np.zeros((0, 4), dtype=np.int64),
        tri_surface=np.zeros(2, dtype=np.int64),
        quad_surface=np.zeros(0, dtype=np.int64),
    )
    assert len(element_adjacency(mesh)) == 0


def test_size_uses_edge_length_not_area():
    """同じ辺長でも四角形の面積は正三角形の 2.3 倍あるので、面積では比べない。"""
    h = 50.0
    nodes = np.array([
        [0.0, 0.0], [h, 0.0], [h, h], [0.0, h],
        [h, 0.0], [2 * h, 0.0], [1.5 * h, h * np.sqrt(3) / 2],
    ])
    mesh = Mesh(
        nodes=nodes,
        triangles=np.array([[4, 5, 6]]),
        quads=np.array([[0, 1, 2, 3]]),
        tri_surface=np.zeros(1, dtype=np.int64),
        quad_surface=np.zeros(1, dtype=np.int64),
    )
    size = element_size_estimate(evaluate_quality(mesh))
    assert size[0] == pytest.approx(h, rel=1e-6)  # 正三角形
    assert size[1] == pytest.approx(h, rel=1e-6)  # 正方形


def test_a_ratio_within_the_limit_is_not_flagged():
    mesh = _grid_mesh([50.0, 90.0])
    coarse, target = neighbor_ratio_violations(mesh, evaluate_quality(mesh), 2.0)
    assert not coarse.any()
    assert np.isinf(target).all()


def test_the_coarse_side_is_flagged_with_a_target():
    mesh = _grid_mesh([40.0, 120.0])
    coarse, target = neighbor_ratio_violations(mesh, evaluate_quality(mesh), 2.0)

    assert coarse.tolist() == [False, True]
    assert target[1] == pytest.approx(80.0)  # 細かい側 40 m の 2 倍
    assert np.isinf(target[0])


def test_the_strictest_neighbour_wins():
    """複数の相手と接するときは、最も細かい相手に合わせる。"""
    mesh = _grid_mesh([30.0, 200.0, 50.0])
    coarse, target = neighbor_ratio_violations(mesh, evaluate_quality(mesh), 2.0)

    assert coarse.tolist() == [False, True, False]
    assert target[1] == pytest.approx(60.0)  # 30 m の 2 倍。50 m 側ではない


def test_stats_report_the_worst_ratio():
    mesh = _grid_mesh([40.0, 120.0])
    stats = neighbor_ratio_stats(mesh, evaluate_quality(mesh), 2.0)

    assert stats["neighbor_pairs"] == 1
    assert stats["neighbor_ratio_max"] == pytest.approx(3.0)
    assert stats["viol_neighbor_ratio"] == 1


def test_the_criterion_can_be_switched_off():
    mesh = _grid_mesh([40.0, 400.0])
    coarse, _ = neighbor_ratio_violations(mesh, evaluate_quality(mesh), 0.0)
    assert not coarse.any()


def test_a_ratio_of_one_or_less_is_rejected(tmp_path):
    body = """
    input:
      domain: /tmp/none.gpkg
      dem: /tmp/none.tif
    mesh:
      max_neighbor_element_ratio: 1.0
    """
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_neighbor_element_ratio"):
        load_config(path)


def test_refinement_shrinks_the_coarse_side(tmp_path):
    """細かい側は動かさない。粗くすると地形基準を満たさなくなるため。"""
    mesh = _grid_mesh([40.0, 160.0])
    quality = evaluate_quality(mesh)
    cfg = _cfg(tmp_path)

    sf = SizeField.uniform(
        bounds=(-100.0, -100.0, 500.0, 1000.0),
        spacing=20.0, value=160.0, floor=25.0, ceiling=200.0,
    )
    refine_size_field(sf, mesh, quality, None, cfg)

    centroids = mesh.element_centroids()
    got = sf.sample(centroids[:, 0], centroids[:, 1])
    assert got[1] <= 80.0 + 1e-6
    assert got[0] >= 40.0  # 細かい側は縮められていない


def test_the_boundary_band_is_not_expected_to_shrink(tmp_path):
    """帯は Transfinite で分割数が決まっており、サイズ場では縮まない。

    それを違反として数え続けると反復が終わらないので、解消不能に分類する。
    """
    cfg = _cfg(tmp_path)

    interior = _grid_mesh([40.0, 160.0])
    _, counts = has_converged(interior, evaluate_quality(interior), None, cfg)
    assert counts["viol_neighbor_ratio"] == 1
    assert counts["blocked_by_area_floor"] == 0  # 内部なら縮められる

    band = _grid_mesh([40.0, 160.0])
    band.quad_surface = np.array([0, 9])
    band.surface_roles = {9: "quad_boundary"}
    _, counts = has_converged(band, evaluate_quality(band), None, cfg)
    assert counts["viol_neighbor_ratio"] == 1
    assert counts["blocked_by_area_floor"] == 1  # 帯なので諦める


def _mesh_with_interior_triangle() -> Mesh:
    """矩形の外形線の内側に、外形線に接しない三角形が 1 枚だけあるメッシュ。

    三角形 [6, 7, 4] はどの辺も外形線の両端点を結ばない（内部）。
    三角形 [6, 4, 5] は辺 (4, 5) がそのまま外形線に一致する（境界に接する）。
    """
    nodes = np.array([
        [0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [200.0, 100.0],
        [100.0, 100.0], [0.0, 100.0], [98.0, 48.0], [102.0, 48.0],
    ])
    quads = np.array([[0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 4, 7]])
    tris = np.array([[6, 7, 4], [6, 4, 5]])
    return Mesh(
        nodes=nodes, triangles=tris, quads=quads,
        tri_surface=np.zeros(len(tris), dtype=np.int64),
        quad_surface=np.zeros(len(quads), dtype=np.int64),
    )


def test_boundary_node_mask_identifies_the_outer_ring_only():
    """外形線上の節点だけが対象。内部の 2 頂点は含まれない。"""
    mesh = _mesh_with_interior_triangle()
    mask = mesh.boundary_node_mask()
    assert mask[:6].all()  # 矩形の外周 6 頂点
    assert not mask[6] and not mask[7]  # 内部の 2 頂点


def test_boundary_touching_element_mask_spares_a_genuinely_interior_triangle():
    """外形線に辺を持たない三角形は対象外。頂点が 1 つ触れるだけでは対象にしない。"""
    mesh = _mesh_with_interior_triangle()
    touching = mesh.boundary_touching_element_mask()
    n_tri = len(mesh.triangles)
    assert not touching[0]  # [6,7,4]: どの辺も外形線の両端点を結ばない
    assert touching[1]  # [6,4,5]: 辺 (4,5) がそのまま外形線
    assert touching[n_tri:].all()  # 四角形はすべて外形線に接する


def test_boundary_touching_element_mask_ors_in_the_band_role():
    """band_element_mask（quad_boundary ロール）の要素は、外形線に接しなくても対象。"""
    mesh = _mesh_with_interior_triangle()
    mesh.tri_surface = np.array([9, 0])  # 内部三角形 [6,7,4] に帯用の面 id を付ける
    mesh.surface_roles = {9: "quad_boundary"}
    touching = mesh.boundary_touching_element_mask()
    assert touching[0]  # 実際には外形線に接していないが、band ロールにより対象


def _band_quad_with_small_triangle_neighbour(surface_roles=None) -> Mesh:
    """帯の四角形 1 枚に、辺 (1,2) を共有する小さい三角形が 1 枚接するメッシュ。

    四角形は 100x100（代表辺長 100）、三角形は底辺 100・高さ 10 で
    面積 500（代表辺長 34）。比は約 2.9 で上限 2.0 を超える。
    """
    nodes = np.array([
        [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0],  # quad
        [110.0, 50.0],  # triangle apex
    ])
    return Mesh(
        nodes=nodes,
        triangles=np.array([[1, 4, 2]]),
        quads=np.array([[0, 1, 2, 3]]),
        tri_surface=np.zeros(1, dtype=np.int64),
        quad_surface=np.array([9]),
        surface_roles={9: "quad_boundary"} if surface_roles is None else surface_roles,
    )


def test_a_band_quad_far_coarser_than_its_triangle_neighbour_is_split(tmp_path):
    """帯の四角形が隣の三角形よりずっと粗いなら、対角線で三角形に割ってよい。"""
    mesh = _band_quad_with_small_triangle_neighbour()
    cfg = _cfg(tmp_path, min_area=100.0)

    out, n = split_band_quads_for_ratio(mesh, cfg)

    assert n == 1
    assert len(out.quads) == 0
    assert len(out.triangles) == 3
    # 新しい三角形も帯の面 id を引き継ぐ
    assert (out.tri_surface == 9).sum() == 2


def test_a_band_quad_split_is_refused_if_it_would_break_the_area_floor(tmp_path):
    """割った半分が面積下限を割るなら、帯の四角形はそのまま残す。"""
    mesh = _band_quad_with_small_triangle_neighbour()
    cfg = _cfg(tmp_path, min_area=6000.0)  # 四角形の半分(5000)より大きい下限

    out, n = split_band_quads_for_ratio(mesh, cfg)

    assert n == 0
    assert len(out.quads) == 1
    assert len(out.triangles) == 1


def test_a_non_band_quad_is_left_to_the_other_repair_passes(tmp_path):
    """帯扱いでない四角形はこの関数の対象外（他の修復経路に任せる）。"""
    mesh = _band_quad_with_small_triangle_neighbour(surface_roles={})
    cfg = _cfg(tmp_path, min_area=100.0)

    out, n = split_band_quads_for_ratio(mesh, cfg)

    assert n == 0
    assert len(out.quads) == 1


def _pair_mesh(**kwargs) -> Mesh:
    """共有辺 (0, 1) を挟んで背中合わせに並ぶ 2 つの三角形。"""
    return Mesh(
        nodes=np.array([[0.0, 0.0], [200.0, 0.0], [100.0, 160.0], [100.0, -160.0]]),
        triangles=np.array([[0, 1, 2], [0, 3, 1]]),
        quads=np.zeros((0, 4), dtype=np.int64),
        tri_surface=np.zeros(2, dtype=np.int64),
        quad_surface=np.zeros(0, dtype=np.int64),
        **kwargs,
    )


def _edge_use(mesh: Mesh) -> dict[tuple[int, int], int]:
    use: dict[tuple[int, int], int] = {}
    for conn in (mesh.triangles, mesh.quads):
        for elem in conn:
            m = len(elem)
            for k in range(m):
                a, b = int(elem[k]), int(elem[(k + 1) % m])
                key = (min(a, b), max(a, b))
                use[key] = use.get(key, 0) + 1
    return use


def test_bisection_splits_the_partner_too_so_no_node_is_left_hanging(tmp_path):
    """相手を割らずに中点を入れると、相手の辺の途中に節点がぶら下がる。"""
    mesh = _pair_mesh()

    out, n, _ = _bisect_pass(mesh, np.array([0]), _cfg(tmp_path))

    assert n == 1
    assert len(out.triangles) == 4  # 2 つとも割れている
    # どの辺もちょうど 1 つか 2 つの要素に使われている＝宙ぶらりんの節点がない
    assert set(_edge_use(out).values()) <= {1, 2}


def test_a_bisected_constraint_stays_on_mesh_edges(tmp_path):
    """拘束辺を二等分しても、中点は辺の上なので線はメッシュ辺に載ったまま。"""
    mesh = _pair_mesh(constrained_edges=np.array([[0, 1]]))

    out, n, _ = _bisect_pass(mesh, np.array([0]), _cfg(tmp_path))

    assert n == 1
    assert len(out.triangles) == 4
    # 元の (0, 1) が (0, mid) と (mid, 1) の 2 本に置き換わっている
    edges = out.constrained_edge_set()
    assert (0, 1) not in edges
    assert len(edges) == 2
    mid = out.n_nodes - 1
    assert edges == {(0, mid), (1, mid)}
    assert np.allclose(out.nodes[mid], [100.0, 0.0])
    # 拘束辺は実際に要素の辺として残っている
    assert edges <= set(_edge_use(out))


def test_the_band_is_never_bisected(tmp_path):
    """外周帯は Transfinite で構造が決まっているので触らない。"""
    mesh = _pair_mesh()
    mesh.tri_surface = np.array([0, 9])
    mesh.surface_roles = {9: "quad_boundary"}

    before = mesh.triangles[1].copy()
    out, n, _ = _bisect_pass(mesh, np.array([0]), _cfg(tmp_path))

    # 共有辺 (0, 1) は帯と接するので使わず、別の辺で割る
    assert n == 1
    assert np.array_equal(out.triangles[1], before)
    assert out.tri_surface[1] == 9


def test_a_quad_neighbour_blocks_bisection(tmp_path):
    """四角形を割ると外周帯や統合結果が崩れるので、相手が四角形なら諦める。"""
    tri = np.array([[0.0, 0.0], [200.0, 0.0], [100.0, 160.0]])
    # 三角形の 3 辺すべてを四角形に接させる
    outer = np.array([[0.0, -150.0], [200.0, -150.0], [320.0, 90.0],
                      [180.0, 300.0], [-80.0, 240.0], [-120.0, 20.0]])
    mesh = Mesh(
        nodes=np.vstack([tri, outer]),
        triangles=np.array([[0, 1, 2]]),
        quads=np.array([[0, 3, 4, 1], [1, 5, 6, 2], [2, 7, 8, 0]]),
        tri_surface=np.zeros(1, dtype=np.int64),
        quad_surface=np.zeros(3, dtype=np.int64),
    )

    out, n, why = _bisect_pass(mesh, np.array([0]), _cfg(tmp_path))

    assert n == 0
    assert why["相手が四角形"] == 1
    assert len(out.quads) == 3


def test_a_split_that_would_break_the_area_floor_is_refused(tmp_path):
    """面積下限は最優先。子が下限を割るなら二等分しない。"""
    mesh = _pair_mesh()
    mesh.nodes = np.array([[0.0, 0.0], [60.0, 0.0], [30.0, 40.0], [30.0, -40.0]])
    # 親の面積 1200 m^2、子は 600 m^2 で下限 625 m^2 を割る
    out, n, why = _bisect_pass(mesh, np.array([0]), _cfg(tmp_path))

    assert n == 0
    assert why["形状基準を満たさない"] == 1
    assert len(out.triangles) == 2


def test_the_split_is_skipped_when_the_criterion_is_off(tmp_path):
    mesh = _pair_mesh()
    out, n = split_coarse_triangles(mesh, _cfg(tmp_path, ratio=0.0))
    assert n == 0
    assert out is mesh


def _cfg(tmp_path, min_area: float = 625.0, ratio: float = 2.0):
    floor = math.sqrt(4.0 * min_area / math.sqrt(3.0))
    global_min_size = max(45.0, floor * 1.01)
    body = f"""
    input:
      domain: /tmp/none.gpkg
      dem: /tmp/none.tif
    mesh:
      min_element_area: {min_area}
      max_neighbor_element_ratio: {ratio}
      global_min_size: {global_min_size}
      boundary_quad_band:
        enabled: false
    output:
      directory: /tmp/out
    """
    path = tmp_path / f"balance_{min_area}_{ratio}.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_config(path)
