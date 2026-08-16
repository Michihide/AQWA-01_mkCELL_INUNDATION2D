from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LinearRing, Polygon

from src.boundary_quad_band import _sharp_interface_corners, band_report, build_polygon_band
from src.geometry_cleaning import (
    clean_polygon,
    clean_polygons,
    polygon_rings_resampled,
    remove_narrow_features,
    resample_ring,
)


def rect(w=1000.0, h=600.0) -> Polygon:
    return Polygon([(0, 0), (w, 0), (w, h), (0, h)])


def test_resample_ring_is_evenly_spaced():
    pts = resample_ring(rect().exterior, 50.0)
    d = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)
    assert d.min() == pytest.approx(d.max(), rel=1e-6)
    assert d.mean() == pytest.approx(3200.0 / len(pts), rel=1e-6)


def test_exterior_is_ccw_and_holes_are_cw():
    outer = rect(1000, 800).exterior
    hole = [(300, 300), (600, 300), (600, 500), (300, 500)]
    poly = Polygon(outer, [hole])
    ext, holes = polygon_rings_resampled(poly, 50.0)

    def signed(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * (x * np.roll(y, -1) - y * np.roll(x, -1)).sum()

    assert signed(ext) > 0
    assert len(holes) == 1
    assert signed(holes[0]) < 0


def test_clean_polygon_removes_duplicate_vertices():
    poly = Polygon([(0, 0), (0, 0), (100, 0), (100, 100), (100, 100), (0, 100)])
    cleaned = clean_polygon(poly, min_area=1.0, min_hole_area=1.0)
    assert len(cleaned.exterior.coords) == 5  # 閉じ点を含めて 4 頂点


def test_remove_narrow_features_drops_thin_neck():
    # 中央が幅 10 m でつながった 2 つの正方形
    poly = Polygon([
        (0, 0), (200, 0), (200, 95), (400, 95), (400, 0), (600, 0),
        (600, 300), (400, 300), (400, 105), (200, 105), (200, 300), (0, 300),
    ])
    parts = remove_narrow_features(poly, radius=30.0, min_area=100.0)
    assert len(parts) == 2
    assert all(p.area > 40000 for p in parts)


# Hii の実データ（ポリゴン #17）を simplify(15) で単純化した実例。
# 開処理後の単一ブロブを元ポリゴンと交差させると 2 つに分裂するケースで、
# 大きい方だけを残すと領域の 4 割超がまるごと欠落していた（実データでは
# 約 28 万 m^2 の消失）。
_HII_POLY17_COORDS = [
    [1973.0, 1061.0], [2008.0, 1015.0], [2046.0, 1028.0], [2039.0, 1071.0],
    [2085.0, 1066.0], [2039.0, 1233.0], [2059.0, 1318.0], [2154.0, 1297.0],
    [2091.0, 1040.0], [1868.0, 897.0], [1783.0, 810.0], [739.0, 361.0],
    [1074.0, 625.0], [1655.0, 868.0], [1735.0, 844.0], [1656.0, 874.0],
    [1072.0, 630.0], [733.0, 382.0], [730.0, 437.0], [760.0, 492.0],
    [1226.0, 852.0], [1253.0, 887.0], [1234.0, 965.0], [1248.0, 1059.0],
    [1239.0, 1095.0], [1228.0, 973.0], [1242.0, 883.0], [741.0, 485.0],
    [722.0, 438.0], [720.0, 383.0], [219.0, 215.0], [121.0, 134.0],
    [25.0, 0.0], [0.0, 131.0], [109.0, 234.0], [280.0, 342.0], [307.0, 391.0],
    [253.0, 648.0], [411.0, 671.0], [400.0, 736.0], [425.0, 751.0],
    [417.0, 785.0], [445.0, 838.0], [490.0, 824.0], [514.0, 727.0],
    [535.0, 715.0], [571.0, 746.0], [583.0, 847.0], [624.0, 848.0],
    [672.0, 761.0], [743.0, 798.0], [792.0, 760.0], [769.0, 889.0],
    [732.0, 911.0], [790.0, 956.0], [929.0, 835.0], [1004.0, 861.0],
    [1036.0, 927.0], [969.0, 962.0], [971.0, 985.0], [1103.0, 1005.0],
    [1096.0, 1109.0], [1205.0, 1130.0], [1266.0, 1130.0], [1266.0, 1031.0],
    [1307.0, 1002.0], [1356.0, 1009.0], [1385.0, 1092.0], [1427.0, 1078.0],
    [1432.0, 962.0], [1472.0, 945.0], [1543.0, 974.0], [1576.0, 960.0],
    [1581.0, 1012.0], [1666.0, 950.0], [1673.0, 1026.0], [1704.0, 1028.0],
    [1723.0, 934.0], [1760.0, 915.0], [1861.0, 933.0], [1903.0, 965.0],
    [1893.0, 1032.0], [1973.0, 1061.0],
]


def test_clean_polygons_keeps_polygon_when_narrow_opening_erases_it():
    """開処理で全体が消えた場合は、開処理前の形状を保持する。"""
    small = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    out = clean_polygons([small], min_area=0.0, narrow_feature_radius=13.0)
    assert len(out) == 1
    assert out[0].area == pytest.approx(400.0)


def test_remove_narrow_features_keeps_both_halves_of_a_split_blob():
    poly = Polygon(_HII_POLY17_COORDS)
    assert poly.is_valid and poly.area > 6.5e5

    parts = remove_narrow_features(poly, radius=31.5, min_area=1000.0)
    total = sum(p.area for p in parts)

    # 開処理後のブロブを元ポリゴンと交差させると 2 つに割れる。どちらも
    # 有効な領域なので、小さい方を捨てずに両方残すこと。
    assert len(parts) >= 2
    assert total > 0.9 * poly.area


def test_band_quads_meet_area_floor():
    band = build_polygon_band(rect(1200, 900), width=30.0, spacing=50.0,
                              min_element_area=625.0)
    assert band.ok
    r = band_report(band, 625.0)
    assert r["below_min"] == 0
    assert r["area_min"] >= 625.0


def test_band_quads_are_convex_and_positive_area():
    band = build_polygon_band(rect(1200, 900), width=30.0, spacing=50.0,
                              min_element_area=625.0)
    for ring in band.rings():
        for i in ring.quad_indices():
            quad = ring.quad_coords(i)
            assert Polygon(quad).is_simple
            e1 = quad - np.roll(quad, 1, axis=0)
            e2 = np.roll(quad, -1, axis=0) - quad
            cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            assert (cross > 0).all()


def test_band_interior_is_inside_polygon():
    poly = rect(1200, 900)
    band = build_polygon_band(poly, width=30.0, spacing=50.0, min_element_area=625.0)
    assert band.interior is not None
    assert poly.buffer(1e-6).contains(band.interior)
    assert band.interior.area == pytest.approx(poly.area - 30.0 * poly.length + 4 * 900, rel=0.02)


def test_band_handles_hole():
    outer = rect(1600, 1200).exterior
    hole = [(600, 450), (1000, 450), (1000, 750), (600, 750)][::-1]
    poly = Polygon(outer, [hole])
    band = build_polygon_band(poly, width=30.0, spacing=50.0, min_element_area=625.0)
    assert band.ok
    assert len(band.holes) == 1
    assert band.interior is not None
    assert len(band.interior.interiors) == 1


def test_sharp_corner_detection():
    # 頂点 1 が約 20 度の鋭角で、そこに置ける最大の三角形は 450 m^2
    pts = np.array([(0.0, 0.0), (50.0, 0.0), (0.0, 18.0)])
    idx = _sharp_interface_corners(pts, min_element_area=625.0, max_angle_deg=120.0)
    assert 1 in idx.tolist()


def test_sharp_corner_ignores_roomy_corner():
    # 同じ 20 度でも辺が長ければ 625 m^2 の三角形が入る
    pts = np.array([(0.0, 0.0), (100.0, 0.0), (0.0, 36.4)])
    idx = _sharp_interface_corners(pts, min_element_area=625.0, max_angle_deg=120.0)
    assert 1 not in idx.tolist()


def test_sharp_corner_ignores_straight_boundary():
    # ほぼ直線上の頂点は、三角形面積が小さくても対象外
    pts = np.array([(0.0, 0.0), (50.0, 0.5), (100.0, 0.0), (100.0, 500.0), (0.0, 500.0)])
    idx = _sharp_interface_corners(pts, min_element_area=625.0, max_angle_deg=120.0)
    assert 1 not in idx.tolist()


def test_band_on_sharp_wedge_still_meets_area_floor():
    wedge = Polygon([(0, 0), (1500, 60), (1500, 700), (200, 700)])
    band = build_polygon_band(wedge, width=30.0, spacing=50.0, min_element_area=625.0)
    assert band.ok
    assert band_report(band, 625.0)["below_min"] == 0


def test_a_sharp_wedge_is_neither_skipped_nor_cut():
    """鋭角コーナーでも、内向きオフセットが自己交差しない限り区間をスキップしない。

    境界の形状再現性を優先するため面積下限は境界要素の対象外にしており、鋭角
    コーナーを理由にした区間スキップ（旧 _sharp_interface_corners 判定）も
    頂点の切り落としも行わない。
    """
    wedge = Polygon([(0, 0), (1500, 60), (1500, 700), (200, 700)])
    band = build_polygon_band(wedge, width=30.0, spacing=50.0, min_element_area=625.0)
    ring = band.exterior

    assert band.corner_cuts == 0  # 境界の頂点は 1 つも落としていない
    assert not ring.skipped  # 自己交差しない限りスキップしない
    # 外周の頂点列そのものは変わっていない（形状を保っている）
    np.testing.assert_allclose(ring.outer, polygon_rings_resampled(wedge, 50.0)[0])


def _grid_ring_band(n: int = 12, width: float = 30.0, spacing: float = 50.0):
    """円周上に等間隔で並ぶ頂点からなる、素直な RingBand を作る。

    コーナー検出の癖に左右されない、スキップ操作そのものを試すための土台。
    """
    from src.boundary_quad_band import RingBand

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = spacing * n / (2.0 * np.pi)
    outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    inner = outer * (1.0 - width / r)
    return RingBand(outer=outer, inner=inner, widths=np.full(n, width))


def test_the_interior_ring_is_a_simple_closed_polygon_with_a_skip_run():
    """内側領域の境界（Γ1 + スキップ区間の Γ0）は自己交差しない閉曲線になる。"""
    ring = _grid_ring_band(16)
    ring.skipped = {3, 4, 5}
    ring.close_ends()

    iface = ring.interior_ring()
    assert len(iface) == ring.n
    poly = Polygon(iface)
    assert poly.is_valid
    assert poly.is_simple


def test_interior_walk_covers_every_segment_exactly_once():
    """inner/outer は帯から外した区間の分だけ入れ替わり、全区間を 1 回ずつ辿る。

    閉じの三角形区間（cell_indices にあって quad_indices には無い区間）も、
    そこでは Γ1 が Γ0 に着地しているので inner 側として辿る。
    """
    ring = _grid_ring_band(16)
    ring.skipped = {3, 4, 5}
    ring.close_ends()
    walk = ring.interior_walk()

    inner_hits = sorted(i for kind, i in walk if kind == "inner")
    outer_hits = sorted(i for kind, i in walk if kind == "outer")
    assert inner_hits == sorted(ring.cell_indices())
    assert outer_hits == sorted(ring.skipped)
    assert len(walk) == ring.n


def test_two_skip_runs_each_get_their_own_down_and_up_transition():
    """close_ends を呼ばない生の状態では、各スキップ区間の両端に降り/上りが要る。"""
    ring = _grid_ring_band(16)
    ring.skipped = {2, 3, 9, 10, 11}
    walk = ring.interior_walk()

    assert [k for k, _ in walk].count("down") == 2
    assert [k for k, _ in walk].count("up") == 2


def test_close_ends_removes_the_down_and_up_transitions():
    """close_ends で接合点の幅を 0 にすると、降り/上りの径方向線は不要になる。"""
    ring = _grid_ring_band(16)
    ring.skipped = {2, 3, 9, 10, 11}
    ring.close_ends()
    walk = ring.interior_walk()

    assert [k for k, _ in walk].count("down") == 0
    assert [k for k, _ in walk].count("up") == 0


def test_close_ends_collapses_width_at_skip_boundaries():
    """スキップ区間の端では、Γ1 が Γ0 に着地するよう幅が 0 になる。"""
    ring = _grid_ring_band(16)
    ring.skipped = {3, 4, 5}
    ring.close_ends()

    assert ring.collapsed
    for v in ring.collapsed:
        np.testing.assert_allclose(ring.inner_pt(v), ring.outer[v])
    # 閉じの三角形自体は帯要素として残る
    closing = ring.closing_indices()
    assert closing
    assert set(closing) & ring.skipped == set()


def test_close_ends_widens_a_skip_run_whose_closing_triangle_is_degenerate(monkeypatch):
    """閉じの三角形が真に退化していれば、その区間も帯から外して 1 つ手前で閉じる。

    面積下限では区間を広げなくなったため（境界形状の再現性を優先）、ここでは
    退化検出の閾値（_DEGENERATE_AREA_EPS）を極端に高くして、閉じの三角形が
    「退化」と判定される状況を人為的に作る。
    """
    import src.boundary_quad_band as bqb

    ring = _grid_ring_band(16)
    ring.skipped = {5}
    monkeypatch.setattr(bqb, "_DEGENERATE_AREA_EPS", 1e12)
    # 通常なら 1 区間だけ閉じるが、退化閾値を極端に高くして拒否させる
    ring.close_ends()
    assert ring.skipped >= {4, 5, 6}


def test_a_ring_with_no_skips_walks_pure_inner():
    ring = _grid_ring_band(16)
    ring.close_ends()
    assert not ring.skipped
    assert not ring.collapsed
    walk = ring.interior_walk()
    assert walk == [("inner", i) for i in range(ring.n)]


def test_a_fully_skipped_ring_falls_back_to_pure_outer():
    ring = _grid_ring_band(8)
    ring.skipped = set(range(ring.n))
    walk = ring.interior_walk()
    assert walk == [("outer", i) for i in range(ring.n)]
