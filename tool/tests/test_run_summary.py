"""run_summary のユニットテスト。"""

from __future__ import annotations

from src.config import MeshConfig, TerrainConfig
from src.run_summary import aggregate_block_summaries, quality_comment_lines


class _Cfg:
    mesh = MeshConfig(
        min_element_area=200.0,
        max_neighbor_size_ratio=2.0,
    )
    terrain = TerrainConfig(
        plane_fit_rmse_max=0.3,
        slope_direction_spread_max_deg=30.0,
    )


def test_quality_comment_lines_includes_overall():
    summary = {
        "n_elements": 1000,
        "n_triangles": 800,
        "n_quads": 200,
        "area_total_km2": 2.0,
        "area_min": 140.0,
        "area_max": 9000.0,
        "area_median": 650.0,
        "viol_area_below_min": 0,
        "viol_any": 5,
        "viol_any_ratio": 0.005,
        "viol_neighbor_ratio": 0,
        "neighbor_ratio_max": 1.5,
        "band_quad_count": 200,
        "plane_fit_rmse_area_weighted": 0.25,
        "slope_spread_area_weighted": 48.0,
        "terrain_viol_any": 800,
        "terrain_viol_any_ratio": 0.8,
        "n_iterations": 8,
    }
    lines = quality_comment_lines(summary, _Cfg())
    text = "\n".join(lines)
    assert "--- セル品質 ---" in text
    assert "総合" in text
    assert "面積下限" in text
    assert "OK" in text


def test_aggregate_block_summaries_sums_elements():
    blocks = [
        {"n_elements": 100, "n_triangles": 80, "n_quads": 20, "area_total_km2": 1.0,
         "area_min": 200, "area_max": 1000, "area_median": 500,
         "viol_any": 1, "terrain_viol_any": 50,
         "plane_fit_rmse_area_weighted": 0.2, "slope_spread_area_weighted": 40.0,
         "n_iterations": 6, "band_quads": {"count": 20}, "viol_neighbor_ratio": 0},
        {"n_elements": 200, "n_triangles": 150, "n_quads": 50, "area_total_km2": 2.0,
         "area_min": 150, "area_max": 2000, "area_median": 700,
         "viol_any": 2, "terrain_viol_any": 100,
         "plane_fit_rmse_area_weighted": 0.4, "slope_spread_area_weighted": 50.0,
         "n_iterations": 8, "band_quads": {"count": 30}, "viol_neighbor_ratio": 0},
    ]
    agg = aggregate_block_summaries(blocks)
    assert agg["n_elements"] == 300
    assert agg["n_quads"] == 70
    assert agg["band_quad_count"] == 50
    assert agg["n_iterations"] == 8
