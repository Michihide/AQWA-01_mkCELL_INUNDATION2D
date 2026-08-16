"""DEM からの盛り土検出。

合成 DEM で、盛り土だけを拾い、丘や山腹を拾わないことを確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from src.embankment_detection import (
    DetectionParams,
    detect_embankments,
    embankment_mask,
    skeleton_to_paths,
    too_sinuous,
    top_hat,
)

PX = 2.0


def _flat(shape=(200, 300), value=10.0) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _add_ridge(z, row, half_width_px, height):
    """行 row を中心に、東西に走る盛り土を足す。"""
    z[row - half_width_px : row + half_width_px + 1, :] += height
    return z


def test_top_hat_extracts_a_narrow_ridge():
    z = _add_ridge(_flat(), row=100, half_width_px=5, height=3.0)
    relief = top_hat(z, size_px=31)
    assert relief[100, 150] == pytest.approx(3.0, abs=0.01)
    assert relief[20, 150] == pytest.approx(0.0, abs=0.01)


def test_top_hat_ignores_a_hill_wider_than_the_structuring_element():
    z = _flat()
    z[60:140, :] += 3.0  # 幅 160 m。構造要素 62 m より広い
    relief = top_hat(z, size_px=31)
    assert relief.max() == pytest.approx(0.0, abs=0.01)


def test_mask_rejects_a_ridge_on_steep_ground():
    """山腹の上の細い高まりは、周囲が平坦でないので盛り土としない。"""
    z = _flat()
    z += np.linspace(0.0, 60.0, z.shape[0])[:, None]  # 200 m で 60 m 上がる斜面
    z = _add_ridge(z, row=100, half_width_px=5, height=3.0)
    inside = np.ones_like(z, dtype=bool)
    mask, _ = embankment_mask(z, PX, inside, DetectionParams())
    assert not mask.any()


def test_mask_rejects_relief_beyond_the_upper_limit():
    z = _add_ridge(_flat(), row=100, half_width_px=5, height=40.0)
    inside = np.ones_like(z, dtype=bool)
    mask, _ = embankment_mask(z, PX, inside, DetectionParams(max_relief=15.0))
    assert not mask.any()


def test_mask_is_confined_to_the_domain():
    z = _add_ridge(_flat(), row=100, half_width_px=5, height=3.0)
    inside = np.zeros_like(z, dtype=bool)
    inside[:, :100] = True
    mask, _ = embankment_mask(z, PX, inside, DetectionParams())
    assert mask[:, :100].any()
    assert not mask[:, 100:].any()


def test_skeleton_of_a_straight_line_is_one_path():
    skel = np.zeros((20, 40), dtype=bool)
    skel[10, 5:35] = True
    paths = skeleton_to_paths(skel)
    assert len(paths) == 1
    assert len(paths[0]) == 30


def test_skeleton_splits_at_a_junction():
    skel = np.zeros((30, 30), dtype=bool)
    skel[15, 5:25] = True
    skel[5:15, 15] = True  # T 字
    paths = skeleton_to_paths(skel)
    assert len(paths) == 3
    # 交点 (15,15) を含めて、左 11 画素・右 10 画素・上 11 画素
    assert sorted(len(p) for p in paths) == [10, 11, 11]


def test_skeleton_handles_a_closed_loop():
    skel = np.zeros((30, 30), dtype=bool)
    skel[10, 10:20] = True
    skel[19, 10:20] = True
    skel[10:20, 10] = True
    skel[10:20, 19] = True
    paths = skeleton_to_paths(skel)
    assert len(paths) == 1
    assert paths[0][0] == paths[0][-1]


def test_sinuosity_rejects_a_loop_and_keeps_a_road():
    loop = LineString([(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)])
    straight = LineString([(0, 0), (300, 20)])
    assert too_sinuous(loop, 2.0)
    assert not too_sinuous(straight, 2.0)


def test_detect_returns_a_line_along_the_embankment():
    z = _add_ridge(_flat(), row=100, half_width_px=5, height=3.0)
    inside = np.ones_like(z, dtype=bool)
    lines, attrs, _ = detect_embankments(
        z, PX, x_min=0.0, y_max=400.0, inside=inside, params=DetectionParams()
    )
    assert len(lines) == 1
    line = lines[0]
    # 盛り土の中心は row=100 なので y = 400 - 100.5*2 = 199 付近
    ys = np.asarray(line.coords)[:, 1]
    assert ys.min() == pytest.approx(199.0, abs=PX)
    assert ys.max() == pytest.approx(199.0, abs=PX)
    assert line.length > 500.0
    assert attrs[0]["relief_median"] == pytest.approx(3.0, abs=0.05)


def test_detect_finds_nothing_on_flat_ground():
    inside = np.ones((200, 300), dtype=bool)
    lines, _, _ = detect_embankments(
        _flat(), PX, 0.0, 400.0, inside, DetectionParams()
    )
    assert lines == []


def test_detect_separates_two_parallel_embankments():
    z = _flat()
    _add_ridge(z, row=60, half_width_px=4, height=2.0)
    _add_ridge(z, row=140, half_width_px=4, height=2.0)
    inside = np.ones_like(z, dtype=bool)
    lines, _, _ = detect_embankments(z, PX, 0.0, 400.0, inside, DetectionParams())
    assert len(lines) == 2
    ys = sorted(np.asarray(ln.coords)[0, 1] for ln in lines)
    assert ys[1] - ys[0] == pytest.approx(160.0, abs=2 * PX)
