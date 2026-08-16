#!/usr/bin/env python3
"""Smoke test for mkcell_native ctypes bindings."""
from __future__ import annotations

import numpy as np

import mkcell_native as mn

assert mn.load_native() is not None, "native library not built"

# elevation_stats
vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
stats = mn.elevation_stats(vals)
assert abs(stats["median"] - 30.0) < 1e-6

# slab_dem_stats
slab = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
valid = np.ones(4, dtype=np.uint8)
domain = np.ones(4, dtype=np.uint8)
out = mn.slab_dem_stats(slab, valid, domain, -9999.0, True)
assert out is not None and out["count"] == 4

# quadtree_refine (tiny synthetic DEM)
dem = np.linspace(0, 10, 16, dtype=np.float32).reshape(4, 4)
valid = np.ones((4, 4), dtype=np.uint8)
domain = np.ones((4, 4), dtype=np.uint8)

class _T:
    a = 1.0
    e = -1.0
    c = 0.0
    f = 4.0

leaves = mn.quadtree_refine(
    dem,
    valid,
    domain,
    _T(),
    -9999.0,
    0.0,
    0.0,
    4.0,
    0.0,
    0.0,
    4.0,
    4.0,
    0.5,
    2.0,
    1.0,
    2,
)
assert len(leaves) > 0

keys = mn.grid_clip_candidates(0.0, 0.0, 4.0, 2, 0.5, 0.5, 3.5, 3.5)
assert len(keys) > 0

print("mkcell_native smoke test: OK")
