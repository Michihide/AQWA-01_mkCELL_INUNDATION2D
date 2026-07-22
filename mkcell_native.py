"""ctypes bindings for native/mkcell_native."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
USE_NATIVE = os.environ.get("MKCELL_USE_NATIVE", "1") not in {"0", "false", "False"}


def _lib_path() -> Path:
    here = Path(__file__).resolve().parent
    native_dir = here / "native"
    for name in ("libmkcell_native.dylib", "libmkcell_native.so"):
        p = native_dir / name
        if p.is_file():
            return p
    return native_dir / "libmkcell_native.so"


def load_native() -> Optional[ctypes.CDLL]:
    global _LIB
    if not USE_NATIVE:
        return None
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    if not path.is_file():
        return None
    lib = ctypes.CDLL(str(path))
    _setup_signatures(lib)
    _LIB = lib
    return lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    lib.mkcell_accumulate_codes.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mkcell_accumulate_codes.restype = None

    lib.mkcell_elevation_stats.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mkcell_elevation_stats.restype = None

    lib.mkcell_slab_dem_stats.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mkcell_slab_dem_stats.restype = None

    lib.mkcell_quadtree_refine.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mkcell_quadtree_refine.restype = ctypes.c_int

    lib.mkcell_grid_clip_candidates.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mkcell_grid_clip_candidates.restype = ctypes.c_int


def accumulate_codes(codes: np.ndarray, weights: np.ndarray, min_code: int, max_code: int) -> np.ndarray:
    lib = load_native()
    out_len = max_code - min_code + 1
    out = np.zeros(out_len, dtype=np.float64)
    if lib is None:
        for code, w in zip(codes.astype(int), weights):
            if min_code <= code <= max_code:
                out[code - min_code] += w
        return out

    codes_i = np.ascontiguousarray(codes, dtype=np.int32)
    weights_d = np.ascontiguousarray(weights, dtype=np.float64)
    lib.mkcell_accumulate_codes(
        codes_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        weights_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(codes_i)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(out_len),
        ctypes.c_int(min_code),
        ctypes.c_int(max_code),
    )
    return out


def elevation_stats(values: np.ndarray) -> dict:
    lib = load_native()
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p05": 0.0, "p95": 0.0}
    vals = np.ascontiguousarray(values.astype(np.float64))
    mean = ctypes.c_double()
    median = ctypes.c_double()
    stdv = ctypes.c_double()
    p05 = ctypes.c_double()
    p95 = ctypes.c_double()
    if lib is None:
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        }
    lib.mkcell_elevation_stats(
        vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(vals)),
        ctypes.byref(mean),
        ctypes.byref(median),
        ctypes.byref(stdv),
        ctypes.byref(p05),
        ctypes.byref(p95),
    )
    return {
        "mean": mean.value,
        "median": median.value,
        "std": stdv.value,
        "p05": p05.value,
        "p95": p95.value,
    }


def slab_dem_stats(slab: np.ndarray, valid: np.ndarray, domain: np.ndarray, nodata: float, need_relief: bool):
    lib = load_native()
    n = slab.size
    slab_f = np.ascontiguousarray(slab.astype(np.float32).ravel())
    valid_b = np.ascontiguousarray(valid.astype(np.uint8).ravel())
    domain_b = np.ascontiguousarray(domain.astype(np.uint8).ravel())
    stdv = ctypes.c_double()
    relief = ctypes.c_double()
    count = ctypes.c_int()
    full = ctypes.c_int()
    if lib is None:
        from importlib import import_module

        # fallback to existing python helper via lazy import
        return None

    lib.mkcell_slab_dem_stats(
        slab_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        valid_b.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        domain_b.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.c_size_t(n),
        ctypes.c_float(float(nodata)),
        ctypes.c_int(1 if need_relief else 0),
        ctypes.byref(stdv),
        ctypes.byref(relief),
        ctypes.byref(count),
        ctypes.byref(full),
    )
    return {
        "std": None if count.value == 0 else stdv.value,
        "relief": None if count.value == 0 else relief.value,
        "count": count.value,
        "full_cover": bool(full.value),
    }


def quadtree_refine(
    dem: np.ndarray,
    valid: np.ndarray,
    domain_mask: np.ndarray,
    transform,
    nodata: float,
    origin_x: float,
    origin_y: float,
    base_cell: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    std_threshold: float,
    relief_threshold: float,
    min_area: float,
    max_depth: int,
    max_out: int = 1_000_000,
) -> list[tuple[int, int, int]]:
    """Run DEM-adaptive quadtree refinement in native code."""
    lib = load_native()
    if lib is None:
        raise RuntimeError("native library not available")

    height, width = dem.shape
    dem_f = np.ascontiguousarray(dem.astype(np.float32))
    valid_b = np.ascontiguousarray(valid.astype(np.uint8))
    domain_b = np.ascontiguousarray(domain_mask.astype(np.uint8))
    out_levels = np.zeros(max_out, dtype=np.int32)
    out_is = np.zeros(max_out, dtype=np.int32)
    out_js = np.zeros(max_out, dtype=np.int32)
    out_count = ctypes.c_int(0)

    rc = lib.mkcell_quadtree_refine(
        dem_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        valid_b.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        domain_b.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.c_int(height),
        ctypes.c_int(width),
        ctypes.c_double(float(transform.a)),
        ctypes.c_double(float(transform.e)),
        ctypes.c_double(float(transform.c)),
        ctypes.c_double(float(transform.f)),
        ctypes.c_float(float(nodata)),
        ctypes.c_double(origin_x),
        ctypes.c_double(origin_y),
        ctypes.c_double(base_cell),
        ctypes.c_double(xmin),
        ctypes.c_double(ymin),
        ctypes.c_double(xmax),
        ctypes.c_double(ymax),
        ctypes.c_double(std_threshold),
        ctypes.c_double(relief_threshold),
        ctypes.c_double(min_area),
        ctypes.c_int(max_depth),
        out_levels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_is.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_js.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(max_out),
        ctypes.byref(out_count),
    )
    if rc != 0:
        raise RuntimeError(f"mkcell_quadtree_refine failed (rc={rc})")

    n = out_count.value
    return [
        (int(out_levels[k]), int(out_is[k]), int(out_js[k]))
        for k in range(n)
    ]


def grid_clip_candidates(
    origin_x: float,
    origin_y: float,
    base_cell: float,
    max_level: int,
    poly_minx: float,
    poly_miny: float,
    poly_maxx: float,
    poly_maxy: float,
    max_out: int = 500_000,
) -> list[tuple[int, int, int]]:
    """Enumerate quadtree leaf keys overlapping a polygon bbox."""
    lib = load_native()
    if lib is None:
        raise RuntimeError("native library not available")

    out_levels = np.zeros(max_out, dtype=np.int32)
    out_is = np.zeros(max_out, dtype=np.int32)
    out_js = np.zeros(max_out, dtype=np.int32)
    out_count = ctypes.c_int(0)
    rc = lib.mkcell_grid_clip_candidates(
        ctypes.c_double(origin_x),
        ctypes.c_double(origin_y),
        ctypes.c_double(base_cell),
        ctypes.c_int(max_level),
        ctypes.c_double(poly_minx),
        ctypes.c_double(poly_miny),
        ctypes.c_double(poly_maxx),
        ctypes.c_double(poly_maxy),
        out_levels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_is.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_js.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(max_out),
        ctypes.byref(out_count),
    )
    if rc != 0:
        raise RuntimeError(f"mkcell_grid_clip_candidates failed (rc={rc})")

    n = out_count.value
    return [
        (int(out_levels[k]), int(out_is[k]), int(out_js[k]))
        for k in range(n)
    ]
