"""氾濫 LTS セルレベルをメッシュ生成時に付与（mesh.bin AQWALTS1 トレーラー）。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .mesh_tables import NormalizedMesh

Binning = Literal["log", "linear", "quantile"]

BINNING_TO_CODE = {"log": 0, "linear": 1, "quantile": 2}


def face_min_edge_length(mesh: NormalizedMesh) -> np.ndarray:
    """各面の最短辺長（三角形 alt 相当）。"""
    nodes = mesh.nodes
    edges = mesh.edges
    nface = len(mesh.faces)
    out = np.full(nface, np.inf, dtype=np.float64)
    for fi in range(nface):
        for eid in mesh.faces.edge_ids[fi]:
            ei = int(eid) - 1
            v1, v2 = int(edges.v1[ei]), int(edges.v2[ei])
            dx = nodes.x[v1 - 1] - nodes.x[v2 - 1]
            dy = nodes.y[v1 - 1] - nodes.y[v2 - 1]
            out[fi] = min(out[fi], math.hypot(dx, dy))
    out = np.where(np.isfinite(out) & (out > 0.0), out, 1.0)
    return out


def _levels_from_dt(dt: np.ndarray, max_level: int) -> np.ndarray:
    """ソルバー level_from_dt と同じ 2 進ラダー（dt 大 → レベル大 = 粗い）。"""
    cap = max(int(max_level), 0)
    dt_base = 1.0
    dt_min = dt_base / float(2**cap) if cap > 0 else dt_base
    dt_fine = dt_min
    levels = np.zeros(len(dt), dtype=np.int32)
    for ic, dtc in enumerate(np.asarray(dt, dtype=np.float64)):
        lev = 0
        for k in range(cap, -1, -1):
            if dt_fine * (2.0**k) <= dtc * (1.0 + 1.0e-12):
                lev = k
                break
        levels[ic] = lev
    return levels


def compute_face_dt(
    char_len: np.ndarray,
    max_level: int,
    binning: Binning,
) -> np.ndarray:
    """基準 dt=1 の相対 dt（ソルバーは ndiv で実 dt にスケール）。"""
    n = len(char_len)
    cap = max(int(max_level), 0)
    dt_base = 1.0
    dt_min = dt_base / float(2**cap) if cap > 0 else dt_base
    lens = np.asarray(char_len, dtype=np.float64)
    lens = np.maximum(lens, 1.0e-12)

    if binning == "linear":
        len_max = float(lens.max())
        scale = lens / len_max
        return np.clip(dt_base * scale, dt_min, dt_base)

    if binning == "quantile":
        order = np.argsort(lens, kind="stable")
        dt = np.empty(n, dtype=np.float64)
        dt_fine = dt_min
        if n <= 1:
            dt[:] = dt_fine
            return dt
        for rank, ic in enumerate(order):
            lev = min(cap, (rank * (cap + 1)) // n)
            dt[ic] = dt_fine * float(2**lev)
        return dt

    # log: p5–p95 パーセンタイル間で幾何補間
    order = np.argsort(lens, kind="stable")
    ilo = max(0, int(0.05 * (n - 1)))
    ihi = max(0, int(0.95 * (n - 1)))
    lo = float(lens[order[ilo]])
    hi = float(lens[order[ihi]])
    if hi <= lo * (1.0 + 1.0e-12):
        hi = lo * 1.000001
    log_lo, log_hi = math.log(lo), math.log(hi)
    ratio = dt_base / dt_min
    t = (np.log(lens) - log_lo) / (log_hi - log_lo)
    t = np.clip(t, 0.0, 1.0)
    return dt_min * (ratio**t)


def compute_lts_levels(
    mesh: NormalizedMesh,
    *,
    max_level: int = 2,
    binning: Binning = "quantile",
    smooth_passes: int = 8,
) -> tuple[np.ndarray, int]:
    """面ごとの LTS レベル（0=最細）と使用 lmax。"""
    char_len = face_min_edge_length(mesh)
    cap = max(int(max_level), 0)
    n = len(char_len)
    if binning == "quantile" and n > 0:
        order = np.argsort(char_len, kind="stable")
        levels = np.zeros(n, dtype=np.int32)
        if n == 1:
            levels[0] = 0
        else:
            for rank, ic in enumerate(order):
                levels[ic] = min(cap, (rank * (cap + 1)) // n)
    else:
        dt = compute_face_dt(char_len, max_level, binning)
        levels = _levels_from_dt(dt, max_level)
    if smooth_passes > 0:
        levels = smooth_lts_levels(levels, mesh, passes=smooth_passes)
    lmax = int(levels.max()) if len(levels) else 0
    return levels.astype(np.int32), lmax


def _face_neighbors(mesh: NormalizedMesh) -> list[list[int]]:
    nface = len(mesh.faces)
    nbr: list[list[int]] = [[] for _ in range(nface)]
    for ei in range(len(mesh.edges)):
        fl = int(mesh.edges.face_left[ei])
        fr = int(mesh.edges.face_right[ei])
        if fl > 0 and fr > 0:
            a, b = fl - 1, fr - 1
            nbr[a].append(b)
            nbr[b].append(a)
    return nbr


def smooth_lts_levels(
    levels: np.ndarray,
    mesh: NormalizedMesh,
    *,
    passes: int = 8,
) -> np.ndarray:
    """隣接面のレベル差を 1 以内に（ソルバー smooth_inun 相当）。"""
    lev = np.asarray(levels, dtype=np.int32).copy()
    nbr = _face_neighbors(mesh)
    nface = len(lev)
    for _ in range(passes):
        changed = False
        for ic in range(nface):
            for nb in nbr[ic]:
                if lev[ic] > lev[nb] + 1:
                    lev[ic] = lev[nb] + 1
                    changed = True
        if not changed:
            break
    return lev
