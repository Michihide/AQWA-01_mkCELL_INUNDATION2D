"""要素の形状品質と面積の評価。

面積下限 625 m^2 は最優先の制約なので、形状指標とは別に必ず集計する。
四角形は凸性（scaled Jacobian が全隅で正）を必須とし、凹・反転を許さない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .mesh_parser import Mesh


@dataclass
class QualityReport:
    """要素ごとの品質指標。三角形 -> 四角形の順に並ぶ。"""

    area: np.ndarray
    kind: np.ndarray  # 3 or 4
    min_angle_deg: np.ndarray
    max_angle_deg: np.ndarray
    aspect_ratio: np.ndarray
    radius_ratio: np.ndarray  # 三角形のみ。四角形は nan
    scaled_jacobian: np.ndarray  # 四角形のみ。三角形は nan
    is_convex: np.ndarray

    def __len__(self) -> int:
        return len(self.area)


def polygon_areas(coords: np.ndarray) -> np.ndarray:
    """(M, K, 2) の多角形群の符号付き面積。"""
    x = coords[..., 0]
    y = coords[..., 1]
    return 0.5 * (x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1)


def _interior_angles(coords: np.ndarray) -> np.ndarray:
    """(M, K, 2) の多角形群の内角 [deg]。頂点は反時計回りを想定。"""
    prev = np.roll(coords, 1, axis=1)
    nxt = np.roll(coords, -1, axis=1)
    v1 = prev - coords
    v2 = nxt - coords
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(axis=-1)
    # atan2 で 0..2pi。反時計回りなら凸頂点で cross < 0 になるため符号を反転する
    ang = np.arctan2(-cross, dot)
    ang = np.where(ang < 0.0, ang + 2.0 * np.pi, ang)
    return np.degrees(ang)


def _edge_lengths(coords: np.ndarray) -> np.ndarray:
    d = np.roll(coords, -1, axis=1) - coords
    return np.hypot(d[..., 0], d[..., 1])


def triangle_quality(nodes: np.ndarray, tris: np.ndarray) -> dict[str, np.ndarray]:
    coords = nodes[tris]
    area = polygon_areas(coords)
    edges = _edge_lengths(coords)
    a, b, c = edges[:, 0], edges[:, 1], edges[:, 2]
    s = 0.5 * (a + b + c)
    abs_area = np.abs(area)

    with np.errstate(divide="ignore", invalid="ignore"):
        # 半径比 q = 8 A^2 / (s a b c)。正三角形で 1。
        radius_ratio = np.where(s > 0, 8.0 * abs_area**2 / (s * a * b * c), 0.0)
        # アスペクト比 R / (2 r)。正三角形で 1。
        r_in = np.where(s > 0, abs_area / s, 0.0)
        r_circ = np.where(abs_area > 0, a * b * c / (4.0 * abs_area), np.inf)
        aspect = np.where(r_in > 0, r_circ / (2.0 * r_in), np.inf)

    angles = _interior_angles(coords)
    return {
        "area": area,
        "min_angle": angles.min(axis=1),
        "max_angle": angles.max(axis=1),
        "aspect_ratio": aspect,
        "radius_ratio": np.nan_to_num(radius_ratio, nan=0.0),
    }


def quad_quality(nodes: np.ndarray, quads: np.ndarray) -> dict[str, np.ndarray]:
    coords = nodes[quads]
    area = polygon_areas(coords)
    edges = _edge_lengths(coords)

    prev = np.roll(coords, 1, axis=1)
    nxt = np.roll(coords, -1, axis=1)
    e1 = coords - prev
    e2 = nxt - coords
    n1 = np.hypot(e1[..., 0], e1[..., 1])
    n2 = np.hypot(e2[..., 0], e2[..., 1])
    cross = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        corner_jac = np.where((n1 > 0) & (n2 > 0), cross / (n1 * n2), 0.0)

    # 反時計回りで全隅の外積が正 <=> 凸。1 つでも非正なら凹または反転。
    scaled_jacobian = corner_jac.min(axis=1)
    is_convex = scaled_jacobian > 0.0

    angles = _interior_angles(coords)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = edges.max(axis=1) / np.where(edges.min(axis=1) > 0, edges.min(axis=1), np.nan)

    return {
        "area": area,
        "min_angle": angles.min(axis=1),
        "max_angle": angles.max(axis=1),
        "aspect_ratio": np.nan_to_num(aspect, nan=np.inf),
        "scaled_jacobian": scaled_jacobian,
        "is_convex": is_convex,
    }


def evaluate_quality(mesh: Mesh) -> QualityReport:
    n_tri, n_quad = len(mesh.triangles), len(mesh.quads)
    tri = triangle_quality(mesh.nodes, mesh.triangles) if n_tri else None
    quad = quad_quality(mesh.nodes, mesh.quads) if n_quad else None

    def cat(key: str, tri_default=None, quad_default=None) -> np.ndarray:
        parts = []
        if n_tri:
            parts.append(tri[key] if key in tri else np.full(n_tri, tri_default))
        if n_quad:
            parts.append(quad[key] if key in quad else np.full(n_quad, quad_default))
        return np.concatenate(parts) if parts else np.empty(0)

    area = cat("area")
    tri_convex = np.ones(n_tri, dtype=bool) if n_tri else np.empty(0, dtype=bool)
    is_convex = (
        np.concatenate([tri_convex, quad["is_convex"]]) if n_quad
        else tri_convex
    )

    return QualityReport(
        area=np.abs(area),
        kind=mesh.element_kinds(),
        min_angle_deg=cat("min_angle"),
        max_angle_deg=cat("max_angle"),
        aspect_ratio=cat("aspect_ratio"),
        radius_ratio=cat("radius_ratio", quad_default=np.nan),
        scaled_jacobian=cat("scaled_jacobian", tri_default=np.nan),
        is_convex=is_convex,
    )


def quality_violations(
    report: QualityReport, cfg: Config, mesh: Mesh | None = None
) -> dict[str, np.ndarray]:
    """基準を満たさない要素のブールマスク。

    mesh を渡すと、外周に接する要素（`mesh.boundary_touching_element_mask()`）を
    area_below_min から除外する。境界（堤防など）の形状再現性を優先し、面積
    下限はこれらの要素には課さないため。形状指標（角度・アスペクト比等）は
    従来どおりすべての要素に課す。
    """
    q, m = cfg.quality, cfg.mesh
    is_tri = report.kind == 3
    is_quad = ~is_tri

    area_below_min = report.area < m.min_element_area
    if mesh is not None:
        area_below_min = area_below_min & ~mesh.boundary_touching_element_mask()

    return {
        "area_below_min": area_below_min,
        "tri_min_angle": is_tri & (report.min_angle_deg < q.triangle_min_angle_deg),
        "tri_radius_ratio": is_tri & (report.radius_ratio < q.triangle_radius_ratio_min),
        "tri_aspect_ratio": is_tri & (report.aspect_ratio > q.triangle_aspect_ratio_max),
        "quad_nonconvex": is_quad & ~report.is_convex,
        "quad_min_angle": is_quad & (report.min_angle_deg < q.quadrilateral_min_interior_angle_deg),
        "quad_max_angle": is_quad & (report.max_angle_deg > q.quadrilateral_max_interior_angle_deg),
        "quad_aspect_ratio": is_quad & (report.aspect_ratio > q.quadrilateral_aspect_ratio_max),
        "quad_jacobian": is_quad
        & (report.scaled_jacobian < q.quadrilateral_scaled_jacobian_min),
    }


def element_size_estimate(report: QualityReport) -> np.ndarray:
    """要素の代表辺長。三角形は正三角形換算、四角形は正方形換算。

    面積そのままで比べてはいけない。同じ辺長でも四角形の面積は正三角形の
    2.3 倍あり、三角形と四角形が隣り合うだけで比が 1.5 倍に見えてしまう。
    """
    is_tri = report.kind == 3
    size = np.empty(len(report))
    size[is_tri] = np.sqrt(4.0 * report.area[is_tri] / np.sqrt(3.0))
    size[~is_tri] = np.sqrt(report.area[~is_tri])
    return size


def element_adjacency(mesh: Mesh) -> np.ndarray:
    """辺を共有する要素の組 (m, 2)。番号は三角形 -> 四角形の順。

    頂点だけを共有する要素は隣とみなさない。要素ごとに時間刻みを変えるとき、
    フラックスをやり取りするのは辺を共有する相手だけであるため。
    """
    owners: dict[tuple[int, int], list[int]] = {}
    offset = 0
    for conn in (mesh.triangles, mesh.quads):
        m = conn.shape[1] if len(conn) else 0
        for local, elem in enumerate(conn):
            for k in range(m):
                a, b = int(elem[k]), int(elem[(k + 1) % m])
                owners.setdefault((a, b) if a < b else (b, a), []).append(offset + local)
        offset += len(conn)
    pairs = [v for v in owners.values() if len(v) == 2]
    return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def neighbor_ratio_violations(
    mesh: Mesh, report: QualityReport, max_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """隣接要素の辺長比が上限を超える箇所。

    Returns:
        (粗い側の要素マスク, 要素ごとの目標辺長)。目標は隣の細かい側の
        辺長 * max_ratio で、これ以下にすれば比が収まる。対象外は inf。
    """
    n = len(report)
    coarse_mask = np.zeros(n, dtype=bool)
    target = np.full(n, np.inf)
    if max_ratio <= 0.0 or n == 0:
        return coarse_mask, target

    pairs = element_adjacency(mesh)
    if not len(pairs):
        return coarse_mask, target

    size = element_size_estimate(report)
    s0, s1 = size[pairs[:, 0]], size[pairs[:, 1]]
    big = np.where(s0 >= s1, pairs[:, 0], pairs[:, 1])
    small = np.where(s0 >= s1, pairs[:, 1], pairs[:, 0])
    ratio = np.maximum(s0, s1) / np.maximum(np.minimum(s0, s1), 1e-12)

    bad = ratio > max_ratio
    if not bad.any():
        return coarse_mask, target

    coarse_mask[big[bad]] = True
    # 同じ要素が複数の相手と接していることがある。最も厳しい要求を採る。
    np.minimum.at(target, big[bad], size[small[bad]] * max_ratio)
    return coarse_mask, target


def neighbor_ratio_stats(
    mesh: Mesh, report: QualityReport, max_ratio: float
) -> dict[str, float]:
    """隣接要素の辺長比の分布。レベル差が 1 以内かの確認に使う。"""
    pairs = element_adjacency(mesh)
    if not len(pairs):
        return {}
    size = element_size_estimate(report)
    s0, s1 = size[pairs[:, 0]], size[pairs[:, 1]]
    ratio = np.maximum(s0, s1) / np.maximum(np.minimum(s0, s1), 1e-12)
    over = int((ratio > max_ratio).sum()) if max_ratio > 0.0 else 0
    return {
        "neighbor_pairs": len(pairs),
        "neighbor_ratio_median": float(np.median(ratio)),
        "neighbor_ratio_p99": float(np.percentile(ratio, 99)),
        "neighbor_ratio_max": float(ratio.max()),
        "viol_neighbor_ratio": over,
    }


def summarize(report: QualityReport, cfg: Config, mesh: Mesh | None = None) -> dict[str, float]:
    v = quality_violations(report, cfg, mesh)
    n = max(len(report), 1)
    out: dict[str, float] = {
        "n_elements": len(report),
        "n_triangles": int((report.kind == 3).sum()),
        "n_quads": int((report.kind == 4).sum()),
        "area_min": float(report.area.min()) if len(report) else float("nan"),
        "area_median": float(np.median(report.area)) if len(report) else float("nan"),
        "area_max": float(report.area.max()) if len(report) else float("nan"),
        "area_total_km2": float(report.area.sum()) / 1e6,
    }
    for name, mask in v.items():
        out[f"viol_{name}"] = int(mask.sum())
    out["viol_any"] = int(np.logical_or.reduce(list(v.values())).sum()) if v else 0
    out["viol_any_ratio"] = out["viol_any"] / n
    return out
