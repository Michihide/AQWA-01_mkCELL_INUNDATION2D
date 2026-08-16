"""反復局所細分化。

各反復で、地形基準・品質基準を満たさない要素の位置にだけ小さい目標サイズを
書き込み、再メッシュする。面積下限 625 m^2 は細分化の床であり、下限に達した
要素はそれ以上細かくしない。逆に、基準を十分に満たしている大きな空き領域では
サイズを緩めて要素数を減らす（粗大化）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .mesh_parser import Mesh
from .quality_metrics import (
    QualityReport,
    element_size_estimate,
    neighbor_ratio_violations,
    quality_violations,
)
from .size_field import SizeField
from .terrain_metrics import TerrainReport, terrain_violations
from .utils import get_logger


@dataclass
class RefinementStep:
    """1 反復分の記録。"""

    iteration: int
    n_elements: int
    n_triangles: int
    n_quads: int
    area_min: float
    n_refined: int
    n_coarsened: int
    n_at_area_floor: int
    violations: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {
            "iteration": self.iteration,
            "n_elements": self.n_elements,
            "n_triangles": self.n_triangles,
            "n_quads": self.n_quads,
            "area_min": self.area_min,
            "n_refined": self.n_refined,
            "n_coarsened": self.n_coarsened,
            "n_at_area_floor": self.n_at_area_floor,
        }
        row.update(self.violations)
        return row


def refine_size_field(
    size_field: SizeField,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport | None,
    cfg: Config,
) -> tuple[int, int, np.ndarray]:
    """違反要素の位置で目標サイズを縮め、余裕のある要素では緩める。

    Returns:
        (細分化した要素数, 粗大化した要素数, 細分化対象マスク)
    """
    m = cfg.mesh
    centroids = mesh.element_centroids()
    current = element_size_estimate(quality)

    q_viol = quality_violations(quality, cfg, mesh)
    # 面積下限違反は「細かすぎる」ことが原因なので細分化の対象にしない
    refine_mask = np.zeros(len(quality), dtype=bool)
    for name, mask in q_viol.items():
        if name == "area_below_min":
            continue
        refine_mask |= mask
    if terrain is not None:
        for mask in terrain_violations(terrain, cfg).values():
            refine_mask |= mask

    # すでに面積下限に達している要素はこれ以上細かくできない
    at_floor = quality.area <= m.min_element_area * 1.05
    refine_mask &= ~at_floor

    # 細分化しても解消しない基準（勾配方向のばらつきなど）で無限に細かくしないよう、
    # 同じ場所を細分化する回数に上限を設ける
    if refine_mask.any():
        refine_mask &= size_field.refinable(
            centroids[:, 0], centroids[:, 1], m.max_refine_attempts
        )

    target = np.full(len(quality), np.inf)
    target[refine_mask] = current[refine_mask] * m.refinement_factor

    # 隣り合う要素の辺長比を上限以下にする。粗い側だけを縮めればよく、細かい側を
    # 粗くすると地形基準を満たさなくなる。帯の四角形は Transfinite で分割数が
    # 決まっており縮まないので、そちらが粗い側の組は諦める。
    balance_mask, balance_target = neighbor_ratio_violations(
        mesh, quality, m.max_neighbor_element_ratio
    )
    balance_mask &= ~at_floor & ~mesh.band_element_mask()
    if balance_mask.any():
        # 上限ちょうどを狙うと、gmsh が目標より大きい要素を作った分だけ超える。
        # 通常の細分化幅も併せて課し、下振れさせる。
        want = np.minimum(
            balance_target[balance_mask], current[balance_mask] * m.refinement_factor
        )
        # 帯の内側境界や拘束線の付近ではサイズ下限を高くしてあるが、それが比を
        # 守れない原因になっているなら、その位置だけ緩める。面積下限に直結する
        # global_min_size は下回らない。
        size_field.lower_floor(
            centroids[balance_mask, 0], centroids[balance_mask, 1],
            want, m.global_min_size,
        )
        target[balance_mask] = np.minimum(target[balance_mask], want)
        refine_mask |= balance_mask

    apply = np.isfinite(target)
    n_refined = 0
    if apply.any():
        size_field.apply_min(centroids[apply, 0], centroids[apply, 1], target[apply])
        n_refined = int(apply.sum())

    # 粗大化: 面積下限を割った要素と、基準に十分な余裕がある要素。
    # 境界に接する要素は面積下限の対象外なので、ここでも除外する
    # （境界の形状再現性を優先し、サイズ場をいじって形を変えない）。
    coarsen_mask = (quality.area < m.min_element_area) & ~mesh.boundary_touching_element_mask()
    n_coarsened = 0
    if coarsen_mask.any():
        target = np.maximum(
            current[coarsen_mask] / m.refinement_factor,
            np.sqrt(4.0 * m.min_element_area / np.sqrt(3.0)),
        )
        size_field.apply_max(
            centroids[coarsen_mask, 0], centroids[coarsen_mask, 1], target
        )
        n_coarsened = int(coarsen_mask.sum())

    size_field.clamp()
    size_field.smooth(m.max_neighbor_size_ratio)
    return n_refined, n_coarsened, refine_mask


def has_converged(
    mesh: Mesh, quality: QualityReport, terrain: TerrainReport | None, cfg: Config
) -> tuple[bool, dict[str, int]]:
    """全基準を満たしているか。満たしていない項目の件数も返す。"""
    counts: dict[str, int] = {}
    for name, mask in quality_violations(quality, cfg, mesh).items():
        counts[f"viol_{name}"] = int(mask.sum())
    if terrain is not None:
        for name, mask in terrain_violations(terrain, cfg).items():
            counts[f"terrain_viol_{name}"] = int(mask.sum())

    # 面積下限に達したために細分化できない要素は「解消不能」として扱う
    at_floor = quality.area <= cfg.mesh.min_element_area * 1.05
    blocked = 0
    if terrain is not None:
        for mask in terrain_violations(terrain, cfg).values():
            blocked += int((mask & at_floor).sum())

    coarse, _ = neighbor_ratio_violations(
        mesh, quality, cfg.mesh.max_neighbor_element_ratio
    )
    counts["viol_neighbor_ratio"] = int(coarse.sum())
    # 帯の四角形と面積下限に達した要素が粗い側なら、細分化しても縮まない
    blocked += int((coarse & (at_floor | mesh.band_element_mask())).sum())
    counts["blocked_by_area_floor"] = blocked

    unresolved = sum(v for k, v in counts.items() if k != "blocked_by_area_floor") - blocked
    return unresolved <= 0, counts


def summarize_unresolvable(
    mesh: Mesh, quality: QualityReport, terrain: TerrainReport | None, size_field,
    cfg: Config, centroids: np.ndarray,
) -> dict[str, int]:
    """細分化では解消できない違反の内訳。

    面積下限に達した要素と、細分化回数の上限に達した位置の要素を分けて数える。
    隣接比については、サイズ場の下限に張り付いている要素も分けて数える。外周帯の
    内側境界付近では下限を高くしているため、そこで比が開くと縮められない。
    """
    at_floor = quality.area <= cfg.mesh.min_element_area * 1.05
    exhausted = ~size_field.refinable(
        centroids[:, 0], centroids[:, 1], cfg.mesh.max_refine_attempts
    )
    viol = np.zeros(len(quality), dtype=bool)
    for name, mask in quality_violations(quality, cfg, mesh).items():
        if name != "area_below_min":
            viol |= mask
    if terrain is not None:
        for mask in terrain_violations(terrain, cfg).values():
            viol |= mask
    coarse, _ = neighbor_ratio_violations(
        mesh, quality, cfg.mesh.max_neighbor_element_ratio
    )
    size_floor = size_field.at_floor(centroids[:, 0], centroids[:, 1])
    return {
        "unresolvable_area_floor": int((viol & at_floor).sum()),
        "unresolvable_refine_limit": int((viol & ~at_floor & exhausted).sum()),
        "resolvable_remaining": int((viol & ~at_floor & ~exhausted).sum()),
        "neighbor_ratio_at_size_floor": int((coarse & size_floor).sum()),
        "neighbor_ratio_at_band": int((coarse & mesh.band_element_mask()).sum()),
    }


def log_step(step: RefinementStep) -> None:
    logger = get_logger()
    viol = ", ".join(
        f"{k}={v}" for k, v in step.violations.items() if isinstance(v, int) and v
    )
    logger.info(
        "反復 %d: 要素 %d (三角 %d / 四角 %d), 最小面積 %.0f m^2,"
        " 細分化 %d, 粗大化 %d%s",
        step.iteration, step.n_elements, step.n_triangles, step.n_quads,
        step.area_min, step.n_refined, step.n_coarsened,
        f" | 違反: {viol}" if viol else " | 違反なし",
    )
