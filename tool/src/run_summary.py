"""run.py の最終出力向けセル品質サマリー。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .quality_metrics import neighbor_ratio_stats, summarize
from .terrain_metrics import summarize_terrain

if TYPE_CHECKING:
    from .config import Config
    from .mesh_parser import Mesh
    from .quality_metrics import QualityReport
    from .terrain_metrics import TerrainReport


def build_quality_summary(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport | None,
    *,
    extra: dict | None = None,
) -> dict:
    """メッシュ評価結果を summary.json 互換の dict にまとめる。"""
    out = summarize(quality, cfg, mesh)
    out.update(neighbor_ratio_stats(mesh, quality, cfg.mesh.max_neighbor_element_ratio))
    if terrain is not None:
        out.update(summarize_terrain(terrain, cfg, quality.area))
    if extra:
        out.update(extra)
    return out


def aggregate_block_summaries(summaries: list[dict]) -> dict:
    """ブロック summary.json のリストから全域集計を作る（merged mesh が無い場合の補助）。"""
    if not summaries:
        return {}

    def col(k: str) -> list:
        return [s[k] for s in summaries if k in s]

    n_el = sum(col("n_elements"))
    area_km2 = sum(col("area_total_km2"))
    viol_any = sum(col("viol_any"))
    terrain_viol = sum(col("terrain_viol_any"))

    rmse_w = 0.0
    spread_w = 0.0
    if area_km2 > 0:
        for s in summaries:
            w = s.get("area_total_km2", 0.0)
            rmse_w += s.get("plane_fit_rmse_area_weighted", 0.0) * w
            spread_w += s.get("slope_spread_area_weighted", 0.0) * w
        rmse_w /= area_km2
        spread_w /= area_km2

    band_count = sum(s.get("band_quads", {}).get("count", 0) for s in summaries)

    return {
        "n_elements": n_el,
        "n_triangles": sum(col("n_triangles")),
        "n_quads": sum(col("n_quads")),
        "area_total_km2": area_km2,
        "area_min": min(col("area_min")) if col("area_min") else float("nan"),
        "area_max": max(col("area_max")) if col("area_max") else float("nan"),
        "area_median": _area_weighted_median(
            [(s["area_median"], s["area_total_km2"]) for s in summaries if "area_median" in s]
        ),
        "viol_area_below_min": sum(col("viol_area_below_min")),
        "viol_any": viol_any,
        "viol_any_ratio": viol_any / max(n_el, 1),
        "terrain_viol_any": terrain_viol,
        "terrain_viol_any_ratio": terrain_viol / max(n_el, 1),
        "plane_fit_rmse_area_weighted": rmse_w,
        "slope_spread_area_weighted": spread_w,
        "n_iterations": max(col("n_iterations")) if col("n_iterations") else 0,
        "band_quad_count": band_count,
        "viol_neighbor_ratio": sum(col("viol_neighbor_ratio")),
    }


def _area_weighted_median(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return float("nan")
    pairs = sorted(pairs, key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float(pairs[len(pairs) // 2][0])
    half = total / 2.0
    acc = 0.0
    for med, w in pairs:
        acc += w
        if acc >= half:
            return float(med)
    return float(pairs[-1][0])


def quality_comment_lines(summary: dict, cfg: Config) -> list[str]:
    """最終出力に付けるセル品質コメント行。"""
    n = int(summary.get("n_elements", 0))
    if n <= 0:
        return []

    m = cfg.mesh
    t = cfg.terrain
    lines = ["", "--- セル品質 ---"]

    n_tri = int(summary.get("n_triangles", 0))
    n_quad = int(summary.get("n_quads", 0))
    lines.append(f"要素数        : {n:,} (三角 {n_tri:,} / 四角 {n_quad:,})")

    if "area_total_km2" in summary:
        density = n / summary["area_total_km2"] if summary["area_total_km2"] > 0 else 0.0
        lines.append(
            f"面積          : {summary['area_total_km2']:.3f} km²"
            f" ({density:,.0f} 要素/km²)"
        )

    if "area_min" in summary:
        lines.append(
            f"要素面積      : {summary['area_min']:.0f} - {summary['area_max']:.0f} m²"
            f" (中央値 {summary.get('area_median', float('nan')):.0f})"
        )

    viol_area = int(summary.get("viol_area_below_min", 0))
    lines.append(
        f"面積下限      : {viol_area} 件違反"
        f" (下限 {m.min_element_area:.0f} m²)"
        + (" — OK" if viol_area == 0 else " — 要確認")
    )

    viol_any = int(summary.get("viol_any", 0))
    viol_ratio = summary.get("viol_any_ratio", viol_any / max(n, 1))
    lines.append(
        f"形状品質      : {viol_any} 件 ({100.0 * viol_ratio:.2f}%)"
        + (" — OK" if viol_ratio < 0.01 else " — 外周四角形の角度・縦横比等を確認")
    )

    viol_nb = int(summary.get("viol_neighbor_ratio", 0))
    nb_max = summary.get("neighbor_ratio_max")
    nb_suffix = f", 最大比 {nb_max:.2f}" if nb_max is not None else ""
    lines.append(
        f"隣接サイズ比  : {viol_nb} 件違反 (上限 {m.max_neighbor_size_ratio:.1f}{nb_suffix})"
        + (" — OK" if viol_nb == 0 else "")
    )

    band_n = summary.get("band_quad_count")
    if band_n is None and "band_quads" in summary:
        band_n = summary["band_quads"].get("count")
    if band_n is not None:
        lines.append(f"外周四角形    : {int(band_n):,} 個")

    if "plane_fit_rmse_area_weighted" in summary:
        rmse_w = summary["plane_fit_rmse_area_weighted"]
        rmse_ok = rmse_w <= t.plane_fit_rmse_max
        lines.append(
            f"地形 RMSE     : 面積加重 {rmse_w:.3f} m"
            f" (基準 {t.plane_fit_rmse_max} m)"
            + (" — OK" if rmse_ok else " — 基準超過（急斜面・面積下限律速の可能性）")
        )

    if "slope_spread_area_weighted" in summary:
        spread_w = summary["slope_spread_area_weighted"]
        spread_ok = spread_w <= t.slope_direction_spread_max_deg
        lines.append(
            f"勾配方向      : 面積加重 {spread_w:.1f}°"
            f" (基準 {t.slope_direction_spread_max_deg}°)"
            + (" — OK" if spread_ok else " — 基準超過")
        )

    if "terrain_viol_any" in summary:
        tv = int(summary["terrain_viol_any"])
        tv_ratio = summary.get("terrain_viol_any_ratio", tv / max(n, 1))
        lines.append(
            f"地形基準(要素): {tv:,} 件 ({100.0 * tv_ratio:.1f}%)"
            " — 要素単位。面積下限で細分化不能な領域を含む"
        )

    if "n_iterations" in summary:
        lines.append(f"反復回数      : {int(summary['n_iterations'])}")

    lines.append("")
    lines.append(f"総合          : {_overall_comment(summary, cfg)}")
    return lines


def _overall_comment(summary: dict, cfg: Config) -> str:
    n = max(int(summary.get("n_elements", 0)), 1)
    parts: list[str] = []

    if summary.get("viol_area_below_min", 0) == 0:
        parts.append("面積下限は全要素で充足")
    else:
        parts.append("面積下限違反あり")

    viol_ratio = summary.get("viol_any_ratio", summary.get("viol_any", 0) / n)
    if viol_ratio < 0.01:
        parts.append("形状品質は良好")
    elif viol_ratio < 0.05:
        parts.append("形状品質はおおむね良好（外周四角形に少数の警告）")
    else:
        parts.append("形状品質に要確認箇所あり")

    rmse_w = summary.get("plane_fit_rmse_area_weighted")
    if rmse_w is not None:
        if rmse_w <= cfg.terrain.plane_fit_rmse_max:
            parts.append("代表標高（面積加重 RMSE）は基準内")
        else:
            parts.append("代表標高（面積加重 RMSE）は基準超過")

    tv_ratio = summary.get("terrain_viol_any_ratio")
    if tv_ratio is not None:
        if tv_ratio < 0.05:
            parts.append("要素単位の地形基準もほぼ充足")
        elif tv_ratio < 0.5:
            parts.append("要素単位の地形基準は一部未達（平坦部中心なら許容しうる）")
        else:
            parts.append(
                "要素単位の地形基準は多数未達"
                "（min_element_area 律速、または terrain 閾値が厳しすぎる可能性）"
            )

    return "。".join(parts) + "。"
