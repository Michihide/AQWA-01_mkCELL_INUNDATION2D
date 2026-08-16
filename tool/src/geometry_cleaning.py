"""解析領域ポリゴンの修復とリサンプリング。

重要な地形境界を壊す過度な simplify は行わない。除去するのは、gmsh へ渡すと
退化要素の原因になる重複頂点・極小穴・スパイクのみ。
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.validation import make_valid

from .utils import get_logger

MITRE_LIMIT = 3.0


def clean_polygon(
    poly: Polygon,
    min_area: float,
    min_hole_area: float,
    duplicate_tol: float = 1e-6,
) -> Polygon | None:
    """1 ポリゴンを修復する。修復不能・極小なら None。"""
    if poly is None or poly.is_empty:
        return None

    if not poly.is_valid:
        fixed = make_valid(poly)
        if fixed.is_empty:
            fixed = poly.buffer(0)
        poly = _largest_polygon(fixed)
        if poly is None:
            return None

    exterior = _dedupe_ring(poly.exterior, duplicate_tol)
    if exterior is None:
        return None

    holes = []
    for ring in poly.interiors:
        if Polygon(ring).area < min_hole_area:
            continue
        cleaned = _dedupe_ring(ring, duplicate_tol)
        if cleaned is not None:
            holes.append(cleaned)

    result = Polygon(exterior, holes)
    if not result.is_valid:
        result = _largest_polygon(make_valid(result))
        if result is None:
            return None
    if result.area < min_area:
        return None
    return result


def clean_polygons(
    polys: list[Polygon],
    min_area: float,
    min_hole_area: float | None = None,
    narrow_feature_radius: float = 0.0,
) -> list[Polygon]:
    logger = get_logger()
    if min_hole_area is None:
        min_hole_area = min_area
    out: list[Polygon] = []
    n_dropped = 0
    area_before = 0.0
    for poly in polys:
        cleaned = clean_polygon(poly, min_area, min_hole_area)
        if cleaned is None:
            n_dropped += 1
            continue
        area_before += cleaned.area
        if narrow_feature_radius > 0.0:
            parts = remove_narrow_features(cleaned, narrow_feature_radius, min_area)
            if not parts:
                logger.warning(
                    "細部除去（半径 %.1f m）でポリゴンが消失したため、"
                    "開処理前の形状を保持します（面積 %.0f m²）",
                    narrow_feature_radius, cleaned.area,
                )
                out.append(cleaned)
            else:
                out.extend(parts)
        else:
            out.append(cleaned)

    if n_dropped:
        logger.warning("修復できない/極小なポリゴンを %d 個除外しました", n_dropped)
    if narrow_feature_radius > 0.0 and area_before > 0.0:
        area_after = sum(p.area for p in out)
        logger.info(
            "細部除去（半径 %.1f m）: %d -> %d ポリゴン、面積 %.3f -> %.3f km^2 (損失 %.2f%%)",
            narrow_feature_radius, len(polys), len(out),
            area_before / 1e6, area_after / 1e6,
            100.0 * (1.0 - area_after / area_before),
        )
    return out


def remove_narrow_features(poly: Polygon, radius: float, min_area: float) -> list[Polygon]:
    """幅 2*radius 未満のくびれ・スパイクを除去する（モルフォロジー的開処理）。

    外周四角形帯は帯幅 2 つ分より細い部分に入れないため、そのままだと内向き
    オフセットが自己交差する。ここで落としておく。角を丸めないよう mitre 継ぎで
    処理し、結果が複数に分かれた場合はそれぞれを独立したポリゴンとして返す。
    """
    if radius <= 0.0:
        return [poly]
    opened = (
        poly.buffer(-radius, join_style="mitre", mitre_limit=MITRE_LIMIT)
        .buffer(radius, join_style="mitre", mitre_limit=MITRE_LIMIT)
    )
    if opened.is_empty:
        return []
    parts = list(opened.geoms) if isinstance(opened, MultiPolygon) else [opened]
    out = []
    for part in parts:
        if not isinstance(part, Polygon) or part.area < min_area:
            continue
        # 開処理は元領域からはみ出さないが、数値誤差の分だけ内側へ寄せる。
        # くびれの両側が別々の大きな塊として残ることがあるので、交差の
        # 断片は最大のものだけでなく全部を候補として残す（そうしないと
        # 領域が丸ごと 1 個抜け落ちる）。
        clipped = part.intersection(poly)
        for piece in _all_polygons(clipped):
            if piece.area < min_area:
                continue
            cleaned = clean_polygon(piece, min_area, min_area)
            if cleaned is not None:
                out.append(cleaned)
    return out


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        return max(parts, key=lambda g: g.area) if parts else None
    if hasattr(geom, "geoms"):
        parts = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        return max(parts, key=lambda g: g.area) if parts else None
    return None


def _all_polygons(geom) -> list[Polygon]:
    """geom に含まれる Polygon をすべて返す（GeometryCollection も辿る）。

    くびれの開処理で本体を元領域と交差させると、くびれの両側が別々の
    Polygon に分かれることがある。最大のものだけを残すと領域が欠ける。
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(_all_polygons(g))
        return out
    return []


def _dedupe_ring(ring: LinearRing, tol: float) -> LinearRing | None:
    coords = np.asarray(ring.coords)[:-1]  # 閉じ点を除く
    if len(coords) < 3:
        return None
    keep = [0]
    for i in range(1, len(coords)):
        if np.hypot(*(coords[i] - coords[keep[-1]])) > tol:
            keep.append(i)
    # 先頭と末尾が重なる場合は末尾を捨てる
    while len(keep) > 3 and np.hypot(*(coords[keep[-1]] - coords[keep[0]])) <= tol:
        keep.pop()
    if len(keep) < 3:
        return None
    return LinearRing(coords[keep])


def resample_ring(ring: LinearRing, spacing: float) -> np.ndarray:
    """リングを指定間隔で等間隔リサンプリングし、閉じ点を含まない (N, 2) を返す。

    元の頂点位置は保持しない。長辺は分割され、密集した頂点は間引かれるため、
    外周四角形帯の接線方向サイズを一定に保てる。
    """
    length = ring.length
    n = max(4, int(round(length / spacing)))
    distances = np.linspace(0.0, length, n, endpoint=False)
    pts = np.array([ring.interpolate(d).coords[0] for d in distances])
    return pts


def polygon_rings_resampled(poly: Polygon, spacing: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """外周と穴をリサンプリングして返す。

    外周は反時計回り (CCW)、穴は時計回り (CW) に揃える。オフセット方向の計算を
    向きに依存させないため。
    """
    exterior = resample_ring(poly.exterior, spacing)
    if _signed_area(exterior) < 0:
        exterior = exterior[::-1]

    holes = []
    for ring in poly.interiors:
        pts = resample_ring(ring, spacing)
        if _signed_area(pts) > 0:
            pts = pts[::-1]
        holes.append(pts)
    return exterior, holes


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
