"""参照レイヤ（OSM 道路等）をサイズ場へ反映する。

拘束ブレークラインと違い、メッシュ辺として固定はしない。外周四角形帯の内側
（`PolygonBand.interior`）にクリップした線に沿って目標サイズを下げ、道路付近だけ
細かくする。解析領域の外周（Γ0/Γ1 帯）は対象外。
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .boundary_quad_band import PolygonBand
from .config import Config
from .io_vector import ReferenceLayers
from .size_field import SizeField
from .utils import get_logger


def _densify(coords: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    out = [coords[:1]]
    for i, length in enumerate(seg):
        n = max(1, int(np.ceil(length / step)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(coords[i] + t * (coords[i + 1] - coords[i]))
    return np.vstack(out)


def _interior_union(bands: list[PolygonBand]) -> BaseGeometry | None:
    parts = [
        b.interior for b in bands
        if b.interior is not None and not b.interior.is_empty
    ]
    if not parts:
        return None
    return unary_union(parts)


def _lines_from_geom(geom: BaseGeometry) -> list[LineString]:
    if isinstance(geom, LineString):
        return [geom] if len(geom.coords) >= 2 else []
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if len(g.coords) >= 2]
    if geom.geom_type == "GeometryCollection":
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_lines_from_geom(g))
        return out
    return []


def _layer_target_size(cfg: Config, layer_name: str) -> float:
    key = layer_name.lower()
    if "levee" in key or "embankment" in key:
        return cfg.mesh.levee_target_size
    return cfg.mesh.road_target_size


def apply_reference_layers_to_size_field(
    cfg: Config,
    size_field: SizeField,
    bands: list[PolygonBand],
    references: ReferenceLayers,
    *,
    count_attempts: bool = False,
) -> int:
    """内側領域にクリップした参照線に沿ってサイズ場を細かくする。"""
    if not cfg.features.reference_layers_for_size_field or references.is_empty():
        return 0

    interior = _interior_union(bands)
    if interior is None or interior.is_empty:
        return 0

    logger = get_logger()
    step = max(cfg.mesh.size_field_grid * 0.5, 1.0)
    changed = 0
    clipped_km = 0.0

    for name, gdf in references.layers.items():
        target = float(np.clip(
            _layer_target_size(cfg, name),
            cfg.mesh.global_min_size,
            cfg.mesh.global_max_size,
        ))
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            clipped = geom.intersection(interior)
            if clipped.is_empty:
                continue
            clipped_km += float(getattr(clipped, "length", 0.0)) / 1000.0
            for line in _lines_from_geom(clipped):
                xy = _densify(np.asarray(line.coords), step)
                changed += size_field.apply_min(
                    xy[:, 0], xy[:, 1],
                    np.full(len(xy), target),
                    count_attempts=count_attempts,
                )

    if changed:
        size_field.smooth(cfg.mesh.max_neighbor_size_ratio)
        logger.info(
            "参照レイヤをサイズ場へ反映: %d 格子点を更新（内側 %.1f km, target≈%.0f m）",
            changed, clipped_km, cfg.mesh.road_target_size,
        )
    return changed
