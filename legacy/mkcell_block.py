"""Flood-block helpers for parallel mkCELL processing."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import geopandas as gpd
from pyproj import CRS, Geod, Transformer


def area_m2(geom, crs) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    crs_obj = CRS(crs)
    if crs_obj.is_projected:
        factor = crs_obj.axis_info[0].unit_conversion_factor if crs_obj.axis_info else 1.0
        return float(geom.area) * float(factor) * float(factor)
    geod = crs_obj.get_geod() if hasattr(crs_obj, "get_geod") else Geod(ellps="WGS84")
    area_m2_val, _ = geod.geometry_area_perimeter(geom)
    return abs(float(area_m2_val))


def load_target_blocks(
    input_file: str,
    block_index: Optional[int] = None,
    min_block_area_m2: float = 100.0,
) -> Tuple[gpd.GeoDataFrame, int, list[int]]:
    """Load target area polygons, optionally selecting one block by row index."""
    gdf = gpd.read_file(input_file)
    if len(gdf) == 0:
        raise ValueError(f"target area is empty: {input_file}")

    valid_indices = []
    for idx, row in gdf.iterrows():
        if area_m2(row.geometry, gdf.crs) >= min_block_area_m2:
            valid_indices.append(int(idx))

    if block_index is not None:
        if block_index < 0 or block_index >= len(gdf):
            raise IndexError(f"block_index {block_index} out of range (0..{len(gdf) - 1})")
        if block_index not in valid_indices and area_m2(gdf.geometry.iloc[block_index], gdf.crs) < min_block_area_m2:
            raise ValueError(
                f"block {block_index} area {area_m2(gdf.geometry.iloc[block_index], gdf.crs):.2f} m² "
                f"< min_block_area_m2={min_block_area_m2}"
            )
        out = gdf.iloc[[block_index]].copy().reset_index(drop=True)
        out["block_index"] = block_index
        return out, 1, valid_indices

    if min_block_area_m2 > 0:
        gdf = gdf.iloc[valid_indices].copy().reset_index(drop=True)
        gdf["block_index"] = valid_indices

    return gdf, len(gdf), valid_indices


def block_output_dir(base_output: str, block_index: int) -> str:
    project_root = os.path.dirname(base_output)
    return os.path.join(project_root, "blocks", f"block_{block_index:03d}")


def _bbox_with_pad(bbox: Tuple[float, float, float, float], pad_ratio: float = 0.02):
    minx, miny, maxx, maxy = bbox
    dx = (maxx - minx) * pad_ratio
    dy = (maxy - miny) * pad_ratio
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def read_vector_bbox(path: str, bbox, target_crs) -> gpd.GeoDataFrame:
    """Read vector data intersecting bbox; bbox is in target_crs coordinates."""
    minx, miny, maxx, maxy = _bbox_with_pad(tuple(bbox))

    try:
        import pyogrio

        info = pyogrio.read_info(path)
        src_crs = info.get("crs")
    except Exception:
        src_crs = None

    if src_crs is not None and target_crs is not None and not CRS(src_crs).equals(CRS(target_crs)):
        transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
        xs, ys = transformer.transform([minx, maxx], [miny, maxy])
        read_bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        read_bbox = (minx, miny, maxx, maxy)

    try:
        gdf = gpd.read_file(path, bbox=read_bbox)
    except Exception:
        gdf = gpd.read_file(path)
        gdf = gdf.cx[minx:maxx, miny:maxy]

    if len(gdf) == 0:
        return gdf
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf
