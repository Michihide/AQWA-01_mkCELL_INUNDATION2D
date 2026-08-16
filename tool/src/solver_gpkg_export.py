"""01_mkMESH_INUN2DH と同形式の face/edge GeoPackage・CSV 出力。

output_gpkg_csv/face.gpkg, edge.gpkg（および face.csv, edge.csv）を書き出す。
cell.bin のヘッダーに埋め込むパスもここで決める。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from .cell_bin_export import (
    LANDUSE_CODES,
    SOIL_CODES,
    CellAttributes,
    _BOUNDARY_EDGE_SENTINEL,
    _find_neighbors,
    _local_edge_arrays,
    compute_cell_attributes,
)
from .config import Config
from .io_raster import DemGrid
from .mesh_parser import Mesh
from .quality_metrics import QualityReport
from .terrain_metrics import TerrainReport
from .utils import get_logger


def _node_elevations(mesh: Mesh, dem: DemGrid | None) -> np.ndarray:
    if mesh.node_z is not None and np.any(mesh.node_z != 0):
        z = mesh.node_z.copy()
    else:
        z = np.zeros(mesh.n_nodes)
    if dem is not None:
        sampled = dem.sample(mesh.nodes[:, 0], mesh.nodes[:, 1])
        z = np.where(np.isfinite(sampled), sampled, z)
    return z


def _physical_line_ids(node_a: np.ndarray, node_b: np.ndarray) -> np.ndarray:
    lo = np.minimum(node_a, node_b)
    hi = np.maximum(node_a, node_b)
    n_nodes = int(max(node_a.max(), node_b.max())) + 1 if len(node_a) else 0
    keys = lo.astype(np.int64) * (n_nodes + 1) + hi.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    line_id = np.zeros(len(keys), dtype=np.int32)
    current = 0
    prev = None
    for idx in order:
        key = keys[idx]
        if key != prev:
            current += 1
            prev = key
        line_id[idx] = current
    return line_id


def write_face_gpkg(
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    attrs: CellAttributes,
    crs: CRS,
    path: Path,
) -> Path:
    polys = [Polygon(coords) for coords in mesh.element_polygons()]
    n = len(polys)
    areas = quality.area
    centroids = mesh.element_centroids()
    elevation = np.where(np.isfinite(terrain.elevation), terrain.elevation, 0.0)
    bill_area = attrs.bld_ratio * areas
    ratio = np.divide(
        bill_area, areas, out=np.zeros_like(bill_area), where=areas > 0,
    )
    ratio = np.minimum(ratio, 0.95)

    lu_cols = {code: attrs.landuse_areas[:, i] for i, code in enumerate(LANDUSE_CODES)}
    majority = np.array([
        int(LANDUSE_CODES[int(np.argmax(attrs.landuse_areas[i]))])
        if attrs.landuse_areas[i].sum() > 0 else 0
        for i in range(n)
    ])

    data = {
        "CalMesh": np.zeros(n, dtype=np.int32),
        "area": areas,
        "CN": np.arange(1, n + 1, dtype=np.int32),
        "bill_area": bill_area,
        "bill_perimeter": attrs.bld_peri,
        "ratio": ratio,
        "_median": elevation,
        "50": lu_cols["50"],
        "_majority": majority,
        **{code: lu_cols[code] for code in LANDUSE_CODES if code != "50"},
        **{f"soil_{i}_area": attrs.soil_areas[:, i - 1] for i in range(1, 18)},
    }
    gdf = gpd.GeoDataFrame(data, geometry=polys, crs=crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG", layer="poly")
    get_logger().info("face GeoPackage を出力: %s", path)
    return path


def _edge_incidence(
    mesh: Mesh,
    quality: QualityReport,
    dem: DemGrid | None,
) -> dict[str, np.ndarray]:
    """AQWA edge.gpkg / edge.csv 用の辺レコード（面×局所辺）を組み立てる。"""
    owner, node_a, node_b = _local_edge_arrays(mesh)
    neighbor = _find_neighbors(owner, node_a, node_b)
    line_ids = _physical_line_ids(node_a, node_b)
    node_z = _node_elevations(mesh, dem)
    areas = quality.area[owner]
    merge2 = 0.5 * (node_z[node_a] + node_z[node_b])
    is_boundary = neighbor == _BOUNDARY_EDGE_SENTINEL
    return {
        "owner": owner,
        "node_a": node_a,
        "node_b": node_b,
        "line_ids": line_ids,
        "node_z": node_z,
        "areas": areas,
        "merge2": merge2,
        "is_boundary": is_boundary,
    }


def write_edge_gpkg(
    mesh: Mesh,
    quality: QualityReport,
    dem: DemGrid | None,
    crs: CRS,
    path: Path,
) -> Path:
    inc = _edge_incidence(mesh, quality, dem)
    owner = inc["owner"]
    node_a, node_b = inc["node_a"], inc["node_b"]
    line_ids = inc["line_ids"]

    x_a, y_a = mesh.nodes[node_a, 0], mesh.nodes[node_a, 1]
    x_b, y_b = mesh.nodes[node_b, 0], mesh.nodes[node_b, 1]
    geoms = [LineString([(x_a[i], y_a[i]), (x_b[i], y_b[i])]) for i in range(len(owner))]

    gdf = gpd.GeoDataFrame(
        {
            "CalMesh": np.zeros(len(owner), dtype=np.int32),
            "area": inc["areas"],
            "CN": owner.astype(np.int32) + 1,
            "LineID": line_ids,
            "StartNode": node_a.astype(np.int32) + 1,
            "EndNode": node_b.astype(np.int32) + 1,
            # AQWA では LN と LineID は同一の全局 ID
            "LN": line_ids,
            "ID": np.where(inc["is_boundary"], 1, 0).astype(np.int16),
            "merge2_mn": inc["merge2"],
        },
        geometry=geoms,
        crs=crs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG", layer="line")
    get_logger().info("edge GeoPackage を出力: %s (%d 本)", path, len(gdf))
    return path


def write_face_csv(
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    attrs: CellAttributes,
    path: Path,
) -> Path:
    n = mesh.n_elements
    areas = quality.area
    centroids = mesh.element_centroids()
    polys = mesh.element_polygons()
    elevation = np.where(np.isfinite(terrain.elevation), terrain.elevation, 0.0)
    bill_area = attrs.bld_ratio * areas
    ratio = np.divide(
        bill_area, areas, out=np.zeros_like(bill_area), where=areas > 0,
    )
    ratio = np.minimum(ratio, 0.95)
    df = pd.DataFrame({
        "CN": np.arange(1, n + 1, dtype=np.int32),
        "bill_area": bill_area,
        "area": areas,
        "ratio": ratio,
        "_median": elevation,
        "xcoord": centroids[:, 0],
        "ycoord": centroids[:, 1],
        "CalMesh": np.zeros(n, dtype=np.int32),
        "bill_perimeter": attrs.bld_peri,
        "wkt": [Polygon(c).wkt for c in polys],
        **{code: attrs.landuse_areas[:, i] for i, code in enumerate(LANDUSE_CODES)},
        **{f"soil_{i}_area": attrs.soil_areas[:, i - 1] for i in range(1, 18)},
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    get_logger().info("face CSV を出力: %s", path.name)
    return path


def write_edge_csv(
    mesh: Mesh,
    quality: QualityReport,
    dem: DemGrid | None,
    path: Path,
) -> Path:
    inc = _edge_incidence(mesh, quality, dem)
    owner, node_a, node_b = inc["owner"], inc["node_a"], inc["node_b"]
    line_ids, node_z, merge2 = inc["line_ids"], inc["node_z"], inc["merge2"]

    rows = []
    for i in range(len(owner)):
        cn = int(owner[i]) + 1
        edge_id = 1 if inc["is_boundary"][i] else 0
        ln = int(line_ids[i])
        sn, en = int(node_a[i]) + 1, int(node_b[i]) + 1
        m2 = float(merge2[i])
        for node_idx, nid in ((node_a[i], sn), (node_b[i], en)):
            rows.append({
                "CN": cn,
                "ID": edge_id,
                "LN": ln,
                "StartNode": sn,
                "EndNode": en,
                "xcoord": float(mesh.nodes[node_idx, 0]),
                "ycoord": float(mesh.nodes[node_idx, 1]),
                "node_id": nid,
                "merge1_mn": float(node_z[node_idx]),
                "merge2_mn": m2,
            })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    get_logger().info("edge CSV を出力: %s (%d 行)", path.name, len(df))
    return path


def write_solver_gpkg_csv(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    crs: CRS,
    dem: DemGrid | None,
) -> tuple[Path, Path]:
    """face/edge の GeoPackage と CSV を output.gpkg_csv で指定した場所へ書き出す。"""
    gc = cfg.output.gpkg_csv
    out_dir = cfg.gpkg_csv_dir
    attrs = compute_cell_attributes(cfg, mesh, quality, crs)

    face_gpkg = out_dir / gc.face_gpkg
    edge_gpkg = out_dir / gc.edge_gpkg
    write_face_gpkg(mesh, quality, terrain, attrs, crs, face_gpkg)
    write_edge_gpkg(mesh, quality, dem, crs, edge_gpkg)
    if gc.enabled:
        write_face_csv(mesh, quality, terrain, attrs, out_dir / gc.face_csv)
        write_edge_csv(mesh, quality, dem, out_dir / gc.edge_csv)
    return face_gpkg, edge_gpkg
