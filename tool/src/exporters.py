"""メッシュと診断情報の出力。

msh / VTU / XDMF は meshio、GeoPackage は geopandas、要素一覧は CSV で書き出す。
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import meshio
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from .mesh_parser import Mesh
from .quality_metrics import QualityReport
from .utils import get_logger


def _to_meshio(mesh: Mesh, cell_data: dict[str, list[np.ndarray]] | None = None) -> meshio.Mesh:
    points = np.column_stack([
        mesh.nodes,
        mesh.node_z if mesh.node_z is not None else np.zeros(mesh.n_nodes),
    ])
    cells = []
    if len(mesh.triangles):
        cells.append(("triangle", mesh.triangles))
    if len(mesh.quads):
        cells.append(("quad", mesh.quads))
    return meshio.Mesh(points=points, cells=cells, cell_data=cell_data or {})


def _split_cell_data(mesh: Mesh, values: np.ndarray) -> list[np.ndarray]:
    """要素順（三角形 -> 四角形）の配列を meshio のセルブロック単位に分ける。"""
    n_tri = len(mesh.triangles)
    out = []
    if n_tri:
        out.append(values[:n_tri])
    if len(mesh.quads):
        out.append(values[n_tri:])
    return out


def write_mesh_files(
    mesh: Mesh,
    report: QualityReport,
    out_dir: Path,
    basename: str,
    write_msh: bool = True,
    write_vtu: bool = True,
    write_xdmf: bool = True,
) -> list[Path]:
    logger = get_logger()
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_data = {
        "area": _split_cell_data(mesh, report.area),
        "min_angle_deg": _split_cell_data(mesh, report.min_angle_deg),
        "aspect_ratio": _split_cell_data(mesh, np.nan_to_num(report.aspect_ratio, posinf=1e6)),
        "is_convex": _split_cell_data(mesh, report.is_convex.astype(np.int8)),
    }
    mio = _to_meshio(mesh, cell_data)

    written: list[Path] = []
    if write_vtu:
        p = out_dir / f"{basename}.vtu"
        mio.write(p)
        written.append(p)
    if write_xdmf:
        p = out_dir / f"{basename}.xdmf"
        mio.write(p)
        written.append(p)
    if write_msh:
        p = out_dir / f"{basename}.msh"
        # gmsh 形式は cell_data の型が限られるため、幾何のみ書き出す
        _to_meshio(mesh).write(p, file_format="gmsh22", binary=False)
        written.append(p)

    logger.info("メッシュを出力: %s", ", ".join(p.name for p in written))
    return written


def write_element_gpkg(
    mesh: Mesh, report: QualityReport, crs: CRS, path: Path, layer: str = "elements"
) -> Path:
    """要素をポリゴンとして GeoPackage へ。QGIS で品質を可視化するために使う。"""
    polys = [Polygon(coords) for coords in mesh.element_polygons()]
    gdf = gpd.GeoDataFrame(
        {
            "element_id": np.arange(len(polys)),
            "kind": np.where(report.kind == 3, "triangle", "quad"),
            "area": report.area,
            "min_angle_deg": report.min_angle_deg,
            "max_angle_deg": report.max_angle_deg,
            "aspect_ratio": np.nan_to_num(report.aspect_ratio, posinf=1e6),
            "radius_ratio": report.radius_ratio,
            "scaled_jacobian": report.scaled_jacobian,
            "is_convex": report.is_convex,
            "surface": mesh.element_surfaces(),
        },
        geometry=polys,
        crs=crs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, layer=layer, driver="GPKG")
    get_logger().info("要素 GeoPackage を出力: %s", path.name)
    return path


def write_element_csv(mesh: Mesh, report: QualityReport, path: Path) -> Path:
    centroids = mesh.element_centroids()
    df = pd.DataFrame({
        "element_id": np.arange(len(report)),
        "kind": np.where(report.kind == 3, "triangle", "quad"),
        "centroid_x": centroids[:, 0],
        "centroid_y": centroids[:, 1],
        "area": report.area,
        "min_angle_deg": report.min_angle_deg,
        "max_angle_deg": report.max_angle_deg,
        "aspect_ratio": np.nan_to_num(report.aspect_ratio, posinf=1e6),
        "radius_ratio": report.radius_ratio,
        "scaled_jacobian": report.scaled_jacobian,
        "is_convex": report.is_convex,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def write_node_csv(mesh: Mesh, path: Path) -> Path:
    df = pd.DataFrame({
        "node_id": np.arange(mesh.n_nodes),
        "x": mesh.nodes[:, 0],
        "y": mesh.nodes[:, 1],
        "z": mesh.node_z if mesh.node_z is not None else np.zeros(mesh.n_nodes),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def write_connectivity_csv(mesh: Mesh, path: Path) -> Path:
    rows = []
    for i, tri in enumerate(mesh.triangles):
        rows.append((i, 3, tri[0], tri[1], tri[2], -1))
    offset = len(mesh.triangles)
    for i, quad in enumerate(mesh.quads):
        rows.append((offset + i, 4, quad[0], quad[1], quad[2], quad[3]))
    df = pd.DataFrame(rows, columns=["element_id", "n_nodes", "n0", "n1", "n2", "n3"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_summary_json(summary: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=_json_default)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"JSON にできない型です: {type(obj)}")


def write_iteration_log(rows: list[dict], path: Path) -> Path:
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    return path
