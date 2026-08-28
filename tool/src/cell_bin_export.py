"""AQWA-INUNDATION2D の cell.bin（Fortran stream unformatted）出力。

02_Solver/compile/src/inun2dh/inun2dh_io.f90 がそのまま読み込めるバイナリを
書き出す。旧 cell.bin（Fortran stream unformatted）と同じ並び
（リトルエンディアン）。

  ヘッダー: face_gpkg_path(1000B), edge_gpkg_path(1000B),
            total_face(i4), total_edge(i4), total_face_index(i4)
  セル × total_face:
    face_number, total_edge_in_face (i4×2)
    x, y, A, bl, bld_ratio, bld_peri (r8×6)
    A_lu10..A_lu255 (r8×13, 固定コード順)
    A_soil1..A_soil17 (r8×17, 固定コード順)
    dummy_face（CalMesh。ダミーメッシュを使わないため常に 0）(i4)
    辺 × total_edge_in_face:
      en, bc_flag, fn_self, fn_adj (i4×4)
      A_CV_edge, dl, dxdl, dydl, weight_self, weight_adj, weight_face (r8×7)
      vertex_number_st, vertex_number_en (i4×2)
      x_vertex_st, y_vertex_st, bl_vertex_st,
        x_vertex_en, y_vertex_en, bl_vertex_en,
        x_edge_cnt, y_edge_cnt, bl_edge_cnt, edge_length (r8×10)

en（辺レコード ID）は edge.gpkg の LN / LineID と同じ共有辺 ID を書く。
LFP 連成は LN で edge_length を参照するため、通し番号だと溢水時に破綻する。
内部の辺は隣接 2 面から同じ en で 1 レコードずつ書き出される。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS
from shapely.geometry import Polygon

from .config import CellBinConfig, Config
from .io_raster import DemGrid
from .mesh_parser import Mesh
from .quality_metrics import QualityReport
from .terrain_metrics import TerrainReport
from .utils import get_logger
from .zonal_stats import (
    building_ratio_perimeter,
    load_buildings_bbox,
    raster_code_areas,
    vegetation_chi_and_params,
)

LANDUSE_CODES = ["10", "20", "50", "60", "70", "91", "92", "100", "110", "140", "150", "160", "255"]
SOIL_CODES = [str(i) for i in range(1, 18)]

_FACE_STRUCT = struct.Struct("<ii6d13d17di")
_EDGE_STRUCT = struct.Struct("<iiii7dii10d")
_PATH_FIELD_LEN = 1000
_BOUNDARY_EDGE_SENTINEL = -1


@dataclass
class CellAttributes:
    landuse_areas: np.ndarray
    soil_areas: np.ndarray
    bld_ratio: np.ndarray
    bld_peri: np.ndarray
    chi_veg: np.ndarray
    a_veg: np.ndarray
    H_veg: np.ndarray
    Cd_veg: np.ndarray


def compute_cell_attributes(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    crs: CRS,
) -> CellAttributes:
    """mesh.bin / face.gpkg 共通の土地利用・土壌・建物・植生属性を求める。"""
    logger = get_logger()
    cb = cfg.output.cell_bin
    polygons = [Polygon(coords) for coords in mesh.element_polygons()]
    n = len(polygons)

    if cb.landuse_raster:
        path = cfg.resolve(cb.landuse_raster)
        logger.info("土地利用ラスタの重なり面積を計算中: %s", path)
        landuse_areas = raster_code_areas(
            polygons, crs, path, LANDUSE_CODES, fallback_epsg=cb.landuse_epsg,
        )
    else:
        logger.warning("output.cell_bin.landuse_raster が未指定のため土地利用面積を 0 で埋めます")
        landuse_areas = np.zeros((n, len(LANDUSE_CODES)))

    if cb.soil_raster:
        path = cfg.resolve(cb.soil_raster)
        logger.info("土壌ラスタの重なり面積を計算中: %s", path)
        soil_areas = raster_code_areas(
            polygons, crs, path, SOIL_CODES, fallback_epsg=cb.soil_epsg,
        )
    else:
        logger.warning("output.cell_bin.soil_raster が未指定のため土壌面積を 0 で埋めます")
        soil_areas = np.zeros((n, len(SOIL_CODES)))

    if cb.buildings:
        path = cfg.resolve(cb.buildings)
        mins = mesh.nodes.min(axis=0)
        maxs = mesh.nodes.max(axis=0)
        bounds = (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))
        logger.info("建物ポリゴンを読み込み中: %s", path)
        buildings = load_buildings_bbox(path, crs, bounds)
        bld_ratio, bld_peri = building_ratio_perimeter(
            polygons, quality.area, buildings, ratio_max=cb.building_ratio_max,
        )
    else:
        logger.warning("output.cell_bin.buildings が未指定のため bld_ratio/bld_peri を 0 で埋めます")
        bld_ratio = np.zeros(n)
        bld_peri = np.zeros(n)

    if cb.vegetation:
        path = cfg.resolve(cb.vegetation)
        mins = mesh.nodes.min(axis=0)
        maxs = mesh.nodes.max(axis=0)
        bounds = (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))
        logger.info("植生ポリゴンを読み込み中: %s", path)
        vegetation = load_buildings_bbox(path, crs, bounds)
        chi_veg, a_veg, H_veg, Cd_veg = vegetation_chi_and_params(
            polygons, quality.area, vegetation,
            ratio_max=cb.vegetation_ratio_max,
            field_cd=cb.veg_field_cd,
            field_a=cb.veg_field_a,
            field_hv=cb.veg_field_hv,
        )
    else:
        logger.warning("output.cell_bin.vegetation が未指定のため植生抵抗を 0 で埋めます")
        chi_veg = np.zeros(n)
        a_veg = np.zeros(n)
        H_veg = np.zeros(n)
        Cd_veg = np.zeros(n)

    return CellAttributes(
        landuse_areas, soil_areas, bld_ratio, bld_peri,
        chi_veg, a_veg, H_veg, Cd_veg,
    )


def _pad_path(path_str: str, length: int = _PATH_FIELD_LEN) -> bytes:
    b = path_str.encode("utf-8", errors="replace")
    return b[:length].ljust(length, b" ")


def _element_node_lists(mesh: Mesh) -> list[np.ndarray]:
    out = [row for row in mesh.triangles]
    out.extend(row for row in mesh.quads)
    return out


def _local_edge_arrays(
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(owner_element, node_a, node_b) を面 -> 局所辺の順に並べて返す。"""
    owners: list[np.ndarray] = []
    node_a: list[np.ndarray] = []
    node_b: list[np.ndarray] = []
    offset = 0
    for conn in (mesh.triangles, mesh.quads):
        if len(conn):
            m = conn.shape[1]
            owners.append(np.repeat(np.arange(len(conn)) + offset, m))
            node_a.append(conn.ravel())
            node_b.append(np.roll(conn, -1, axis=1).ravel())
        offset += len(conn)
    if not owners:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(owners), np.concatenate(node_a), np.concatenate(node_b)


def _physical_line_ids(node_a: np.ndarray, node_b: np.ndarray) -> np.ndarray:
    """共有辺に同じ LN（1-origin）を振る。edge.gpkg の LineID と同じ規則。"""
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


def _find_neighbors(owner: np.ndarray, node_a: np.ndarray, node_b: np.ndarray) -> np.ndarray:
    """各局所辺について、辺を共有する隣接要素番号を返す（境界は -1）。"""
    logger = get_logger()
    n = len(owner)
    neighbor = np.full(n, _BOUNDARY_EDGE_SENTINEL, dtype=np.int64)
    if n == 0:
        return neighbor

    lo = np.minimum(node_a, node_b)
    hi = np.maximum(node_a, node_b)
    n_nodes = int(max(node_a.max(), node_b.max())) + 1 if n else 0
    keys = lo.astype(np.int64) * (n_nodes + 1) + hi.astype(np.int64)

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    n_nonmanifold = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_keys[j] == sorted_keys[i]:
            j += 1
        group = order[i:j]
        if j - i == 2 and owner[group[0]] != owner[group[1]]:
            neighbor[group[0]] = owner[group[1]]
            neighbor[group[1]] = owner[group[0]]
        elif j - i != 1:
            n_nonmanifold += 1
        i = j

    if n_nonmanifold:
        logger.warning(
            "cell.bin: 非多様体な辺が %d 本あります（境界扱いにします）", n_nonmanifold,
        )
    return neighbor


def write_cell_bin(
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    dem: DemGrid,
    landuse_areas: np.ndarray,
    soil_areas: np.ndarray,
    bld_ratio: np.ndarray,
    bld_peri: np.ndarray,
    face_gpkg_path: str,
    edge_gpkg_path: str,
    output_path: Path,
) -> Path:
    """cell.bin を書き出す。"""
    logger = get_logger()
    n = mesh.n_elements
    elem_nodes = _element_node_lists(mesh)
    centroids = mesh.element_centroids()
    areas = quality.area
    elevation = np.where(np.isfinite(terrain.elevation), terrain.elevation, 0.0)
    if not np.isfinite(terrain.elevation).all():
        logger.warning(
            "cell.bin: 代表標高が欠損している要素が %d 個あります（0 で埋めます）",
            int((~np.isfinite(terrain.elevation)).sum()),
        )

    owner, node_a, node_b = _local_edge_arrays(mesh)
    neighbor = _find_neighbors(owner, node_a, node_b)
    line_ids = _physical_line_ids(node_a, node_b)
    total_edge = len(owner)
    total_face = n
    total_face_index = n  # ダミーメッシュ未対応のため、全セルを計算対象として扱う

    x_a, y_a = mesh.nodes[node_a, 0], mesh.nodes[node_a, 1]
    x_b, y_b = mesh.nodes[node_b, 0], mesh.nodes[node_b, 1]
    z_a = mesh.node_z[node_a] if mesh.node_z is not None else np.zeros(len(node_a))
    z_b = mesh.node_z[node_b] if mesh.node_z is not None else np.zeros(len(node_b))
    x_cnt = 0.5 * (x_a + x_b)
    y_cnt = 0.5 * (y_a + y_b)
    edge_length = np.hypot(x_b - x_a, y_b - y_a)
    z_cnt_dem = dem.sample(x_cnt, y_cnt)
    z_cnt = np.where(np.isfinite(z_cnt_dem), z_cnt_dem, 0.5 * (z_a + z_b))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(_pad_path(face_gpkg_path))
        f.write(_pad_path(edge_gpkg_path))
        f.write(struct.pack("<iii", total_face, total_edge, total_face_index))

        edge_cursor = 0
        for ei in range(n):
            m = len(elem_nodes[ei])
            cn = ei + 1
            record = bytearray(_FACE_STRUCT.pack(
                cn, m,
                float(centroids[ei, 0]), float(centroids[ei, 1]), float(areas[ei]),
                float(elevation[ei]), float(bld_ratio[ei]), float(bld_peri[ei]),
                *[float(v) for v in landuse_areas[ei]],
                *[float(v) for v in soil_areas[ei]],
                0,  # dummy_face（CalMesh）
            ))

            for k in range(m):
                idx = edge_cursor + k
                en = int(line_ids[idx])
                nb = int(neighbor[idx])
                has_nb = nb != _BOUNDARY_EDGE_SENTINEL
                cn_end = nb + 1 if has_nb else 0
                bc_flag = 0 if has_nb else 1

                area_own = float(areas[ei])
                dist_own_to_cnt = float(np.hypot(
                    x_cnt[idx] - centroids[ei, 0], y_cnt[idx] - centroids[ei, 1],
                ))

                if has_nb:
                    x_nb, y_nb = float(centroids[nb, 0]), float(centroids[nb, 1])
                    total_area = area_own + float(areas[nb])
                    dx_cell = x_nb - centroids[ei, 0]
                    dy_cell = y_nb - centroids[ei, 1]
                    dl_val = float(np.hypot(dx_cell, dy_cell))
                    cos_x, cos_y = (dx_cell / dl_val, dy_cell / dl_val) if dl_val > 1e-10 else (0.0, 0.0)

                    dist_nb_to_cnt = float(np.hypot(x_cnt[idx] - x_nb, y_cnt[idx] - y_nb))
                    if dist_own_to_cnt > 1e-10 and dist_nb_to_cnt > 1e-10:
                        inv_o = 1.0 / dist_own_to_cnt
                        inv_n = 1.0 / dist_nb_to_cnt
                        s = inv_o + inv_n
                        weight_self, weight_adj = inv_o / s, inv_n / s
                    elif dist_own_to_cnt <= 1e-10:
                        weight_self, weight_adj = 1.0, 0.0
                    else:
                        weight_self, weight_adj = 0.0, 1.0
                else:
                    total_area = area_own
                    dl_val = dist_own_to_cnt
                    dx_cell = x_cnt[idx] - centroids[ei, 0]
                    dy_cell = y_cnt[idx] - centroids[ei, 1]
                    cos_x, cos_y = (dx_cell / dl_val, dy_cell / dl_val) if dl_val > 1e-10 else (0.0, 0.0)
                    weight_self, weight_adj = 1.0, 0.0

                weight_face = (
                    (1.0 / dist_own_to_cnt) / (1.0 + 1.0 / dist_own_to_cnt)
                    if dist_own_to_cnt > 1e-10 else 1.0
                )

                record.extend(_EDGE_STRUCT.pack(
                    en, bc_flag, cn, cn_end,
                    total_area, dl_val, cos_x, cos_y,
                    weight_self, weight_adj, weight_face,
                    int(node_a[idx]) + 1, int(node_b[idx]) + 1,
                    float(x_a[idx]), float(y_a[idx]), float(z_a[idx]),
                    float(x_b[idx]), float(y_b[idx]), float(z_b[idx]),
                    float(x_cnt[idx]), float(y_cnt[idx]), float(z_cnt[idx]),
                    float(edge_length[idx]),
                ))
            f.write(record)
            edge_cursor += m

    logger.info(
        "cell.bin を出力: %s（セル数 %d, 辺レコード数 %d）",
        output_path.name, total_face, total_edge,
    )
    return output_path


def build_and_write_cell_bin(
    cfg: Config,
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    crs: CRS,
    dem: DemGrid,
    out_dir: Path,
    face_gpkg_path: Path,
    edge_gpkg_path: Path,
    attrs: CellAttributes | None = None,
) -> Path | None:
    """設定に従って土地利用・土壌・建物の統計を求め、cell.bin を出力する。"""
    _ = out_dir  # 互換のため残す。出力先は cfg.cell_bin_dir。
    cb: CellBinConfig = cfg.output.cell_bin
    if not cb.enabled:
        return None

    if attrs is None:
        attrs = compute_cell_attributes(cfg, mesh, quality, crs)

    filename = cb.filename or f"{cfg.output.basename}.bin"
    output_path = cfg.cell_bin_dir / filename
    return write_cell_bin(
        mesh, quality, terrain, dem,
        attrs.landuse_areas, attrs.soil_areas, attrs.bld_ratio, attrs.bld_peri,
        face_gpkg_path=str(face_gpkg_path.resolve()),
        edge_gpkg_path=str(edge_gpkg_path.resolve()),
        output_path=output_path,
    )
