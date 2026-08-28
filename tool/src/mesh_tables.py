"""Mesh から正規化 node / edge / face 配列を作る。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cell_bin_export import (
    _BOUNDARY_EDGE_SENTINEL,
    _find_neighbors,
    _local_edge_arrays,
    _physical_line_ids,
)
from .io_raster import DemGrid
from .mesh_parser import Mesh
from .quality_metrics import QualityReport
from .terrain_metrics import TerrainReport
from .utils import get_logger

_EPS = 1e-10


@dataclass
class NodeTable:
    id: np.ndarray  # i32, 1-origin
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __len__(self) -> int:
        return int(len(self.id))


@dataclass
class EdgeTable:
    id: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    face_left: np.ndarray
    face_right: np.ndarray
    z_crest: np.ndarray

    def __len__(self) -> int:
        return int(len(self.id))


@dataclass
class FaceTable:
    id: np.ndarray
    dummy: np.ndarray
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray
    z_bed: np.ndarray
    n_sides: np.ndarray
    edge_ids: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return int(len(self.id))


@dataclass
class NormalizedMesh:
    nodes: NodeTable
    edges: EdgeTable
    faces: FaceTable


def build_normalized_tables(
    mesh: Mesh,
    quality: QualityReport,
    terrain: TerrainReport,
    dem: DemGrid | None = None,
    dummy: np.ndarray | None = None,
) -> NormalizedMesh:
    """共有辺を 1 本に畳み、点座標は node だけに置く。"""
    logger = get_logger()
    n_face = mesh.n_elements
    n_node = mesh.n_nodes
    centroids = mesh.element_centroids()
    areas = quality.area
    elevation = np.where(np.isfinite(terrain.elevation), terrain.elevation, 0.0)
    if not np.isfinite(terrain.elevation).all():
        logger.warning(
            "mesh.bin: 代表標高が欠損している要素が %d 個あります（0 で埋めます）",
            int((~np.isfinite(terrain.elevation)).sum()),
        )

    node_z = mesh.node_z if mesh.node_z is not None else np.zeros(n_node)
    nodes = NodeTable(
        id=np.arange(1, n_node + 1, dtype=np.int32),
        x=np.asarray(mesh.nodes[:, 0], dtype=np.float64),
        y=np.asarray(mesh.nodes[:, 1], dtype=np.float64),
        z=np.asarray(node_z, dtype=np.float64),
    )

    owner, node_a, node_b = _local_edge_arrays(mesh)
    neighbor = _find_neighbors(owner, node_a, node_b)
    line_ids = _physical_line_ids(node_a, node_b)
    n_local = len(owner)
    n_edge = int(line_ids.max()) if n_local else 0

    v1 = np.zeros(n_edge, dtype=np.int32)
    v2 = np.zeros(n_edge, dtype=np.int32)
    face_left = np.zeros(n_edge, dtype=np.int32)
    face_right = np.zeros(n_edge, dtype=np.int32)
    z_crest = np.zeros(n_edge, dtype=np.float64)
    seen = np.zeros(n_edge, dtype=bool)

    x_a, y_a = mesh.nodes[node_a, 0], mesh.nodes[node_a, 1]
    x_b, y_b = mesh.nodes[node_b, 0], mesh.nodes[node_b, 1]
    z_a = node_z[node_a]
    z_b = node_z[node_b]
    x_cnt = 0.5 * (x_a + x_b)
    y_cnt = 0.5 * (y_a + y_b)
    if dem is not None:
        z_cnt_dem = dem.sample(x_cnt, y_cnt)
        z_cnt = np.where(np.isfinite(z_cnt_dem), z_cnt_dem, 0.5 * (z_a + z_b))
    else:
        z_cnt = 0.5 * (z_a + z_b)

    face_edge_ids: list[list[int]] = [[] for _ in range(n_face)]
    for i in range(n_local):
        eid = int(line_ids[i])
        fi = int(owner[i])
        face_edge_ids[fi].append(eid)
        if not seen[eid - 1]:
            seen[eid - 1] = True
            v1[eid - 1] = int(node_a[i]) + 1
            v2[eid - 1] = int(node_b[i]) + 1
            face_left[eid - 1] = fi + 1
            nb = int(neighbor[i])
            face_right[eid - 1] = nb + 1 if nb != _BOUNDARY_EDGE_SENTINEL else 0
            z_crest[eid - 1] = float(z_cnt[i])
        else:
            nb = int(neighbor[i])
            other = nb + 1 if nb != _BOUNDARY_EDGE_SENTINEL else 0
            if face_right[eid - 1] == 0 and other != face_left[eid - 1]:
                face_right[eid - 1] = other

    if n_edge and not seen.all():
        missing = int((~seen).sum())
        raise RuntimeError(f"辺 id に欠番があります（{missing} 本）")

    if dummy is None:
        dummy_arr = np.zeros(n_face, dtype=np.int32)
    else:
        dummy_arr = np.asarray(dummy, dtype=np.int32)
        if dummy_arr.shape != (n_face,):
            raise ValueError("dummy の長さが面数と一致しません")

    faces = FaceTable(
        id=np.arange(1, n_face + 1, dtype=np.int32),
        dummy=dummy_arr,
        x=np.asarray(centroids[:, 0], dtype=np.float64),
        y=np.asarray(centroids[:, 1], dtype=np.float64),
        A=np.asarray(areas, dtype=np.float64),
        z_bed=np.asarray(elevation, dtype=np.float64),
        n_sides=np.array([len(e) for e in face_edge_ids], dtype=np.int32),
        edge_ids=[np.asarray(e, dtype=np.int32) for e in face_edge_ids],
    )
    edges = EdgeTable(
        id=np.arange(1, n_edge + 1, dtype=np.int32),
        v1=v1,
        v2=v2,
        face_left=face_left,
        face_right=face_right,
        z_crest=z_crest,
    )
    return NormalizedMesh(nodes=nodes, edges=edges, faces=faces)


def inverse_distance_weights(
    dist_own: float,
    dist_nb: float | None,
) -> tuple[float, float, float]:
    """旧 cell.bin と同じ逆距離重み (self, adj, face)。"""
    if dist_nb is None:
        w_self, w_adj = 1.0, 0.0
    elif dist_own > _EPS and dist_nb > _EPS:
        inv_o = 1.0 / dist_own
        inv_n = 1.0 / dist_nb
        s = inv_o + inv_n
        w_self, w_adj = inv_o / s, inv_n / s
    elif dist_own <= _EPS:
        w_self, w_adj = 1.0, 0.0
    else:
        w_self, w_adj = 0.0, 1.0
    w_face = (1.0 / dist_own) / (1.0 + 1.0 / dist_own) if dist_own > _EPS else 1.0
    return w_self, w_adj, w_face
