#!/usr/bin/env python3
"""旧 cell.bin を作業ファイルと mesh.bin に変換する。"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cell_bin_export import LANDUSE_CODES, SOIL_CODES, _EDGE_STRUCT, _FACE_STRUCT, _PATH_FIELD_LEN  # noqa: E402
from src.mesh_bin_io import (  # noqa: E402
    MeshAttributes,
    building_from_dense,
    dense_areas_to_sparse,
    pack_work_directory,
    write_work_files,
)
from src.mesh_tables import EdgeTable, FaceTable, NodeTable, NormalizedMesh  # noqa: E402


def read_cell_bin(path: Path) -> tuple[NormalizedMesh, MeshAttributes]:
    """重複辺を en で畳み、点座標は node だけに置く。"""
    with path.open("rb") as f:
        f.read(_PATH_FIELD_LEN)
        f.read(_PATH_FIELD_LEN)
        total_face, _total_edge, _total_face_index = struct.unpack("<iii", f.read(12))

        face_ids = np.empty(total_face, dtype=np.int32)
        dummy = np.empty(total_face, dtype=np.int32)
        fx = np.empty(total_face)
        fy = np.empty(total_face)
        area = np.empty(total_face)
        z_bed = np.empty(total_face)
        n_sides = np.empty(total_face, dtype=np.int32)
        bld_ratio = np.empty(total_face)
        bld_peri = np.empty(total_face)
        landuse = np.zeros((total_face, len(LANDUSE_CODES)))
        soil = np.zeros((total_face, len(SOIL_CODES)))
        edge_ids: list[list[int]] = [[] for _ in range(total_face)]

        edges: dict[int, dict] = {}
        nodes: dict[int, tuple[float, float, float]] = {}

        for i in range(total_face):
            raw = f.read(_FACE_STRUCT.size)
            fields = _FACE_STRUCT.unpack(raw)
            cn, n_edge = int(fields[0]), int(fields[1])
            face_ids[i] = cn
            fx[i], fy[i], area[i], z_bed[i] = fields[2:6]
            bld_ratio[i], bld_peri[i] = fields[6:8]
            landuse[i] = fields[8:21]
            soil[i] = fields[21:38]
            dummy[i] = int(fields[38])
            n_sides[i] = n_edge
            for _k in range(n_edge):
                ef = _EDGE_STRUCT.unpack(f.read(_EDGE_STRUCT.size))
                en = int(ef[0])
                cn_self = int(ef[2])
                cn_adj = int(ef[3])
                v_st, v_en = int(ef[11]), int(ef[12])
                x_st, y_st, z_st = float(ef[13]), float(ef[14]), float(ef[15])
                x_en, y_en, z_en = float(ef[16]), float(ef[17]), float(ef[18])
                z_cnt = float(ef[21])
                edge_ids[i].append(en)
                nodes.setdefault(v_st, (x_st, y_st, z_st))
                nodes.setdefault(v_en, (x_en, y_en, z_en))
                if en not in edges:
                    edges[en] = {
                        "v1": v_st, "v2": v_en,
                        "left": cn_self, "right": cn_adj,
                        "z_crest": z_cnt,
                    }
                else:
                    rec = edges[en]
                    if rec["right"] == 0 and cn_self != rec["left"]:
                        rec["right"] = cn_self

        leftover = f.read()
        if leftover:
            raise ValueError(f"cell.bin の末尾に {len(leftover)} B 残っています")

    if not nodes:
        raise ValueError("cell.bin に頂点がありません")
    max_nid = max(nodes)
    if sorted(nodes) != list(range(1, max_nid + 1)):
        # 欠番は座標 0 で埋める（孤立点は旧形式に出ない）
        for nid in range(1, max_nid + 1):
            nodes.setdefault(nid, (0.0, 0.0, 0.0))

    node_ids = np.arange(1, max_nid + 1, dtype=np.int32)
    node_xyz = np.array([nodes[i] for i in node_ids], dtype=np.float64)
    node_table = NodeTable(node_ids, node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2])

    max_eid = max(edges) if edges else 0
    if edges and sorted(edges) != list(range(1, max_eid + 1)):
        raise ValueError("辺 id (en) に欠番があります")
    eids = np.arange(1, max_eid + 1, dtype=np.int32)
    edge_table = EdgeTable(
        id=eids,
        v1=np.array([edges[i]["v1"] for i in eids], dtype=np.int32),
        v2=np.array([edges[i]["v2"] for i in eids], dtype=np.int32),
        face_left=np.array([edges[i]["left"] for i in eids], dtype=np.int32),
        face_right=np.array([edges[i]["right"] for i in eids], dtype=np.int32),
        z_crest=np.array([edges[i]["z_crest"] for i in eids], dtype=np.float64),
    )
    # 面 id が 1..n でない場合は並びに合わせる
    order = np.argsort(face_ids)
    face_table = FaceTable(
        id=face_ids[order].astype(np.int32),
        dummy=dummy[order].astype(np.int32),
        x=fx[order],
        y=fy[order],
        A=area[order],
        z_bed=z_bed[order],
        n_sides=n_sides[order].astype(np.int32),
        edge_ids=[np.asarray(edge_ids[i], dtype=np.int32) for i in order],
    )
    mesh_attrs = MeshAttributes(
        landuse=dense_areas_to_sparse(landuse[order], [int(c) for c in LANDUSE_CODES]),
        soil=dense_areas_to_sparse(soil[order], [int(c) for c in SOIL_CODES]),
        building=building_from_dense(bld_ratio[order], bld_peri[order]),
    )
    return NormalizedMesh(node_table, edge_table, face_table), mesh_attrs


def convert(cell_bin: Path, out_dir: Path, *, epsg: int = 0) -> Path:
    mesh, attrs = read_cell_bin(cell_bin)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_work_files(out_dir, mesh, attrs)
    mesh_path = out_dir / "mesh.bin"
    pack_work_directory(out_dir, mesh_path, epsg=epsg)
    return mesh_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_bin", type=Path, help="旧 cell.bin")
    parser.add_argument("-o", "--out-dir", type=Path, required=True, help="作業ファイルと mesh.bin の出力先")
    parser.add_argument("--epsg", type=int, default=0)
    args = parser.parse_args(argv)
    path = convert(args.cell_bin, args.out_dir, epsg=args.epsg)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
