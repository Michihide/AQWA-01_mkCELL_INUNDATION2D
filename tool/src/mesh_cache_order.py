"""面・辺・頂点番号をキャッシュ向けに詰め直す。

面は連結成分ごとの逆 Cuthill-McKee。辺は隣接面番号の小さい順。
頂点は、その頂点を最初に使う面の番号順。幾何と属性の対応は保つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh_bin_io import (
    BuildingTable,
    CoupleTable,
    MeshAttributes,
    SparseTriple,
    SpecialEdgeTable,
    VegTable,
    _section_payload_end,
    decode_block_trailer,
    decode_lts_trailer,
    pack_mesh,
    unpack_mesh,
)
from .mesh_tables import EdgeTable, FaceTable, NodeTable, NormalizedMesh


@dataclass
class CachePermutation:
    """添字は新しい番号、値は古い番号（どちらも 0 始まり）。"""

    face_new_to_old: np.ndarray
    edge_new_to_old: np.ndarray
    vertex_new_to_old: np.ndarray
    span_before: float
    span_after: float
    step_before: float
    step_after: float


def internal_edge_locality(face_left: np.ndarray, face_right: np.ndarray) -> tuple[float, float]:
    """内部辺の |左面-右面| 平均と、内部辺を並んだ順に見た |Δ左面| 平均。"""
    left = np.asarray(face_left)
    right = np.asarray(face_right)
    both = (left > 0) & (right > 0)
    if not np.any(both):
        return 0.0, 0.0
    diff = np.abs(left[both].astype(np.int64) - right[both].astype(np.int64))
    seq = left[both].astype(np.int64)
    step = np.abs(np.diff(seq)) if seq.size > 1 else np.zeros(1, dtype=np.int64)
    return float(diff.mean()), float(step.mean())


def apply_cache_order(
    mesh: NormalizedMesh,
    attrs: MeshAttributes | None = None,
) -> tuple[NormalizedMesh, MeshAttributes, CachePermutation]:
    """面・辺・頂点を詰め直したコピーを返す。入力は変えない。"""
    attrs = attrs or MeshAttributes()
    nface = len(mesh.faces)
    nedge = len(mesh.edges)
    nnode = len(mesh.nodes)
    span_before, step_before = internal_edge_locality(
        mesh.edges.face_left, mesh.edges.face_right,
    )
    if nface == 0:
        perm = CachePermutation(
            face_new_to_old=np.empty(0, dtype=np.int32),
            edge_new_to_old=np.empty(0, dtype=np.int32),
            vertex_new_to_old=np.empty(0, dtype=np.int32),
            span_before=span_before,
            span_after=span_before,
            step_before=step_before,
            step_after=step_before,
        )
        return mesh, attrs, perm

    face_perm = _rcm_face_perm(nface, mesh.edges.face_left, mesh.edges.face_right)
    inv_face = _inverse(face_perm)

    edge_ids = [_as_i32(mesh.faces.edge_ids[int(old)]) for old in face_perm]
    left = _remap_positive(mesh.edges.face_left, inv_face, nface)
    right = _remap_positive(mesh.edges.face_right, inv_face, nface)

    vperm = _vertex_perm(nnode, nface, mesh.edges.v1, mesh.edges.v2, left, right)
    inv_vertex = _inverse(vperm)
    v1 = _remap_positive(mesh.edges.v1, inv_vertex, nnode)
    v2 = _remap_positive(mesh.edges.v2, inv_vertex, nnode)

    edge_perm = _edge_perm(left, right)
    inv_edge = _inverse(edge_perm)
    edge_ids = [_remap_positive(ids, inv_edge, nedge) for ids in edge_ids]

    faces = FaceTable(
        id=np.arange(1, nface + 1, dtype=np.int32),
        dummy=_take(mesh.faces.dummy, face_perm),
        x=_take(mesh.faces.x, face_perm),
        y=_take(mesh.faces.y, face_perm),
        A=_take(mesh.faces.A, face_perm),
        z_bed=_take(mesh.faces.z_bed, face_perm),
        n_sides=_take(mesh.faces.n_sides, face_perm),
        edge_ids=edge_ids,
        block=_take(mesh.faces.block_ids(), face_perm),
    )
    edges = EdgeTable(
        id=np.arange(1, nedge + 1, dtype=np.int32),
        v1=v1[edge_perm].copy(),
        v2=v2[edge_perm].copy(),
        face_left=left[edge_perm].copy(),
        face_right=right[edge_perm].copy(),
        z_crest=_take(mesh.edges.z_crest, edge_perm),
        fr=_take(mesh.edges.froude(), edge_perm),
    )
    nodes = NodeTable(
        id=np.arange(1, nnode + 1, dtype=np.int32),
        x=_take(mesh.nodes.x, vperm),
        y=_take(mesh.nodes.y, vperm),
        z=_take(mesh.nodes.z, vperm),
    )
    new_attrs = MeshAttributes(
        landuse=_remap_sparse(attrs.landuse, inv_face),
        soil=_remap_sparse(attrs.soil, inv_face),
        building=_remap_building(attrs.building, inv_face),
        veg=_remap_veg(attrs.veg, inv_face),
        special_edges=_remap_special(attrs.special_edges, inv_edge, nedge),
        couple=_remap_couple(attrs.couple, inv_edge, nedge),
    )
    span_after, step_after = internal_edge_locality(edges.face_left, edges.face_right)
    perm = CachePermutation(
        face_new_to_old=face_perm,
        edge_new_to_old=edge_perm,
        vertex_new_to_old=vperm,
        span_before=span_before,
        span_after=span_after,
        step_before=step_before,
        step_after=step_after,
    )
    return NormalizedMesh(nodes, edges, faces), new_attrs, perm


def read_lts_trailer(data: bytes, nface: int) -> tuple[np.ndarray, int, int] | None:
    """mesh.bin 全体から LTS トレーラーを読む。無ければ None。"""
    tail = data[_section_payload_end(data):]
    block = decode_block_trailer(tail, nface)
    offset = 12 + 4 * nface if block is not None else 0
    return decode_lts_trailer(tail[offset:], nface)


def reorder_mesh_bin(src: Path, dst: Path) -> CachePermutation:
    """既存 mesh.bin を詰め直して dst に書く。src は変更しない。

    面の対応は dst と同じディレクトリの `<dst名>.face_perm.npy`
    （新しい面番号 → 古い面番号、0 始まり）に残す。
    """
    src = Path(src)
    dst = Path(dst)
    data = src.read_bytes()
    packed = unpack_mesh(data)
    lts = read_lts_trailer(data, len(packed.mesh.faces))
    mesh, attrs, perm = apply_cache_order(packed.mesh, packed.attrs)
    levels = None
    lmax = 0
    code = 2
    if lts is not None:
        levels = np.asarray(lts[0], dtype=np.int32)[perm.face_new_to_old]
        lmax = int(lts[1])
        code = int(lts[2])
    blob = pack_mesh(
        mesh, attrs, epsg=packed.epsg,
        lts_levels=levels, lts_lmax=lmax, lts_binning_code=code,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(dst)
    np.save(dst.with_name(dst.name + ".face_perm.npy"), perm.face_new_to_old)
    return perm


def permute_rain_csv(src: Path, dst: Path, face_new_to_old: np.ndarray) -> None:
    """降雨 CSV の面列を face_new_to_old（新しい列 → 古い列）で並べ替える。

    先頭 3 行（開始日時、ステップ数、コメント）はそのまま。数値の文字列表記も保つ。
    """
    src = Path(src)
    dst = Path(dst)
    perm = np.asarray(face_new_to_old, dtype=np.int64)
    nface = int(perm.size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with src.open(encoding="utf-8", newline="") as fin, tmp.open("w", encoding="utf-8", newline="") as fout:
        for _ in range(3):
            line = fin.readline()
            if line == "":
                raise ValueError(f"降雨 CSV のヘッダが 3 行ありません: {src}")
            fout.write(line)
        for lineno, line in enumerate(fin, start=4):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) != nface + 1:
                raise ValueError(
                    f"{src}:{lineno}: 列数が {len(parts) - 1} です（面 {nface} を期待）"
                )
            fout.write(parts[0])
            for old in perm:
                fout.write(",")
                fout.write(parts[int(old) + 1])
            fout.write("\n")
    tmp.replace(dst)


def align_solver_tables(
    face_gpkg: Path,
    edge_gpkg: Path,
    face_x: np.ndarray,
    face_y: np.ndarray,
    face_new_to_old: np.ndarray,
    *,
    face_csv: Path | None = None,
    edge_csv: Path | None = None,
    tol_m: float = 1.0e-3,
) -> bool:
    """並べ替え前の面重心が一致するときだけ、gpkg / csv の CN を新しい面番号に合わせる。

    一致しなければファイルは変えない。
    """
    face_gpkg = Path(face_gpkg)
    if not face_gpkg.is_file():
        return False
    import geopandas as gpd

    perm = np.asarray(face_new_to_old, dtype=np.int64)
    face = gpd.read_file(face_gpkg, layer="poly")
    if "CN" not in face.columns or len(face) != perm.size:
        return False
    face = face.sort_values("CN").reset_index(drop=True)
    cn = face["CN"].to_numpy()
    if not np.array_equal(cn, np.arange(1, perm.size + 1)):
        return False
    cent = face.geometry.centroid
    # perm[new] = old。gpkg は古い CN 順なので、新しい面座標は cent[perm] と一致する。
    dx = float(np.max(np.abs(cent.x.to_numpy()[perm] - np.asarray(face_x))))
    dy = float(np.max(np.abs(cent.y.to_numpy()[perm] - np.asarray(face_y))))
    if dx > tol_m or dy > tol_m:
        raise ValueError(
            f"face.gpkg の重心が mesh.bin と合いません (dx={dx:.4g} m, dy={dy:.4g} m)。"
            "CN は変更していません"
        )

    inv = _inverse(perm.astype(np.int32))
    face = face.iloc[perm].reset_index(drop=True)
    face["CN"] = np.arange(1, len(face) + 1, dtype=np.int32)
    _replace_gpkg(face_gpkg, face, "poly")

    if edge_gpkg.is_file():
        edge = gpd.read_file(edge_gpkg, layer="line")
        if "CN" in edge.columns:
            old = edge["CN"].to_numpy(dtype=np.int64)
            if np.any((old < 1) | (old > perm.size)):
                raise ValueError("edge.gpkg の CN が面数の範囲外です")
            edge = edge.copy()
            edge["CN"] = inv[old - 1] + 1
            _replace_gpkg(edge_gpkg, edge, "line")

    _remap_csv(face_csv, perm, inv, reorder_rows=True)
    _remap_csv(edge_csv, perm, inv, reorder_rows=False)
    return True


def _replace_gpkg(path: Path, frame, layer: str) -> None:
    tmp = path.with_name(path.stem + ".reindex.gpkg")
    if tmp.exists():
        tmp.unlink()
    frame.to_file(tmp, driver="GPKG", layer=layer)
    tmp.replace(path)


def _remap_csv(
    path: Path | None,
    perm: np.ndarray,
    inv: np.ndarray,
    *,
    reorder_rows: bool,
) -> None:
    if path is None or not Path(path).is_file():
        return
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(path)
    if "CN" not in df.columns:
        return
    if reorder_rows:
        df = df.sort_values("CN").reset_index(drop=True)
        if len(df) != perm.size:
            return
        if not np.array_equal(df["CN"].to_numpy(), np.arange(1, perm.size + 1)):
            return
        df = df.iloc[perm].reset_index(drop=True)
        df["CN"] = np.arange(1, len(df) + 1, dtype=np.int32)
    else:
        old = df["CN"].to_numpy(dtype=np.int64)
        if np.any((old < 1) | (old > perm.size)):
            return
        df = df.copy()
        df["CN"] = inv[old - 1] + 1
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _rcm_face_perm(nface: int, face_left: np.ndarray, face_right: np.ndarray) -> np.ndarray:
    neighbors = _adjacency(nface, face_left, face_right)
    degree = np.fromiter((len(nbr) for nbr in neighbors), dtype=np.int32, count=nface)
    neighbors = _sort_by_degree(neighbors, degree)
    visited = np.zeros(nface, dtype=bool)
    order = np.empty(nface, dtype=np.int32)
    pos = 0
    for seed in range(nface):
        if visited[seed]:
            continue
        nodes = _reachable(seed, neighbors, visited)
        start0 = min(nodes, key=lambda i: (int(degree[i]), i))
        start = _pseudo_peripheral(start0, neighbors, degree, visited)
        queue = [start]
        visited[start] = True
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            order[pos] = u
            pos += 1
            for v in neighbors[u]:
                v = int(v)
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
    if pos != nface:
        raise RuntimeError("RCM が全ての面を訪れませんでした")
    return order[::-1].copy()


def _adjacency(nface: int, face_left: np.ndarray, face_right: np.ndarray) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(nface)]
    for left, right in zip(face_left.tolist(), face_right.tolist()):
        if left <= 0 or right <= 0:
            continue
        a = int(left) - 1
        b = int(right) - 1
        if a >= nface or b >= nface:
            raise ValueError(f"面 ID が範囲外です: {left}, {right}")
        buckets[a].append(b)
        buckets[b].append(a)
    out: list[np.ndarray] = []
    for xs in buckets:
        if not xs:
            out.append(np.empty(0, dtype=np.int32))
            continue
        arr = np.asarray(xs, dtype=np.int32)
        arr.sort()
        if arr.size > 1:
            arr = arr[np.concatenate(([True], arr[1:] != arr[:-1]))]
        out.append(arr)
    return out


def _sort_by_degree(neighbors: list[np.ndarray], degree: np.ndarray) -> list[np.ndarray]:
    ordered: list[np.ndarray] = []
    for nbr in neighbors:
        if nbr.size <= 1:
            ordered.append(nbr)
            continue
        ordered.append(nbr[np.lexsort((nbr, degree[nbr]))])
    return ordered


def _reachable(seed: int, neighbors: list[np.ndarray], visited: np.ndarray) -> list[int]:
    seen = np.zeros(len(neighbors), dtype=bool)
    seen[seed] = True
    queue = [seed]
    head = 0
    nodes: list[int] = []
    while head < len(queue):
        u = queue[head]
        head += 1
        if visited[u]:
            continue
        nodes.append(u)
        for v in neighbors[u]:
            v = int(v)
            if not seen[v] and not visited[v]:
                seen[v] = True
                queue.append(v)
    return nodes


def _farthest(
    start: int,
    neighbors: list[np.ndarray],
    degree: np.ndarray,
    blocked: np.ndarray,
) -> tuple[int, int]:
    depth = np.full(len(neighbors), -1, dtype=np.int32)
    depth[start] = 0
    queue = [start]
    head = 0
    max_depth = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        nxt = int(depth[u]) + 1
        for v in neighbors[u]:
            v = int(v)
            if blocked[v] or depth[v] >= 0:
                continue
            depth[v] = nxt
            if nxt > max_depth:
                max_depth = nxt
            queue.append(v)
    best = start
    best_degree = 10**9
    for node in queue:
        if int(depth[node]) != max_depth:
            continue
        deg = int(degree[node])
        if deg < best_degree or (deg == best_degree and node < best):
            best = node
            best_degree = deg
    return best, max_depth


def _pseudo_peripheral(
    seed: int,
    neighbors: list[np.ndarray],
    degree: np.ndarray,
    blocked: np.ndarray,
) -> int:
    root = seed
    nxt, ecc = _farthest(root, neighbors, degree, blocked)
    for _ in range(len(neighbors)):
        cand, ecc_nxt = _farthest(nxt, neighbors, degree, blocked)
        if ecc_nxt > ecc:
            root = nxt
            nxt = cand
            ecc = ecc_nxt
            continue
        return root
    return root


def _vertex_perm(
    nnode: int,
    nface: int,
    v1: np.ndarray,
    v2: np.ndarray,
    face_left: np.ndarray,
    face_right: np.ndarray,
) -> np.ndarray:
    first = np.full(nnode, nface + 1, dtype=np.int32)
    for left, right, a, b in zip(
        face_left.tolist(), face_right.tolist(), v1.tolist(), v2.tolist(),
    ):
        touch = nface + 1
        if left > 0:
            touch = int(left)
        if right > 0:
            touch = min(touch, int(right))
        for vertex in (int(a), int(b)):
            if vertex <= 0:
                continue
            if vertex > nnode:
                raise ValueError(f"頂点 ID が範囲外です: {vertex}")
            slot = vertex - 1
            if touch < int(first[slot]):
                first[slot] = touch
    return np.lexsort((np.arange(nnode, dtype=np.int32), first)).astype(np.int32)


def _edge_perm(face_left: np.ndarray, face_right: np.ndarray) -> np.ndarray:
    left = face_left.astype(np.int32, copy=False)
    right = face_right.astype(np.int32, copy=False)
    both = (left > 0) & (right > 0)
    lo = np.where(both, np.minimum(left, right), np.maximum(left, right))
    hi = np.where(both, np.maximum(left, right), np.int32(0))
    return np.lexsort((np.arange(left.size, dtype=np.int32), hi, lo)).astype(np.int32)


def _inverse(perm: np.ndarray) -> np.ndarray:
    inv = np.empty(perm.size, dtype=np.int32)
    inv[np.asarray(perm, dtype=np.int64)] = np.arange(perm.size, dtype=np.int32)
    return inv


def _take(values: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(values)[perm].copy()


def _as_i32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int32).copy()


def _remap_positive(ids: np.ndarray, inverse: np.ndarray, n_old: int) -> np.ndarray:
    src = np.asarray(ids, dtype=np.int32)
    out = np.zeros(src.shape, dtype=np.int32)
    pos = src > 0
    if np.any(src[pos] > n_old):
        raise ValueError("ID が範囲外です")
    out[pos] = inverse[src[pos] - 1] + 1
    return out


def _remap_id_column(face_id: np.ndarray, inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(face_id, dtype=np.int64)
    if src.size and (np.any(src < 1) or np.any(src > inverse.size)):
        raise ValueError("属性の面 ID が範囲外です")
    new_id = inverse[src - 1] + 1 if src.size else np.empty(0, dtype=np.int32)
    order = np.argsort(new_id, kind="mergesort")
    return np.asarray(new_id, dtype=np.int32)[order], order


def _remap_sparse(table: SparseTriple, inverse: np.ndarray) -> SparseTriple:
    if len(table) == 0:
        return SparseTriple()
    face_id, order = _remap_id_column(table.face_id, inverse)
    return SparseTriple(
        face_id=face_id,
        code=np.asarray(table.code)[order].astype(np.int32, copy=True),
        area=np.asarray(table.area)[order].astype(np.float64, copy=True),
    )


def _remap_building(table: BuildingTable, inverse: np.ndarray) -> BuildingTable:
    if len(table) == 0:
        return BuildingTable()
    face_id, order = _remap_id_column(table.face_id, inverse)
    return BuildingTable(
        face_id=face_id,
        chi_bld=np.asarray(table.chi_bld)[order].astype(np.float64, copy=True),
        bld_peri=np.asarray(table.bld_peri)[order].astype(np.float64, copy=True),
    )


def _remap_veg(table: VegTable, inverse: np.ndarray) -> VegTable:
    if len(table) == 0:
        return VegTable()
    face_id, order = _remap_id_column(table.face_id, inverse)
    return VegTable(
        face_id=face_id,
        chi_veg=np.asarray(table.chi_veg)[order].astype(np.float64, copy=True),
        a=np.asarray(table.a)[order].astype(np.float64, copy=True),
        H_v=np.asarray(table.H_v)[order].astype(np.float64, copy=True),
        C_D=np.asarray(table.C_D)[order].astype(np.float64, copy=True),
    )


def _remap_special(table: SpecialEdgeTable, inverse: np.ndarray, nedge: int) -> SpecialEdgeTable:
    if len(table) == 0:
        return SpecialEdgeTable()
    edge_id = _remap_positive(table.edge_id, inverse, nedge)
    order = np.argsort(edge_id, kind="mergesort")
    def col(values, dtype):
        return np.asarray(values)[order].astype(dtype, copy=True)
    return SpecialEdgeTable(
        edge_id=edge_id[order].astype(np.int32, copy=True),
        kind=col(table.kind, np.int32),
        zc=col(table.zc, np.float64),
        H=col(table.H, np.float64),
        z_road=col(table.z_road, np.float64),
        qgroup=col(table.qgroup, np.int32),
        B=col(table.B, np.float64) if len(table.B) == len(table) else np.empty(0, dtype=np.float64),
        fr=col(table.fr, np.float64) if len(table.fr) == len(table) else np.empty(0, dtype=np.float64),
    )


def _remap_couple(table: CoupleTable, inverse: np.ndarray, nedge: int) -> CoupleTable:
    if len(table) == 0:
        return CoupleTable()
    edge_id = _remap_positive(table.edge_id, inverse, nedge)
    order = np.argsort(edge_id, kind="mergesort")
    return CoupleTable(
        edge_id=edge_id[order].astype(np.int32, copy=True),
        river_link=np.asarray(table.river_link)[order].astype(np.int32, copy=True),
        kp=np.asarray(table.kp)[order].astype(np.float64, copy=True),
        bank=np.asarray(table.bank)[order].astype(np.int32, copy=True),
    )
