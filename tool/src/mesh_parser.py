"""gmsh の生成結果を numpy 配列へ取り出す。"""

from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np

GMSH_LINE = 1
GMSH_TRIANGLE = 2
GMSH_QUADRANGLE = 3


@dataclass
class Mesh:
    """2 次元ハイブリッドメッシュ。

    nodes は (N, 2)。triangles は (T, 3)、quads は (Q, 4) で、いずれも nodes への
    0 始まりインデックス。*_surface は各要素が属する gmsh 面のタグ。
    """

    nodes: np.ndarray
    triangles: np.ndarray
    quads: np.ndarray
    tri_surface: np.ndarray
    quad_surface: np.ndarray
    node_z: np.ndarray | None = None
    surface_roles: dict[int, str] = field(default_factory=dict)
    # 拘束ブレークライン上のメッシュ辺 (M, 2)。修復処理はこれを壊してはならない。
    constrained_edges: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    # 面ごとの氾濫ブロック id（1 始まり）。未設定なら pack 時に 1。
    block_id: np.ndarray | None = None

    @property
    def n_elements(self) -> int:
        return len(self.triangles) + len(self.quads)

    def constrained_node_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_nodes, dtype=bool)
        if len(self.constrained_edges):
            mask[self.constrained_edges.ravel()] = True
        return mask

    def constrained_edge_set(self) -> set[tuple[int, int]]:
        return {
            (int(a), int(b)) if a < b else (int(b), int(a))
            for a, b in self.constrained_edges
        }

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def element_centroids(self) -> np.ndarray:
        """全要素の重心。三角形 -> 四角形の順に並ぶ。"""
        parts = []
        if len(self.triangles):
            parts.append(self.nodes[self.triangles].mean(axis=1))
        if len(self.quads):
            parts.append(self.nodes[self.quads].mean(axis=1))
        return np.vstack(parts) if parts else np.empty((0, 2))

    def element_polygons(self) -> list[np.ndarray]:
        out = [self.nodes[t] for t in self.triangles]
        out.extend(self.nodes[q] for q in self.quads)
        return out

    def element_surfaces(self) -> np.ndarray:
        return np.concatenate([self.tri_surface, self.quad_surface])

    def element_kinds(self) -> np.ndarray:
        """要素種別。3 = 三角形、4 = 四角形。"""
        return np.concatenate([
            np.full(len(self.triangles), 3, dtype=np.int8),
            np.full(len(self.quads), 4, dtype=np.int8),
        ])

    def band_element_mask(self) -> np.ndarray:
        """外周四角形帯に属する要素。

        帯は Transfinite で分割数が決まっているため、サイズ場を下げても細かく
        ならない。細分化で解消できない違反を切り分けるのに使う。
        """
        band = [s for s, role in self.surface_roles.items() if role == "quad_boundary"]
        if not band:
            return np.zeros(self.n_elements, dtype=bool)
        return np.isin(self.element_surfaces(), band)

    def boundary_node_mask(self) -> np.ndarray:
        """解析領域の外形線（Γ0）上にある節点。

        外形線に接する辺は要素 1 個だけに属するため、その頂点として特定
        できる。ここを動かすと解析領域の形が変わるため、修復処理はこれらの
        節点を固定する。
        """
        parts = []
        for elems in (self.triangles, self.quads):
            if len(elems):
                m = elems.shape[1]
                parts.append(
                    np.stack([elems, np.roll(elems, -1, axis=1)], axis=2).reshape(-1, 2)
                )
        if not parts:
            return np.zeros(self.n_nodes, dtype=bool)

        edges = np.sort(np.vstack(parts), axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)

        on_boundary = np.zeros(self.n_nodes, dtype=bool)
        on_boundary[uniq[counts == 1].ravel()] = True
        return on_boundary

    def boundary_touching_element_mask(self) -> np.ndarray:
        """外周に接する要素。面積下限のチェック・修復の対象外として扱う。

        外周四角形帯の要素に加え、区間スキップにより Γ1 が Γ0 まで降りた
        内側の三角形（外形線上に辺を持つ三角形を含む）も対象にする。
        境界（堤防など）の形状再現性を、面積下限より優先するための除外。
        """
        mask = self.band_element_mask().copy()
        bnode = self.boundary_node_mask()
        if not bnode.any():
            return mask

        offset = 0
        for elems in (self.triangles, self.quads):
            n = len(elems)
            if n:
                m = elems.shape[1]
                on_edge = np.zeros(n, dtype=bool)
                for k in range(m):
                    a = elems[:, k]
                    b = elems[:, (k + 1) % m]
                    on_edge |= bnode[a] & bnode[b]
                mask[offset:offset + n] |= on_edge
            offset += n
        return mask


def extract_mesh(
    surface_roles: dict[int, str] | None = None,
    breakline_curves: list[int] | None = None,
) -> Mesh:
    """現在の gmsh モデルから 2 次元メッシュを取り出す。"""
    gmsh.model.mesh.renumberNodes()
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    node_tags = np.asarray(node_tags, dtype=np.int64)

    # renumberNodes 後もタグは 1 始まりだが順序は保証されないため写像を作る
    order = np.argsort(node_tags)
    nodes_xy = coords[order, :2]
    node_index = np.zeros(node_tags.max() + 1, dtype=np.int64)
    node_index[node_tags[order]] = np.arange(len(node_tags))

    tris: list[np.ndarray] = []
    quads: list[np.ndarray] = []
    tri_surf: list[np.ndarray] = []
    quad_surf: list[np.ndarray] = []

    for _, surf_tag in gmsh.model.getEntities(2):
        etypes, _, enodes = gmsh.model.mesh.getElements(2, surf_tag)
        for etype, conn in zip(etypes, enodes):
            conn = node_index[np.asarray(conn, dtype=np.int64)]
            if etype == GMSH_TRIANGLE:
                conn = conn.reshape(-1, 3)
                tris.append(conn)
                tri_surf.append(np.full(len(conn), surf_tag, dtype=np.int64))
            elif etype == GMSH_QUADRANGLE:
                conn = conn.reshape(-1, 4)
                quads.append(conn)
                quad_surf.append(np.full(len(conn), surf_tag, dtype=np.int64))

    mesh = Mesh(
        nodes=nodes_xy,
        triangles=np.vstack(tris) if tris else np.empty((0, 3), dtype=np.int64),
        quads=np.vstack(quads) if quads else np.empty((0, 4), dtype=np.int64),
        tri_surface=np.concatenate(tri_surf) if tri_surf else np.empty(0, dtype=np.int64),
        quad_surface=np.concatenate(quad_surf) if quad_surf else np.empty(0, dtype=np.int64),
        surface_roles=surface_roles or {},
        constrained_edges=_extract_constrained_edges(breakline_curves, node_index),
    )
    orient_ccw(mesh)
    return mesh


def _extract_constrained_edges(
    curve_tags: list[int] | None, node_index: np.ndarray
) -> np.ndarray:
    """拘束曲線上に生成された 1 次元要素を、メッシュ辺として取り出す。"""
    if not curve_tags:
        return np.zeros((0, 2), dtype=np.int64)

    edges: list[np.ndarray] = []
    for tag in curve_tags:
        etypes, _, enodes = gmsh.model.mesh.getElements(1, tag)
        for etype, conn in zip(etypes, enodes):
            if etype == GMSH_LINE:
                edges.append(node_index[np.asarray(conn, dtype=np.int64)].reshape(-1, 2))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.sort(np.vstack(edges), axis=1), axis=0)


def orient_ccw(mesh: Mesh) -> None:
    """全要素の節点順を反時計回りに揃える。

    内角・凸性・scaled Jacobian の判定はすべて反時計回りを前提にしているため、
    gmsh の面の向きに依存しないようここで正規化する。
    """
    for conn in (mesh.triangles, mesh.quads):
        if len(conn) == 0:
            continue
        pts = mesh.nodes[conn]
        x, y = pts[..., 0], pts[..., 1]
        signed = 0.5 * (x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1)
        flip = signed < 0.0
        if flip.any():
            conn[flip] = conn[flip][:, ::-1]


def drop_unused_nodes(mesh: Mesh) -> Mesh:
    """どの要素にも使われていない節点を除去する。"""
    used = np.zeros(mesh.n_nodes, dtype=bool)
    if len(mesh.triangles):
        used[mesh.triangles.ravel()] = True
    if len(mesh.quads):
        used[mesh.quads.ravel()] = True
    if used.all():
        return mesh

    remap = np.full(mesh.n_nodes, -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    edges = mesh.constrained_edges
    if len(edges):
        edges = remap[edges]
        edges = edges[(edges >= 0).all(axis=1)]
    return Mesh(
        nodes=mesh.nodes[used],
        triangles=remap[mesh.triangles] if len(mesh.triangles) else mesh.triangles,
        quads=remap[mesh.quads] if len(mesh.quads) else mesh.quads,
        tri_surface=mesh.tri_surface,
        quad_surface=mesh.quad_surface,
        node_z=mesh.node_z[used] if mesh.node_z is not None else None,
        surface_roles=mesh.surface_roles,
        constrained_edges=edges,
    )


def load_mesh_from_msh(path: str | Path) -> Mesh:
    """meshio で .msh を読み、Mesh オブジェクトを返す（face/edge 再出力用）。"""
    import meshio

    mio = meshio.read(str(path))
    points = np.asarray(mio.points, dtype=np.float64)
    if points.shape[1] >= 3:
        node_z = points[:, 2].copy()
        nodes = points[:, :2]
    else:
        node_z = None
        nodes = points[:, :2] if points.shape[1] >= 2 else points.reshape(-1, 2)

    triangles = np.empty((0, 3), dtype=np.int64)
    quads = np.empty((0, 4), dtype=np.int64)
    for block in mio.cells:
        if block.type == "triangle":
            triangles = np.vstack([triangles, np.asarray(block.data, dtype=np.int64)])
        elif block.type == "quad":
            quads = np.vstack([quads, np.asarray(block.data, dtype=np.int64)])

    if len(triangles) == 0 and len(quads) == 0:
        raise ValueError(f"{path}: 三角形・四角形要素がありません")

    return Mesh(
        nodes=nodes,
        triangles=triangles,
        quads=quads,
        tri_surface=np.ones(len(triangles), dtype=np.int64),
        quad_surface=np.ones(len(quads), dtype=np.int64),
        node_z=node_z,
        surface_roles={1: "interior"},
        constrained_edges=np.zeros((0, 2), dtype=np.int64),
    )
