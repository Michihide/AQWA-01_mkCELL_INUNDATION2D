"""生成後のメッシュ修復。

外周四角形帯の内側境界は節点間隔が固定されているため、鋭角部では面積下限を
割る三角形がどうしても残る。ここでは、そうした三角形を隣接三角形と統合して
1 個の凸四角形にする。内部の凸四角形は許容されるため、三角形へ戻す必要はない。
統合により要素数も減る。それでも消えない三角形は辺の縮約で潰す。

拘束ブレークライン上の辺は、統合しても縮約してもならない。線がメッシュ辺から
外れてしまうためで、Mesh.constrained_edges を見て除外する。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .mesh_parser import Mesh
from .quality_metrics import (
    element_size_estimate,
    evaluate_quality,
    neighbor_ratio_violations,
    polygon_areas,
    quad_quality,
    triangle_quality,
)
from .utils import get_logger


@dataclass
class RepairStats:
    merged_pairs: int = 0
    relaxed_merges: int = 0
    collapsed_edges: int = 0
    remaining_undersized: int = 0
    rejected_nonconvex: int = 0
    sacrificed_constraints: int = 0
    split_quads: int = 0
    split_band_quads: int = 0
    split_band_quads_ratio: int = 0
    split_coarse: int = 0

    def add(self, other: RepairStats) -> None:
        self.merged_pairs += other.merged_pairs
        self.relaxed_merges += other.relaxed_merges
        self.collapsed_edges += other.collapsed_edges
        self.rejected_nonconvex += other.rejected_nonconvex
        self.sacrificed_constraints += other.sacrificed_constraints
        self.split_quads += other.split_quads
        self.remaining_undersized = other.remaining_undersized


def _triangle_edge_map(tris: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """無向辺 -> [(三角形番号, 局所辺番号)]。"""
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ti, tri in enumerate(tris):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append((ti, k))
    return edge_map


def _merged_quad(tri1: np.ndarray, k: int, tri2: np.ndarray) -> np.ndarray | None:
    """辺を共有する 2 三角形を 1 個の四角形へ。向きは反時計回りを保つ。"""
    a, b = int(tri1[k]), int(tri1[(k + 1) % 3])
    c = int(tri1[(k + 2) % 3])
    apex = [int(n) for n in tri2 if int(n) not in (a, b)]
    if len(apex) != 1:
        return None
    return np.array([c, a, apex[0], b], dtype=np.int64)


def merge_small_triangles(
    mesh: Mesh,
    min_area: float,
    max_aspect_ratio: float,
    min_interior_angle_deg: float,
    max_interior_angle_deg: float,
    min_scaled_jacobian: float,
    relaxed: bool = False,
    sacrifice_constraints: bool = False,
) -> tuple[Mesh, RepairStats]:
    """面積下限を割る三角形を、隣接三角形と統合して凸四角形にする。

    relaxed=True では凸性と面積下限だけを課し、形状品質の閾値を外す。面積下限は
    ハード制約なので、鋭角部で品質基準と両立しない場合はこちらを優先する。

    sacrifice_constraints=True では拘束辺をまたぐ統合も許す。そこだけ拘束線が
    メッシュ辺から外れるが、面積下限を優先するという方針に従う。

    外周に接する三角形（mesh.boundary_touching_element_mask()）は面積下限の
    対象外なので、ここでも修復（統合）の対象にしない。境界形状の再現性を
    優先するため。
    """
    logger = get_logger()
    stats = RepairStats()
    if len(mesh.triangles) == 0:
        return mesh, stats

    tris = mesh.triangles
    areas = np.abs(polygon_areas(mesh.nodes[tris]))
    boundary_touch = mesh.boundary_touching_element_mask()
    undersized = np.flatnonzero((areas < min_area) & ~boundary_touch[:len(tris)])
    if undersized.size == 0:
        return mesh, stats

    edge_map = _triangle_edge_map(tris)
    constrained = set() if sacrifice_constraints else mesh.constrained_edge_set()
    consumed = np.zeros(len(tris), dtype=bool)
    new_quads: list[np.ndarray] = []
    new_quad_surface: list[int] = []

    # 面積の小さい三角形から処理する
    for ti in undersized[np.argsort(areas[undersized])]:
        ti = int(ti)
        if consumed[ti]:
            continue

        best: tuple[float, np.ndarray, int] | None = None
        for k in range(3):
            a, b = int(tris[ti, k]), int(tris[ti, (k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            if key in constrained:
                continue  # 統合するとブレークラインがメッシュ辺から消える
            for tj, _ in edge_map.get(key, []):
                if tj == ti or consumed[tj]:
                    continue
                quad = _merged_quad(tris[ti], k, tris[tj])
                if quad is None:
                    continue
                q = quad_quality(mesh.nodes, quad[None, :])
                if not bool(q["is_convex"][0]):
                    stats.rejected_nonconvex += 1
                    continue
                if float(q["area"][0]) < min_area:
                    continue
                if not relaxed:
                    if float(q["aspect_ratio"][0]) > max_aspect_ratio:
                        continue
                    if float(q["min_angle"][0]) < min_interior_angle_deg:
                        continue
                    if float(q["max_angle"][0]) > max_interior_angle_deg:
                        continue
                    if float(q["scaled_jacobian"][0]) < min_scaled_jacobian:
                        continue
                score = float(q["scaled_jacobian"][0])
                if best is None or score > best[0]:
                    best = (score, quad, tj)

        if best is None:
            continue
        _, quad, tj = best
        consumed[ti] = True
        consumed[tj] = True
        new_quads.append(quad)
        new_quad_surface.append(int(mesh.tri_surface[ti]))
        stats.merged_pairs += 1
        if relaxed:
            stats.relaxed_merges += 1

    if stats.merged_pairs == 0:
        stats.remaining_undersized = int(undersized.size)
        return mesh, stats

    kept = ~consumed
    merged = Mesh(
        nodes=mesh.nodes,
        triangles=tris[kept],
        quads=np.vstack([mesh.quads, np.array(new_quads)]) if len(mesh.quads)
        else np.array(new_quads),
        tri_surface=mesh.tri_surface[kept],
        quad_surface=np.concatenate([mesh.quad_surface, np.array(new_quad_surface)])
        if len(mesh.quad_surface) else np.array(new_quad_surface),
        node_z=mesh.node_z,
        surface_roles=mesh.surface_roles,
        constrained_edges=mesh.constrained_edges,
    )

    tri_area = np.abs(polygon_areas(merged.nodes[merged.triangles])) if len(merged.triangles) else np.empty(0)
    quad_area = np.abs(polygon_areas(merged.nodes[merged.quads])) if len(merged.quads) else np.empty(0)
    merged_boundary = merged.boundary_touching_element_mask()
    n_tri = len(merged.triangles)
    stats.remaining_undersized = int(
        (tri_area[~merged_boundary[:n_tri]] < min_area).sum()
        + (quad_area[~merged_boundary[n_tri:]] < min_area).sum()
    )

    return merged, stats


def _boundary_nodes(mesh: Mesh) -> np.ndarray:
    """外形線上の節点。ここを動かすと解析領域の形が変わるため固定する。"""
    return mesh.boundary_node_mask()


def _apply_collapse(mesh: Mesh, u: int, v: int, target: np.ndarray) -> Mesh:
    """節点 v を u へ統合し、u を target へ移す。

    退化した三角形は削除し、退化した四角形は三角形へ落とす。
    """
    nodes = mesh.nodes.copy()
    nodes[u] = target
    remap = np.arange(len(nodes), dtype=np.int64)
    remap[v] = u

    tris = remap[mesh.triangles] if len(mesh.triangles) else mesh.triangles
    tri_keep = (
        np.array([len(set(map(int, t))) == 3 for t in tris], dtype=bool)
        if len(tris) else np.zeros(0, dtype=bool)
    )
    new_tris = [tris[tri_keep]] if len(tris) else []
    new_tri_surf = [mesh.tri_surface[tri_keep]] if len(tris) else []

    # v を含む四角形だけが潰れうる。それ以外はそのまま残す。
    if len(mesh.quads):
        affected = (mesh.quads == v).any(axis=1) | (mesh.quads == u).any(axis=1)
        quads = remap[mesh.quads]
        keep_quads = [quads[~affected]]
        keep_quad_surf = [mesh.quad_surface[~affected]]
        demoted, demoted_surf = [], []
        for q, s in zip(quads[affected], mesh.quad_surface[affected]):
            uniq = list(dict.fromkeys(int(x) for x in q))
            if len(uniq) == 4:
                keep_quads.append(q[None, :])
                keep_quad_surf.append(np.array([s]))
            elif len(uniq) == 3:
                demoted.append(np.array(uniq, dtype=np.int64))
                demoted_surf.append(s)
            # 2 点以下に潰れた四角形は削除
        if demoted:
            new_tris.append(np.array(demoted))
            new_tri_surf.append(np.array(demoted_surf))
        new_quads = np.vstack(keep_quads)
        new_quad_surf = np.concatenate(keep_quad_surf)
    else:
        new_quads = np.zeros((0, 4), dtype=np.int64)
        new_quad_surf = np.zeros(0, dtype=np.int64)

    return Mesh(
        nodes=nodes,
        triangles=np.vstack(new_tris) if new_tris else np.zeros((0, 3), dtype=np.int64),
        quads=new_quads,
        tri_surface=np.concatenate(new_tri_surf) if new_tri_surf
        else np.zeros(0, dtype=np.int64),
        quad_surface=new_quad_surf,
        node_z=mesh.node_z,
        surface_roles=mesh.surface_roles,
        # 拘束辺は v を含まないため（v は非拘束節点に限る）付け替え不要
        constrained_edges=mesh.constrained_edges,
    )


def _affected_stats(
    mesh: Mesh, nodes: tuple[int, ...], min_area: float, check_shape: bool
) -> tuple[int, float] | None:
    """指定の節点に接する要素の (面積下限を割る数, 最小面積)。

    メッシュ全体を見ると、無関係な既存の品質違反が常に縮約を却下してしまうため、
    影響が及ぶ範囲だけを対象にする。check_shape=True で反転・凹四角形を弾く。
    """
    bad = 0
    smallest = np.inf

    if len(mesh.triangles):
        mask = np.isin(mesh.triangles, nodes).any(axis=1)
        if mask.any():
            area = polygon_areas(mesh.nodes[mesh.triangles[mask]])
            if check_shape and (area <= 0.0).any():
                return None
            bad += int((np.abs(area) < min_area).sum())
            smallest = min(smallest, float(np.abs(area).min()))

    if len(mesh.quads):
        mask = np.isin(mesh.quads, nodes).any(axis=1)
        if mask.any():
            q = quad_quality(mesh.nodes, mesh.quads[mask])
            if check_shape and (not bool(q["is_convex"].all()) or (q["area"] <= 0.0).any()):
                return None
            bad += int((q["area"] < min_area).sum())
            smallest = min(smallest, float(q["area"].min()))

    return bad, smallest


def _collapse_targets(
    mesh: Mesh, a: int, b: int, pinned: np.ndarray
) -> list[tuple[int, int, np.ndarray]]:
    """縮約の (残す節点, 消す節点, 移動先) の候補。

    外形線上とブレークライン上の節点は固定する。前者を動かすと解析領域の形が
    変わり、後者を動かすと拘束線がメッシュ辺から外れるためで、どちらも消せない。
    片側だけが固定なら、動かせる側をそちらへ寄せる一択。どちらも自由なら中点と
    両端点の 3 通りを試す。中点だと隣の四角形が凹む配置があるため端点も要る。
    """
    if pinned[a] and pinned[b]:
        return []
    if pinned[a]:
        return [(a, b, mesh.nodes[a])]
    if pinned[b]:
        return [(b, a, mesh.nodes[b])]
    return [
        (a, b, 0.5 * (mesh.nodes[a] + mesh.nodes[b])),
        (a, b, mesh.nodes[a].copy()),
        (b, a, mesh.nodes[b].copy()),
    ]


def collapse_small_triangles(
    mesh: Mesh, min_area: float, max_collapses: int = 200,
    sacrifice_constraints: bool = False,
) -> tuple[Mesh, RepairStats]:
    """隣が四角形ばかりで統合できない小三角形を、最短辺の縮約で消す。

    外形線上とブレークライン上の節点は動かさない。片側だけが固定の辺は、自由な
    側を固定節点へ寄せることで外形と拘束線を保つ。

    sacrifice_constraints=True では拘束線上の節点も動かす。外形線上の節点だけは
    動かさない。解析領域の形が変わってしまうためで、こちらは譲れない。

    外周に接する三角形（mesh.boundary_touching_element_mask()）は面積下限の
    対象外なので、ここでも修復（縮約）の対象にしない。境界形状の再現性を
    優先するため。
    """
    stats = RepairStats()
    if len(mesh.triangles) == 0:
        return mesh, stats

    # 縮約しても外形線と拘束線の節点は動かないため、一度だけ求めれば足りる
    pinned = _boundary_nodes(mesh)
    if not sacrifice_constraints:
        pinned = pinned | mesh.constrained_node_mask()

    for _ in range(max_collapses):
        tri_area = polygon_areas(mesh.nodes[mesh.triangles])
        boundary_touch = mesh.boundary_touching_element_mask()[:len(mesh.triangles)]
        undersized = np.flatnonzero((tri_area < min_area) & ~boundary_touch)
        if undersized.size == 0:
            break

        applied = False
        for ti in undersized[np.argsort(tri_area[undersized])]:
            tri = mesh.triangles[int(ti)]
            edges = [(int(tri[k]), int(tri[(k + 1) % 3])) for k in range(3)]
            lengths = [float(np.linalg.norm(mesh.nodes[a] - mesh.nodes[b])) for a, b in edges]

            for _, (a, b) in sorted(zip(lengths, edges), key=lambda p: p[0]):
                before = _affected_stats(mesh, (a, b), min_area, check_shape=False)
                if before is None:
                    continue
                bad_before, min_before = before

                for u, v, target in _collapse_targets(mesh, a, b, pinned):
                    candidate = _apply_collapse(mesh, u, v, target)
                    after = _affected_stats(candidate, (u,), min_area, check_shape=True)
                    if after is None:
                        continue
                    bad_after, min_after = after
                    # 下限割れが減るか、減らせなくても最小面積が改善するなら進める。
                    # 後者でも要素数は必ず減るので、繰り返しても止まらないことはない
                    improved = bad_after < bad_before or (
                        bad_after == bad_before and min_after > min_before * (1.0 + 1e-9)
                    )
                    if improved:
                        mesh = candidate
                        stats.collapsed_edges += 1
                        applied = True
                        break
                if applied:
                    break
            if applied:
                break
        if not applied:
            break

    tri_area = polygon_areas(mesh.nodes[mesh.triangles]) if len(mesh.triangles) else np.empty(0)
    quad_area = np.abs(polygon_areas(mesh.nodes[mesh.quads])) if len(mesh.quads) else np.empty(0)
    final_boundary = mesh.boundary_touching_element_mask()
    n_tri = len(mesh.triangles)
    stats.remaining_undersized = int(
        (tri_area[~final_boundary[:n_tri]] < min_area).sum()
        + (quad_area[~final_boundary[n_tri:]] < min_area).sum()
    )
    return mesh, stats


def repair_mesh(mesh: Mesh, cfg, max_passes: int = 5) -> tuple[Mesh, RepairStats]:
    """面積下限違反を修復する。

    まず品質基準を満たす統合だけを行い、それでも残る違反に対して品質基準を
    外した統合を試みる。面積下限がハード制約であるため、この順序にしている。
    """
    logger = get_logger()
    q = cfg.quality
    kwargs = dict(
        min_area=cfg.mesh.min_element_area,
        max_aspect_ratio=q.quadrilateral_aspect_ratio_max,
        min_interior_angle_deg=q.quadrilateral_min_interior_angle_deg,
        max_interior_angle_deg=q.quadrilateral_max_interior_angle_deg,
        min_scaled_jacobian=q.quadrilateral_scaled_jacobian_min,
    )

    total = RepairStats()
    for _ in range(max_passes):
        done_before = total.merged_pairs + total.collapsed_edges

        for relaxed in (False, True):
            for _ in range(max_passes):
                mesh, stats = merge_small_triangles(mesh, relaxed=relaxed, **kwargs)
                total.add(stats)
                if stats.merged_pairs == 0:
                    break

        if total.remaining_undersized:
            # 縮約で潰した小三角形の隣に、統合できる相手が新しく現れることがある
            mesh, stats = collapse_small_triangles(mesh, cfg.mesh.min_element_area)
            total.add(stats)

        if total.remaining_undersized == 0:
            break
        if total.merged_pairs + total.collapsed_edges == done_before:
            # 四角形に囲まれて動けない小三角形が残っているかもしれない。
            # 隣を三角形へ戻して組み直しの余地を作り、もう一巡する。
            mesh, split = split_quads_around_small_triangles(
                mesh, cfg.mesh.min_element_area
            )
            total.split_quads += split
            if split == 0:
                break

    # ここまで残る違反は、拘束辺が統合も縮約も塞いでいる。埋め込み曲線の脇では
    # サイズ場を上げても gmsh が小さい要素を作るため、これは避けられない。
    # 面積下限を優先する方針に従い、その場に限って拘束を諦める。
    if total.remaining_undersized:
        live_before = _live_constrained_edges(mesh)
        for _ in range(max_passes):
            done_before = total.merged_pairs + total.collapsed_edges
            for _ in range(max_passes):
                mesh, stats = merge_small_triangles(
                    mesh, relaxed=True, sacrifice_constraints=True, **kwargs
                )
                total.add(stats)
                if stats.merged_pairs == 0:
                    break
            if total.remaining_undersized:
                mesh, stats = collapse_small_triangles(
                    mesh, cfg.mesh.min_element_area, sacrifice_constraints=True
                )
                total.add(stats)
            if total.remaining_undersized == 0:
                break
            if total.merged_pairs + total.collapsed_edges == done_before:
                # ここまで来たら外周帯の四角形も割る。境界を四角形で覆うことより
                # 面積下限のほうが優先という方針に従う。
                mesh, split = split_quads_around_small_triangles(
                    mesh, cfg.mesh.min_element_area, allow_band=True
                )
                total.split_quads += split
                total.split_band_quads += split
                if split == 0:
                    break
        total.sacrificed_constraints = live_before - _live_constrained_edges(mesh)

    # 面積下限を満たしたうえで、隣接比の残りを詰める。外周帯の四角形が粗い側
    # なら先に割って三角形にし（面積下限までは割ってよい）、それでも残る
    # 三角形どうしの粗さは重心分割ではなく最長辺の二等分で詰める。分割は
    # 要素を小さくするだけなので、ここまでで満たした面積下限を壊さない。
    mesh, split_band = split_band_quads_for_ratio(mesh, cfg)
    total.split_quads += split_band
    total.split_band_quads_ratio += split_band
    if split_band:
        logger.info(
            "隣接比を詰めるため、外周帯の四角形 %d 個を三角形に割りました",
            split_band,
        )

    mesh, total.split_coarse = split_coarse_triangles(mesh, cfg)
    if total.split_coarse:
        logger.info(
            "隣接比を詰めるため、粗い三角形 %d 個の最長辺を二等分しました",
            total.split_coarse,
        )

    if total.merged_pairs or total.collapsed_edges:
        logger.info(
            "小面積要素の修復: 統合 %d 組（うち品質緩和 %d）、辺の縮約 %d 回、"
            "四角形の再分割 %d 個、残存違反 %d",
            total.merged_pairs, total.relaxed_merges, total.collapsed_edges,
            total.split_quads, total.remaining_undersized,
        )
    if total.sacrificed_constraints:
        logger.warning(
            "面積下限を満たすため、拘束辺 %d 本をメッシュ辺から外しました",
            total.sacrificed_constraints,
        )
    if total.split_band_quads:
        logger.warning(
            "面積下限を満たすため、外周帯の四角形 %d 個を三角形に割りました",
            total.split_band_quads,
        )
    if total.split_band_quads_ratio:
        logger.info(
            "隣接比を満たすため、外周帯の四角形 %d 個を三角形に割りました",
            total.split_band_quads_ratio,
        )
    return mesh, total


def split_coarse_triangles(mesh: Mesh, cfg, max_passes: int = 8) -> tuple[Mesh, int]:
    """隣より粗すぎる三角形の最長辺を二等分し、隣接比を上限以下へ詰める。

    サイズ場をいくら整えても、出来上がりの要素寸法は目標どおりにはならない。
    特に外周四角形帯の内側では、帯の四角形が Transfinite で固定されている一方、
    そこから内側へ伸びる三角形が目標より大きく作られ、比が上限を超えて残る。

    最長辺の二等分は面積を半分（代表辺長を 1/sqrt(2)）にする。重心分割と違って
    細長い子を作らないので、形状基準を保ったまま比を詰められる。宙ぶらりんの
    節点を作らないよう、辺を挟んだ相手も同じ辺で割る。挿入する節点は辺の中点で
    あり、拘束線も領域界も直線上を動かないため、形状としては何も変わらない。
    """
    limit = cfg.mesh.max_neighbor_element_ratio
    if limit <= 0.0:
        return mesh, 0

    total = 0
    for _ in range(max_passes):
        report = evaluate_quality(mesh)
        coarse, _ = neighbor_ratio_violations(mesh, report, limit)
        coarse &= report.kind == 3
        coarse &= ~mesh.band_element_mask()
        picked = np.flatnonzero(coarse)
        if not len(picked):
            break
        # 粗い順に処理する。1 回の掃引では、隣り合う組が二重に割られないよう
        # 触れた要素を記録して飛ばす。
        picked = picked[np.argsort(-report.area[picked])]

        mesh, n, why = _bisect_pass(mesh, picked, cfg)
        if n == 0:
            if why.get("相手が四角形"):
                # 四角形だけに囲まれていると辺を割れない。隣の四角形を三角形へ
                # 戻して足場を作り、次の掃引で二等分する。
                mesh, opened = _open_quads_for_coarse(mesh, cfg, coarse, report)
                if opened:
                    continue
            if any(why.values()):
                get_logger().info(
                    "隣接比を詰められなかった三角形 %d 個（内訳: %s）",
                    sum(why.values()),
                    ", ".join(f"{k} {v}" for k, v in why.items() if v),
                )
            break
        total += n
    return mesh, total


def split_band_quads_for_ratio(mesh: Mesh, cfg) -> tuple[Mesh, int]:
    """隣の三角形よりずっと粗い外周帯の四角形を対角線で三角形へ割る。

    帯は Transfinite で寸法が固定されており、通常は隣接比の緩和対象から
    除外している（split_coarse_triangles、_open_quads_for_coarse）。しかし
    地形細分化で内側の三角形が面積下限近くまで小さくなると、帯との比が
    大きく開くことがある。境界を四角形で覆うことより隣接比を優先してよい
    という指示に従い、面積下限を割らない範囲でここも割る。
    """
    limit = cfg.mesh.max_neighbor_element_ratio
    if limit <= 0.0 or len(mesh.quads) == 0 or len(mesh.triangles) == 0:
        return mesh, 0

    band = mesh.band_element_mask()
    if not band[len(mesh.triangles):].any():
        return mesh, 0

    report = evaluate_quality(mesh)
    coarse, _ = neighbor_ratio_violations(mesh, report, limit)
    n_tri = len(mesh.triangles)
    victims = np.flatnonzero(coarse[n_tri:] & band[n_tri:])
    if not len(victims):
        return mesh, 0

    keep = np.ones(len(mesh.quads), dtype=bool)
    new_tris: list[np.ndarray] = []
    new_surf: list[int] = []
    for qi in map(int, victims):
        halves = _split_quad(mesh.nodes, mesh.quads[qi], min_area=cfg.mesh.min_element_area)
        if halves is None:
            continue
        keep[qi] = False
        new_tris.extend(halves)
        new_surf.extend([int(mesh.quad_surface[qi])] * 2)

    if not new_tris:
        return mesh, 0

    return replace(
        mesh,
        triangles=np.vstack([mesh.triangles, np.array(new_tris)]),
        quads=mesh.quads[keep],
        tri_surface=np.concatenate([mesh.tri_surface, np.array(new_surf)]),
        quad_surface=mesh.quad_surface[keep],
        node_z=None,
    ), len(new_tris) // 2


def _open_quads_for_coarse(
    mesh: Mesh, cfg, coarse: np.ndarray, report
) -> tuple[Mesh, int]:
    """四角形だけに囲まれた粗い三角形の隣を、対角線で三角形へ戻す。

    割った四角形は代表辺長が 1/sqrt(2) になる。比を開かせている細かい側を割ると
    かえって差が広がるので、粗い三角形より十分細かい四角形は対象にしない。
    外周四角形帯は Transfinite で構造が決まっているため割らない。
    """
    limit = cfg.mesh.max_neighbor_element_ratio
    size = element_size_estimate(report)
    n_tri = len(mesh.triangles)
    band = {s for s, role in mesh.surface_roles.items() if role == "quad_boundary"}

    quad_of_edge: dict[tuple[int, int], int] = {}
    for qi, quad in enumerate(mesh.quads):
        if int(mesh.quad_surface[qi]) in band:
            continue
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            quad_of_edge[(min(a, b), max(a, b))] = qi

    victims: dict[int, None] = {}
    for e in np.flatnonzero(coarse[:n_tri]):
        tri = mesh.triangles[int(e)]
        best, best_size = None, 0.0
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            qi = quad_of_edge.get((min(a, b), max(a, b)))
            if qi is None:
                continue
            qs = size[n_tri + qi]
            # 割ったあとも三角形との比が上限に収まる相手だけを選ぶ
            if qs * limit < size[e] or qs <= best_size:
                continue
            halves = _split_quad(mesh.nodes, mesh.quads[qi])
            # 対角線は半分ずつになるとは限らない。面積下限を割る割り方はしない。
            if halves is None or not _pieces_pass(halves, mesh.nodes, {}, cfg):
                continue
            best, best_size = qi, qs
        if best is not None:
            victims[best] = None

    if not victims:
        return mesh, 0

    keep = np.ones(len(mesh.quads), dtype=bool)
    new_tris: list[np.ndarray] = []
    new_surf: list[int] = []
    for qi in victims:
        halves = _split_quad(mesh.nodes, mesh.quads[qi])
        if halves is None:
            continue
        keep[qi] = False
        new_tris.extend(halves)
        new_surf.extend([int(mesh.quad_surface[qi])] * 2)

    if not new_tris:
        return mesh, 0
    return (
        replace(
            mesh,
            triangles=np.vstack([mesh.triangles, np.array(new_tris)]),
            quads=mesh.quads[keep],
            tri_surface=np.concatenate([mesh.tri_surface, np.array(new_surf)]),
            quad_surface=mesh.quad_surface[keep],
            node_z=None,
        ),
        len(new_tris) // 2,
    )


def _bisect_pass(mesh: Mesh, picked: np.ndarray, cfg) -> tuple[Mesh, int, dict]:
    """選んだ三角形の最長辺を二等分する掃引を 1 回。

    Returns:
        (メッシュ, 二等分した個数, 見送った理由の内訳)
    """
    tris = [row.copy() for row in mesh.triangles]
    surf = list(mesh.tri_surface)
    extra: dict[int, np.ndarray] = {}
    constrained = mesh.constrained_edge_set()
    tri_owners, quad_edges = _edge_owners(mesh)
    band = mesh.band_element_mask()
    touched: set[int] = set()
    why = {"相手が四角形": 0, "相手が帯か処理済み": 0, "形状基準を満たさない": 0}
    n_done = 0

    for e in map(int, picked):
        if e in touched:
            continue
        xy = mesh.nodes[tris[e]]
        lengths = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)

        plan = None
        reason = ""
        # 長い辺から順に試す。長辺を割るほど子の形が良いが、相手が四角形なら
        # 割れないので次の辺へ回る。
        for k in np.argsort(-lengths):
            a, b, c = (int(tris[e][(int(k) + i) % 3]) for i in range(3))
            key = (min(a, b), max(a, b))
            if key in quad_edges:
                # 四角形を崩すと外周帯や統合結果が壊れる
                reason = reason or "相手が四角形"
                continue
            other = next((o for o in tri_owners[key] if o != e), None)
            if other is not None and (other in touched or band[other]):
                reason = reason or "相手が帯か処理済み"
                continue

            mid = mesh.n_nodes + len(extra)
            centre = 0.5 * (mesh.nodes[a] + mesh.nodes[b])
            pieces: list[tuple[int | None, tuple[int, int, int]]] = [
                (e, (a, mid, c)), (None, (mid, b, c))
            ]
            if other is not None:
                d = _opposite_vertex(tris[other], a, b)
                if d is None:
                    continue
                pieces += [(other, (b, mid, d)), (None, (mid, a, d))]

            if not _pieces_pass(
                [conn for _, conn in pieces], mesh.nodes, {**extra, mid: centre}, cfg
            ):
                reason = reason or "形状基準を満たさない"
                continue
            plan = (key, mid, centre, other, pieces)
            break

        if plan is None:
            why[reason] = why.get(reason, 0) + 1
            continue

        key, mid, centre, other, pieces = plan
        extra[mid] = centre
        for slot, conn in pieces:
            if slot is None:
                tris.append(np.array(conn, dtype=mesh.triangles.dtype))
                surf.append(surf[e])
            else:
                tris[slot] = np.array(conn, dtype=mesh.triangles.dtype)
        touched.update({e} if other is None else {e, other})
        if key in constrained:
            # 中点は辺の上にあるので、線は 2 本の辺として残る
            constrained.discard(key)
            constrained.add((min(key[0], mid), max(key[0], mid)))
            constrained.add((min(key[1], mid), max(key[1], mid)))
        n_done += 1

    if n_done == 0:
        return mesh, 0, why

    added = np.array([extra[i] for i in sorted(extra)])
    edges = (
        np.array(sorted(constrained), dtype=np.int64) if constrained
        else np.zeros((0, 2), dtype=np.int64)
    )
    return (
        replace(
            mesh,
            nodes=np.vstack([mesh.nodes, added]),
            triangles=np.array(tris),
            tri_surface=np.array(surf, dtype=mesh.tri_surface.dtype),
            constrained_edges=edges,
            node_z=None,
        ),
        n_done,
        why,
    )


def _edge_owners(mesh: Mesh) -> tuple[dict, set]:
    """辺 -> それを持つ三角形の一覧と、四角形が持つ辺の集合。"""
    owners: dict[tuple[int, int], list[int]] = {}
    for i, tri in enumerate(mesh.triangles):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            owners.setdefault((min(a, b), max(a, b)), []).append(i)
    quad_edges = set()
    for quad in mesh.quads:
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            quad_edges.add((min(a, b), max(a, b)))
    return owners, quad_edges


def _opposite_vertex(tri: np.ndarray, a: int, b: int) -> int | None:
    rest = [int(v) for v in tri if int(v) not in (a, b)]
    return rest[0] if len(rest) == 1 else None


def _pieces_pass(conns, nodes: np.ndarray, extra: dict, cfg) -> bool:
    """二等分で出来る三角形が面積下限と形状基準を満たすか。"""
    q = cfg.quality
    coords = np.array([
        [extra[v] if v in extra else nodes[v] for v in conn] for conn in conns
    ]).reshape(-1, 2)
    metrics = triangle_quality(coords, np.arange(len(coords)).reshape(-1, 3))
    return bool(
        np.all(np.abs(metrics["area"]) >= cfg.mesh.min_element_area)
        and np.all(metrics["min_angle"] >= q.triangle_min_angle_deg)
        and np.all(metrics["radius_ratio"] >= q.triangle_radius_ratio_min)
        and np.all(metrics["aspect_ratio"] <= q.triangle_aspect_ratio_max)
    )


def split_quads_around_small_triangles(
    mesh: Mesh, min_area: float, allow_band: bool = False
) -> tuple[Mesh, int]:
    """四角形だけに囲まれて動けない小三角形の隣を、三角形へ戻す。

    統合段が作った四角形が、結果として小三角形を閉じ込めてしまうことがある。
    統合には三角形の相手が要り、縮約は周囲の四角形が凹むため通らず、手詰まりに
    なる。隣の四角形を対角線で割って三角形へ戻せば、次の統合段でその半分と
    組み直せる。

    外周四角形帯は通常は割らない。境界は四角形で覆いたいためだが、面積下限の
    ほうが優先なので、他に手が無くなった段では allow_band=True で帯も割る。
    """
    if len(mesh.triangles) == 0 or len(mesh.quads) == 0:
        return mesh, 0

    areas = np.abs(polygon_areas(mesh.nodes[mesh.triangles]))
    undersized = np.flatnonzero(areas < min_area)
    if undersized.size == 0:
        return mesh, 0

    band = set() if allow_band else {
        s for s, role in mesh.surface_roles.items() if role == "quad_boundary"
    }
    quad_of_edge: dict[tuple[int, int], int] = {}
    for qi, quad in enumerate(mesh.quads):
        if int(mesh.quad_surface[qi]) in band:
            continue
        for k in range(4):
            a, b = int(quad[k]), int(quad[(k + 1) % 4])
            quad_of_edge[(a, b) if a < b else (b, a)] = qi

    victims: dict[int, None] = {}
    for ti in undersized:
        tri = mesh.triangles[int(ti)]
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            qi = quad_of_edge.get((a, b) if a < b else (b, a))
            if qi is not None:
                victims[qi] = None
                break

    if not victims:
        return mesh, 0

    keep = np.ones(len(mesh.quads), dtype=bool)
    new_tris: list[np.ndarray] = []
    new_surf: list[int] = []
    for qi in victims:
        quad = mesh.quads[qi]
        halves = _split_quad(mesh.nodes, quad, min_area=min_area)
        if halves is None:
            continue
        keep[qi] = False
        new_tris.extend(halves)
        new_surf.extend([int(mesh.quad_surface[qi])] * 2)

    if not new_tris:
        return mesh, 0

    return Mesh(
        nodes=mesh.nodes,
        triangles=np.vstack([mesh.triangles, np.array(new_tris)]),
        quads=mesh.quads[keep],
        tri_surface=np.concatenate([mesh.tri_surface, np.array(new_surf)]),
        quad_surface=mesh.quad_surface[keep],
        node_z=mesh.node_z,
        surface_roles=mesh.surface_roles,
        constrained_edges=mesh.constrained_edges,
    ), len(new_tris) // 2


def _split_quad(
    nodes: np.ndarray, quad: np.ndarray, min_area: float = 0.0
) -> list[np.ndarray] | None:
    """四角形を対角線で 2 つの三角形へ。面積の偏りが小さい方の対角線を選ぶ。

    どちらの対角線でも半分が面積下限を割るなら None を返す。四角形を割った
    結果が新たな面積下限違反になっては、割る意味がない。
    """
    a, b, c, d = (int(n) for n in quad)
    best = None
    for tris in ([np.array([a, b, c]), np.array([a, c, d])],
                 [np.array([a, b, d]), np.array([b, c, d])]):
        area = np.abs(polygon_areas(nodes[np.array(tris)]))
        if (area <= 0.0).any() or (area < min_area).any():
            continue
        score = float(area.min())
        if best is None or score > best[0]:
            best = (score, tris)
    return None if best is None else best[1]


def _live_constrained_edges(mesh: Mesh) -> int:
    """拘束辺のうち、実際に要素の辺として残っている本数。"""
    if not len(mesh.constrained_edges):
        return 0
    edges = set()
    for elems in (mesh.triangles, mesh.quads):
        for elem in elems:
            m = len(elem)
            for k in range(m):
                a, b = int(elem[k]), int(elem[(k + 1) % m])
                edges.add((a, b) if a < b else (b, a))
    return sum(
        1 for a, b in mesh.constrained_edges
        if ((int(a), int(b)) if a < b else (int(b), int(a))) in edges
    )
