"""DEM から盛り土（道路盛土・堤防）の天端線を検出する。

DEM だけで判定する。OpenStreetMap は形状に一切使わない。検出した線が道路と
一致するかは事後の照合にしか用いず、その結果を線に反映させることもしない。

原理はトップハット変換である。構造要素より幅の狭い凸部だけが残るので、盛り土や
堤防は取り出され、山腹や段丘のような幅の広い高まりは原理的に落ちる。ただし山の
尾根も「幅の狭い凸部」なので、それだけでは足りない。素の地形が広域的に平坦で
あること、比高が構造物として妥当な範囲にあること、解析領域の内側にあることを
併せて課す。残った領域を細線化して天端線にする。

解像度は 1-2 m が望ましいが、5 m でも実用になる。球磨川の 5 m DEM では検出線の
70.7% が OSM 道路の 15 m 以内、10.1% が堤防にあたり、OSM を使わない検出として
妥当な結果が得られている。平滑化のぶん比高は小さく出る（中央値 1.75 m に対し、
斐伊川の 1 m DEM では 3.10 m）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_closing,
    grey_opening,
    label,
    maximum_filter,
    minimum_filter,
)
from shapely.geometry import LineString
from skimage.morphology import remove_small_objects, skeletonize

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# 交差数を数えるための、8 近傍を一周する順序
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


@dataclass
class DetectionParams:
    """既定値は 1-5 m の DEM と、都市河川の氾濫原を想定している。"""

    min_relief: float = 1.0
    max_relief: float = 15.0
    width: float = 60.0
    min_area: float = 400.0
    min_length: float = 100.0
    max_sinuosity: float = 2.0
    flat_relief: float = 10.0
    flat_scale: float = 200.0
    simplify: float = 3.0


def top_hat(z: np.ndarray, size_px: int) -> np.ndarray:
    """幅が size_px 未満の凸部の高さ。

    正方形の構造要素なら scipy が 1 次元 2 回に分解して処理するため、円形と違って
    大きな DEM でも実用的な速度で回る。
    """
    filled = np.where(np.isfinite(z), z, np.nanmin(z))
    return filled - grey_opening(filled, size=(size_px, size_px), mode="nearest")


def embankment_mask(
    z: np.ndarray, px: float, inside: np.ndarray, params: DetectionParams
) -> tuple[np.ndarray, np.ndarray]:
    """盛り土とみなす画素のマスクと、その比高を返す。"""
    size_px = max(3, int(round(params.width / px)) | 1)
    relief = top_hat(z, size_px)

    # トップハットを引いた残りが「素の地形」。その広域起伏で氾濫原を選ぶ。
    base = np.where(np.isfinite(z), z, np.nanmin(z)) - relief
    w = max(3, int(round(params.flat_scale / px)) | 1)
    flat = (maximum_filter(base, w) - minimum_filter(base, w)) <= params.flat_relief

    mask = (
        (relief >= params.min_relief)
        & (relief <= params.max_relief)
        & flat
        & inside
    )
    # 横断歩道や樹木で切れた区間を繋ぐ
    mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    # max_size は「これ以下を消す」。面積下限そのものではないので 1 引く。
    mask = remove_small_objects(mask, max_size=int(params.min_area / (px * px)) - 1)
    return mask, relief


def skeleton_to_paths(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """細線化した画素集合を、分岐点で切った画素列に分解する。

    隣接画素の数をそのまま次数として使うと、階段状に並んだ画素が分岐点に見える。
    斜めの隣接があるためで、実際には枝分かれしていない。そこで 8 近傍を一周した
    ときの 0 から 1 への変化の回数（交差数）で判定する。これなら端点が 1、
    通過点が 2、分岐点が 3 以上になる。
    """
    pix = set(map(tuple, np.argwhere(skel)))

    def nbrs(p: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = p
        return [q for q in ((r + dr, c + dc) for dr, dc in NEIGHBORS) if q in pix]

    crossing = {p: _crossing_number(p, pix) for p in pix}
    nodes = {p for p in pix if crossing[p] != 2}

    paths: list[list[tuple[int, int]]] = []
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def walk(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, first]
        seen = {start, first}
        used.add((start, first))
        used.add((first, start))
        cur, prev = first, start
        while crossing[cur] == 2:
            cands = [q for q in nbrs(cur) if q not in seen]
            if not cands:
                # 一周して戻ってきたなら閉じる
                if start in nbrs(cur) and len(path) > 2:
                    path.append(start)
                break
            # 階段状の並びでは斜めと直交の両方が候補になる。直交を選べば飛ばさない。
            nxt = min(cands, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            path.append(nxt)
            seen.add(nxt)
            used.add((cur, nxt))
            used.add((nxt, cur))
            prev, cur = cur, nxt
        return path

    for node in nodes:
        for nb in nbrs(node):
            if (node, nb) not in used:
                paths.append(walk(node, nb))

    # どこにも分岐点が無い閉ループが残る
    remaining = pix - {p for path in paths for p in path}
    while remaining:
        start = next(iter(remaining))
        nb = nbrs(start)
        if not nb:
            remaining.discard(start)
            continue
        path = walk(start, nb[0])
        paths.append(path)
        remaining -= set(path)

    return paths


def _crossing_number(p: tuple[int, int], pix: set[tuple[int, int]]) -> int:
    """8 近傍を一周したときに、空から画素へ変わる回数。"""
    r, c = p
    ring = [(r + dr, c + dc) for dr, dc in RING]
    on = [q in pix for q in ring]
    return sum(1 for i in range(len(on)) if on[i] and not on[i - 1])


def too_sinuous(line: LineString, limit: float) -> bool:
    """細長い構造物ではなく、丘の頂部を拾ってしまった線を弾く。

    盛り土は道路や堤防なので、うねっていても両端は離れる。丘の頂部を細線化すると
    その場でぐるぐる回る線や閉ループになり、全長のわりに両端が近い。
    """
    ends = np.asarray(line.coords)[[0, -1]]
    span = float(np.linalg.norm(ends[1] - ends[0]))
    return span < 1e-6 or line.length / span > limit


def detect_embankments(
    z: np.ndarray,
    px: float,
    x_min: float,
    y_max: float,
    inside: np.ndarray,
    params: DetectionParams,
) -> tuple[list[LineString], list[dict], np.ndarray]:
    """天端線と、その属性、盛り土マスクを返す。"""
    mask, relief = embankment_mask(z, px, inside, params)

    lines: list[LineString] = []
    attrs: list[dict] = []
    for path in skeleton_to_paths(skeletonize(mask)):
        rc = np.array(path)
        xy = np.stack([
            x_min + (rc[:, 1] + 0.5) * px,
            y_max - (rc[:, 0] + 0.5) * px,
        ], axis=1)
        line = LineString(xy).simplify(params.simplify, preserve_topology=False)
        if line.length < params.min_length or too_sinuous(line, params.max_sinuosity):
            continue
        values = relief[rc[:, 0], rc[:, 1]]
        lines.append(line)
        attrs.append({
            "source": "dem_tophat",
            "relief_median": float(np.median(values)),
            "relief_max": float(values.max()),
        })
    return lines, attrs, mask


def count_blobs(mask: np.ndarray) -> int:
    return int(label(mask)[1])
