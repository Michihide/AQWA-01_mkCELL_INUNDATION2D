#!/usr/bin/env python3
"""要素を大きさでレベル分けしたとき、隣り合う要素のレベル差が 1 以内かを調べる。

ネスト格子では 1 段階までに抑えるのが普通なので、非構造格子でも同じ見方で
粗密の付き方を確かめる。レベルは「1 段で一辺が半分」と定義し、基準の辺長 L0 を
level 0 として level = floor(log2(L0 / L)) とする。

大きさの尺度には辺長 L を使い、面積そのものは使わない。三角形と四角形が混在
するためで、同じ辺長でも四角形の面積は正三角形の 2.3 倍になる。面積で比べると
四角形と三角形が隣り合うだけで比が 1.5 倍に見えてしまう。

隣接は辺の共有で判定する。頂点だけを共有する要素は隣とみなさない。

    python scripts/check_level_balance.py --mesh output/kuma_hitoyoshi/kuma_mesh.vtu
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import meshio
import numpy as np


def element_areas(nodes: np.ndarray, cells: list[np.ndarray]) -> np.ndarray:
    """多角形の面積（シューレースの公式）。"""
    out = []
    for conn in cells:
        xy = nodes[conn]
        x, y = xy[:, :, 0], xy[:, :, 1]
        out.append(0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        ))
    return np.concatenate(out)


def element_sizes(nodes: np.ndarray, cells: list[np.ndarray]) -> np.ndarray:
    """要素の代表辺長。三角形は正三角形換算、四角形は正方形換算。

    src.quality_metrics.element_size_estimate と同じ定義にしてある。反復ループが
    この寸法で判定しているので、検査だけ別の物差しを使うと噛み合わない。
    """
    out = []
    for conn in cells:
        xy = nodes[conn]
        x, y = xy[:, :, 0], xy[:, :, 1]
        area = 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        )
        out.append(
            np.sqrt(area) if conn.shape[1] == 4
            else np.sqrt(4.0 * area / np.sqrt(3.0))
        )
    return np.concatenate(out)


def edge_adjacency(cells: list[np.ndarray]) -> np.ndarray:
    """辺を共有する要素の組 (m, 2)。"""
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    offset = 0
    for conn in cells:
        m = conn.shape[1]
        for local, elem in enumerate(conn):
            for k in range(m):
                a, b = int(elem[k]), int(elem[(k + 1) % m])
                owners[(a, b) if a < b else (b, a)].append(offset + local)
        offset += len(conn)
    pairs = [tuple(v) for v in owners.values() if len(v) == 2]
    return np.array(pairs, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--base-size", type=float, default=0.0,
                    help="level 0 とする辺長 [m]。0 で最大要素の辺長を使う")
    ap.add_argument("--step", type=float, default=2.0,
                    help="1 レベルあたりの辺長の比。既定の 2 は面積 1/4 に相当")
    args = ap.parse_args()

    mesh = meshio.read(args.mesh)
    nodes = mesh.points[:, :2]
    cells = [
        np.asarray(block.data)
        for block in mesh.cells
        if block.type in ("triangle", "quad")
    ]
    if not cells:
        raise SystemExit("三角形・四角形が見つかりません")

    area = element_areas(nodes, cells)
    size = element_sizes(nodes, cells)
    base = args.base_size if args.base_size > 0.0 else float(size.max())
    level = np.floor(np.log2(base / size) / np.log2(args.step)).astype(int)
    level -= level.min()

    pairs = edge_adjacency(cells)
    diff = np.abs(level[pairs[:, 0]] - level[pairs[:, 1]])
    ratio = np.maximum(size[pairs[:, 0]] / size[pairs[:, 1]],
                       size[pairs[:, 1]] / size[pairs[:, 0]])

    print(f"要素 {len(area):,} / 隣接する組 {len(pairs):,}")
    print(f"要素面積 {area.min():.0f} - {area.max():.0f} m^2"
          f" / 代表辺長 {size.min():.1f} - {size.max():.1f} m")
    print(f"level 0 の辺長 {base:.1f} m、1 レベルあたり {args.step} 倍\n")

    print("レベルごとの要素数:")
    for lv, n in zip(*np.unique(level, return_counts=True)):
        lo = base / args.step ** (lv + 1)
        print(f"  level {lv}: {n:>8,} 個  ({100 * n / len(area):5.1f}%)"
              f"  辺長 {lo:6.1f} - {base / args.step ** lv:6.1f} m")

    print("\n隣り合う要素のレベル差:")
    for d, n in zip(*np.unique(diff, return_counts=True)):
        print(f"  差 {d}: {n:>9,} 組  ({100 * n / len(pairs):6.2f}%)")

    over = int((diff > 1).sum())
    print(f"\n差が 2 以上の組: {over:,} 組"
          f"（{100 * over / len(pairs):.3f}%）")
    print(f"隣接要素の辺長比: 中央値 {np.median(ratio):.2f} / "
          f"99 パーセンタイル {np.percentile(ratio, 99):.2f} / 最大 {ratio.max():.2f}")

    # レベルの区切りをどこに置くかで結果は変わる。基準辺長を 1 段ぶん動かして
    # 総当たりし、どの置き方でも差が 1 以内に収まるかを見る。
    worst = 0
    worst_base = base
    for f in np.linspace(1.0, args.step, 64, endpoint=False):
        lv = np.floor(np.log2(base * f / size) / np.log2(args.step)).astype(int)
        d = int(np.abs(lv[pairs[:, 0]] - lv[pairs[:, 1]]).max())
        if d > worst:
            worst, worst_base = d, base * f
    print(f"\n区切り位置を総当たりしたときの最大レベル差: {worst}"
          f"（level 0 の辺長を {worst_base:.1f} m に置いた場合）")
    exceed = int((ratio > args.step).sum())
    print(f"辺長比が {args.step} を超える組: {exceed:,} 組"
          f"（{100 * exceed / len(pairs):.3f}%）"
          f" ← これが 0 なら区切り位置によらず差 1 以内")


if __name__ == "__main__":
    main()
