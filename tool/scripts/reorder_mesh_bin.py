#!/usr/bin/env python3
"""mesh.bin の面・辺・頂点番号を詰め、降雨 CSV の列も同じ順に並べ替える。

元ファイルは変更しない。面の対応は出力 bin の隣に
`<名前>.face_perm.npy`（新しい面番号 → 古い面番号、0 始まり）として残る。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_cache_order import permute_rain_csv, reorder_mesh_bin  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="詰め直す前の mesh.bin")
    parser.add_argument("dst", type=Path, help="詰め直した mesh.bin の出力先")
    parser.add_argument(
        "--rain",
        nargs=2,
        metavar=("IN", "OUT"),
        type=Path,
        help="降雨 CSV を面の新しい順に並べ替えて OUT に書く",
    )
    args = parser.parse_args()
    perm = reorder_mesh_bin(args.src, args.dst)
    print(
        f"内部辺 |L-R| 平均 {perm.span_before:.1f} -> {perm.span_after:.1f}"
        f"（{perm.span_before / perm.span_after:.1f} 倍）"
        if perm.span_after > 0
        else f"内部辺 |L-R| 平均 {perm.span_before:.1f} -> {perm.span_after:.1f}"
    )
    print(
        f"辺走査 |ΔL| 平均 {perm.step_before:.1f} -> {perm.step_after:.1f}"
    )
    print(f"wrote {args.dst}")
    if args.rain is not None:
        rain_in, rain_out = args.rain
        permute_rain_csv(rain_in, rain_out, perm.face_new_to_old)
        print(f"wrote {rain_out}")


if __name__ == "__main__":
    main()
