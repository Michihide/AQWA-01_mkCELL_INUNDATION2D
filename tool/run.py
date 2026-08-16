#!/usr/bin/env python3
"""地形適合型ハイブリッド非構造格子生成ツールのエントリポイント。

    python run.py --config config/kuma_hitoyoshi.yaml
    python run.py --config config/hii.yaml --export-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.main import export_solver_outputs, main  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.block_runner import run_blocks_parallel  # noqa: E402
from src.run_summary import quality_comment_lines  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", default="config/kuma_hitoyoshi.yaml", help="設定 YAML のパス",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="既存 .msh から face.gpkg / edge.gpkg / cell.bin のみ出力する",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="--export-only 時に読み込む .msh（省略時は output 配下の basename.msh）",
    )
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="氾濫ブロック単位で並列生成（parallel.enabled 相当）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="--blocks 時の並列プロセス数",
    )
    parser.add_argument(
        "--block-indices",
        type=int,
        nargs="*",
        default=None,
        help="指定したブロック index のみ実行",
    )
    return parser.parse_args(argv)


def _print_summary(summary: dict, cfg=None) -> None:
    print()
    if "face_gpkg" in summary:
        print(f"face.gpkg     : {summary['face_gpkg']}")
        print(f"edge.gpkg     : {summary['edge_gpkg']}")
        if "cell_bin" in summary:
            print(f"cell.bin      : {summary['cell_bin']}")
        if "n_blocks" in summary:
            print(f"ブロック数    : {summary['n_blocks']} (workers={summary.get('workers')})")
    elif "n_elements" not in summary or cfg is None:
        if "n_elements" in summary:
            print(f"要素数        : {summary['n_elements']:,}"
                  f" (三角 {summary.get('n_triangles', 0):,}"
                  f" / 四角 {summary.get('n_quads', 0):,})")
        return

    if cfg is not None and "n_elements" in summary:
        for line in quality_comment_lines(summary, cfg):
            print(line)
        return

    # cfg なしのフォールバック（旧形式）
    print(f"要素数        : {summary['n_elements']:,}"
          f" (三角 {summary['n_triangles']:,} / 四角 {summary['n_quads']:,})")
    print(f"面積          : {summary['area_total_km2']:.3f} km^2")
    print(f"要素面積      : {summary['area_min']:.0f} - {summary['area_max']:.0f} m^2"
          f" (中央値 {summary['area_median']:.0f})")
    print(f"面積下限違反  : {summary['viol_area_below_min']}")
    print(f"品質違反      : {summary['viol_any']}")
    if "terrain_viol_any" in summary:
        print(f"地形基準違反  : {summary['terrain_viol_any']}")
    print(f"反復回数      : {summary['n_iterations']}")


if __name__ == "__main__":
    args = parse_args()
    cfg = None
    try:
        if args.export_only:
            cfg = load_config(args.config)
            setup_logging(cfg.output_dir)
            summary = export_solver_outputs(cfg, args.mesh)
        elif args.blocks:
            cfg = load_config(args.config)
            setup_logging(cfg.output_dir)
            summary = run_blocks_parallel(
                cfg, Path(args.config),
                workers=args.workers,
                block_indices=args.block_indices,
            )
        else:
            cfg = load_config(args.config)
            setup_logging(cfg.output_dir)
            summary = main(args.config)
    except ConfigError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    _print_summary(summary, cfg)
