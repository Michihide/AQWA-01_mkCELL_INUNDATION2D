#!/usr/bin/env python3
"""Copy completed Kuma_Hitoyoshi block outputs into a regression snapshot."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

NEED = ("face.csv", "face.gpkg", "edge.csv", "edge.gpkg")


def snapshot_blocks(src_blocks: Path, dst_blocks: Path) -> tuple[list[str], list[str], list[dict]]:
    complete: list[str] = []
    incomplete: list[str] = []
    in_progress: list[dict] = []
    dst_blocks.mkdir(parents=True, exist_ok=True)

    for block_dir in sorted(src_blocks.glob("block_*")):
        files = [f for f in NEED if (block_dir / f).exists()]
        if len(files) == len(NEED):
            dst = dst_blocks / block_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(block_dir, dst)
            complete.append(block_dir.name)
        elif files:
            in_progress.append({"block": block_dir.name, "files": files})
        else:
            incomplete.append(block_dir.name)
    return complete, incomplete, in_progress


def main() -> int:
    parser = argparse.ArgumentParser(description="Update regression snapshot with completed blocks")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("_regression_snapshots/Kuma_Hitoyoshi_python_baseline_20260715_1621"),
        help="Snapshot root (relative to 01_mkMESH_INUN2DH)",
    )
    parser.add_argument("--project", default="Kuma_Hitoyoshi")
    parser.add_argument("--wait", action="store_true", help="Poll until 31 blocks complete or timeout")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    snapshot = args.snapshot_dir if args.snapshot_dir.is_absolute() else root / args.snapshot_dir
    src_blocks = root / "Cell" / args.project / "blocks"
    dst_blocks = snapshot / "blocks"
    snapshot.mkdir(parents=True, exist_ok=True)

    import time

    t0 = time.time()
    while True:
        complete, incomplete, in_progress = snapshot_blocks(src_blocks, dst_blocks)
        meta = {
            "project": args.project,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "blocks_total": 31,
            "blocks_complete": len(complete),
            "blocks_complete_list": complete,
            "blocks_in_progress": in_progress,
            "blocks_incomplete": incomplete,
        }
        (snapshot / "snapshot_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        log_src = Path("/tmp/kuma_hitoyoshi_mkcell.log")
        if log_src.is_file():
            shutil.copy2(log_src, snapshot / "logs" / "run.log")

        merged = root / "Cell" / args.project / "output_gpkg_csv"
        if merged.is_dir():
            dst_merged = snapshot / "merged_output"
            if dst_merged.exists():
                shutil.rmtree(dst_merged)
            shutil.copytree(merged, dst_merged)

        bin_src = root / "Cell" / args.project / f"{args.project}.bin"
        if bin_src.is_file():
            shutil.copy2(bin_src, snapshot / f"{args.project}.bin")

        print(json.dumps(meta, indent=2, ensure_ascii=False))
        if not args.wait or len(complete) >= 31 or time.time() - t0 > args.timeout_sec:
            break
        time.sleep(30)

    return 0 if len(complete) >= 31 else 1


if __name__ == "__main__":
    raise SystemExit(main())
