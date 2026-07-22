#!/usr/bin/env python3
"""Run mkCELL mesh generation per flood block in parallel."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from mkcell_block import area_m2, load_target_blocks


def _replace_project_name(obj, project_name):
    if isinstance(obj, dict):
        return {k: _replace_project_name(v, project_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_project_name(item, project_name) for item in obj]
    if isinstance(obj, str):
        return obj.replace("{project_name}", project_name)
    return obj


def _run_block(
    script_dir: Path,
    yaml_path: Path,
    block_index: int,
    min_area: float,
    log_dir: Path,
) -> tuple[int, float]:
    cmd = [
        sys.executable,
        str(script_dir / "01_mkINPUT_Face_Edge.py"),
        str(yaml_path),
        "--block-index",
        str(block_index),
        "--min-block-area-m2",
        str(min_area),
    ]
    env = os.environ.copy()
    env.update({
        "MKCELL_USE_NATIVE": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "GDAL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    log_path = log_dir / f"block_{block_index:03d}.log"
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"started={datetime.now().isoformat()}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(script_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        elapsed = time.perf_counter() - started
        log.write(f"\nexit={proc.returncode} wall_sec={elapsed:.3f}\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"block {block_index} failed (exit={proc.returncode}); log={log_path}"
        )
    return block_index, elapsed


def _default_workers(requested: int) -> int:
    """Conservative process count for memory-heavy GeoPandas workers."""
    cpu_count = os.cpu_count() or 4
    return max(1, min(requested, max(1, cpu_count - 2), 8))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel flood-block mesh generation")
    parser.add_argument("config_file", type=str)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=None,
        help="optional block indices for targeted profiling",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    yaml_path = Path(args.config_file)
    if not yaml_path.is_absolute():
        yaml_path = script_dir / yaml_path

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_name = config["project"]["name"]
    config = _replace_project_name(config, project_name)
    base_dir = config["project"]["base_dir"]
    if not os.path.isabs(base_dir):
        base_dir = str(script_dir / base_dir)

    parallel = config.get("parallel", {})
    requested_workers = int(parallel.get("workers", os.cpu_count() or 4))
    workers = args.workers if args.workers is not None else _default_workers(requested_workers)
    min_area = float(parallel.get("min_block_area_m2", 100.0))

    target_file = os.path.join(
        base_dir,
        config["input"]["target_area"]["dir"],
        config["input"]["target_area"]["file"],
    )
    blocks, _, valid_indices = load_target_blocks(
        target_file, block_index=None, min_block_area_m2=min_area
    )
    if args.blocks is not None:
        requested_blocks = set(args.blocks)
        invalid = requested_blocks.difference(valid_indices)
        if invalid:
            raise ValueError(f"Invalid block indices: {sorted(invalid)}")
        valid_indices = [idx for idx in valid_indices if idx in requested_blocks]
        blocks = blocks[blocks["block_index"].isin(valid_indices)].copy()
    block_areas = {
        int(row.block_index): area_m2(row.geometry, blocks.crs)
        for row in blocks.itertuples()
    }
    scheduled_indices = sorted(
        valid_indices, key=lambda idx: block_areas.get(idx, 0.0), reverse=True
    )

    project_dir = Path(os.path.dirname(config["output"]["dir"]))
    if not project_dir.is_absolute():
        project_dir = Path(base_dir) / project_dir
    log_dir = project_dir / "block_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"run_blocks: project={project_name}, blocks={len(valid_indices)}, "
        f"workers={workers}, native=off, threads_per_worker=1",
        flush=True,
    )
    print(f"run_blocks: logs={log_dir}", flush=True)

    failures = []
    started = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_block, script_dir, yaml_path, idx, min_area, log_dir): idx
            for idx in scheduled_indices
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _, elapsed = fut.result()
                completed += 1
                print(
                    f"block {idx:03d} done wall={elapsed:.1f}s "
                    f"progress={completed}/{len(valid_indices)} "
                    f"total_elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"block {idx:03d} failed: {exc}", flush=True)
                failures.append(idx)

    if failures:
        print(f"Failed blocks: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
