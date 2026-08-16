#!/usr/bin/env python3
"""Compare face.csv / edge.csv between baseline and candidate block outputs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_csv(
    a: Path,
    b: Path,
    key_cols: list[str],
    rtol: float = 1e-9,
    atol: float = 1e-6,
    ignore_columns: set[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    if not a.is_file():
        return [f"missing: {a}"]
    if not b.is_file():
        return [f"missing: {b}"]

    if file_hash(a) == file_hash(b):
        return issues

    da = pd.read_csv(a)
    db = pd.read_csv(b)
    # Block-mode candidates carry geometry as WKT so merge can build the
    # final GPKG without writing one GPKG per block.
    optional_columns = {"wkt"} | (ignore_columns or set())
    da = da.drop(columns=[c for c in optional_columns if c in da.columns])
    db = db.drop(columns=[c for c in optional_columns if c in db.columns])
    if len(da) != len(db):
        issues.append(f"row count {len(da)} != {len(db)}")
        return issues

    cols = [c for c in da.columns if c in db.columns]
    if cols != list(da.columns) or cols != list(db.columns):
        issues.append(f"column mismatch: {list(da.columns)} vs {list(db.columns)}")

    sort_cols = [c for c in key_cols if c in da.columns]
    if sort_cols:
        da = da.sort_values(sort_cols).reset_index(drop=True)
        db = db.sort_values(sort_cols).reset_index(drop=True)

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(da[c])]
    for c in num_cols:
        diff = (da[c].astype(float) - db[c].astype(float)).abs()
        bad = diff > (atol + rtol * db[c].astype(float).abs())
        if bad.any():
            n = int(bad.sum())
            issues.append(f"{c}: {n} rows differ (max diff={diff.max():.6g})")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression compare for mkCELL block outputs")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--blocks", nargs="*", default=None, help="block names, default: intersection")
    parser.add_argument(
        "--ignore-columns",
        nargs="*",
        default=[],
        help="columns intentionally changed by a correctness fix",
    )
    args = parser.parse_args()

    base_blocks = args.baseline / "blocks"
    cand_blocks = args.candidate / "blocks"
    blocks = args.blocks or sorted(
        {p.name for p in base_blocks.glob("block_*")} & {p.name for p in cand_blocks.glob("block_*")}
    )

    failed = 0
    for block in blocks:
        print(f"=== {block} ===")
        for name, keys in (("face.csv", ["CN"]), ("edge.csv", ["CN", "LN"])):
            issues = compare_csv(
                base_blocks / block / name,
                cand_blocks / block / name,
                keys,
                ignore_columns=set(args.ignore_columns),
            )
            if issues:
                failed += 1
                print(f"  {name}: FAIL")
                for msg in issues:
                    print(f"    - {msg}")
            else:
                print(f"  {name}: OK")
    print(f"summary: {len(blocks)} blocks checked, {failed} file mismatches")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
