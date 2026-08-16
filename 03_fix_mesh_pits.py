#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メッシュグラフ上の人工的な袋状凹地（pit）を緩和する。

三角メッシュの辺が急斜面をまたぐと、DEM 上は浅いのに
「メッシュ隣接だけ見ると深い袋」になり、漫流で水が過剰に溜まる。
min(メッシュ隣 bed) - 自セル bed が閾値を超えるセルについて、
床高 (_median) を溢流口直前まで引き上げる（擬似バーチング）。

使用例:
  bash run_mesh_phase1.sh yaml/Kuma_Hitoyoshi.yaml
  python 03_fix_mesh_pits.py yaml/Kuma_Hitoyoshi.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from mesh_config import read_mesh_config


def read_config(config_file: str) -> dict:
    """後方互換の薄いラッパ。"""
    return read_mesh_config(config_file)


def mesh_neighbors(edge_df: pd.DataFrame) -> dict[int, set[int]]:
    pair_to_faces: dict[tuple[int, int], set[int]] = defaultdict(set)
    face_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in edge_df.itertuples(index=False):
        cn = int(row.CN)
        n1, n2 = sorted((int(row.StartNode), int(row.EndNode)))
        pair_to_faces[(n1, n2)].add(cn)
        face_edges[cn].append((n1, n2))

    out: dict[int, set[int]] = {}
    for cn, pairs in face_edges.items():
        nbs: set[int] = set()
        for pair in pairs:
            nbs.update(f for f in pair_to_faces[pair] if f != cn)
        out[cn] = nbs
    return out


def fix_mesh_pits(
    bed: dict[int, float],
    neighbors: dict[int, set[int]],
    max_spill: float,
    eps: float = 0.01,
    max_iter: int = 50,
) -> tuple[dict[int, float], list[int]]:
    """袋状凹地の床高を溢流口手前まで引き上げる。変更セル CN リストを返す。"""
    bed = dict(bed)
    changed_cns: list[int] = []

    for _ in range(max_iter):
        updated: list[int] = []
        for cn, b in list(bed.items()):
            nbs = neighbors.get(cn)
            if not nbs:
                continue
            nb_beds = [bed[n] for n in nbs if n in bed]
            if not nb_beds:
                continue
            spill = min(nb_beds) - b
            if spill > max_spill:
                new_b = min(nb_beds) - eps
                if new_b > b + 1e-6:
                    bed[cn] = new_b
                    updated.append(cn)
        if not updated:
            break
        changed_cns.extend(updated)

    return bed, sorted(set(changed_cns))


def main() -> None:
    parser = argparse.ArgumentParser(description="メッシュ袋状凹地の床高修正")
    parser.add_argument("config_file", help="yaml/Kuma_Hitoyoshi.yaml 等")
    parser.add_argument(
        "--max-spill",
        type=float,
        default=3.0,
        help="この溢流高差 [m] 超の袋のみ修正（default: 3.0）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="face.csv を書き換えず統計のみ表示",
    )
    args = parser.parse_args()

    paths = read_config(args.config_file)
    face_csv = Path(paths["face_csv"])
    edge_csv = Path(paths["edge_csv"])
    face_gpkg = Path(paths["face_gpkg"])

    df_face = pd.read_csv(face_csv)
    df_edge = pd.read_csv(edge_csv)
    neighbors = mesh_neighbors(df_edge)

    bed = df_face.set_index("CN")["_median"].astype(float).to_dict()
    bed = {int(k): v for k, v in bed.items()}
    new_bed, changed = fix_mesh_pits(bed, neighbors, max_spill=args.max_spill)

    print(f"\n=== メッシュ pit 修正 (max_spill={args.max_spill} m) ===")
    print(f"修正セル数: {len(changed)} / {len(bed)}")

    if changed:
        deltas = [new_bed[c] - bed[c] for c in changed]
        print(f"床高引き上げ量 [m]: min={min(deltas):.2f} max={max(deltas):.2f} mean={sum(deltas)/len(deltas):.2f}")
        print("例 (最大10件):")
        top = sorted(changed, key=lambda c: new_bed[c] - bed[c], reverse=True)[:10]
        for cn in top:
            nbs = neighbors.get(cn, set())
            nb_min = min(bed[n] for n in nbs) if nbs else float("nan")
            print(
                f"  CN{cn}: {bed[cn]:.2f} -> {new_bed[cn]:.2f} m  "
                f"(mesh spill was {nb_min - bed[cn]:.1f} m)"
            )

    if args.dry_run:
        print("\n(dry-run: ファイルは未変更)")
        return

    if not changed:
        print("修正不要。終了。")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = face_csv.with_suffix(f".csv.bak_pitfix_{ts}")
    shutil.copy2(face_csv, backup)
    print(f"\nバックアップ: {backup}")

    df_face["_median"] = df_face["CN"].astype(int).map(new_bed)
    df_face.to_csv(face_csv, index=False, encoding="utf-8")
    print(f"更新: {face_csv}")

    if face_gpkg.is_file():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(face_gpkg)
            gdf["CN"] = gdf["CN"].astype(int)
            gdf["_median"] = gdf["CN"].map(new_bed)
            gpkg_bak = face_gpkg.with_suffix(f".gpkg.bak_pitfix_{ts}")
            shutil.copy2(face_gpkg, gpkg_bak)
            gdf.to_file(face_gpkg, layer="poly", driver="GPKG")
            print(f"更新: {face_gpkg} (backup: {gpkg_bak})")
        except Exception as exc:
            print(f"警告: face.gpkg 更新をスキップ: {exc}")

    print("\n次: python 02_mkCELL.py", args.config_file)
    print("     02_Solver/inun2dh/01_Input_cell/ へ bin を配置して再計算")


if __name__ == "__main__":
    main()
