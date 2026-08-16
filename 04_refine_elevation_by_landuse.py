#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土地利用に応じてセル代表標高 (_median) を再計算する（第1段階・地形表現改善）。

森林・荒地など谷筋を median が見落としやすい土地利用について、
DEM 上の低パーセンタイル (p10 等) を代表標高に採用する。
ソルバー本体は変更せず face.csv / face.gpkg のみ更新。

推奨パイプライン:
  python 04_refine_elevation_by_landuse.py yaml/Kuma_Hitoyoshi.yaml
  python 03_fix_mesh_pits.py yaml/Kuma_Hitoyoshi.yaml
  python 02_mkCELL.py yaml/Kuma_Hitoyoshi.yaml

使用例:
  python 04_refine_elevation_by_landuse.py yaml/Kuma_Hitoyoshi.yaml --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

from mesh_config import read_mesh_config

LANDUSE_CODES = [
    "10", "20", "50", "60", "70", "91", "92", "100", "110", "140", "150", "160", "255",
]


def dominant_landuse(row: pd.Series, codes: list[str]) -> tuple[int | None, float]:
    """面積最大の土地利用コードとその面積比 (bill_area 基準) を返す。"""
    areas = {c: float(row.get(c, 0.0) or 0.0) for c in codes}
    total = sum(areas.values())
    if total <= 0:
        return None, 0.0
    code = max(areas, key=areas.get)
    return int(code), areas[code] / total


def dem_stat_for_geom(src, geom, nodata: float, stat: str) -> float | None:
    """ポリゴン内 DEM 画素から指定統計量 [m] を返す。"""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    try:
        out_img, _ = rio_mask(src, [mapping(geom)], crop=True, nodata=nodata, filled=True)
    except ValueError:
        return None
    band = out_img[0]
    if np.issubdtype(band.dtype, np.floating):
        valid = np.isfinite(band) & (band != nodata)
    else:
        valid = band != nodata
    vals = band[valid].astype(np.float64)
    if vals.size == 0:
        return None
    if stat == "min":
        return float(vals.min())
    if stat == "p10":
        return float(np.percentile(vals, 10))
    if stat == "mean":
        return float(vals.mean())
    if stat == "median":
        return float(np.median(vals))
    raise ValueError(f"未対応 stat: {stat}")


def refine_elevations(
    df_face: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    dem_tif: str,
    low_codes: set[int],
    low_stat: str,
    min_area_ratio: float,
    dry_run: bool,
) -> tuple[dict[int, float], list[int]]:
    """対象セルの _median を再計算。新 bed dict と変更 CN リストを返す。"""
    cn_to_idx = {int(cn): i for i, cn in enumerate(gdf["CN"].astype(int))}
    bed = df_face.set_index("CN")["_median"].astype(float).to_dict()
    bed = {int(k): float(v) for k, v in bed.items()}
    changed: list[int] = []

    with rasterio.open(dem_tif) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        raster_crs = src.crs
        gdf_dem = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf

        for _, row in df_face.iterrows():
            cn = int(row["CN"])
            dom_code, ratio = dominant_landuse(row, LANDUSE_CODES)
            if dom_code is None or dom_code not in low_codes:
                continue
            if ratio < min_area_ratio:
                continue

            idx = cn_to_idx.get(cn)
            if idx is None:
                continue
            geom = gdf_dem.geometry.iloc[idx]
            new_z = dem_stat_for_geom(src, geom, nodata, low_stat)
            if new_z is None:
                continue
            old_z = bed[cn]
            if new_z < old_z - 1e-6:
                bed[cn] = new_z
                changed.append(cn)

    return bed, sorted(set(changed))


def main() -> None:
    parser = argparse.ArgumentParser(description="土地利用別標高 (_median) 再計算")
    parser.add_argument("config_file", help="yaml/Kuma_Hitoyoshi.yaml 等")
    parser.add_argument("--dry-run", action="store_true", help="統計のみ、ファイル未変更")
    args = parser.parse_args()

    cfg = read_mesh_config(args.config_file)
    lu_cfg = cfg["elevation_stat_by_landuse"]
    if not lu_cfg.get("enabled", False):
        print("elevation_stat_by_landuse.enabled が false のためスキップ")
        return

    low_codes = {int(c) for c in lu_cfg.get("low_codes", [50, 60])}
    low_stat = str(lu_cfg.get("low_stat", "p10")).lower()
    min_area_ratio = float(lu_cfg.get("min_area_ratio", 0.5))
    if low_stat not in ("min", "p10", "mean", "median"):
        raise SystemExit(f"low_stat '{low_stat}' は min/p10/mean/median のいずれか")

    face_csv = Path(cfg["face_csv"])
    face_gpkg = Path(cfg["face_gpkg"])
    dem_tif = Path(cfg["dem_tif"])

    if not face_csv.is_file():
        raise SystemExit(f"face.csv がありません: {face_csv}")
    if not dem_tif.is_file():
        raise SystemExit(f"DEM がありません: {dem_tif}")

    df_face = pd.read_csv(face_csv)
    gdf = gpd.read_file(face_gpkg)
    gdf["CN"] = gdf["CN"].astype(int)

    old_bed = df_face.set_index("CN")["_median"].astype(float).to_dict()
    new_bed, changed = refine_elevations(
        df_face, gdf, str(dem_tif), low_codes, low_stat, min_area_ratio, args.dry_run,
    )

    print(f"\n=== 土地利用別標高再計算 (low_codes={sorted(low_codes)}, stat={low_stat}) ===")
    print(f"対象条件: 優占面積比 >= {min_area_ratio:.0%}")
    print(f"引き下げセル数: {len(changed)} / {len(old_bed)}")

    if changed:
        deltas = [new_bed[c] - old_bed[c] for c in changed]
        print(f"標高変化 [m]: min={min(deltas):.2f} max={max(deltas):.2f} mean={sum(deltas)/len(deltas):.2f}")
        print("例 (最大10件):")
        top = sorted(changed, key=lambda c: old_bed[c] - new_bed[c], reverse=True)[:10]
        for cn in top:
            row = df_face.loc[df_face["CN"] == cn].iloc[0]
            dom, ratio = dominant_landuse(row, LANDUSE_CODES)
            print(
                f"  CN{cn}: {old_bed[cn]:.2f} -> {new_bed[cn]:.2f} m  "
                f"(LU={dom}, ratio={ratio:.0%})"
            )

    if args.dry_run:
        print("\n(dry-run: ファイルは未変更)")
        return

    if not changed:
        print("変更なし。終了。")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = face_csv.with_suffix(f".csv.bak_lustat_{ts}")
    shutil.copy2(face_csv, backup)
    print(f"\nバックアップ: {backup}")

    df_face["_median"] = df_face["CN"].astype(int).map(new_bed)
    df_face.to_csv(face_csv, index=False, encoding="utf-8")
    print(f"更新: {face_csv}")

    gdf["_median"] = gdf["CN"].map(new_bed)
    gpkg_bak = face_gpkg.with_suffix(f".gpkg.bak_lustat_{ts}")
    shutil.copy2(face_gpkg, gpkg_bak)
    gdf.to_file(face_gpkg, layer="poly", driver="GPKG")
    print(f"更新: {face_gpkg} (backup: {gpkg_bak})")

    print("\n次: python 03_fix_mesh_pits.py", args.config_file)
    print("     python 02_mkCELL.py", args.config_file)


if __name__ == "__main__":
    main()
