#!/usr/bin/env python3
"""球磨川の左右岸線を、MLIT の定期横断測量データから作る。

各横断面には左岸・右岸の距離杭座標が入っている。これを河川ごとに KP 順で
繋げば、左岸線と右岸線という測量由来の縦断線になる。堤防・河岸に沿うため、
拘束ブレークラインとして使える。OpenStreetMap は使わない。

KP が飛んでいる区間をそのまま繋ぐと、実在しない位置に直線を引いてしまう。
kp_gap_max を超える間隔があれば線を分割する。

    python scripts/make_kuma_breaklines.py --out tests/../data/kuma_riverbanks.gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

CROSS_SECTIONS = Path(
    "/Users/test/Desktop/AQWA/01_mkMESH_RIVER1DH/01_Make_rivdata/output/"
    "Kuma_Hitoyoshi/gpkg_etc/selected_cross_section.gpkg"
)
DOMAIN = Path(
    "/Users/test/Desktop/AQWA/01_mkMESH_INUN2DH/Cell/Kuma_Hitoyoshi/"
    "input_gpkg_tif/input2.gpkg"
)
TARGET_EPSG = 6670

COL_CODE = "河川コード"
COL_NAME = "河川名"
COL_KP = "KP"
BANKS = {
    "left": ("左岸距離杭X座標", "左岸距離杭Y座標"),
    "right": ("右岸距離杭X座標", "右岸距離杭Y座標"),
}


def build_bank_lines(gdf: gpd.GeoDataFrame, kp_gap_max: float) -> gpd.GeoDataFrame:
    records = []
    for code, group in gdf.groupby(COL_CODE):
        group = group.sort_values(COL_KP)
        kp = group[COL_KP].to_numpy(dtype=float)
        # KP が飛んでいるところで切る
        breaks = np.flatnonzero(np.diff(kp) > kp_gap_max) + 1
        segments = np.split(np.arange(len(group)), breaks)

        for side, (cx, cy) in BANKS.items():
            xy = group[[cx, cy]].to_numpy(dtype=float)
            for seg in segments:
                pts = xy[seg]
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) < 3:
                    continue
                records.append({
                    "river_code": str(code),
                    "river_name": str(group[COL_NAME].iloc[0]),
                    "side": side,
                    "kp_min": float(kp[seg].min()),
                    "kp_max": float(kp[seg].max()),
                    "geometry": LineString(pts),
                })
    return gpd.GeoDataFrame(records, geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kp-gap-max", type=float, default=0.5,
                    help="この KP 間隔を超えたら線を分割する [km]")
    ap.add_argument("--clip-to-domain", action="store_true", default=True)
    args = ap.parse_args()

    gdf = gpd.read_file(CROSS_SECTIONS).to_crs(TARGET_EPSG)
    lines = build_bank_lines(gdf, args.kp_gap_max)

    if args.clip_to_domain:
        domain = gpd.read_file(DOMAIN).to_crs(TARGET_EPSG).union_all()
        lines = gpd.clip(lines, domain).explode(index_parts=False)
        lines = lines[lines.geom_type == "LineString"].reset_index(drop=True)
        lines = lines[lines.length > 0.0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines.to_file(args.out, layer="riverbanks", driver="GPKG")

    print(f"横断面 {len(gdf)} 断面 -> 左右岸線 {len(lines)} 本")
    print(f"総延長 {lines.length.sum() / 1000.0:.2f} km")
    print(f"河川 {lines['river_name'].nunique()} 水系 / {lines['river_code'].nunique()} 区間")
    print(f"出力: {args.out}")


if __name__ == "__main__":
    main()
