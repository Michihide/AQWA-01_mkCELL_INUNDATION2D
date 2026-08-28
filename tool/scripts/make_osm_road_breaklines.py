#!/usr/bin/env python3
"""OpenStreetMap の道路から拘束ブレークラインを作る（次善策）。

本来は scripts/detect_embankments.py で DEM から盛り土を検出し、その天端線を
拘束する。しかしそれには 1 m 級の DEM が要る。球磨川のように 5 m DEM しか
無い区域では、確実に盛り土である堤防ですら周囲と区別できず、検出が成立しない。
そこで次善策として OSM の幾何をそのまま使う。

OSM は測量成果ではないので、位置精度はここで担保できない。氾濫の流れを実際に
制御する幹線だけに絞り、生活道路は含めない。球磨川では全道路 1,848 km に対し
幹線は 211 km で、全部を拘束すると面積下限 625 m^2 と両立しないためでもある。

    python scripts/make_osm_road_breaklines.py --config config/kuma_hitoyoshi.yaml \
        --roads <gis_osm_roads_free_1.shp> --out data/kuma_osm_major_roads.gpkg

全道路を含める場合は `--all-classes`（最小長 30 m、橋・トンネル除外は同じ）:

    python scripts/make_osm_road_breaklines.py ... --out data/kuma_osm_all_roads.gpkg --all-classes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.io_vector import load_domain  # noqa: E402
from src.special_edges import pick_clear_indices  # noqa: E402

# 盛土構造を持ちやすく、氾濫流を実際に分断する種別
DEFAULT_CLASSES = ["motorway", "trunk", "primary", "secondary"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--roads", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                    help="fclass で絞り込む（--all-classes 時は無視）")
    ap.add_argument("--all-classes", action="store_true",
                    help="fclass による種別フィルタを行わず全道路を含める")
    ap.add_argument("--min-length", type=float, default=None,
                    help="これより短い線分を除外 [m]（省略時: 幹線 200 / 全道路 30）")
    ap.add_argument("--min-clearance", type=float, default=0.0,
                    help="これより近い線（交差以外）は長い方を残して落とす [m]")
    args = ap.parse_args()

    min_length = args.min_length
    if min_length is None:
        min_length = 30.0 if args.all_classes else 200.0

    cfg = load_config(args.config)
    polys, crs = load_domain(cfg)
    domain = shapely.union_all(np.asarray(polys))

    bbox = tuple(gpd.GeoSeries([domain], crs=crs).to_crs(4326).total_bounds)
    roads = gpd.read_file(args.roads, bbox=bbox).to_crs(crs)
    if not args.all_classes and "fclass" in roads.columns:
        roads = roads[roads["fclass"].isin(args.classes)]
    roads = gpd.clip(roads, domain).explode(index_parts=False)
    roads = roads[roads.geom_type == "LineString"]
    roads = roads[roads.length >= min_length].reset_index(drop=True)

    keep = [c for c in ("fclass", "name", "bridge", "tunnel") if c in roads.columns]
    out = roads[keep + ["geometry"]].copy()
    out["source"] = "osm_road"

    # 橋とトンネルは地表の盛り土ではないので拘束しない
    for col in ("bridge", "tunnel"):
        if col in out.columns:
            out = out[~out[col].astype(str).str.upper().isin(["T", "TRUE", "YES", "1"])]
    out = out.reset_index(drop=True)

    if args.min_clearance > 0.0 and len(out) > 1:
        geoms = [g for g in out.geometry]
        keep_i = pick_clear_indices(geoms, args.min_clearance)
        dropped = len(out) - len(keep_i)
        out = out.iloc[keep_i].reset_index(drop=True)
        print(
            f"近接間引き: 下限 {args.min_clearance:.0f} m、"
            f"{dropped} 本を落とし {len(out)} 本を残した"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.out, layer="roads", driver="GPKG")
    if args.all_classes:
        print("対象種別: 全 fclass")
    else:
        print(f"対象種別: {', '.join(args.classes)}")
    print(f"最小長: {min_length:.0f} m")
    print(f"拘束候補: {len(out)} 本 / {out.length.sum() / 1000:.1f} km"
          f"（解析領域 {domain.area / 1e6:.1f} km^2）")
    print(f"出力: {args.out}")


if __name__ == "__main__":
    main()
