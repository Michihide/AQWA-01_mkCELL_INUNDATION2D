#!/usr/bin/env python3
"""DEM から盛り土（道路盛土・堤防）の天端線を検出して GeoPackage に書き出す。

検出そのものは src/embankment_detection.py にある。ここは入出力だけを担う。

    python scripts/detect_embankments.py --config config/hii.yaml \
        --out data/hii_embankments.gpkg --min-relief 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.embankment_detection import (  # noqa: E402
    DetectionParams,
    count_blobs,
    detect_embankments,
)
from src.io_raster import load_dem  # noqa: E402
from src.io_vector import load_domain  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-relief", type=float, default=1.0,
                    help="周囲との比高の下限 [m]")
    ap.add_argument("--max-relief", type=float, default=15.0,
                    help="比高の上限 [m]。これを超えるものは構造物ではなく地形")
    ap.add_argument("--width", type=float, default=60.0,
                    help="構造要素の幅 [m]。これより幅の広い高まりは盛り土とみなさない")
    ap.add_argument("--min-length", type=float, default=100.0,
                    help="採用する天端線の長さの下限 [m]")
    ap.add_argument("--min-area", type=float, default=400.0,
                    help="盛り土領域の面積の下限 [m^2]")
    ap.add_argument("--max-sinuosity", type=float, default=2.0,
                    help="全長 / 両端間距離 の上限。丘の頂部を弾く")
    ap.add_argument("--flat-relief", type=float, default=10.0,
                    help="盛り土を探す範囲の、素の地形の起伏の上限 [m]")
    ap.add_argument("--flat-scale", type=float, default=200.0,
                    help="上の起伏を測る窓の大きさ [m]")
    ap.add_argument("--simplify", type=float, default=3.0)
    ap.add_argument("--dem-px", type=float, default=2.0)
    ap.add_argument("--label-with", type=Path, default=None,
                    help="検出線に道路との距離を注記するためだけの参照レイヤ。"
                         "線の形状には一切使わない")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.output_dir)
    cfg.terrain.dem_target_px_m = args.dem_px

    polys, crs = load_domain(cfg)
    domain = shapely.union_all(np.asarray(polys))
    dem = load_dem(cfg, crs, tuple(shapely.bounds(domain)), buffer_m=200.0)
    if dem.px > 5.5:
        print(f"警告: DEM 解像度 {dem.px:.1f} m は粗すぎる")
    elif dem.px > 2.5:
        # 球磨川の 5 m でも実用になるが、平滑化で比高が小さく出る
        # （中央値 1.75 m。斐伊川の 1 m DEM では 3.10 m）
        print(f"注意: DEM 解像度 {dem.px:.1f} m。比高が小さく出るため検出は控えめになる")

    inside = rasterize(
        [(domain, 1)], out_shape=dem.array.shape, fill=0, all_touched=True,
        dtype="uint8",
        transform=from_origin(dem.x_min, dem.y_max, dem.px, dem.px),
    ).astype(bool)

    params = DetectionParams(
        min_relief=args.min_relief, max_relief=args.max_relief, width=args.width,
        min_area=args.min_area, min_length=args.min_length,
        max_sinuosity=args.max_sinuosity, flat_relief=args.flat_relief,
        flat_scale=args.flat_scale, simplify=args.simplify,
    )
    lines, attrs, mask = detect_embankments(
        dem.array.astype(np.float32), dem.px, dem.x_min, dem.y_max, inside, params
    )
    print(f"盛り土候補: {mask.sum() * dem.px ** 2 / 1e6:.3f} km^2 / {count_blobs(mask)} 塊"
          f"（解析領域 {domain.area / 1e6:.2f} km^2 の"
          f"{100 * mask.sum() * dem.px ** 2 / domain.area:.1f}%）")

    out = gpd.GeoDataFrame(attrs, geometry=lines, crs=crs)
    if len(out):
        out = gpd.clip(out, domain).explode(index_parts=False)
        out = out[(out.geom_type == "LineString") & (out.length >= args.min_length)]
        out = out.reset_index(drop=True)
    if args.label_with is not None and len(out):
        out["near_road_m"] = _distance_to(args.label_with, out, domain, crs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.out, layer="embankments", driver="GPKG")

    print(f"天端線: {len(out)} 本 / {out.length.sum() / 1000:.2f} km")
    if len(out):
        print(f"比高 中央値 {out['relief_median'].median():.2f} m / "
              f"最大 {out['relief_max'].max():.2f} m")
        if "near_road_m" in out:
            near = out["near_road_m"] < 15.0
            print(f"うち参照レイヤから 15 m 以内: {near.sum()} 本"
                  f" / {out.loc[near].length.sum() / 1000:.2f} km（注記のみ）")
    print(f"出力: {args.out}")


def _distance_to(path: Path, lines: gpd.GeoDataFrame, domain, crs) -> np.ndarray:
    bbox = tuple(gpd.GeoSeries([domain], crs=crs).to_crs(4326).total_bounds)
    ref = gpd.read_file(path, bbox=bbox).to_crs(crs)
    ref = ref[ref.geom_type.isin(["LineString", "MultiLineString"])]
    if ref.empty:
        return np.full(len(lines), np.inf)
    tree = shapely.STRtree(ref.geometry.values)
    mids = shapely.line_interpolate_point(lines.geometry.values, 0.5, normalized=True)
    return shapely.distance(mids, ref.geometry.values[tree.nearest(mids)])


if __name__ == "__main__":
    main()
