"""既存メッシュ（AQWA の face.gpkg）と本ツールの出力を同じ地形指標で比較する。

同じ DEM・同じ判定式で評価しないと粗密の良し悪しは論じられないため、
双方をポリゴン列に落として compute_polygon_terrain_metrics に通す。

    python scripts/compare_with_existing.py config/kuma_hitoyoshi.yaml
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
from src.io_raster import load_dem  # noqa: E402
from src.io_vector import load_domain  # noqa: E402
from src.terrain_metrics import (  # noqa: E402
    compute_polygon_terrain_metrics,
    summarize_terrain,
    terrain_violations,
)

EXISTING = Path("/Users/test/Desktop/AQWA/01_mkMESH_INUN2DH/Cell")


def polygons_from_gpkg(path: Path, crs: str) -> np.ndarray:
    gdf = gpd.read_file(path).to_crs(crs)
    geoms = gdf.geometry.explode(index_parts=False).values
    return np.asarray([g for g in geoms if g is not None and not g.is_empty])


def evaluate(name: str, geoms: np.ndarray, dem, cfg) -> dict[str, float]:
    area = shapely.area(geoms)
    centroids = shapely.get_coordinates(shapely.centroid(geoms))
    report = compute_polygon_terrain_metrics(geoms, centroids, dem, cfg)
    stats = summarize_terrain(report, cfg, area)
    viol = terrain_violations(report, cfg)
    return {
        "name": name,
        "n": len(geoms),
        "area_mean": float(area.mean()),
        "area_min": float(area.min()),
        "below_floor": int((area < cfg.mesh.min_element_area).sum()),
        "rmse_aw": stats["plane_fit_rmse_area_weighted"],
        "spread_aw": stats["slope_spread_area_weighted"],
        "viol_area": 100.0 * stats["terrain_viol_area_ratio"],
        "viol_rmse": int(viol["plane_fit_rmse"].sum()),
        "viol_spread": int(viol["slope_direction_spread"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--ours", type=Path, nargs="*", default=[])
    ap.add_argument(
        "--existing", nargs="*", default=["Kuma_Hitoyoshi", "Kuma_Hitoyoshi_hex"]
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    polys, crs = load_domain(cfg)
    bounds = shapely.bounds(shapely.union_all(np.asarray(polys)))
    dem = load_dem(cfg, crs, tuple(bounds))

    rows = []
    for tag in args.existing:
        path = EXISTING / tag / "output_gpkg_csv" / "face.gpkg"
        if not path.exists():
            print(f"skip {tag}: {path} がありません")
            continue
        rows.append(evaluate(f"既存 {tag}", polygons_from_gpkg(path, crs), dem, cfg))

    for path in args.ours:
        rows.append(evaluate(f"本ツール {path.parent.name}", polygons_from_gpkg(path, crs), dem, cfg))

    hdr = (
        f"{'メッシュ':<28}{'要素数':>9}{'平均面積':>9}{'最小面積':>9}{'625未満':>8}"
        f"{'RMSE(面積重み)':>15}{'方向(面積重み)':>15}{'違反面積率':>10}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['name']:<28}{r['n']:>9,}{r['area_mean']:>9.0f}{r['area_min']:>9.0f}"
            f"{r['below_floor']:>8,}{r['rmse_aw']:>15.3f}{r['spread_aw']:>15.1f}{r['viol_area']:>9.0f}%"
        )


if __name__ == "__main__":
    main()
