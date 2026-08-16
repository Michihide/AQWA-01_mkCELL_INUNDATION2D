#!/usr/bin/env python3
"""検出した盛り土がメッシュ辺として再現されているかを図にする。

左は DEM の陰影図に検出線を重ねたもの、右は同じ範囲のメッシュである。
拘束が効いていれば、右で赤い線の上に要素の辺が並ぶ。

    python scripts/plot_embankment_constraints.py --config config/hii.yaml \
        --out output/hii_constraints.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import shapely

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.font_utils import setup_japanese_font  # noqa: E402
from src.io_raster import load_dem  # noqa: E402
from src.io_vector import load_domain  # noqa: E402


def _hillshade(ax, dem, cfg, crs, box):
    minx, miny, maxx, maxy = box.bounds
    d = load_dem(cfg, crs, (minx - 50, miny - 50, maxx + 50, maxy + 50), buffer_m=0.0)
    z = np.nan_to_num(d.array, nan=float(np.nanmedian(d.array)))
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    ax.imshow(
        ls.hillshade(z, vert_exag=3, dx=d.px, dy=d.px), cmap="gray",
        extent=(d.x_min, d.x_min + z.shape[1] * d.px,
                d.y_max - z.shape[0] * d.px, d.y_max),
        origin="upper",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--half-width", type=float, default=350.0)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--dem-px", type=float, default=2.0)
    args = ap.parse_args()

    setup_japanese_font()
    cfg = load_config(args.config)
    cfg.terrain.dem_target_px_m = args.dem_px
    polys, crs = load_domain(cfg)
    domain = shapely.union_all(np.asarray(polys))

    out_dir = cfg.output_dir
    base = cfg.output.basename
    elements = gpd.read_file(out_dir / f"{base}_elements.gpkg")
    constrained = gpd.read_file(out_dir / f"{base}_breaklines.gpkg")
    if constrained.empty:
        raise SystemExit("拘束線がありません")

    # 長い拘束線のまわりを選ぶ
    picks = constrained.assign(L=constrained.length).nlargest(args.rows, "L")

    fig, axes = plt.subplots(args.rows, 2, figsize=(12, 5.8 * args.rows),
                             squeeze=False)
    hw = args.half_width
    for row, (_, line) in enumerate(picks.iterrows()):
        c = line.geometry.interpolate(0.5, normalized=True)
        box = shapely.box(c.x - hw, c.y - hw, c.x + hw, c.y + hw)

        left, right = axes[row]
        _hillshade(left, None, cfg, crs, box)
        gpd.clip(constrained, box).plot(ax=left, color="#dc2626", linewidth=2.2)
        left.set_title("DEM 陰影図と検出した盛り土天端")

        sub = elements[elements.intersects(box)]
        sub.plot(ax=right, facecolor="none", edgecolor="#9ca3af", linewidth=0.5)
        sub[sub["kind"] == "quad"].plot(
            ax=right, facecolor="#fde68a", edgecolor="#a16207",
            linewidth=0.5, alpha=0.55,
        )
        gpd.GeoSeries([domain.boundary], crs=crs).clip(box).plot(
            ax=right, color="#16a34a", linewidth=1.6
        )
        gpd.clip(constrained, box).plot(ax=right, color="#dc2626", linewidth=2.2)
        right.set_title("生成したメッシュ（黄: 四角形 / 緑: 領域界）")

        for ax in (left, right):
            ax.set_xlim(c.x - hw, c.x + hw)
            ax.set_ylim(c.y - hw, c.y + hw)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"内部の道路・盛り土をメッシュ境界として表現（{base}）\n"
        f"赤: DEM から検出した天端線。メッシュ辺がこれに一致している",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"出力: {args.out}")


if __name__ == "__main__":
    main()
