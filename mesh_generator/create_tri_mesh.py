#!/usr/bin/env python
"""地形適合 三角形メッシュ 生成 CLI。

QGIS/GIS データ（解析領域ポリゴン・DEM・建物など）から、氾濫解析向けの
三角形メッシュを生成する。scipy.spatial.Delaunay を用いた実装。

使い方:
    python create_tri_mesh.py --config configs/sample_config.yaml
    python create_tri_mesh.py --area area.gpkg --dem dem.tif --out output_mesh

決定性: 乱数を使用しないため、同一入力・同一環境では常に同一の出力となる。
"""

from __future__ import annotations

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# meshlib をインポート（スクリプト直接実行・パッケージ両対応）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from meshlib import config as config_mod
from meshlib import crs as crs_mod
from meshlib import io as io_mod
from meshlib import points as points_mod
from meshlib import buildings as buildings_mod
from meshlib import elevation as elevation_mod
from meshlib import triangulation as tri_mod
from meshlib import quality as quality_mod
from meshlib import adjacency as adj_mod
from meshlib import plotting as plot_mod


def parse_args():
    p = argparse.ArgumentParser(
        description="地形適合 三角形メッシュ 生成プログラム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="設定ファイル（YAML/JSON）")
    p.add_argument("--area", type=str, default=None, help="解析対象領域ポリゴン")
    p.add_argument("--dem", type=str, default=None, help="DEM ラスタ（GeoTIFF）")
    p.add_argument("--buildings", type=str, default=None, help="建物ポリゴン（任意）")
    p.add_argument("--rivers", type=str, default=None, help="河川ライン（任意）")
    p.add_argument("--roads", type=str, default=None, help="道路ライン（任意）")
    p.add_argument("--facilities", type=str, default=None, help="重要施設ポイント（任意）")
    p.add_argument("--out", type=str, default=None, help="出力ディレクトリ")
    p.add_argument("--min-distance", type=float, default=None, help="最小点間距離 [m]")
    p.add_argument("--interior-spacing", type=float, default=None, help="内部点間隔 [m]")
    p.add_argument("--building-mode", type=str, default=None,
                   choices=["building_as_roughness", "building_as_hole"],
                   help="建物の扱い")
    p.add_argument("--no-refine", action="store_true", help="標高ベースの反復細分化を無効化")
    return p.parse_args()


def build_overrides(args) -> dict:
    """CLI 引数を設定上書き dict に変換する。"""
    ov: dict = {"input": {}, "output": {}, "mesh": {}, "elevation": {}}
    if args.area:
        ov["input"]["analysis_area"] = args.area
    if args.dem:
        ov["input"]["dem"] = args.dem
    if args.buildings:
        ov["input"]["buildings"] = args.buildings
    if args.rivers:
        ov["input"]["rivers"] = args.rivers
    if args.roads:
        ov["input"]["roads"] = args.roads
    if args.facilities:
        ov["input"]["facilities"] = args.facilities
    if args.out:
        ov["output"]["directory"] = args.out
    if args.min_distance is not None:
        ov["mesh"]["min_distance"] = args.min_distance
    if args.interior_spacing is not None:
        ov["mesh"]["interior_spacing"] = args.interior_spacing
    if args.building_mode:
        ov["mesh"]["building_mode"] = args.building_mode
    if args.no_refine:
        ov["elevation"]["slope_refine"] = False
        ov["elevation"]["max_refine_iter"] = 0
    # 空 dict は除去
    return {k: v for k, v in ov.items() if v}


def generate_base_points(cfg, area_geom, buildings_gdf, rivers_gdf, roads_gdf, facilities_gdf, dem_path, target_epsg):
    """初期点群（境界・建物・河川道路・施設・内部・勾配）を生成して統合・間引く。"""
    mesh = cfg["mesh"]
    elev = cfg["elevation"]
    crs = f"EPSG:{target_epsg}"
    gdfs = []

    # 1. 解析領域境界点
    bnd = points_mod.densify_polygon_boundary(area_geom, mesh["boundary_spacing"])
    gdfs.append(points_mod.points_to_gdf(bnd, "boundary", crs))
    print(f"  解析境界点: {len(bnd)}")

    # 2-3. 建物境界点・建物バッファ点
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        bb = buildings_mod.building_boundary_points(buildings_gdf, mesh["building_boundary_spacing"])
        gdfs.append(points_mod.points_to_gdf(bb, "building_boundary", crs))
        bbuf = buildings_mod.building_buffer_points(
            buildings_gdf, mesh["building_buffer"], mesh["building_buffer_spacing"]
        )
        gdfs.append(points_mod.points_to_gdf(bbuf, "building_buffer", crs))
        print(f"  建物境界点: {len(bb)}, 建物バッファ点: {len(bbuf)}")

    # 6. 河川・道路点
    if rivers_gdf is not None and len(rivers_gdf) > 0:
        rp = points_mod.densify_lines_gdf(rivers_gdf, mesh["river_spacing"])
        gdfs.append(points_mod.points_to_gdf(rp, "river", crs))
        print(f"  河川点: {len(rp)}")
    if roads_gdf is not None and len(roads_gdf) > 0:
        rdp = points_mod.densify_lines_gdf(roads_gdf, mesh["road_spacing"])
        gdfs.append(points_mod.points_to_gdf(rdp, "road", crs))
        print(f"  道路点: {len(rdp)}")

    # 7. 重要施設周辺点
    if facilities_gdf is not None and len(facilities_gdf) > 0:
        fac_pts = []
        for geom in facilities_gdf.geometry:
            base = geom.centroid if geom.geom_type != "Point" else geom
            buf = base.buffer(mesh["facility_buffer"])
            fac_pts.extend(points_mod.densify_polygon_boundary(buf, mesh["facility_spacing"]))
            fac_pts.append(base)
        gdfs.append(points_mod.points_to_gdf(fac_pts, "facility", crs))
        print(f"  施設周辺点: {len(fac_pts)}")

    # 4. 内部規則点（穴モードでは建物内部を除外）
    holes = None
    if cfg["mesh"]["building_mode"] == "building_as_hole":
        holes = buildings_mod.get_building_holes(buildings_gdf)
    interior = points_mod.generate_regular_points_within_polygon(
        area_geom, mesh["interior_spacing"], holes=holes
    )
    gdfs.append(points_mod.points_to_gdf(interior, "interior", crs))
    print(f"  内部規則点: {len(interior)}")

    # 5. DEM 勾配大領域の追加点
    if elev.get("slope_refine", True):
        slope_pts = elevation_mod.generate_slope_points(
            area_geom, dem_path, elev["slope_point_spacing"], elev["slope_threshold"], crs
        )
        gdfs.append(points_mod.points_to_gdf(slope_pts, "slope", crs))
        print(f"  勾配追加点: {len(slope_pts)}")

    merged = points_mod.merge_point_gdfs(gdfs, crs)
    print(f"  統合点数: {len(merged)}")
    thinned = points_mod.thin_points(merged, mesh["min_distance"])
    print(f"  間引き後: {len(thinned)}")
    return thinned


def attach_all_attributes(tris, coords, cfg, buildings_gdf, dem_path):
    """三角形に品質・標高・建物属性・refine_flag を付与する。"""
    tris = quality_mod.compute_quality(tris, coords)
    tris = quality_mod.flag_quality(
        tris,
        cfg["quality"]["min_edge_length"],
        cfg["quality"]["min_area"],
        cfg["quality"]["min_angle"],
        cfg["quality"]["max_aspect_ratio"],
    )
    tris = elevation_mod.compute_elevation_stats(
        tris, dem_path, all_touched=cfg["elevation"].get("all_touched", False)
    )
    tris = buildings_mod.compute_building_attributes(tris, buildings_gdf)

    refine_mask = elevation_mod.find_refine_triangles(
        tris,
        cfg["elevation"]["z_range_threshold"],
        cfg["elevation"]["z_std_threshold"],
        cfg["elevation"]["edge_max_for_refine"],
        cfg["elevation"]["z_range_for_long_edge"],
    )
    tris["refine_flag"] = refine_mask.astype(int)
    return tris


def load_aux_clipped(path, area_gdf_target, target_epsg, margin=0.0):
    """補助レイヤを解析領域の範囲（bbox）で絞って読み込み、領域にクリップする。

    巨大な広域データ（OSM 等）対策として、まずファイル自身の CRS の bbox で
    フィーチャを絞り込んでから読み込み、対象 EPSG に変換して領域でクリップする。
    """
    if io_mod._is_none_path(path) or not os.path.exists(str(path)):
        if path and not os.path.exists(str(path)) and not io_mod._is_none_path(path):
            print(f"  警告: ファイルが見つかりません（スキップ）: {path}")
        return None

    area_geom_t = area_gdf_target.geometry.union_all()
    if margin > 0:
        area_geom_t = area_geom_t.buffer(margin)

    # ファイル CRS の bbox を計算
    file_crs = io_mod.get_file_crs(str(path))
    try:
        area_in_file = gpd.GeoSeries([area_geom_t], crs=f"EPSG:{target_epsg}")
        if file_crs is not None:
            area_in_file = area_in_file.to_crs(file_crs)
        bbox = tuple(area_in_file.total_bounds)
    except Exception:
        bbox = None

    gdf = io_mod.read_vector(str(path), bbox=bbox)
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = crs_mod.ensure_crs(gdf, target_epsg)
    # 領域（マージン込み）にクリップ
    try:
        clip_poly = gpd.GeoDataFrame(geometry=[area_geom_t], crs=f"EPSG:{target_epsg}")
        gdf = gpd.clip(gdf, clip_poly)
        gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].reset_index(drop=True)
    except Exception:
        pass
    return gdf


def main():
    args = parse_args()
    overrides = build_overrides(args)
    cfg = config_mod.load_config(args.config, overrides)
    config_mod.validate_config(cfg)

    out_dir = cfg["output"]["directory"]
    io_mod.ensure_dir(out_dir)
    gis_dir = io_mod.ensure_dir(os.path.join(out_dir, "gis"))
    csv_dir = io_mod.ensure_dir(os.path.join(out_dir, "csv"))
    png_dir = io_mod.ensure_dir(os.path.join(out_dir, "preview"))

    print("=" * 60)
    print("三角形メッシュ生成 開始")
    print("=" * 60)

    # --- 入力読み込み ---
    print("[1] 入力データ読み込み")
    area_gdf = io_mod.read_vector(cfg["input"]["analysis_area"])
    if area_gdf is None or len(area_gdf) == 0:
        print("エラー: 解析領域ポリゴンが読み込めません。")
        sys.exit(1)
    dem_path = cfg["input"]["dem"]
    if not os.path.exists(dem_path):
        print(f"エラー: DEM が見つかりません: {dem_path}")
        sys.exit(1)

    # --- CRS 決定・変換（解析領域） ---
    print("[2] 座標系の決定と変換")
    target_epsg = crs_mod.resolve_target_epsg(area_gdf, cfg["crs"])
    print(f"  対象 EPSG: {target_epsg}")
    area_gdf = crs_mod.ensure_crs(area_gdf, target_epsg)
    area_geom = area_gdf.geometry.union_all()

    # --- 補助レイヤを領域 bbox で絞り込み読み込み（巨大データ対策） ---
    build_margin = cfg["mesh"].get("building_buffer", 10.0)
    buildings_gdf = load_aux_clipped(cfg["input"]["buildings"], area_gdf, target_epsg, margin=build_margin)
    rivers_gdf = load_aux_clipped(cfg["input"]["rivers"], area_gdf, target_epsg)
    roads_gdf = load_aux_clipped(cfg["input"]["roads"], area_gdf, target_epsg)
    facilities_gdf = load_aux_clipped(cfg["input"]["facilities"], area_gdf, target_epsg)
    for name, g in [("建物", buildings_gdf), ("河川", rivers_gdf), ("道路", roads_gdf), ("施設", facilities_gdf)]:
        if g is not None:
            print(f"  {name}: {len(g)} 件（領域内）")

    # --- 初期点群生成 ---
    print("[3] 点群生成・統合・間引き")
    pts_gdf = generate_base_points(
        cfg, area_geom, buildings_gdf, rivers_gdf, roads_gdf,
        facilities_gdf, dem_path, target_epsg
    )

    # --- 初期三角形分割 ---
    print("[4] Delaunay 三角形分割")
    mode = cfg["mesh"]["building_mode"]
    tris, coords = tri_mod.create_delaunay_triangles(
        pts_gdf, area_geom, buildings_gdf, mode=mode
    )
    print(f"  三角形数（境界内）: {len(tris)}")
    if len(tris) == 0:
        print("エラー: 三角形が生成されませんでした。spacing を見直してください。")
        sys.exit(1)

    # --- 属性付与 ---
    print("[5] 標高・建物・品質属性の付与")
    tris = quality_mod.compute_quality(tris, coords)
    tris = buildings_mod.apply_building_mode(tris, mode)
    tris = attach_all_attributes(tris, coords, cfg, buildings_gdf, dem_path)

    # --- 反復細分化 ---
    max_iter = int(cfg["elevation"].get("max_refine_iter", 0))
    if max_iter > 0:
        print("[6] 標高分布に基づく反復細分化")
        pts_gdf = refine_loop(
            cfg, pts_gdf, tris, coords, area_geom, buildings_gdf, dem_path, mode, target_epsg
        )
        # 最終メッシュ再構築
        tris, coords = tri_mod.create_delaunay_triangles(
            pts_gdf, area_geom, buildings_gdf, mode=mode
        )
        tris = quality_mod.compute_quality(tris, coords)
        tris = buildings_mod.apply_building_mode(tris, mode)
        tris = attach_all_attributes(tris, coords, cfg, buildings_gdf, dem_path)
        print(f"  細分化後 三角形数: {len(tris)}")

    # --- 使用ノードの再採番 ---
    tris, coords = tri_mod.reindex_used_nodes(tris, coords)
    tris["triangle_id"] = np.arange(len(tris), dtype=int)

    # --- 隣接テーブル ---
    print("[7] node / cell / edge テーブル作成")
    node_df = adj_mod.build_node_table(coords)
    cell_df = adj_mod.build_cell_table(tris)
    edge_df = adj_mod.build_edge_table(
        tris, coords, boundary_polygon=area_geom, buildings_gdf=buildings_gdf
    )
    summary = quality_mod.quality_summary(tris)

    # --- 出力 ---
    print("[8] 出力")
    crs = f"EPSG:{target_epsg}"
    io_mod.write_vector(tris, os.path.join(gis_dir, "triangles.gpkg"), layer="triangles")
    io_mod.write_vector(pts_gdf, os.path.join(gis_dir, "points.gpkg"), layer="points")
    flagged = tris[tris.get("quality_flag", False) | (tris.get("refine_flag", 0) == 1)]
    if len(flagged) > 0:
        io_mod.write_vector(flagged, os.path.join(gis_dir, "flagged_triangles.gpkg"), layer="flagged")

    io_mod.write_csv(node_df, os.path.join(csv_dir, "nodes.csv"))
    io_mod.write_csv(cell_df, os.path.join(csv_dir, "cells.csv"))
    io_mod.write_csv(edge_df, os.path.join(csv_dir, "edges.csv"))
    io_mod.write_csv(pd.DataFrame([summary]), os.path.join(csv_dir, "mesh_quality_summary.csv"))

    # WGS84 でも出力（任意）
    if cfg["crs"].get("also_write_wgs84"):
        io_mod.write_vector(tris.to_crs(4326), os.path.join(gis_dir, "triangles_wgs84.gpkg"), layer="triangles")

    # --- 可視化 ---
    print("[9] プレビュー画像出力")
    plot_mod.plot_mesh(tris, os.path.join(png_dir, "mesh_preview.png"), area_gdf)
    plot_mod.plot_field(tris, "z_range", os.path.join(png_dir, "elevation_range_preview.png"),
                        "Elevation Range (z_range) [m]", cmap="magma")
    plot_mod.plot_field(tris, "building_fraction", os.path.join(png_dir, "building_fraction_preview.png"),
                        "Building Fraction", cmap="OrRd")
    plot_mod.plot_quality_flags(tris, os.path.join(png_dir, "quality_flags_preview.png"))

    print("=" * 60)
    print("完了")
    print(f"  三角形数: {summary.get('n_triangles')}")
    print(f"  ノード数: {len(node_df)}, 辺数: {len(edge_df)}")
    print(f"  品質警告: {summary.get('n_quality_flag', 0)}")
    print(f"  出力先: {out_dir}")
    print("=" * 60)


def refine_loop(cfg, pts_gdf, tris, coords, area_geom, buildings_gdf, dem_path, mode, target_epsg):
    """標高分布が大きい三角形に追加点を入れ、再メッシュ化を繰り返す。"""
    elev = cfg["elevation"]
    max_iter = int(elev["max_refine_iter"])
    max_add = int(elev["max_add_points_per_iter"])
    min_distance = cfg["mesh"]["min_distance"]
    crs = f"EPSG:{target_epsg}"

    current_pts = pts_gdf
    current_tris = tris

    for it in range(max_iter):
        refine_mask = elevation_mod.find_refine_triangles(
            current_tris,
            elev["z_range_threshold"],
            elev["z_std_threshold"],
            elev["edge_max_for_refine"],
            elev["z_range_for_long_edge"],
        )
        n_refine = int(refine_mask.sum())
        if n_refine == 0:
            print(f"  反復 {it+1}: 細分化対象なし。終了。")
            break

        # 対象三角形の重心に追加点（z_range 降順で上限まで）
        target = current_tris[refine_mask].copy()
        if "z_range" in target:
            target = target.sort_values("z_range", ascending=False)
        target = target.head(max_add)
        add_pts = [Point(r.centroid_x, r.centroid_y) for r in target.itertuples()]
        add_gdf = points_mod.points_to_gdf(add_pts, "refine", crs)
        print(f"  反復 {it+1}: 対象 {n_refine} 三角形, 追加点 {len(add_gdf)}")

        merged = points_mod.merge_point_gdfs([current_pts, add_gdf], crs)
        current_pts = points_mod.thin_points(merged, min_distance)

        current_tris, c2 = tri_mod.create_delaunay_triangles(
            current_pts, area_geom, buildings_gdf, mode=mode
        )
        current_tris = quality_mod.compute_quality(current_tris, c2)
        current_tris = buildings_mod.apply_building_mode(current_tris, mode)
        current_tris = attach_all_attributes(current_tris, c2, cfg, buildings_gdf, dem_path)

    return current_pts


if __name__ == "__main__":
    main()
