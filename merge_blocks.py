#!/usr/bin/env python3
"""Merge per-block face/edge CSV and GeoPackage outputs into project-level files."""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
import shapely
from shapely import wkt as shapely_wkt

NODE_PRECISION = 4


def _replace_project_name(obj, project_name):
    if isinstance(obj, dict):
        return {k: _replace_project_name(v, project_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_project_name(item, project_name) for item in obj]
    if isinstance(obj, str):
        return obj.replace("{project_name}", project_name)
    return obj


def _round_coord(v: float) -> float:
    return round(float(v), NODE_PRECISION)


def _coord_key(x, y) -> tuple[float, float] | None:
    if pd.isna(x) or pd.isna(y):
        return None
    return (_round_coord(x), _round_coord(y))


def _remap_ln(df: pd.DataFrame, edge_to_ln: dict[tuple[int, int], int]) -> pd.Series:
    new_lns = []
    for _, row in df.iterrows():
        if pd.isna(row.get("StartNode")) or pd.isna(row.get("EndNode")):
            new_lns.append(row.get("LN", 0))
            continue
        a, b = int(row["StartNode"]), int(row["EndNode"])
        edge_key = (min(a, b), max(a, b))
        new_lns.append(edge_to_ln[edge_key])
    return pd.Series(new_lns, index=df.index)


def _assign_global_lns(
    df_edge: pd.DataFrame, edge_to_ln: dict[tuple[int, int], int], next_ln: int
) -> int:
    for _, row in df_edge.iterrows():
        if pd.isna(row.get("StartNode")) or pd.isna(row.get("EndNode")):
            continue
        a, b = int(row["StartNode"]), int(row["EndNode"])
        edge_key = (min(a, b), max(a, b))
        if edge_key not in edge_to_ln:
            edge_to_ln[edge_key] = next_ln
            next_ln += 1
    return next_ln


def _reclassify_global_edge_ids(df_edge: pd.DataFrame) -> pd.DataFrame:
    """Rebuild boundary classes after block-local edges receive global LNs.

    Per-block processing marks every block perimeter as an outer boundary
    (ID=1). After merging, an edge shared by two cells is internal (ID=0),
    even when those cells came from different blocks. CalMesh interfaces
    (ID=2) retain precedence and are propagated to both incidences.
    """
    incidence = df_edge[["LN", "CN"]].drop_duplicates()
    cell_counts = incidence.groupby("LN", sort=False)["CN"].size()
    invalid = cell_counts[cell_counts > 2]
    if len(invalid):
        sample = invalid.head().to_dict()
        raise ValueError(
            "Non-manifold merged edges are shared by more than two cells: "
            f"{sample}"
        )

    interface_lns = set(
        df_edge.loc[df_edge["ID"].astype(np.int64) == 2, "LN"].astype(np.int64)
    )
    counts = df_edge["LN"].map(cell_counts).to_numpy()
    is_interface = df_edge["LN"].isin(interface_lns).to_numpy()
    old_ids = df_edge["ID"].astype(np.int64).to_numpy()
    new_ids = np.where(is_interface, 2, np.where(counts == 2, 0, 1))

    changed = int(np.count_nonzero(old_ids != new_ids))
    cross_block_internal = int(np.count_nonzero((old_ids == 1) & (new_ids == 0)))
    exposed_boundary = int(np.count_nonzero((old_ids == 0) & (new_ids == 1)))
    df_edge = df_edge.copy()
    df_edge["ID"] = new_ids.astype(np.int64)
    print(
        "merge_blocks: global edge classes rebuilt "
        f"(changed_rows={changed}, boundary_to_internal={cross_block_internal}, "
        f"internal_to_boundary={exposed_boundary})"
    )
    return df_edge


def merge_blocks(
    blocks_dir: str, output_dir: str, crs=None, wkt_source_crs=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    face_parts = []
    edge_parts = []

    block_dirs = sorted(glob.glob(os.path.join(blocks_dir, "block_*")))
    if not block_dirs:
        raise FileNotFoundError(f"No block outputs under {blocks_dir}")

    cn_offset = 0
    for block_order, block_dir in enumerate(block_dirs):
        face_path = os.path.join(block_dir, "face.csv")
        edge_path = os.path.join(block_dir, "edge.csv")
        if not os.path.isfile(face_path) or not os.path.isfile(edge_path):
            raise FileNotFoundError(f"Missing csv in {block_dir}")

        df_face = pd.read_csv(face_path)
        df_edge = pd.read_csv(edge_path)

        if len(df_face) == 0:
            continue

        df_face = df_face.copy()
        df_face["CN"] = df_face["CN"].astype(np.int64) + cn_offset

        df_edge = df_edge.copy()
        required = ["CN", "LN", "node_id", "StartNode", "EndNode", "xcoord", "ycoord"]
        null_counts = df_edge[required].isna().sum()
        if int(null_counts.sum()):
            raise ValueError(
                f"{block_dir}: invalid edge rows contain NaN: {null_counts.to_dict()}"
            )
        group_sizes = df_edge.groupby(["CN", "LN"], sort=False).size()
        if not bool((group_sizes == 2).all()):
            raise ValueError(f"{block_dir}: every CN+LN must have exactly 2 rows")

        df_edge["CN"] = df_edge["CN"].astype(np.int64) + cn_offset
        df_edge["_block_order"] = block_order
        df_edge["_local_node_id"] = df_edge["node_id"].astype(np.int64)

        face_parts.append(df_face)
        edge_parts.append(df_edge)

        cn_offset = int(df_face["CN"].max())

    merged_face = pd.concat(face_parts, ignore_index=True)
    merged_edge = pd.concat(edge_parts, ignore_index=True)

    # Global node IDs: factorize rounded coordinates in first-seen block/row
    # order. This is equivalent to the former Python dict, without 3M row loops.
    rounded_x = merged_edge["xcoord"].astype(np.float64).round(NODE_PRECISION)
    rounded_y = merged_edge["ycoord"].astype(np.float64).round(NODE_PRECISION)
    coord_keys = pd.MultiIndex.from_arrays([rounded_x, rounded_y])
    node_codes, _ = pd.factorize(coord_keys, sort=False)
    merged_edge["node_id"] = node_codes.astype(np.int64) + 1

    local_node_map = merged_edge.drop_duplicates(
        ["_block_order", "_local_node_id"], keep="first"
    )
    local_node_map = pd.Series(
        local_node_map["node_id"].to_numpy(),
        index=pd.MultiIndex.from_frame(
            local_node_map[["_block_order", "_local_node_id"]]
        ),
    )
    for column in ("StartNode", "EndNode"):
        lookup = pd.MultiIndex.from_arrays([
            merged_edge["_block_order"].to_numpy(),
            merged_edge[column].astype(np.int64).to_numpy(),
        ])
        mapped = local_node_map.reindex(lookup).to_numpy()
        if pd.isna(mapped).any():
            raise ValueError(f"Unable to map all {column} values to global nodes")
        merged_edge[column] = mapped.astype(np.int64)

    edge_min = np.minimum(
        merged_edge["StartNode"].to_numpy(), merged_edge["EndNode"].to_numpy()
    )
    edge_max = np.maximum(
        merged_edge["StartNode"].to_numpy(), merged_edge["EndNode"].to_numpy()
    )
    edge_keys = pd.MultiIndex.from_arrays([edge_min, edge_max])
    edge_codes, _ = pd.factorize(edge_keys, sort=False)
    merged_edge["LN"] = edge_codes.astype(np.int64) + 1
    merged_edge = _reclassify_global_edge_ids(merged_edge)
    merged_edge = merged_edge.drop(columns=["_block_order", "_local_node_id"])
    merged_edge = merged_edge.sort_values(["CN", "LN", "node_id"]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    face_csv_out = merged_face.drop(columns=["wkt"], errors="ignore")
    face_csv_out.to_csv(os.path.join(output_dir, "face.csv"), index=False, encoding="utf-8")
    merged_edge.to_csv(os.path.join(output_dir, "edge.csv"), index=False, encoding="utf-8")

    if "wkt" in merged_face.columns and crs is not None:
        geoms = merged_face["wkt"].map(shapely_wkt.loads)
        merged_face_gpkg = gpd.GeoDataFrame(
            merged_face.drop(columns=["wkt"]),
            geometry=geoms,
            crs=wkt_source_crs or crs,
        )
        # Older optimized blocks stored WKT in the landuse raster CRS. New
        # blocks store target/work CRS directly.
        bounds = merged_face_gpkg.total_bounds
        if (
            wkt_source_crs is not None
            and getattr(crs, "is_projected", False)
            and max(abs(bounds[0]), abs(bounds[2])) < 1000
        ):
            merged_face_gpkg = merged_face_gpkg.to_crs(crs)
        else:
            merged_face_gpkg = merged_face_gpkg.set_crs(crs, allow_override=True)
        merged_face_gpkg.to_file(
            os.path.join(output_dir, "face.gpkg"), layer="poly", driver="GPKG"
        )
        print("merge_blocks: face.gpkg built from face.csv wkt column")
    else:
        print("merge_blocks: warning - no WKT found; skipped face.gpkg output")

    if crs is not None:
        starts = merged_edge[
            merged_edge["node_id"] == merged_edge["StartNode"]
        ].drop_duplicates(["CN", "LN"], keep="first")
        ends = merged_edge[
            merged_edge["node_id"] == merged_edge["EndNode"]
        ].drop_duplicates(["CN", "LN"], keep="first")
        edge_rows = starts.merge(
            ends[["CN", "LN", "xcoord", "ycoord"]],
            on=["CN", "LN"],
            how="inner",
            suffixes=("_start", "_end"),
            validate="one_to_one",
        )
        if len(edge_rows) * 2 != len(merged_edge):
            raise ValueError("Unable to construct one line for every merged CN+LN")
        coords = np.stack(
            [
                edge_rows[["xcoord_start", "ycoord_start"]].to_numpy(),
                edge_rows[["xcoord_end", "ycoord_end"]].to_numpy(),
            ],
            axis=1,
        )
        geometry = shapely.linestrings(coords)
        edge_rows = edge_rows.drop(
            columns=["xcoord_start", "ycoord_start", "xcoord_end", "ycoord_end"]
        )
        merged_edge_gpkg = gpd.GeoDataFrame(edge_rows, geometry=geometry, crs=crs)
        merged_edge_gpkg.to_file(
            os.path.join(output_dir, "edge.gpkg"), layer="line", driver="GPKG"
        )
        print("merge_blocks: edge.gpkg built from edge.csv endpoints")
    else:
        print("merge_blocks: warning - CRS missing; skipped edge.gpkg output")

    return merged_face, merged_edge


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge block face/edge CSV outputs")
    parser.add_argument("config_file", type=str)
    parser.add_argument(
        "--keep-blocks",
        action="store_true",
        help="retain block outputs for regression inspection",
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

    output_dir = os.path.join(base_dir, config["output"]["dir"])
    blocks_dir = os.path.join(os.path.dirname(output_dir), "blocks")

    target_file = os.path.join(
        base_dir,
        config["input"]["target_area"]["dir"],
        config["input"]["target_area"]["file"],
    )
    target_crs = gpd.read_file(target_file).crs

    landuse_epsg = config.get("input", {}).get("landuse", {}).get("epsg")
    wkt_source_crs = (
        f"EPSG:{int(landuse_epsg)}" if landuse_epsg is not None else None
    )
    merged_face, merged_edge = merge_blocks(
        blocks_dir,
        output_dir,
        crs=target_crs,
        wkt_source_crs=wkt_source_crs,
    )
    print(
        f"merge_blocks: face={len(merged_face)} cells, edge={len(merged_edge)} rows -> {output_dir}"
    )
    face_gpkg = os.path.join(output_dir, "face.gpkg")
    edge_gpkg = os.path.join(output_dir, "edge.gpkg")
    if os.path.isfile(face_gpkg):
        print(f"merge_blocks: face.gpkg -> {face_gpkg}")
    if os.path.isfile(edge_gpkg):
        print(f"merge_blocks: edge.gpkg -> {edge_gpkg}")

    if os.path.isdir(blocks_dir) and not args.keep_blocks:
        shutil.rmtree(blocks_dir)
        print(f"merge_blocks: removed blocks dir (capacity save): {blocks_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
