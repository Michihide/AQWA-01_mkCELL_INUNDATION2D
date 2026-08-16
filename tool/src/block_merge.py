"""ブロック別 face/edge CSV・GeoPackage を AQWA 互換形式で統合する。"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely import wkt as shapely_wkt

from .utils import get_logger

NODE_PRECISION = 4


def _reclassify_global_edge_ids(df_edge: pd.DataFrame) -> pd.DataFrame:
    """ブロック境界を跨ぐ内部辺の ID を 0 に直す（01_mkMESH merge_blocks 相当）。"""
    incidence = df_edge[["LN", "CN"]].drop_duplicates()
    cell_counts = incidence.groupby("LN", sort=False)["CN"].size()
    invalid = cell_counts[cell_counts > 2]
    if len(invalid):
        raise ValueError(
            "非多様体な辺（3 面以上が共有）: "
            f"{invalid.head().to_dict()}"
        )

    interface_lns = set(
        df_edge.loc[df_edge["ID"].astype(np.int64) == 2, "LN"].astype(np.int64)
    )
    counts = df_edge["LN"].map(cell_counts).to_numpy()
    is_interface = df_edge["LN"].isin(interface_lns).to_numpy()
    old_ids = df_edge["ID"].astype(np.int64).to_numpy()
    new_ids = np.where(is_interface, 2, np.where(counts == 2, 0, 1))

    df_edge = df_edge.copy()
    df_edge["ID"] = new_ids.astype(np.int64)
    return df_edge


def merge_block_csvs(
    blocks_dir: Path,
    output_dir: Path,
    crs: CRS,
    *,
    block_indices: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """blocks/block_XXX/face.csv, edge.csv を統合する。

    block_indices を指定した場合、そのブロックだけを統合する（部分実行時に
    blocks/ 内の古い block_* と混ざらないようにする）。
    """
    logger = get_logger()
    face_parts: list[pd.DataFrame] = []
    edge_parts: list[pd.DataFrame] = []

    if block_indices is not None:
        block_dirs = [blocks_dir / f"block_{idx:03d}" for idx in block_indices]
        missing = [d for d in block_dirs if not d.is_dir()]
        if missing:
            raise FileNotFoundError(
                "ブロック出力がありません: "
                + ", ".join(str(p) for p in missing)
            )
    else:
        block_dirs = sorted(blocks_dir.glob("block_*"))
        if not block_dirs:
            raise FileNotFoundError(f"ブロック出力がありません: {blocks_dir}")

    cn_offset = 0
    for block_order, block_dir in enumerate(block_dirs):
        face_path = block_dir / "face.csv"
        edge_path = block_dir / "edge.csv"
        if not face_path.is_file() or not edge_path.is_file():
            logger.warning("スキップ（face/edge なし）: %s", block_dir.name)
            continue

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
            raise ValueError(f"{block_dir}: edge.csv に NaN: {null_counts.to_dict()}")

        group_sizes = df_edge.groupby(["CN", "LN"], sort=False).size()
        if not bool((group_sizes == 2).all()):
            raise ValueError(f"{block_dir}: CN+LN ごとに 2 行必要です")

        df_edge["CN"] = df_edge["CN"].astype(np.int64) + cn_offset
        df_edge["_block_order"] = block_order
        df_edge["_local_node_id"] = df_edge["node_id"].astype(np.int64)

        face_parts.append(df_face)
        edge_parts.append(df_edge)
        cn_offset = int(df_face["CN"].max())

    merged_face = pd.concat(face_parts, ignore_index=True)
    merged_edge = pd.concat(edge_parts, ignore_index=True)

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
            raise ValueError(f"{column} を全局 node_id に写像できません")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    face_csv_out = merged_face.drop(columns=["wkt"], errors="ignore")
    face_csv_out.to_csv(output_dir / "face.csv", index=False, encoding="utf-8")
    merged_edge.to_csv(output_dir / "edge.csv", index=False, encoding="utf-8")

    face_gpkg = output_dir / "face.gpkg"
    edge_gpkg = output_dir / "edge.gpkg"
    if face_gpkg.exists():
        face_gpkg.unlink()
    if edge_gpkg.exists():
        edge_gpkg.unlink()

    if "wkt" in merged_face.columns:
        geoms = merged_face["wkt"].map(shapely_wkt.loads)
        gdf_face = gpd.GeoDataFrame(
            merged_face.drop(columns=["wkt"]),
            geometry=geoms,
            crs=crs,
        )
        gdf_face.to_file(face_gpkg, layer="poly", driver="GPKG")
    else:
        logger.warning("face.csv に wkt 列が無いため face.gpkg をスキップします")

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
        raise ValueError("edge.csv から line ジオメトリを構成できません")

    coords = np.stack([
        edge_rows[["xcoord_start", "ycoord_start"]].to_numpy(),
        edge_rows[["xcoord_end", "ycoord_end"]].to_numpy(),
    ], axis=1)
    edge_rows = edge_rows.drop(
        columns=["xcoord_start", "ycoord_start", "xcoord_end", "ycoord_end"]
    )
    edge_rows["LineID"] = edge_rows["LN"]
    edge_rows["CalMesh"] = 0
    if "area" not in edge_rows.columns:
        edge_rows = edge_rows.merge(
            merged_face[["CN", "area"]], on="CN", how="left",
        )
    gdf_edge = gpd.GeoDataFrame(
        edge_rows,
        geometry=shapely.linestrings(coords),
        crs=crs,
    )
    gdf_edge.to_file(edge_gpkg, layer="line", driver="GPKG")

    logger.info(
        "ブロック統合: face=%d, edge_rows=%d -> %s",
        len(merged_face), len(merged_edge), output_dir,
    )
    return merged_face, merged_edge


def merge_block_outputs(
    blocks_dir: Path,
    output_dir: Path,
    crs: CRS,
    *,
    block_indices: list[int] | None = None,
    keep_blocks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = merge_block_csvs(blocks_dir, output_dir, crs, block_indices=block_indices)
    if not keep_blocks and blocks_dir.is_dir():
        shutil.rmtree(blocks_dir)
    return merged
