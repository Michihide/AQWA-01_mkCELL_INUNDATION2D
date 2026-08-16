"""メッシュ後処理スクリプト共通: YAML 設定読み込み."""

from __future__ import annotations

import os
from typing import Any

import yaml


def read_mesh_config(config_file: str) -> dict[str, Any]:
    """project プレースホルダ展開済みのパス・パラメータ dict を返す。"""
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def repl(obj: Any, project_name: str) -> Any:
        if isinstance(obj, dict):
            return {k: repl(v, project_name) for k, v in obj.items()}
        if isinstance(obj, list):
            return [repl(v, project_name) for v in obj]
        if isinstance(obj, str):
            return obj.replace("{project_name}", project_name)
        return obj

    project_name = config["project"]["name"]
    config = repl(config, project_name)
    base_dir = config["project"]["base_dir"]
    output_dir = os.path.join(base_dir, config["output"]["dir"])
    cell_txt_dir = config["output"].get("cell_txt_dir", output_dir)
    cell_txt_filename = config["output"].get("cell_txt", "cell.bin")
    cell_txt_dir_full = (
        os.path.join(base_dir, cell_txt_dir)
        if not os.path.isabs(cell_txt_dir)
        else cell_txt_dir
    )
    dem_dir = os.path.join(base_dir, config["input"]["dem"]["dir"])
    dem_file = config["input"]["dem"]["file"]

    params = config.get("parameters", {})
    lu_stat = params.get("elevation_stat_by_landuse") or {}

    return {
        "project_name": project_name,
        "face_csv": os.path.join(output_dir, "face.csv"),
        "edge_csv": os.path.join(output_dir, "edge.csv"),
        "face_gpkg": os.path.abspath(os.path.join(output_dir, "face.gpkg")),
        "cell_txt": os.path.join(cell_txt_dir_full, cell_txt_filename),
        "dem_tif": os.path.join(dem_dir, dem_file),
        "elevation_stat_by_landuse": lu_stat,
        "landuse_codes": [
            "10", "20", "50", "60", "70", "91", "92", "100", "110", "140", "150", "160", "255",
        ],
    }
