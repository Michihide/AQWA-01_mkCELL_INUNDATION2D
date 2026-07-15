"""設定ファイル（YAML/JSON）の読み込みと既定値の管理。"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# 既定設定。指示書 15 章に対応する。
DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "analysis_area": None,
        "dem": None,
        "buildings": None,
        "rivers": None,
        "roads": None,
        "facilities": None,
    },
    "output": {
        "directory": "output_mesh",
    },
    "crs": {
        "auto_utm": True,
        "target_epsg": None,
        "also_write_wgs84": False,
    },
    "mesh": {
        "min_distance": 5.0,
        "boundary_spacing": 10.0,
        "building_boundary_spacing": 5.0,
        "building_buffer": 10.0,
        "building_buffer_spacing": 5.0,
        "interior_spacing": 20.0,
        "river_spacing": 10.0,
        "road_spacing": 10.0,
        "facility_buffer": 30.0,
        "facility_spacing": 10.0,
        "building_mode": "building_as_roughness",  # or "building_as_hole"
    },
    "elevation": {
        "slope_refine": True,
        "slope_threshold": 0.03,
        "slope_point_spacing": 10.0,
        "z_range_threshold": 0.5,
        "z_std_threshold": 0.2,
        "max_refine_iter": 3,
        "max_add_points_per_iter": 50000,
        "edge_max_for_refine": 20.0,
        "z_range_for_long_edge": 0.3,
        "all_touched": False,
    },
    "quality": {
        "min_edge_length": 5.0,
        "min_area": 12.5,
        "min_angle": 20.0,
        "max_aspect_ratio": 8.0,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """ネストした dict を再帰的にマージする（override 優先）。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """設定ファイルを読み込み、既定値とマージして返す。

    Args:
        path: YAML または JSON 設定ファイルのパス（None なら既定値のみ）。
        overrides: CLI 引数などで上書きする設定（最優先）。

    Returns:
        マージ済みの設定 dict。
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".json"):
                user_cfg = json.load(f)
            else:
                if yaml is None:
                    raise ImportError("PyYAML がインストールされていません。")
                user_cfg = yaml.safe_load(f) or {}
        config = _deep_update(config, user_cfg)

    if overrides:
        config = _deep_update(config, overrides)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """最低限の妥当性チェック（必須入力・モード）。"""
    inp = config.get("input", {})
    if not inp.get("analysis_area"):
        raise ValueError("input.analysis_area が指定されていません。")
    if not inp.get("dem"):
        raise ValueError("input.dem が指定されていません。")

    mode = config["mesh"].get("building_mode")
    if mode not in {"building_as_roughness", "building_as_hole"}:
        raise ValueError(
            f"mesh.building_mode が不正です: {mode} "
            "('building_as_roughness' または 'building_as_hole')"
        )
