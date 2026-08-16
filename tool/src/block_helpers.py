"""氾濫ブロック（解析領域ポリゴン）の列挙とブロック用設定の生成。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import Config
from .io_vector import domain_polygon_indices


def list_block_indices(cfg: Config) -> list[int]:
    """load_domain と同じ explode + 面積フィルタ後の 0..n-1 インデックス。"""
    min_area = (
        cfg.parallel.min_block_area_m2
        if cfg.parallel.min_block_area_m2 is not None
        else None
    )
    return domain_polygon_indices(cfg, min_area_m2=min_area)


def config_for_block(cfg: Config, block_index: int) -> Config:
    """1 ブロックだけを対象にした設定へ差し替える（出力先は blocks/block_XXX/）。"""
    block_dir = cfg.blocks_dir / f"block_{block_index:03d}"
    base = f"block_{block_index:03d}"
    new_input = replace(cfg.input, polygon_filter=[block_index])
    new_gpkg = replace(cfg.output.gpkg_csv, directory=str(block_dir))
    new_output = replace(
        cfg.output,
        directory=str(block_dir),
        basename=base,
        save_diagnostics=False,
        save_iteration_meshes=False,
        write_vtu=False,
        write_xdmf=False,
        gpkg_csv=new_gpkg,
        cell_bin=replace(cfg.output.cell_bin, enabled=False),
    )
    return replace(cfg, input=new_input, output=new_output)


def block_mesh_path(cfg: Config, block_index: int) -> Path:
    return cfg.blocks_dir / f"block_{block_index:03d}" / f"block_{block_index:03d}.msh"
