"""gmsh の初期化・オプション設定・メッシュ生成。"""

from __future__ import annotations

from contextlib import contextmanager

import gmsh

from .config import Config
from .gmsh_geometry import GeometryTags, build_geometry, set_physical_groups
from .boundary_quad_band import PolygonBand
from .mesh_parser import Mesh, extract_mesh
from .size_field import SizeField
from .utils import get_logger

# 2 次元の平面メッシュに対して意味のある最適化手法
_SURFACE_OPTIMIZERS = {"Laplace2D", "Relocate2D", "UntangleMeshGeometry", "HighOrder"}


@contextmanager
def gmsh_session(cfg: Config, model_name: str = "mesh"):
    """gmsh の初期化と終了を対で行う。"""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if cfg.gmsh.verbosity >= 4 else 0)
        gmsh.option.setNumber("General.Verbosity", cfg.gmsh.verbosity)
        gmsh.option.setNumber("General.NumThreads", cfg.gmsh.num_threads)
        gmsh.model.add(model_name)
        yield
    finally:
        gmsh.finalize()


def apply_mesh_options(cfg: Config) -> None:
    o = gmsh.option.setNumber
    o("Mesh.Algorithm", cfg.gmsh.algorithm_2d)
    o("Mesh.RecombinationAlgorithm", cfg.mesh.interior_quads.recombination_algorithm)
    o("Mesh.RecombineAll", 0)

    # サイズは背景場だけで決める。点・曲率・境界からの伝播はすべて切る。
    o("Mesh.MeshSizeExtendFromBoundary", 0)
    o("Mesh.MeshSizeFromPoints", 0)
    o("Mesh.MeshSizeFromCurvature", 0)
    o("Mesh.MeshSizeMin", cfg.mesh.global_min_size)
    o("Mesh.MeshSizeMax", cfg.mesh.global_max_size)


def set_background_field(size_field: SizeField) -> int:
    """サイズ場を Post-processing View 経由で背景メッシュとして設定する。"""
    view = gmsh.view.add("size_field")
    data = size_field.to_view_data()
    n_cells = (size_field.nx - 1) * (size_field.ny - 1)
    gmsh.view.addListData(view, "SQ", n_cells, data)

    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    return field


def optimize_mesh(cfg: Config) -> None:
    if not cfg.gmsh.optimize:
        return
    logger = get_logger()
    method = cfg.gmsh.optimization_method
    if method not in _SURFACE_OPTIMIZERS:
        logger.warning(
            "gmsh.optimization_method='%s' は 3 次元向けで平面メッシュには効きません。"
            " Laplace2D を使います", method,
        )
        method = "Laplace2D"
    # 帯の節点はすべて幾何点上にあるため平滑化で動かない
    gmsh.model.mesh.optimize(method)


def generate_mesh(
    cfg: Config,
    bands: list[PolygonBand],
    size_field: SizeField,
    breaklines=None,
) -> tuple[Mesh, GeometryTags, dict[str, int]]:
    """ジオメトリ構築からメッシュ生成までを 1 回分実行する。"""
    logger = get_logger()

    tags = build_geometry(bands, breaklines)
    groups = set_physical_groups(tags)
    apply_mesh_options(cfg)
    set_background_field(size_field)

    logger.info("Gmsh 2D 生成を開始（algorithm=%d）", cfg.gmsh.algorithm_2d)
    gmsh.model.mesh.generate(2)
    logger.info("Gmsh 2D 生成が完了")
    optimize_mesh(cfg)

    # 帯端の三角形も Transfinite で分割数が決まっており動かせないので帯扱い
    roles = {s: "quad_boundary" for s in (*tags.band_surfaces, *tags.band_tri_surfaces)}
    roles.update({s: "interior" for s in tags.interior_surfaces})
    mesh = extract_mesh(roles, tags.all_breakline_curves())

    logger.info(
        "メッシュ生成: 節点 %d, 三角形 %d, 四角形 %d (合計 %d 要素)",
        mesh.n_nodes, len(mesh.triangles), len(mesh.quads), mesh.n_elements,
    )
    return mesh, tags, groups
