"""YAML 設定の読み込みと検証。

未知のキーはエラーにする。参照レイヤ（OSM）を誤って拘束ブレークライン側へ
書いてしまうような設定ミスを、実行前に検出するため。
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """設定ファイルの内容が不正。"""


# ---------------------------------------------------------------------------
# データクラス定義
# ---------------------------------------------------------------------------


@dataclass
class BreaklineInputs:
    """メッシュ辺として拘束する線。精度が担保されたデータのみ指定する。"""

    levees: str | None = None
    embankments: str | None = None
    riverbanks: str | None = None
    channels: str | None = None
    roads: str | None = None
    other: str | None = None

    def as_mapping(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name)}


@dataclass
class ReferenceLayerInputs:
    """位置の目安としてのみ使う線・面。格子の分割には一切使わない。"""

    osm_roads: str | None = None
    osm_buildings: str | None = None
    other: str | None = None

    def as_mapping(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name)}


@dataclass
class InputConfig:
    domain: str = ""
    domain_layer: str | None = None
    domain_min_area: float = 1000.0
    # 対象ポリゴンの絞り込み。domain_min_area 適用後のファイル順のインデックス。
    # null で全ポリゴン。
    polygon_filter: list[int] | None = None
    dem: str = ""
    breaklines: BreaklineInputs = field(default_factory=BreaklineInputs)
    reference_layers: ReferenceLayerInputs = field(default_factory=ReferenceLayerInputs)


@dataclass
class CrsConfig:
    # メートル単位の投影座標系であること。Web Mercator (3857) は面積が歪むため不可。
    target_epsg: int = 6670


@dataclass
class BoundaryQuadBandConfig:
    enabled: bool = True
    method: str = "transfinite"  # transfinite | boundary_layer
    width: float = 30.0
    target_size: float = 50.0  # 帯の接線方向の節点間隔
    layers: int = 1
    # 帯に接する内側要素のサイズ下限 [m]。0 で target_size * 1.5。
    # 境界四角形の target_size とは独立（大きくしても帯形状は変わらない）。
    interface_size: float = 0.0
    # Γ1 近傍で interface_size を下限として掛ける半径 [m]。0 で size_field_grid。
    # 旧実装の interface_size 半径は全域に細かさが伝播しやすかった。
    interface_floor_radius: float = 0.0
    # 内向きオフセットが自己交差した頂点で幅を縮める際の下限比と試行回数
    min_width_ratio: float = 0.25
    max_shrink_passes: int = 12
    # Γ1 の鋭角コーナー判定を適用する内角の上限
    corner_angle_deg: float = 120.0
    # 面積下限を満たすために境界の頂点を落とすときの、切り落とし量の上限。
    # 0 なら境界形状を一切変えず、四角形を置けない区間は三角形に譲る。
    max_corner_cut: float = 0.0

    def resolved_interface_size(self) -> float:
        return self.interface_size if self.interface_size > 0.0 else self.target_size * 1.5

    def resolved_interface_floor_radius(self, size_field_grid: float) -> float:
        if self.interface_floor_radius > 0.0:
            return self.interface_floor_radius
        return size_field_grid

    def resolved_max_corner_cut(self) -> float:
        return max(self.max_corner_cut, 0.0)


@dataclass
class InteriorQuadConfig:
    enabled: bool = False
    mode: str = "selective"  # none | selective | opportunistic
    target_regions: list[str] = field(default_factory=lambda: ["flat"])
    flatness_plane_fit_rmse_max: float = 0.05
    recombination_algorithm: int = 3
    revert_to_triangles_if_nonconvex: bool = True


@dataclass
class BreaklineProcessingConfig:
    """拘束ブレークラインの前処理。

    測量線は頂点が密なことが多い。そのまま拘束すると頂点間隔がそのまま要素辺長に
    なり、面積下限を割る要素が量産される。ここで間引いてから gmsh へ渡す。
    """

    # 頂点間隔の下限。0 で mesh.global_min_size。
    min_vertex_spacing: float = 0.0
    # Douglas-Peucker の許容誤差 [m]
    simplify_tolerance: float = 1.0
    # これより短い線は拘束しない。0 で min_vertex_spacing * 2。
    min_length: float = 0.0
    # Γ1 から内側へ確保する余白。0 で min_vertex_spacing * 0.5。
    # 帯の四角形は Transfinite なので、線が Γ1 に触れると帯が壊れる。
    interior_margin: float = 0.0

    def resolved_min_vertex_spacing(self, global_min_size: float) -> float:
        return self.min_vertex_spacing if self.min_vertex_spacing > 0.0 else global_min_size

    def resolved_min_length(self, global_min_size: float) -> float:
        if self.min_length > 0.0:
            return self.min_length
        return self.resolved_min_vertex_spacing(global_min_size) * 2.0

    # 拘束線どうし、拘束線と外周帯の間隔の下限。0 で面積下限から導く。
    min_clearance: float = 0.0
    # 拘束線近傍のサイズ場の下限。0 で面積下限から導く。
    size_floor: float = 0.0

    def resolved_size_floor(self, min_element_area: float, global_min_size: float) -> float:
        """拘束線に接する三角形が面積下限を満たすためのサイズ。

        gmsh は拘束曲線を round(L/s) 等分するため、辺長は目標 s の 2/3 まで
        短くなり得る。対頂点までの距離は概ね 0.87s なので、三角形の面積は
        最悪 0.29*s^2 になる。これが面積下限以上になる s を採る。
        """
        if self.size_floor > 0.0:
            return self.size_floor
        return max(global_min_size, 1.1 * (min_element_area / 0.29) ** 0.5)

    def resolved_min_clearance(self, min_element_area: float, min_spacing: float) -> float:
        """これより近づくと、間の要素が面積下限を割る距離。

        幅 w・辺長 s の帯に三角形を並べると 1 枚あたり w*s/2 になる。
        これを面積下限以上にするには w >= 2*area/s が要る。
        """
        if self.min_clearance > 0.0:
            return self.min_clearance
        return 2.0 * min_element_area / min_spacing

    def resolved_interior_margin(
        self, global_min_size: float, boundary_band_width: float = 0.0,
    ) -> float:
        if self.interior_margin > 0.0:
            return self.interior_margin
        # 外周四角形帯より内側だけに拘束線を置く（帯は先に Transfinite で固定）
        if boundary_band_width > 0.0:
            return boundary_band_width
        return self.resolved_min_vertex_spacing(global_min_size) * 0.5


@dataclass
class MeshConfig:
    # 面積 625 m^2 は「下限」。これを下回る要素を作らない。
    min_element_area: float = 625.0
    global_min_size: float = 45.0
    global_max_size: float = 150.0
    boundary_quad_band: BoundaryQuadBandConfig = field(default_factory=BoundaryQuadBandConfig)
    interior_quads: InteriorQuadConfig = field(default_factory=InteriorQuadConfig)
    breakline_processing: BreaklineProcessingConfig = field(
        default_factory=BreaklineProcessingConfig
    )
    road_target_size: float = 45.0
    levee_target_size: float = 45.0
    transition_distance: float = 100.0
    max_iterations: int = 8
    refinement_factor: float = 0.7
    max_neighbor_size_ratio: float = 1.5
    # 出来上がったメッシュで、辺を共有する要素どうしの代表辺長の比の上限。
    # 上の max_neighbor_size_ratio はサイズ場の格子に課すもので、実際の要素には
    # 効かない（帯の四角形は Transfinite で決まり、修復の統合も辺長を変える）。
    # 2.0 なら「1 段で辺長が半分」のレベル分けで隣どうしの差が必ず 1 以内になる。
    # 要素ごとに dt を変える解法では、この段差が時間刻みの段差そのものになる。
    # 0 で無効。
    max_neighbor_element_ratio: float = 2.0
    size_field_grid: float = 40.0
    # 同じ場所を細分化する上限回数。勾配方向のばらつきのように細分化で解消しない
    # 基準があるため、これが無いと面積下限まで一様に細分化されてしまう。
    max_refine_attempts: int = 3
    # 解析領域から除去するくびれ・スパイクの半径。
    # 0 で帯の最小許容幅（width * min_width_ratio）* 1.05。
    #
    # 外周四角形帯は自己交差する頂点で幅を width * min_width_ratio まで縮め、
    # それでも解消しない区間だけを三角形に譲る（境界形状は変えない）。つまり
    # 帯が実際に必要とする最小幅は width そのものではなく、この縮小後の幅。
    # ここを width 基準にすると、width を境界形状再現のために大きくしただけで
    # width の 2 倍未満のくびれ（実際には帯が問題なく縮めて対応できる幅）まで
    # 解析領域から丸ごと削除してしまい、河道が途中で分断される・意図した範囲が
    # 欠損して見える、といった実害が出る。
    narrow_feature_radius: float = 0.0

    def resolved_narrow_feature_radius(self) -> float:
        if self.narrow_feature_radius > 0.0:
            return self.narrow_feature_radius
        if not self.boundary_quad_band.enabled:
            return 0.0
        band = self.boundary_quad_band
        return band.width * band.min_width_ratio * 1.05

    @property
    def triangle_size_floor(self) -> float:
        """面積下限を満たす正三角形の辺長。A = sqrt(3)/4 * l^2。"""
        return math.sqrt(4.0 * self.min_element_area / math.sqrt(3.0))

    @property
    def quad_size_floor(self) -> float:
        """面積下限を満たす正方形の辺長。"""
        return math.sqrt(self.min_element_area)


@dataclass
class TerrainConfig:
    plane_fit_rmse_max: float = 0.10
    slope_direction_spread_max_deg: float = 25.0
    representative_elevation_method: str = "area_weighted_mean"
    minimum_dem_samples_per_element: int = 4
    refine_if_insufficient_dem_samples: bool = True
    # 勾配がこれ以下の要素は方向ばらつき判定を無効化する
    flat_slope_threshold: float = 0.01
    dem_target_px_m: float = 0.0  # 0 で DEM 原解像度


@dataclass
class QualityConfig:
    triangle_min_angle_deg: float = 25.0
    triangle_radius_ratio_min: float = 0.30
    triangle_aspect_ratio_max: float = 4.0
    quadrilateral_aspect_ratio_max: float = 3.0
    quadrilateral_min_interior_angle_deg: float = 20.0
    quadrilateral_max_interior_angle_deg: float = 160.0
    quadrilateral_scaled_jacobian_min: float = 0.30
    allow_local_exception_near_constraints: bool = False


@dataclass
class FeatureConfig:
    breaklines_as_mesh_edges: bool = True
    split_surface_by_breaklines: bool = True
    create_quad_band_along_roads: bool = False
    create_quad_band_along_levees: bool = False
    # 参照レイヤは既定でサイズ場にも影響させない
    reference_layers_for_size_field: bool = False
    reference_layers_in_diagnostics: bool = True


@dataclass
class GmshConfig:
    algorithm_2d: int = 6
    optimize: bool = True
    # 平面メッシュに効く平滑化。Netgen は 3 次元向けなので指定しても無視される。
    optimization_method: str = "Laplace2D"
    save_all: bool = True
    verbosity: int = 0
    num_threads: int = 1


@dataclass
class CellBinConfig:
    """AQWA-INUNDATION2D ソルバーが読み込む cell.bin の出力設定。"""

    enabled: bool = False
    directory: str | None = None
    filename: str = ""
    landuse_raster: str | None = None
    landuse_epsg: int | None = None
    soil_raster: str | None = None
    soil_epsg: int | None = None
    buildings: str | None = None
    building_ratio_max: float = 0.95


@dataclass
class ParallelConfig:
    """氾濫ブロック（解析領域ポリゴン）単位の並列メッシュ生成。"""

    enabled: bool = False
    workers: int = 4
    min_block_area_m2: float | None = None  # None なら input.domain_min_area
    keep_block_outputs: bool = True


@dataclass
class GpkgCsvConfig:
    """01_mkMESH_INUN2DH と同じ output_gpkg_csv/ 配下への face/edge 出力。"""

    enabled: bool = True
    directory: str = "output_gpkg_csv"
    face_gpkg: str = "face.gpkg"
    edge_gpkg: str = "edge.gpkg"
    face_csv: str = "face.csv"
    edge_csv: str = "edge.csv"


@dataclass
class OutputConfig:
    directory: str = "output"
    basename: str = "mesh"
    save_diagnostics: bool = True
    save_iteration_meshes: bool = True
    write_vtu: bool = True
    write_xdmf: bool = True
    write_msh: bool = True
    gpkg_csv: GpkgCsvConfig = field(default_factory=GpkgCsvConfig)
    cell_bin: CellBinConfig = field(default_factory=CellBinConfig)


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    crs: CrsConfig = field(default_factory=CrsConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gmsh: GmshConfig = field(default_factory=GmshConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    # 設定ファイルの位置。相対パスの解決に使う。
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, path_str: str | None) -> Path | None:
        if not path_str:
            return None
        p = Path(path_str)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.output.directory)  # type: ignore[return-value]

    @property
    def gpkg_csv_dir(self) -> Path:
        return self.resolve(self.output.gpkg_csv.directory)  # type: ignore[return-value]

    @property
    def cell_bin_dir(self) -> Path:
        d = self.output.cell_bin.directory or self.output.directory
        return self.resolve(d)  # type: ignore[return-value]

    @property
    def blocks_dir(self) -> Path:
        return self.gpkg_csv_dir.parent / "blocks"


# ---------------------------------------------------------------------------
# dict -> dataclass
# ---------------------------------------------------------------------------


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)

    if origin is list:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ConfigError(f"{path}: リストを期待しましたが {type(value).__name__} でした")
        (item_type,) = get_args(annotation) or (Any,)
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]

    # Optional[X] / X | None
    args = get_args(annotation)
    if origin is not None and type(None) in args:
        if value is None:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(value, inner, path)

    if is_dataclass(annotation):
        return _build(annotation, value or {}, path)

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: 数値を期待しましたが {value!r} でした")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: 整数を期待しましたが {value!r} でした")
        return int(value)
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: true/false を期待しましたが {value!r} でした")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: 文字列を期待しましたが {value!r} でした")
        return value
    return value


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    # `from __future__ import annotations` により fields(cls).type は文字列になるため解決する
    return get_type_hints(cls)


def _build(cls: type, data: Any, path: str = "") -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: マッピングを期待しましたが {type(data).__name__} でした")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: 未知のキー {sorted(unknown)}。"
            f" 使用可能なキー: {sorted(known)}"
        )

    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        sub_path = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(data[name], hints[name], sub_path)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# 読み込みと検証
# ---------------------------------------------------------------------------


# EPSG:3857 (Web Mercator) は単位が m でも面積が緯度依存で歪むため禁止する
_FORBIDDEN_EPSG = {3857, 900913, 4326, 4612, 6668, 6697}


def _validate(cfg: Config) -> None:
    m, q, t = cfg.mesh, cfg.quality, cfg.terrain

    if cfg.crs.target_epsg in _FORBIDDEN_EPSG:
        raise ConfigError(
            f"crs.target_epsg={cfg.crs.target_epsg} は面積評価に使えません。"
            " メートル単位の投影座標系（例: 6670 = JGD2011 平面直角座標系 II系）を指定してください"
        )

    if not cfg.input.domain:
        raise ConfigError("input.domain が未指定です")
    if not cfg.input.dem:
        raise ConfigError("input.dem が未指定です")

    if m.min_element_area <= 0:
        raise ConfigError("mesh.min_element_area は正の値である必要があります")
    if m.global_min_size >= m.global_max_size:
        raise ConfigError("mesh.global_min_size は global_max_size より小さくしてください")
    if not 0.0 < m.refinement_factor < 1.0:
        raise ConfigError("mesh.refinement_factor は 0 < f < 1 の範囲で指定してください")
    if m.max_neighbor_size_ratio <= 1.0:
        raise ConfigError("mesh.max_neighbor_size_ratio は 1 より大きくしてください")
    if m.max_neighbor_element_ratio != 0.0 and m.max_neighbor_element_ratio <= 1.0:
        raise ConfigError(
            "mesh.max_neighbor_element_ratio は 1 より大きくするか、0 で無効にしてください"
        )

    # 内部は三角形で埋めるため、下限は正三角形換算の辺長で判定する
    floor = m.triangle_size_floor
    if m.global_min_size < floor:
        raise ConfigError(
            f"mesh.global_min_size={m.global_min_size} は面積下限 {m.min_element_area} m^2 に"
            f" 対応する正三角形の辺長 {floor:.1f} m を下回っています"
        )
    band = m.boundary_quad_band
    if band.enabled:
        if band.method not in {"transfinite", "boundary_layer"}:
            raise ConfigError(
                f"mesh.boundary_quad_band.method='{band.method}' は transfinite か boundary_layer です"
            )
        if band.width * band.target_size < m.min_element_area:
            raise ConfigError(
                f"外周四角形帯の width={band.width} x target_size={band.target_size}"
                f" = {band.width * band.target_size:.0f} m^2 が面積下限 {m.min_element_area} m^2 を"
                " 下回ります"
            )
        if band.layers < 1:
            raise ConfigError("mesh.boundary_quad_band.layers は 1 以上です")

    iq = m.interior_quads
    if iq.mode not in {"none", "selective", "opportunistic"}:
        raise ConfigError(f"mesh.interior_quads.mode='{iq.mode}' が不正です")
    if iq.recombination_algorithm not in {0, 1, 2, 3}:
        raise ConfigError("mesh.interior_quads.recombination_algorithm は 0-3 です")

    if t.representative_elevation_method not in {
        "area_weighted_mean", "median", "centroid_sample", "fitted_plane_at_centroid",
    }:
        raise ConfigError(
            f"terrain.representative_elevation_method='{t.representative_elevation_method}' が不正です"
        )
    if t.minimum_dem_samples_per_element < 1:
        raise ConfigError("terrain.minimum_dem_samples_per_element は 1 以上です")

    if not 0.0 < q.triangle_radius_ratio_min <= 1.0:
        raise ConfigError("quality.triangle_radius_ratio_min は 0 < q <= 1 です")
    if q.quadrilateral_min_interior_angle_deg >= q.quadrilateral_max_interior_angle_deg:
        raise ConfigError("quality の四角形内角は min < max である必要があります")

    cb = cfg.output.cell_bin
    if not 0.0 < cb.building_ratio_max <= 1.0:
        raise ConfigError("output.cell_bin.building_ratio_max は 0 < r <= 1 です")


def load_config(path: str | Path) -> Config:
    """YAML を読み込み、既定値を補って検証した Config を返す。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: トップレベルはマッピングである必要があります")

    cfg: Config = _build(Config, {k: v for k, v in raw.items() if k != "base_dir"})
    cfg.base_dir = path.parent
    _validate(cfg)
    return cfg
