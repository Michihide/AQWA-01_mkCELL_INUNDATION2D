"""cell.bin に埋め込む土地利用・土壌・建物の面積統計。

旧 face/edge 生成の "fast" モード（ラスタをポリゴンで
クロップし、コード別のピクセル数 × ピクセル面積を積算する方式）に合わせている。
土地利用・土壌ラスタ（100 m 解像度など）はメッシュ要素より粗いことが多く、
サブピクセルの厳密な重なり面積までは求めない。ソルバー側は面粗度の重み付けに
使うだけなので、この精度で十分という既存パイプラインの判断を踏襲する。
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.mask import mask as rio_mask
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .utils import get_logger


def _pixel_area_m2(raster_crs: CRS, res: tuple[float, float], x: float, y: float) -> float:
    """ラスタの 1 ピクセルの実面積 [m^2]。

    投影座標系なら単位換算するだけでよいが、地理座標系（緯度経度）の場合は
    ピクセルの見かけの面積（度^2）が実際の m^2 と全く違うため、代表点付近での
    測地学的な面積（Geod）で計算する。土地利用・土壌ラスタは EPSG:4326/6668 の
    緯度経度で配布されていることが多く、これを踏まないと面積が桁違いになる。
    """
    if raster_crs.is_projected:
        unit_factor = raster_crs.axis_info[0].unit_conversion_factor if raster_crs.axis_info else 1.0
        return abs(res[0]) * abs(res[1]) * unit_factor * unit_factor
    geod = Geod(ellps="WGS84")
    half_w, half_h = abs(res[0]) / 2.0, abs(res[1]) / 2.0
    lons = [x - half_w, x + half_w, x + half_w, x - half_w]
    lats = [y - half_h, y - half_h, y + half_h, y + half_h]
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area)


def raster_code_areas(
    polygons: list[Polygon],
    mesh_crs: CRS,
    raster_path: str | Path,
    codes: list[str],
    fallback_epsg: int | None = None,
) -> np.ndarray:
    """各ポリゴンについて、ラスタのコード別重なり面積 [m^2] を返す。

    戻り値の shape は (len(polygons), len(codes))。ラスタと重ならない・
    データがない要素は 0 のままになる。
    """
    logger = get_logger()
    code_ints = [int(c) for c in codes]
    code_to_col = {c: i for i, c in enumerate(code_ints)}
    min_code, max_code = min(code_ints), max(code_ints)
    out = np.zeros((len(polygons), len(codes)), dtype=np.float64)
    if not polygons:
        return out

    with rasterio.open(raster_path) as src:
        raster_crs = CRS.from_user_input(src.crs) if src.crs else None
        if raster_crs is None:
            if fallback_epsg is None:
                raise ValueError(
                    f"{raster_path}: ラスタの CRS が不明で epsg も未指定です"
                )
            raster_crs = CRS.from_epsg(fallback_epsg)
            logger.warning("%s: CRS 不明のため EPSG:%d を仮定します", raster_path, fallback_epsg)

        nodata = src.nodata if src.nodata is not None else 255

        transformer = None
        if not CRS(mesh_crs).equals(raster_crs):
            transformer = Transformer.from_crs(mesh_crs, raster_crs, always_xy=True)

        # 対象領域の代表点（重心の平均）でピクセル面積を 1 回だけ評価する。
        # 緯度方向のピクセル面積変化は対象領域のスケールでは無視できるほど小さい。
        rep_x = float(np.mean([p.centroid.x for p in polygons]))
        rep_y = float(np.mean([p.centroid.y for p in polygons]))
        if transformer is not None:
            rep_x, rep_y = transformer.transform(rep_x, rep_y)
        pixel_area = _pixel_area_m2(raster_crs, src.res, rep_x, rep_y)
        logger.info(
            "%s: 1 ピクセル面積 ~ %.2f m^2（代表点 %.4f, %.4f）",
            Path(raster_path).name, pixel_area, rep_x, rep_y,
        )

        n_empty = 0
        for i, poly in enumerate(polygons):
            geom = poly
            if transformer is not None:
                xs, ys = transformer.transform(*zip(*poly.exterior.coords))
                geom = Polygon(zip(xs, ys))
            try:
                out_image, _ = rio_mask(
                    src, [mapping(geom)], crop=True, nodata=nodata, all_touched=True,
                )
            except ValueError:
                n_empty += 1
                continue
            band = out_image[0]
            valid = (
                (band != nodata) & ~np.isnan(band)
                if np.issubdtype(band.dtype, np.floating)
                else band != nodata
            )
            if not valid.any():
                n_empty += 1
                continue
            vals = band[valid]
            vals = (np.rint(vals) if np.issubdtype(vals.dtype, np.floating) else vals).astype(np.int64)
            in_range = (vals >= min_code) & (vals <= max_code)
            if not in_range.any():
                n_empty += 1
                continue
            uniq, counts = np.unique(vals[in_range], return_counts=True)
            for code, count in zip(uniq.tolist(), counts.tolist()):
                col = code_to_col.get(code)
                if col is not None:
                    out[i, col] = count * pixel_area

    if n_empty:
        logger.warning(
            "%s: %d / %d 要素がラスタと重ならず面積 0 になりました",
            Path(raster_path).name, n_empty, len(polygons),
        )
    return out


def load_buildings_bbox(
    path: str | Path, target_crs: CRS, domain_bounds: tuple[float, float, float, float],
    buffer_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """解析領域 bbox だけを対象に建物ポリゴンを読み込む。

    県単位などの巨大な OSM シェープファイルをまるごと読み込むと重いため、
    ソース CRS へ変換した bbox で `gpd.read_file(bbox=...)` を使う。
    """
    logger = get_logger()
    # CRS の確認だけを目的とした軽量読み込み（先頭 1 行）
    probe = gpd.read_file(path, rows=1)
    if probe.crs is None:
        raise ValueError(f"{path}: 建物データの CRS が不明です")
    src_crs = CRS.from_user_input(probe.crs)

    min_x, min_y, max_x, max_y = domain_bounds
    min_x -= buffer_m; min_y -= buffer_m; max_x += buffer_m; max_y += buffer_m
    if not src_crs.equals(target_crs):
        to_src = Transformer.from_crs(target_crs, src_crs, always_xy=True)
        xs, ys = to_src.transform([min_x, max_x, min_x, max_x], [min_y, min_y, max_y, max_y])
        bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        bbox = (min_x, min_y, max_x, max_y)

    gdf = gpd.read_file(path, bbox=bbox)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    logger.info("建物ポリゴン %s: %d 件 (bbox 読み込み)", Path(path).name, len(gdf))
    if gdf.empty:
        return gdf
    if not CRS.from_user_input(gdf.crs).equals(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


def building_ratio_perimeter(
    polygons: list[Polygon],
    areas: np.ndarray,
    buildings: gpd.GeoDataFrame | None,
    ratio_max: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """各要素の建物被覆率 (bld_ratio) と建物外周長 (bld_peri) を返す。"""
    n = len(polygons)
    ratio = np.zeros(n, dtype=np.float64)
    perimeter = np.zeros(n, dtype=np.float64)
    if buildings is None or buildings.empty:
        return ratio, perimeter

    geoms = [g for g in buildings.geometry if g is not None and not g.is_empty]
    if not geoms:
        return ratio, perimeter
    tree = STRtree(geoms)

    for i, poly in enumerate(polygons):
        idx = tree.query(poly, predicate="intersects")
        if len(idx) == 0:
            continue
        candidates = [geoms[j] for j in idx]
        union = unary_union(candidates)
        inter = poly.intersection(union)
        if inter.is_empty:
            continue
        area = float(areas[i])
        perimeter[i] = float(inter.length)
        ratio[i] = min(inter.area / area, ratio_max) if area > 0 else 0.0
    return ratio, perimeter


_CD_ALIASES = ("Cd", "C_D", "CD", "cd", "C_d")
_A_ALIASES = ("a", "a_veg", "A")
_HV_ALIASES = ("Hv", "H_v", "H", "Hveg", "h_veg")


def resolve_attribute_field(
    columns: list[str],
    preferred: str,
    aliases: tuple[str, ...],
    *,
    what: str,
) -> str:
    """ポリゴン属性列を、指定名または別名から決める。"""
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    candidates = (preferred, *aliases)
    for name in candidates:
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(
        f"植生ポリゴンに {what} 列がありません（探した名前: {list(candidates)}）。"
        f" ある列: {list(columns)}"
    )


def vegetation_chi_and_params(
    polygons: list[Polygon],
    areas: np.ndarray,
    vegetation: gpd.GeoDataFrame | None,
    *,
    ratio_max: float = 0.95,
    field_cd: str = "Cd",
    field_a: str = "a",
    field_hv: str = "Hv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """各要素の chi_veg と、交差面積で重み付けした C_D, a, H_v を返す。"""
    n = len(polygons)
    chi = np.zeros(n, dtype=np.float64)
    a_out = np.zeros(n, dtype=np.float64)
    hv_out = np.zeros(n, dtype=np.float64)
    cd_out = np.zeros(n, dtype=np.float64)
    if vegetation is None or vegetation.empty:
        return chi, a_out, hv_out, cd_out

    live = vegetation[vegetation.geometry.notna() & ~vegetation.geometry.is_empty]
    if live.empty:
        return chi, a_out, hv_out, cd_out

    col_cd = resolve_attribute_field(list(live.columns), field_cd, _CD_ALIASES, what="C_D")
    col_a = resolve_attribute_field(list(live.columns), field_a, _A_ALIASES, what="a")
    col_hv = resolve_attribute_field(list(live.columns), field_hv, _HV_ALIASES, what="H_v")

    geoms = list(live.geometry)
    cds = np.asarray(live[col_cd], dtype=np.float64)
    avs = np.asarray(live[col_a], dtype=np.float64)
    hvs = np.asarray(live[col_hv], dtype=np.float64)
    tree = STRtree(geoms)

    for i, poly in enumerate(polygons):
        idx = tree.query(poly, predicate="intersects")
        if len(idx) == 0:
            continue
        inter_area = 0.0
        w_cd = 0.0
        w_a = 0.0
        w_hv = 0.0
        for j in idx:
            inter = poly.intersection(geoms[j])
            if inter.is_empty:
                continue
            ia = float(inter.area)
            if ia <= 0.0:
                continue
            inter_area += ia
            if np.isfinite(cds[j]):
                w_cd += float(cds[j]) * ia
            if np.isfinite(avs[j]):
                w_a += float(avs[j]) * ia
            if np.isfinite(hvs[j]):
                w_hv += float(hvs[j]) * ia
        face_area = float(areas[i])
        chi[i] = min(inter_area / face_area, ratio_max) if face_area > 0.0 else 0.0
        if inter_area > 0.0:
            cd_out[i] = w_cd / inter_area
            a_out[i] = w_a / inter_area
            hv_out[i] = w_hv / inter_area
    return chi, a_out, hv_out, cd_out
