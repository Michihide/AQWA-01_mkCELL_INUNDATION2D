from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon

from src.config import load_config
from src.io_vector import ReferenceLayers
from src.reference_size_field import apply_reference_layers_to_size_field
from src.size_field import SizeField


def _bands_with_interior(outer_side: float = 200.0, inset: float = 25.0) -> list:
    inner = Polygon([
        (inset, inset), (outer_side - inset, inset),
        (outer_side - inset, outer_side - inset), (inset, outer_side - inset),
    ])
    return [SimpleNamespace(interior=inner)]


def test_reference_layers_lower_size_along_road():
    cfg = load_config("config/hii.yaml")
    cfg.features.reference_layers_for_size_field = True
    cfg.mesh.road_target_size = 30.0
    cfg.mesh.global_max_size = 150.0

    sf = SizeField.uniform(
        (0, 0, 200, 200), spacing=10.0, value=150.0,
        floor=45.0, ceiling=150.0,
    )
    road = LineString([(50, 100), (150, 100)])
    refs = ReferenceLayers(layers={"osm_roads": gpd.GeoDataFrame(geometry=[road], crs="EPSG:6671")})

    n = apply_reference_layers_to_size_field(cfg, sf, _bands_with_interior(), refs)
    assert n > 0
    on_road = sf.sample(np.array([100.0]), np.array([100.0]))[0]
    far = sf.sample(np.array([180.0]), np.array([180.0]))[0]
    assert on_road < far


def test_disabled_flag_is_noop():
    cfg = load_config("config/hii.yaml")
    cfg.features.reference_layers_for_size_field = False
    sf = SizeField.uniform(
        (0, 0, 200, 200), spacing=10.0, value=150.0,
        floor=45.0, ceiling=150.0,
    )
    road = LineString([(50, 100), (150, 100)])
    refs = ReferenceLayers(layers={"osm_roads": gpd.GeoDataFrame(geometry=[road], crs="EPSG:6671")})
    n = apply_reference_layers_to_size_field(cfg, sf, _bands_with_interior(), refs)
    assert n == 0
    assert sf.sample(np.array([100.0]), np.array([100.0]))[0] >= 149.0
