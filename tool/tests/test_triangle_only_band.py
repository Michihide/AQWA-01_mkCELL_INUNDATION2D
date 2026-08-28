from shapely.geometry import box

from src.boundary_quad_band import triangle_only_polygon_band
from src.config import Config


def test_triangle_only_band_has_no_quads():
    poly = box(0.0, 0.0, 100.0, 80.0)
    band = triangle_only_polygon_band(poly, spacing=20.0)
    assert band.ok
    assert band.quad_count() == 0
    assert band.interior.equals(poly)
    assert len(band.exterior.skipped) == band.exterior.n
    walk = band.exterior.interior_walk()
    assert walk and all(kind == "outer" for kind, _ in walk)


def test_quad_band_disabled_by_default():
    assert Config().mesh.boundary_quad_band.enabled is False
