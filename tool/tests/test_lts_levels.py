"""LTS レベル付与と mesh.bin トレーラー。"""

import numpy as np
import pytest

from src.lts_levels import compute_lts_levels, face_min_edge_length
from src.mesh_bin_io import decode_lts_trailer, encode_lts_trailer, pack_mesh
from tests.test_mesh_bin import _dem, _mesh, _quality, _terrain, build_normalized_tables


def test_lts_trailer_roundtrip():
    levels = np.array([0, 1, 2, 1], dtype=np.int32)
    raw = encode_lts_trailer(levels, lmax=2, binning_code=2)
    dec = decode_lts_trailer(raw, nface=4)
    assert dec is not None
    out, lmax, code = dec
    np.testing.assert_array_equal(out, levels)
    assert lmax == 2
    assert code == 2


def test_quantile_spreads_levels():
    tables = build_normalized_tables(_mesh(), _quality(), _terrain(), _dem())
    char_len = face_min_edge_length(tables)
    assert len(char_len) == len(tables.faces)
    levels, lmax = compute_lts_levels(tables, max_level=2, binning="quantile")
    assert lmax <= 2
    assert len(np.unique(levels)) >= 2
