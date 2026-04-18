import numpy as np

from density2sse.geometry.helix import HelixPrimitive, canonical_ca_positions_local, residue_count_from_length, unit
from density2sse.geometry.helix_builder import build_backbone_atoms


def test_unit():
    v = np.array([3.0, 0.0, 4.0])
    u = unit(v)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-6


def test_residue_count():
    assert residue_count_from_length(9.0) == 6
    assert residue_count_from_length(20.0) >= 6


def test_canonical_ca():
    ca = canonical_ca_positions_local(8)
    assert ca.shape == (8, 3)


def test_backbone_shape():
    p = HelixPrimitive(np.zeros(3), np.array([0.0, 0.0, 1.0]), 20.0)
    b = build_backbone_atoms(p)
    assert b.ndim == 3
    assert b.shape[1] == 4
