import numpy as np

from density2sse.geometry.helix import HelixPrimitive, unit
from density2sse.geometry.helix_builder import build_backbone_atoms
from density2sse.geometry.peptide_build import (
    R_CA_C,
    R_C_N,
    R_C_O,
    R_N_CA,
    build_polyalanine_alpha_helix,
)


def test_polyalanine_helix_bonds_and_peptide():
    bb = build_polyalanine_alpha_helix(24)
    for i in range(24):
        n, ca, c, o = bb[i]
        assert abs(np.linalg.norm(ca - n) - R_N_CA) < 0.02
        assert abs(np.linalg.norm(c - ca) - R_CA_C) < 0.02
        assert abs(np.linalg.norm(o - c) - R_C_O) < 0.02
    for i in range(1, 24):
        assert abs(np.linalg.norm(bb[i, 0] - bb[i - 1, 2]) - R_C_N) < 0.02


def test_backbone_after_kabsch_preserves_bonds():
    p = HelixPrimitive(np.zeros(3), np.array([0.0, 0.0, 1.0]), 22.0)
    b = build_backbone_atoms(p)
    for i in range(b.shape[0]):
        n, ca, c, o = b[i]
        assert abs(np.linalg.norm(ca - n) - R_N_CA) < 0.05
        assert abs(np.linalg.norm(c - ca) - R_CA_C) < 0.05
        assert abs(np.linalg.norm(o - c) - R_C_O) < 0.05
    for i in range(1, b.shape[0]):
        assert abs(np.linalg.norm(b[i, 0] - b[i - 1, 2]) - R_C_N) < 0.05


def test_arbitrary_axis_same_bonds():
    p = HelixPrimitive(np.array([1.0, -2.0, 3.0]), unit(np.array([1.0, 1.0, 0.0])), 18.0)
    b = build_backbone_atoms(p)
    for i in range(b.shape[0]):
        n, ca, c, o = b[i]
        assert abs(np.linalg.norm(ca - n) - R_N_CA) < 0.05
    v = b[1, 1] - b[0, 1]
    assert np.linalg.norm(v) > 1.0
