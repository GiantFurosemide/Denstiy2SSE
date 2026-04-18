"""Ideal poly-Alanine α-helix backbone via Biopython internal coordinates (PIC/IC)."""

from __future__ import annotations

import numpy as np
from Bio.PDB.PICIO import read_PIC_seq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Engh & Huber–style typical targets (used by tests; Biopython IC will be close).
R_N_CA = 1.458
R_CA_C = 1.523
R_C_O = 1.229
R_C_N = 1.329


def build_polyalanine_alpha_helix(
    n_res: int,
    phi_deg: float = -57.0,
    psi_deg: float = -47.0,
    omega_deg: float = 180.0,
) -> np.ndarray:
    """
    Backbone (N, CA, C, O) per residue, shape ``(n_res, 4, 3)``, in Å.

    Builds default internal coordinates from sequence, sets φ/ψ/ω to α-helix
    values (same workflow as ``examples/Biopython_helix.py``), then rebuilds
    Cartesian coordinates.
    """
    if n_res < 1:
        raise ValueError("n_res must be >= 1")

    seq = "A" * n_res
    structure = read_PIC_seq(
        SeqRecord(Seq(seq), id="AHLX", description="idealized alpha helix"),
    )
    chain = structure[0]["A"]
    for residue in chain:
        ric = residue.internal_coord
        if ric is None:
            continue
        if ric.get_angle("phi") is not None:
            ric.set_angle("phi", phi_deg)
        if ric.get_angle("psi") is not None:
            ric.set_angle("psi", psi_deg)
        if ric.get_angle("omega") is not None:
            ric.set_angle("omega", omega_deg)

    structure.internal_to_atom_coordinates()

    out = np.zeros((n_res, 4, 3), dtype=np.float64)
    backbone = ("N", "CA", "C", "O")
    for i, residue in enumerate(chain):
        for j, name in enumerate(backbone):
            if name not in residue:
                raise RuntimeError(f"missing backbone atom {name} at residue index {i}")
            out[i, j] = np.asarray(residue[name].coord, dtype=np.float64)
    return out
