"""Build full backbone (N, CA, C, O) for canonical helices."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from density2sse.geometry.helix import HelixPrimitive, canonical_ca_positions_local, residue_count_from_length, unit
from density2sse.geometry.peptide_build import build_polyalanine_alpha_helix


def _kabsch(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform (R, t) such that ``p @ R.T + t`` matches ``q`` (rows are points)."""
    pc = p.mean(axis=0)
    qc = q.mean(axis=0)
    x = p - pc
    y = q - qc
    c = x.T @ y
    u, _, vt = np.linalg.svd(c)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt
    t = qc - pc @ r.T
    return r, t


def _rotation_align_z_to_v(v: np.ndarray) -> np.ndarray:
    """Return R such that R @ e_z ~= unit(v)."""
    vz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = unit(v)
    if np.allclose(u, vz):
        return np.eye(3, dtype=np.float64)
    if np.allclose(u, -vz):
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    axis = np.cross(vz, u)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.arccos(np.clip(np.dot(vz, u), -1.0, 1.0))
    # Rodrigues
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)
    return r


def build_backbone_atoms(prim: HelixPrimitive) -> np.ndarray:
    """
    Return atom coordinates (N_res, 4, 3) for N, CA, C, O in Å, chain geometry
    aligned to the primitive axis and center.
    """
    length = float(prim.length)
    n_res = residue_count_from_length(length)
    ca_target = canonical_ca_positions_local(n_res)
    # Internal-coordinate α-helix (φ/ψ, trans ω); avoids resampling artifacts on template grids.
    tmpl = build_polyalanine_alpha_helix(n_res)
    p_ca = tmpl[:, 1, :]
    q_ca = ca_target
    r_fit, t_fit = _kabsch(p_ca, q_ca)
    aligned = tmpl @ r_fit.T + t_fit.reshape(1, 1, 3)
    r_axis = _rotation_align_z_to_v(prim.direction)
    centered = aligned @ r_axis.T
    c = np.asarray(prim.center, dtype=np.float64).reshape(1, 1, 3)
    return centered + c
