"""Helix primitive and canonical CA sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HelixPrimitive:
    """Helix axis primitive h = (center, direction, length)."""

    center: np.ndarray  # (3,) Å
    direction: np.ndarray  # (3,) unit
    length: float  # Å

    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (p0, p1) along the axis."""
        v = unit(self.direction)
        c = np.asarray(self.center, dtype=np.float64)
        half = 0.5 * float(self.length) * v
        return c - half, c + half


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(x)
    return x / (n + eps)


def canonical_ca_positions_local(n_res: int, rise: float = 1.5, radius: float = 2.3) -> np.ndarray:
    """
    Ideal alpha-helix CA positions in a local frame (+Z is the helix axis), centered in Z.

    Angular step 100° per residue (implementation pack §7).
    """
    if n_res < 1:
        raise ValueError("n_res must be >= 1")
    deg = np.pi / 180.0
    out = np.zeros((n_res, 3), dtype=np.float64)
    for k in range(n_res):
        th = 100.0 * deg * k
        z = (k - 0.5 * (n_res - 1)) * rise
        out[k] = (radius * np.cos(th), radius * np.sin(th), z)
    return out


def residue_count_from_length(length_angstrom: float, rise: float = 1.5) -> int:
    """N_res = max(6, round(L / 1.5)) per spec §7.2."""
    return max(6, int(round(float(length_angstrom) / rise)))
