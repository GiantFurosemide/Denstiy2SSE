"""Cylinder / tube voxelization for helix primitives."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from density2sse.geometry.helix import HelixPrimitive, unit


def _point_segment_distance_sq(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``points`` (N,3), segment a-b; return squared distances."""
    ab = b - a
    denom = np.dot(ab, ab) + 1e-12
    t = np.dot(points - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a.reshape(1, 3) + t.reshape(-1, 1) * ab.reshape(1, 3)
    d = points - proj
    return np.sum(d * d, axis=1)


def voxel_centers_angstrom(box_size: int, voxel_size: float) -> np.ndarray:
    """Grid voxel centers in Å, origin at box center, shape (D*D*D, 3) with Z,Y,X order flattened."""
    half = 0.5 * box_size * voxel_size
    edges = np.linspace(-half + 0.5 * voxel_size, half - 0.5 * voxel_size, box_size)
    zz, yy, xx = np.meshgrid(edges, edges, edges, indexing="ij")
    pts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    return pts


def render_helices_binary(
    primitives: list[HelixPrimitive],
    box_size: int,
    voxel_size: float,
    tube_radius: float = 2.5,
) -> np.ndarray:
    """
    Rasterize helices as union of cylinders around axis segments.

    Returns binary mask ``(D, D, D)`` with Z,Y,X indexing, dtype ``uint8``.
    """
    pts = voxel_centers_angstrom(box_size, voxel_size)
    occ = np.zeros(len(pts), dtype=np.float32)
    r2 = float(tube_radius) ** 2
    for prim in primitives:
        v = unit(prim.direction)
        half = 0.5 * float(prim.length) * v
        c = np.asarray(prim.center, dtype=np.float64)
        p0 = c - half
        p1 = c + half
        d2 = _point_segment_distance_sq(pts, p0, p1)
        occ = np.maximum(occ, (d2 <= r2).astype(np.float32))
    mask = occ.reshape(box_size, box_size, box_size)
    return (mask > 0.5).astype(np.uint8)


def box_extent_angstrom(box_size: int, voxel_size: float) -> float:
    """Physical edge length of the cubic box in Å."""
    return float(box_size) * float(voxel_size)
