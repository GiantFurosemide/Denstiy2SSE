"""Cylinder / tube voxelization for helix primitives."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

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


def _axis_centers(box_size: int, voxel_size: float) -> np.ndarray:
    half = 0.5 * box_size * voxel_size
    return np.linspace(-half + 0.5 * voxel_size, half - 0.5 * voxel_size, box_size)


def _primitive_index_bounds(
    prim: HelixPrimitive,
    box_size: int,
    voxel_size: float,
    tube_radius: float,
) -> Tuple[int, int, int, int, int, int]:
    """
    Return conservative voxel-index bounds (z0,z1,y0,y1,x0,x1) for one tube.
    """
    half_extent = 0.5 * float(box_size) * float(voxel_size)
    v = unit(prim.direction)
    half_seg = 0.5 * float(prim.length) * v
    c = np.asarray(prim.center, dtype=np.float64)
    p0 = c - half_seg
    p1 = c + half_seg
    lo = np.minimum(p0, p1) - float(tube_radius)
    hi = np.maximum(p0, p1) + float(tube_radius)

    # voxel center coordinate at index i: -half + (i + 0.5) * voxel_size
    z0 = int(np.ceil((lo[0] + half_extent) / voxel_size - 0.5))
    z1 = int(np.floor((hi[0] + half_extent) / voxel_size - 0.5))
    y0 = int(np.ceil((lo[1] + half_extent) / voxel_size - 0.5))
    y1 = int(np.floor((hi[1] + half_extent) / voxel_size - 0.5))
    x0 = int(np.ceil((lo[2] + half_extent) / voxel_size - 0.5))
    x1 = int(np.floor((hi[2] + half_extent) / voxel_size - 0.5))

    z0 = max(0, min(box_size - 1, z0))
    z1 = max(0, min(box_size - 1, z1))
    y0 = max(0, min(box_size - 1, y0))
    y1 = max(0, min(box_size - 1, y1))
    x0 = max(0, min(box_size - 1, x0))
    x1 = max(0, min(box_size - 1, x1))
    return z0, z1, y0, y1, x0, x1


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


def render_helices_binary_sparse(
    primitives: list[HelixPrimitive],
    box_size: int,
    voxel_size: float,
    tube_radius: float = 2.5,
) -> np.ndarray:
    """
    Exact binary rasterization with per-primitive local bounds to reduce full-volume work.
    """
    if not primitives:
        return np.zeros((box_size, box_size, box_size), dtype=np.uint8)
    axis = _axis_centers(box_size, voxel_size)
    occ = np.zeros((box_size, box_size, box_size), dtype=np.uint8)
    r2 = float(tube_radius) ** 2
    for prim in primitives:
        z0, z1, y0, y1, x0, x1 = _primitive_index_bounds(prim, box_size, voxel_size, tube_radius)
        if z1 < z0 or y1 < y0 or x1 < x0:
            continue
        zz, yy, xx = np.meshgrid(axis[z0 : z1 + 1], axis[y0 : y1 + 1], axis[x0 : x1 + 1], indexing="ij")
        pts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        v = unit(prim.direction)
        half = 0.5 * float(prim.length) * v
        c = np.asarray(prim.center, dtype=np.float64)
        p0 = c - half
        p1 = c + half
        d2 = _point_segment_distance_sq(pts, p0, p1)
        local = (d2 <= r2).astype(np.uint8).reshape(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
        occ[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = np.maximum(
            occ[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1],
            local,
        )
    return occ


def render_helices_count_sparse(
    primitives: list[HelixPrimitive],
    box_size: int,
    voxel_size: float,
    tube_radius: float = 2.5,
) -> np.ndarray:
    """
    Exact per-voxel overlap count with local bounds (for clash computation).
    """
    if not primitives:
        return np.zeros((box_size, box_size, box_size), dtype=np.uint16)
    axis = _axis_centers(box_size, voxel_size)
    cnt = np.zeros((box_size, box_size, box_size), dtype=np.uint16)
    r2 = float(tube_radius) ** 2
    for prim in primitives:
        z0, z1, y0, y1, x0, x1 = _primitive_index_bounds(prim, box_size, voxel_size, tube_radius)
        if z1 < z0 or y1 < y0 or x1 < x0:
            continue
        zz, yy, xx = np.meshgrid(axis[z0 : z1 + 1], axis[y0 : y1 + 1], axis[x0 : x1 + 1], indexing="ij")
        pts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        v = unit(prim.direction)
        half = 0.5 * float(prim.length) * v
        c = np.asarray(prim.center, dtype=np.float64)
        p0 = c - half
        p1 = c + half
        d2 = _point_segment_distance_sq(pts, p0, p1)
        local = (d2 <= r2).astype(np.uint16).reshape(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
        cnt[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] += local
    return cnt


def render_helices_count_sparse_torch(
    centers: torch.Tensor,
    directions: torch.Tensor,
    lengths: torch.Tensor,
    box_size: int,
    voxel_size: float,
    tube_radius: float = 2.5,
) -> torch.Tensor:
    """
    Torch backend (CPU/CUDA) for exact per-voxel overlap count with local primitive bounds.
    Inputs shapes: centers (K,3), directions (K,3), lengths (K,).
    """
    dev = centers.device
    dtype = centers.dtype
    if centers.numel() == 0:
        return torch.zeros((box_size, box_size, box_size), dtype=torch.int16, device=dev)
    half_extent = 0.5 * float(box_size) * float(voxel_size)
    axis = torch.linspace(
        -half_extent + 0.5 * voxel_size,
        half_extent - 0.5 * voxel_size,
        box_size,
        device=dev,
        dtype=dtype,
    )
    cnt = torch.zeros((box_size, box_size, box_size), dtype=torch.int16, device=dev)
    r2 = float(tube_radius) ** 2
    eps = 1e-12
    for i in range(centers.shape[0]):
        c = centers[i]
        d = directions[i]
        d = d / (torch.norm(d) + 1e-8)
        half = 0.5 * lengths[i] * d
        p0 = c - half
        p1 = c + half
        lo = torch.minimum(p0, p1) - float(tube_radius)
        hi = torch.maximum(p0, p1) + float(tube_radius)
        z0 = int(torch.ceil((lo[0] + half_extent) / voxel_size - 0.5).item())
        z1 = int(torch.floor((hi[0] + half_extent) / voxel_size - 0.5).item())
        y0 = int(torch.ceil((lo[1] + half_extent) / voxel_size - 0.5).item())
        y1 = int(torch.floor((hi[1] + half_extent) / voxel_size - 0.5).item())
        x0 = int(torch.ceil((lo[2] + half_extent) / voxel_size - 0.5).item())
        x1 = int(torch.floor((hi[2] + half_extent) / voxel_size - 0.5).item())
        z0 = max(0, min(box_size - 1, z0))
        z1 = max(0, min(box_size - 1, z1))
        y0 = max(0, min(box_size - 1, y0))
        y1 = max(0, min(box_size - 1, y1))
        x0 = max(0, min(box_size - 1, x0))
        x1 = max(0, min(box_size - 1, x1))
        if z1 < z0 or y1 < y0 or x1 < x0:
            continue
        zz, yy, xx = torch.meshgrid(axis[z0 : z1 + 1], axis[y0 : y1 + 1], axis[x0 : x1 + 1], indexing="ij")
        pts = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        ab = p1 - p0
        denom = torch.dot(ab, ab) + eps
        t = torch.sum((pts - p0) * ab.reshape(1, 3), dim=-1) / denom
        t = torch.clamp(t, 0.0, 1.0)
        proj = p0.reshape(1, 3) + t.reshape(-1, 1) * ab.reshape(1, 3)
        d2 = torch.sum((pts - proj) ** 2, dim=-1)
        local = (d2 <= r2).reshape(z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1).to(torch.int16)
        cnt[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] += local
    return cnt


def box_extent_angstrom(box_size: int, voxel_size: float) -> float:
    """Physical edge length of the cubic box in Å."""
    return float(box_size) * float(voxel_size)
