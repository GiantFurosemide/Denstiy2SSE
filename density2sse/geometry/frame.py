"""Spatial frame helpers: training/inference use a box-centered Å lab frame."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def centered_box_corner_origin_angstrom(
    shape_zyx: Tuple[int, int, int],
    voxel_size_angstrom_zyx: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Lower corner of voxel [0,0,0] in Å for the same convention as ``voxel_centers_angstrom``:
    voxel centers lie at ``origin + (i + 0.5) * voxel`` per axis (Z, Y, X order).
    """
    nz, ny, nx = shape_zyx
    vz, vy, vx = voxel_size_angstrom_zyx
    return (-0.5 * nz * vz, -0.5 * ny * vy, -0.5 * nx * vx)


def shift_centered_lab_to_mrc_corner_frame(
    origin_corner_angstrom_zyx: Tuple[float, float, float],
    shape_zyx: Tuple[int, int, int],
    voxel_size_angstrom_zyx: Tuple[float, float, float],
) -> np.ndarray:
    """
    Vector to add to coordinates expressed in the centered lab frame (model outputs) so that
    they match the physical frame implied by an MRC header whose ``origin`` is the lower
    corner of voxel ``[0,0,0]`` in Å (Z,Y,X).
    """
    o_can = centered_box_corner_origin_angstrom(shape_zyx, voxel_size_angstrom_zyx)
    o = np.asarray(origin_corner_angstrom_zyx, dtype=np.float64)
    c = np.asarray(o_can, dtype=np.float64)
    return o - c
