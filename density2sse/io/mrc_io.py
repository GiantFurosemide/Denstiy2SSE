"""MRC / MAP read and write via ``mrcfile``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import mrcfile
import numpy as np

from density2sse.geometry.frame import centered_box_corner_origin_angstrom


@dataclass
class MRCData:
    """Binary mask with voxel grid metadata."""

    data: np.ndarray  # (Z, Y, X) convention per spec
    voxel_size: Tuple[float, float, float]  # Å per axis (Z, Y, X)
    # Lower corner of voxel [0,0,0] in Å (Z, Y, X); matches header origin when present.
    origin_corner_angstrom_zyx: Tuple[float, float, float]


def _voxel_size_from_mrc(m: Any) -> Tuple[float, float, float]:
    voxel = getattr(m, "voxel_size", None)
    if voxel is not None:
        if hasattr(voxel, "x"):
            return (float(voxel.x), float(voxel.y), float(voxel.z))
        arr = np.ravel(np.asarray(voxel, dtype=np.float64))
        if arr.size >= 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
        if arr.size == 1:
            t = float(arr[0])
            return (t, t, t)
    return (1.0, 1.0, 1.0)


def _origin_corner_zyx_from_header(m: Any) -> Tuple[float, float, float]:
    h = m.header
    # mrcfile: origin.x/y/z are column/row/section in Å for CCP4-style maps.
    # numpy array index order is (section=z, row=y, col=x).
    return (float(h.origin.z), float(h.origin.y), float(h.origin.x))


def read_mrc(path: str) -> MRCData:
    """Load ``.mrc`` / ``.map``; return ZYX array, voxel size (Å), and corner origin (Z,Y,X)."""
    m = mrcfile.open(path)
    arr = np.asarray(m.data, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D MRC, got shape {arr.shape}")
    vs = _voxel_size_from_mrc(m)
    oz, oy, ox = _origin_corner_zyx_from_header(m)
    return MRCData(data=arr, voxel_size=vs, origin_corner_angstrom_zyx=(oz, oy, ox))


def write_mrc(
    path: str,
    data: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    *,
    origin_corner_angstrom_zyx: Tuple[float, float, float] | None = None,
    convention: str = "centered",
) -> Tuple[float, float, float]:
    """
    Write binary / float volume as MRC2014 using mrcfile.

    ``convention="centered"`` (default): set header origin to the lower corner of voxel
    ``[0,0,0]`` so that voxel centers match ``density2sse.render.cylinder_renderer.voxel_centers_angstrom``
    for the same ``box_size`` and ``voxel_size``.

    Returns the origin corner (Z, Y, X) in Å written to the header.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {arr.shape}")
    nz, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    vs = tuple(float(x) for x in voxel_size)
    if len(vs) != 3:
        raise ValueError("voxel_size must have length 3 (Z,Y,X)")

    if convention not in ("centered", "none"):
        raise ValueError("convention must be 'centered' or 'none'")

    if convention == "centered":
        oz, oy, ox = centered_box_corner_origin_angstrom((nz, ny, nx), (vs[0], vs[1], vs[2]))
    elif origin_corner_angstrom_zyx is not None:
        oz, oy, ox = (
            float(origin_corner_angstrom_zyx[0]),
            float(origin_corner_angstrom_zyx[1]),
            float(origin_corner_angstrom_zyx[2]),
        )
    else:
        oz, oy, ox = (0.0, 0.0, 0.0)

    out = mrcfile.new(path, data=arr, overwrite=True)
    out.voxel_size = vs
    out.update_header_from_data()
    h = out.header
    h.origin.z = oz
    h.origin.y = oy
    h.origin.x = ox
    out.update_header_stats()
    out.flush()
    return (oz, oy, ox)
