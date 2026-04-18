"""MRC / MAP read and write via ``mrcfile``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mrcfile
import numpy as np


@dataclass
class MRCData:
    """Binary mask with voxel grid metadata."""

    data: np.ndarray  # (Z, Y, X) convention per spec
    voxel_size: Tuple[float, float, float]  # Å


def read_mrc(path: str) -> MRCData:
    """Load ``.mrc`` / ``.map``; return ZYX array and voxel size in Å when present."""
    m = mrcfile.open(path)
    arr = np.asarray(m.data, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D MRC, got shape {arr.shape}")
    voxel = getattr(m, "voxel_size", None)
    if voxel is not None:
        if hasattr(voxel, "x"):
            vs = (float(voxel.x), float(voxel.y), float(voxel.z))
        else:
            arr = np.ravel(np.asarray(voxel, dtype=np.float64))
            if arr.size >= 3:
                vs = (float(arr[0]), float(arr[1]), float(arr[2]))
            elif arr.size == 1:
                t = float(arr[0])
                vs = (t, t, t)
            else:
                vs = (1.0, 1.0, 1.0)
    else:
        vs = (1.0, 1.0, 1.0)
    return MRCData(data=arr, voxel_size=vs)


def write_mrc(path: str, data: np.ndarray, voxel_size: Tuple[float, float, float] = (1.5, 1.5, 1.5)) -> None:
    """Write binary / float volume as MRC2014 using mrcfile."""
    arr = np.asarray(data, dtype=np.float32)
    out = mrcfile.new(path, data=arr, overwrite=True)
    vs = tuple(float(x) for x in voxel_size)
    out.voxel_size = vs
    out.update_header_from_data()
    out.update_header_stats()
    out.flush()
