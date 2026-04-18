"""Convert saved NPZ predictions to PDB."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from density2sse.geometry.helix import HelixPrimitive, unit
from density2sse.geometry import helix_builder
from density2sse.io import pdb_io


def export_npz_to_pdb(npz_path: str, out_pdb: str) -> None:
    z = np.load(npz_path)
    k = int(z["K"])
    centers = z["centers"].astype(np.float64)
    directions = z["directions"].astype(np.float64)
    lengths = z["lengths"].astype(np.float64)
    blocks = []
    for i in range(k):
        prim = HelixPrimitive(center=centers[i], direction=unit(directions[i]), length=float(lengths[i]))
        blocks.append(helix_builder.build_backbone_atoms(prim))
    os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
    pdb_io.helices_to_pdb_file(blocks, out_pdb)
