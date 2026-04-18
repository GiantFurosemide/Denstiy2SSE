"""Synthetic helix mask generation with voxel non-overlap."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from density2sse.geometry.helix import HelixPrimitive, unit
from density2sse.geometry import helix_builder
from density2sse.io import mrc_io, pdb_io
from density2sse.render.cylinder_renderer import box_extent_angstrom, render_helices_binary
from density2sse.data import sample_schema as S


@dataclass
class SyntheticConfig:
    num_samples: int
    seed: int
    box_size: int
    voxel_size: float
    K_min: int
    K_max: int
    length_min: float
    length_max: float
    retry_limit: int
    tube_radius: float
    export_mrc: bool
    export_pdb: bool
    num_workers: int = 1


def _random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return (v / norms).astype(np.float64)


def _segment_fits_box(
    center: np.ndarray,
    direction: np.ndarray,
    length: float,
    half: float,
    margin: float,
) -> bool:
    """Check axis segment ± half length stays inside [-half_box+margin, half_box-margin]."""
    v = unit(direction)
    h = 0.5 * float(length)
    p0 = center - h * v
    p1 = center + h * v
    pts = np.stack([p0, p1], axis=0)
    lim = half - margin
    return bool(np.all(pts <= lim) and np.all(pts >= -lim))


def generate_one_sample(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    sample_idx: int,
) -> Tuple[Dict[str, Any], list[HelixPrimitive]]:
    """Return npz field dict and list of helix primitives."""
    box = cfg.box_size
    vs = cfg.voxel_size
    half = 0.5 * box_extent_angstrom(box, vs)
    margin = cfg.tube_radius + 2.0 * vs

    k = int(rng.integers(cfg.K_min, cfg.K_max + 1))
    primitives: List[HelixPrimitive] = []
    combined = np.zeros((box, box, box), dtype=np.uint8)

    for _attempt in range(cfg.retry_limit):
        primitives.clear()
        combined.fill(0)
        failed = False
        for _hi in range(k):
            placed = False
            for _try in range(cfg.retry_limit):
                length = float(rng.uniform(cfg.length_min, cfg.length_max))
                direction = _random_unit_vectors(rng, 1)[0]
                center = rng.uniform(-half + margin, half - margin, size=3).astype(np.float64)
                if not _segment_fits_box(center, direction, length, half, margin):
                    continue
                prim = HelixPrimitive(center=center, direction=direction, length=length)
                new_mask = render_helices_binary([prim], box, vs, tube_radius=cfg.tube_radius)
                if np.any(combined & new_mask):
                    continue
                combined = combined | new_mask
                primitives.append(prim)
                placed = True
                break
            if not placed:
                failed = True
                break
        if not failed and len(primitives) == k:
            break
    else:
        raise RuntimeError(
            f"Could not place {k} non-overlapping helices after {cfg.retry_limit} outer attempts."
        )

    centers = np.stack([p.center for p in primitives], axis=0).astype(np.float32)
    directions = np.stack([unit(p.direction) for p in primitives], axis=0).astype(np.float32)
    lengths = np.array([p.length for p in primitives], dtype=np.float32)
    box_ang = box_extent_angstrom(box, vs)

    sample_id = f"synth_{sample_idx:06d}"
    fields: Dict[str, Any] = {
        S.MASK: combined.astype(np.uint8),
        S.K: np.int32(k),
        S.CENTERS: centers,
        S.DIRECTIONS: directions,
        S.LENGTHS: lengths,
        S.BOX_SIZE_ANGSTROM: np.float32(box_ang),
        S.VOXEL_SIZE_ANGSTROM: np.float32(vs),
        S.SOURCE_TYPE: np.array("synthetic"),
        S.SAMPLE_ID: np.array(sample_id),
    }
    return fields, primitives


def write_sample_npz(path: str, fields: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **fields)


def _split_index(split: str) -> int:
    return {"train": 0, "val": 1, "test": 2}.get(split, 0)


def _sample_rng_seed(base_seed: int, split_idx: int, sample_idx: int) -> int:
    """Deterministic seed per sample (parallel-safe, same results as sequential with this scheme)."""
    return int(base_seed) + split_idx * 1_000_000 + int(sample_idx)


def _write_one_sample_files(
    split: str,
    i: int,
    out_dir: str,
    fields: Dict[str, Any],
    primitives: List[HelixPrimitive],
    cfg: SyntheticConfig,
) -> None:
    base = os.path.join(out_dir, f"{split}_{i:06d}")
    npz_path = base + ".npz"
    meta: Dict[str, Any] = {
        "sample_id": str(fields[S.SAMPLE_ID]),
        "split": split,
        "K": int(fields[S.K]),
        "npz": npz_path,
    }
    if cfg.export_mrc:
        mrc_path = base + ".mrc"
        mrc_io.write_mrc(mrc_path, fields[S.MASK].astype(np.float32), voxel_size=(cfg.voxel_size,) * 3)
        fields[S.MASK_MRC_PATH] = np.array(mrc_path)
        meta["mask_mrc"] = mrc_path
    if cfg.export_pdb:
        blocks = [helix_builder.build_backbone_atoms(p) for p in primitives]
        pdb_path = base + ".pdb"
        pdb_io.helices_to_pdb_file(blocks, pdb_path)
        fields[S.PDB_PATH] = np.array(pdb_path)
        meta["pdb"] = pdb_path
    write_sample_npz(npz_path, fields)
    with open(base + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _synth_worker_payload(args: Tuple[str, int, str, str, Dict[str, Any]]) -> int:
    """
    Top-level worker for multiprocessing (must be picklable).

    Args: ``(split, sample_idx, out_dir, rng_seed_str, cfg_dict)``.
    """
    split, i, out_dir, rng_seed_str, cfg_dict = args
    rng_seed = int(rng_seed_str)
    cfg = SyntheticConfig(**cfg_dict)
    rng = np.random.default_rng(rng_seed)
    fields, primitives = generate_one_sample(rng, cfg, i)
    _write_one_sample_files(split, i, out_dir, fields, primitives, cfg)
    return i


def generate_dataset_split(
    out_dir: str,
    cfg: SyntheticConfig,
    split: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    split_idx = _split_index(split)
    n = cfg.num_samples
    workers = max(1, int(cfg.num_workers))

    if workers <= 1 or n == 0:
        for i in range(n):
            rng = np.random.default_rng(_sample_rng_seed(cfg.seed, split_idx, i))
            fields, primitives = generate_one_sample(rng, cfg, i)
            _write_one_sample_files(split, i, out_dir, fields, primitives, cfg)
        return

    cfg_dict = asdict(cfg)
    max_proc = min(workers, n)
    tasks = []
    for i in range(n):
        seed = _sample_rng_seed(cfg.seed, split_idx, i)
        tasks.append((split, i, out_dir, str(int(seed)), cfg_dict))

    with ProcessPoolExecutor(max_workers=max_proc) as ex:
        futures = [ex.submit(_synth_worker_payload, t) for t in tasks]
        for fut in as_completed(futures):
            fut.result()
