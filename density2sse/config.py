"""YAML configuration loading, defaults, validation, and resolved snapshots."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


DEFAULTS: Dict[str, Any] = {
    "project": {"name": "density2sse", "output_dir": "outputs", "seed": 42},
    "mode": "train",
    "io": {"input_mrc": None, "output_dir": "outputs/example", "save_debug": True},
    "data": {
        "dataset_type": "synthetic",
        "train_dir": "data/train",
        "val_dir": "data/val",
        "test_dir": "data/test",
        "voxel_size": 1.5,
        "box_size": 96,
        "threshold": 0.2,
        "K_min": 2,
        "K_max": 5,
    },
    "synthetic": {
        "enabled": True,
        "num_samples_train": 2000,
        "num_samples_val": 200,
        "num_samples_test": 200,
        "retry_limit": 500,
        "renderer": "cylinder",
        "tube_radius": 2.5,
        "length_min": 12.0,
        "length_max": 30.0,
        "export_mrc": False,
        "export_pdb": False,
        "num_workers": 1,
    },
    "model": {
        "name": "baseline_cnn",
        "in_channels": 1,
        "base_channels": 16,
        "hidden_dim": 256,
        "d_model": 256,
        "nhead": 8,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
    },
    "training": {
        "batch_size": 8,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_workers": 0,
        "device": "auto",
        "tiny_overfit": False,
        "tiny_num_samples": 8,
        "save_every_epoch": True,
        "checkpoint_pattern": "epoch_{epoch:04d}.pt",
        "keep_last_k_epoch_checkpoints": 0,
        "metrics_train_max_batches": 8,
    },
    "loss": {
        "w_pos": 1.0,
        "w_dir": 1.0,
        "w_len": 1.0,
        "w_render": 0.0,
        "w_clash": 0.0,
        "w_boundary": 0.0,
    },
    "inference": {
        "K": 3,
        "checkpoint": "outputs/train/example/checkpoints/best.pt",
        "export_pdb": True,
        "input_mrc": None,
        "output_prefix": "outputs/infer/run",
        "write_frame_json": True,
    },
    "export": {
        "input_npz": None,
        "output_pdb": "outputs/export/model.pdb",
    },
    "run": {"stages": ["generate-data", "train"]},
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    return data


def resolve_config(path: str) -> Dict[str, Any]:
    """Load YAML and merge onto defaults."""
    user = load_yaml(path)
    return deep_merge(DEFAULTS, user)


def validate_config(cfg: Dict[str, Any], purpose: str) -> None:
    """Raise ``ValueError`` if required keys for a command are missing."""
    if purpose == "generate-data":
        for k in ("synthetic", "data"):
            if k not in cfg:
                raise ValueError(f"Missing section {k}")
    elif purpose == "train":
        for k in ("training", "model", "data", "loss"):
            if k not in cfg:
                raise ValueError(f"Missing section {k}")
        if cfg["training"].get("tiny_overfit"):
            pass
    elif purpose == "infer":
        inf = cfg.get("inference", {})
        if not inf.get("input_mrc"):
            raise ValueError("inference.input_mrc is required for infer")
        if not inf.get("checkpoint"):
            raise ValueError("inference.checkpoint is required for infer")
    elif purpose == "export":
        ex = cfg.get("export", {})
        if not ex.get("input_npz"):
            raise ValueError("export.input_npz is required for export")


def save_resolved(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
