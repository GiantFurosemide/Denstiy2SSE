"""YAML-driven model registry."""

from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn

from density2sse.model.baseline_cnn import BaselineHelixCNN
from density2sse.model.detr3d import Detr3DHelix

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline_cnn": BaselineHelixCNN,
    "detr3d": Detr3DHelix,
}


def model_kwargs_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build constructor kwargs from merged YAML ``cfg``."""
    mcfg = cfg["model"]
    data_cfg = cfg["data"]
    max_K = int(data_cfg["K_max"])
    box = int(data_cfg["box_size"])
    vs = float(data_cfg["voxel_size"])
    box_extent = box * vs
    name = str(mcfg.get("name", "baseline_cnn"))
    common = {
        "max_K": max_K,
        "box_size": box,
        "in_channels": int(mcfg["in_channels"]),
        "base_channels": int(mcfg["base_channels"]),
        "hidden_dim": int(mcfg["hidden_dim"]),
        "box_extent_angstrom": float(box_extent),
    }
    if name == "detr3d":
        common["d_model"] = int(mcfg.get("d_model", mcfg.get("hidden_dim", 256)))
        common["nhead"] = int(mcfg.get("nhead", 8))
        common["num_decoder_layers"] = int(mcfg.get("num_decoder_layers", 2))
        common["dim_feedforward"] = int(mcfg.get("dim_feedforward", 512))
    return common


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate ``model.name`` from merged config."""
    mcfg = cfg["model"]
    name = str(mcfg.get("name", "baseline_cnn"))
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model.name={name!r}. Choose one of: {sorted(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[name]
    kwargs = model_kwargs_from_config(cfg)
    return cls(**kwargs)


def model_config_dict_for_checkpoint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten architecture fields stored in checkpoints and used at inference."""
    mcfg = cfg["model"]
    data_cfg = cfg["data"]
    max_K = int(data_cfg["K_max"])
    box = int(data_cfg["box_size"])
    vs = float(data_cfg["voxel_size"])
    box_extent = box * vs
    name = str(mcfg.get("name", "baseline_cnn"))
    out: Dict[str, Any] = {
        "model_name": name,
        "max_K": max_K,
        "box_size": box,
        "in_channels": int(mcfg["in_channels"]),
        "base_channels": int(mcfg["base_channels"]),
        "hidden_dim": int(mcfg["hidden_dim"]),
        "box_extent_angstrom": float(box_extent),
    }
    if name == "detr3d":
        out["d_model"] = int(mcfg.get("d_model", mcfg.get("hidden_dim", 256)))
        out["nhead"] = int(mcfg.get("nhead", 8))
        out["num_decoder_layers"] = int(mcfg.get("num_decoder_layers", 2))
        out["dim_feedforward"] = int(mcfg.get("dim_feedforward", 512))
    return out


def build_model_from_checkpoint_config(mc: Dict[str, Any]) -> nn.Module:
    """Rebuild a module from ``model_config`` stored in a ``.pt`` checkpoint."""
    name = str(mc.get("model_name", "baseline_cnn"))  # legacy checkpoints omit model_name → baseline
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Checkpoint model_name={name!r} is not registered.")
    cls = MODEL_REGISTRY[name]
    if name == "baseline_cnn":
        return cls(
            max_K=int(mc["max_K"]),
            box_size=int(mc["box_size"]),
            in_channels=int(mc["in_channels"]),
            base_channels=int(mc["base_channels"]),
            hidden_dim=int(mc["hidden_dim"]),
            box_extent_angstrom=float(mc["box_extent_angstrom"]),
        )
    if name == "detr3d":
        return cls(
            max_K=int(mc["max_K"]),
            box_size=int(mc["box_size"]),
            in_channels=int(mc["in_channels"]),
            base_channels=int(mc["base_channels"]),
            hidden_dim=int(mc["hidden_dim"]),
            box_extent_angstrom=float(mc["box_extent_angstrom"]),
            d_model=int(mc.get("d_model", mc["hidden_dim"])),
            nhead=int(mc.get("nhead", 8)),
            num_decoder_layers=int(mc.get("num_decoder_layers", 2)),
            dim_feedforward=int(mc.get("dim_feedforward", 512)),
        )
    raise RuntimeError(f"Unhandled model_name {name}")


def describe_model(cfg: Dict[str, Any]) -> str:
    """Short text for ``model.txt``."""
    name = str(cfg["model"].get("name", "baseline_cnn"))
    lines = [f"model.name: {name}"]
    if name not in MODEL_REGISTRY:
        lines.append("Unknown model.name (not in MODEL_REGISTRY).")
        return "\n".join(lines) + "\n"
    lines.append("Registered class: " + str(MODEL_REGISTRY[name].__name__))
    lines.append("Constructor kwargs (effective):")
    for k, v in sorted(model_kwargs_from_config(cfg).items()):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"
