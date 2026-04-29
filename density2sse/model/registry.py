"""YAML-driven model registry."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch.nn as nn

from density2sse.model.baseline_cnn import BaselineHelixCNN
from density2sse.model.detr3d import Detr3DHelix
from density2sse.model.detr3d_multiscale import Detr3DMultiScaleHelix
from density2sse.model.slot_attention3d import SlotAttention3D
from density2sse.model.unet_sethead import UNetSetHead

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "baseline_cnn": BaselineHelixCNN,
    "detr3d": Detr3DHelix,
    "unet_sethead": UNetSetHead,
    "slot_attention3d": SlotAttention3D,
    "detr3d_multiscale": Detr3DMultiScaleHelix,
}


def _arch_get(mcfg: Dict[str, Any], key: str, legacy_key: Optional[str], default: Any) -> Any:
    arch = mcfg.get("arch", {})
    if isinstance(arch, dict) and key in arch:
        return arch[key]
    if legacy_key and legacy_key in mcfg:
        return mcfg[legacy_key]
    return default


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
    if name == "baseline_cnn":
        common["k_embed_dim"] = int(_arch_get(mcfg, "k_embed_dim", None, 32))
        common["mlp_hidden_dim"] = int(_arch_get(mcfg, "mlp_hidden_dim", "hidden_dim", 256))
        common["mlp_num_layers"] = int(_arch_get(mcfg, "mlp_num_layers", None, 2))
        common["mlp_dropout"] = float(_arch_get(mcfg, "mlp_dropout", None, 0.0))
        common["activation"] = str(_arch_get(mcfg, "activation", None, "relu"))
    if name == "detr3d":
        common["d_model"] = int(_arch_get(mcfg, "d_model", "d_model", mcfg.get("hidden_dim", 256)))
        common["nhead"] = int(_arch_get(mcfg, "nhead", "nhead", 8))
        common["num_decoder_layers"] = int(_arch_get(mcfg, "num_decoder_layers", "num_decoder_layers", 2))
        common["dim_feedforward"] = int(_arch_get(mcfg, "dim_feedforward", "dim_feedforward", 512))
        common["transformer_dropout"] = float(_arch_get(mcfg, "transformer_dropout", None, 0.1))
        common["transformer_norm_first"] = bool(_arch_get(mcfg, "transformer_norm_first", None, True))
        common["transformer_activation"] = str(_arch_get(mcfg, "transformer_activation", None, "relu"))
        common["k_embed_mode"] = str(_arch_get(mcfg, "k_embed_mode", None, "add"))
    if name == "unet_sethead":
        common["k_embed_dim"] = int(_arch_get(mcfg, "k_embed_dim", None, 32))
        common["mlp_hidden_dim"] = int(_arch_get(mcfg, "mlp_hidden_dim", "hidden_dim", 256))
        common["mlp_num_layers"] = int(_arch_get(mcfg, "mlp_num_layers", None, 2))
        common["mlp_dropout"] = float(_arch_get(mcfg, "mlp_dropout", None, 0.0))
        common["activation"] = str(_arch_get(mcfg, "activation", None, "relu"))
    if name == "slot_attention3d":
        common["slot_dim"] = int(_arch_get(mcfg, "slot_dim", None, 256))
        common["slot_iters"] = int(_arch_get(mcfg, "slot_iters", None, 3))
        common["token_proj_dim"] = int(_arch_get(mcfg, "token_proj_dim", None, 256))
        common["slot_mlp_hidden"] = int(_arch_get(mcfg, "slot_mlp_hidden", None, 512))
    if name == "detr3d_multiscale":
        common["d_model"] = int(_arch_get(mcfg, "d_model", "d_model", mcfg.get("hidden_dim", 256)))
        common["nhead"] = int(_arch_get(mcfg, "nhead", "nhead", 8))
        common["num_decoder_layers"] = int(_arch_get(mcfg, "num_decoder_layers", "num_decoder_layers", 2))
        common["dim_feedforward"] = int(_arch_get(mcfg, "dim_feedforward", "dim_feedforward", 512))
        common["transformer_dropout"] = float(_arch_get(mcfg, "transformer_dropout", None, 0.1))
        common["transformer_norm_first"] = bool(_arch_get(mcfg, "transformer_norm_first", None, True))
        common["transformer_activation"] = str(_arch_get(mcfg, "transformer_activation", None, "relu"))
        common["k_embed_mode"] = str(_arch_get(mcfg, "k_embed_mode", None, "add"))
        common["multiscale_levels"] = int(_arch_get(mcfg, "multiscale_levels", None, 3))
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
    if name == "baseline_cnn":
        out["k_embed_dim"] = int(_arch_get(mcfg, "k_embed_dim", None, 32))
        out["mlp_hidden_dim"] = int(_arch_get(mcfg, "mlp_hidden_dim", "hidden_dim", 256))
        out["mlp_num_layers"] = int(_arch_get(mcfg, "mlp_num_layers", None, 2))
        out["mlp_dropout"] = float(_arch_get(mcfg, "mlp_dropout", None, 0.0))
        out["activation"] = str(_arch_get(mcfg, "activation", None, "relu"))
    if name == "detr3d":
        out["d_model"] = int(_arch_get(mcfg, "d_model", "d_model", mcfg.get("hidden_dim", 256)))
        out["nhead"] = int(_arch_get(mcfg, "nhead", "nhead", 8))
        out["num_decoder_layers"] = int(_arch_get(mcfg, "num_decoder_layers", "num_decoder_layers", 2))
        out["dim_feedforward"] = int(_arch_get(mcfg, "dim_feedforward", "dim_feedforward", 512))
        out["transformer_dropout"] = float(_arch_get(mcfg, "transformer_dropout", None, 0.1))
        out["transformer_norm_first"] = bool(_arch_get(mcfg, "transformer_norm_first", None, True))
        out["transformer_activation"] = str(_arch_get(mcfg, "transformer_activation", None, "relu"))
        out["k_embed_mode"] = str(_arch_get(mcfg, "k_embed_mode", None, "add"))
    if name == "unet_sethead":
        out["k_embed_dim"] = int(_arch_get(mcfg, "k_embed_dim", None, 32))
        out["mlp_hidden_dim"] = int(_arch_get(mcfg, "mlp_hidden_dim", "hidden_dim", 256))
        out["mlp_num_layers"] = int(_arch_get(mcfg, "mlp_num_layers", None, 2))
        out["mlp_dropout"] = float(_arch_get(mcfg, "mlp_dropout", None, 0.0))
        out["activation"] = str(_arch_get(mcfg, "activation", None, "relu"))
    if name == "slot_attention3d":
        out["slot_dim"] = int(_arch_get(mcfg, "slot_dim", None, 256))
        out["slot_iters"] = int(_arch_get(mcfg, "slot_iters", None, 3))
        out["token_proj_dim"] = int(_arch_get(mcfg, "token_proj_dim", None, 256))
        out["slot_mlp_hidden"] = int(_arch_get(mcfg, "slot_mlp_hidden", None, 512))
    if name == "detr3d_multiscale":
        out["d_model"] = int(_arch_get(mcfg, "d_model", "d_model", mcfg.get("hidden_dim", 256)))
        out["nhead"] = int(_arch_get(mcfg, "nhead", "nhead", 8))
        out["num_decoder_layers"] = int(_arch_get(mcfg, "num_decoder_layers", "num_decoder_layers", 2))
        out["dim_feedforward"] = int(_arch_get(mcfg, "dim_feedforward", "dim_feedforward", 512))
        out["transformer_dropout"] = float(_arch_get(mcfg, "transformer_dropout", None, 0.1))
        out["transformer_norm_first"] = bool(_arch_get(mcfg, "transformer_norm_first", None, True))
        out["transformer_activation"] = str(_arch_get(mcfg, "transformer_activation", None, "relu"))
        out["k_embed_mode"] = str(_arch_get(mcfg, "k_embed_mode", None, "add"))
        out["multiscale_levels"] = int(_arch_get(mcfg, "multiscale_levels", None, 3))
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
            k_embed_dim=int(mc.get("k_embed_dim", 32)),
            mlp_hidden_dim=int(mc.get("mlp_hidden_dim", mc["hidden_dim"])),
            mlp_num_layers=int(mc.get("mlp_num_layers", 2)),
            mlp_dropout=float(mc.get("mlp_dropout", 0.0)),
            activation=str(mc.get("activation", "relu")),
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
            transformer_dropout=float(mc.get("transformer_dropout", 0.1)),
            transformer_norm_first=bool(mc.get("transformer_norm_first", True)),
            transformer_activation=str(mc.get("transformer_activation", "relu")),
            k_embed_mode=str(mc.get("k_embed_mode", "add")),
        )
    if name == "unet_sethead":
        return cls(
            max_K=int(mc["max_K"]),
            box_size=int(mc["box_size"]),
            in_channels=int(mc["in_channels"]),
            base_channels=int(mc["base_channels"]),
            hidden_dim=int(mc["hidden_dim"]),
            box_extent_angstrom=float(mc["box_extent_angstrom"]),
            k_embed_dim=int(mc.get("k_embed_dim", 32)),
            mlp_hidden_dim=int(mc.get("mlp_hidden_dim", mc["hidden_dim"])),
            mlp_num_layers=int(mc.get("mlp_num_layers", 2)),
            mlp_dropout=float(mc.get("mlp_dropout", 0.0)),
            activation=str(mc.get("activation", "relu")),
        )
    if name == "slot_attention3d":
        return cls(
            max_K=int(mc["max_K"]),
            box_size=int(mc["box_size"]),
            in_channels=int(mc["in_channels"]),
            base_channels=int(mc["base_channels"]),
            hidden_dim=int(mc["hidden_dim"]),
            box_extent_angstrom=float(mc["box_extent_angstrom"]),
            slot_dim=int(mc.get("slot_dim", 256)),
            slot_iters=int(mc.get("slot_iters", 3)),
            token_proj_dim=int(mc.get("token_proj_dim", 256)),
            slot_mlp_hidden=int(mc.get("slot_mlp_hidden", 512)),
        )
    if name == "detr3d_multiscale":
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
            transformer_dropout=float(mc.get("transformer_dropout", 0.1)),
            transformer_norm_first=bool(mc.get("transformer_norm_first", True)),
            transformer_activation=str(mc.get("transformer_activation", "relu")),
            k_embed_mode=str(mc.get("k_embed_mode", "add")),
            multiscale_levels=int(mc.get("multiscale_levels", 3)),
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
