"""Run model on an MRC mask."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import torch

from density2sse.geometry.frame import centered_box_corner_origin_angstrom, shift_centered_lab_to_mrc_corner_frame
from density2sse.io import mrc_io
from density2sse.model import registry as model_registry


def _model_from_yaml(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build model from merged YAML (legacy checkpoints without ``model_config``)."""
    return model_registry.build_model(cfg)


def load_model(
    checkpoint_path: str,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load weights and return ``(model, model_config)``.

    If the checkpoint contains ``model_config`` (written during training), the architecture
    is taken from the checkpoint so inference YAML does not need to match exactly.
    """
    ck = torch.load(checkpoint_path, map_location=device)
    if "model_config" in ck and isinstance(ck["model_config"], dict):
        mc = ck["model_config"]
        model = model_registry.build_model_from_checkpoint_config(mc)
    else:
        warnings.warn(
            "Checkpoint has no 'model_config'; building the network from inference YAML. "
            "Those values must match training or loading will fail. Re-train or save checkpoints "
            "with a current density2sse version to embed architecture in the .pt file.",
            UserWarning,
            stacklevel=2,
        )
        model = _model_from_yaml(cfg)
        data_cfg = cfg["data"]
        mcfg = cfg["model"]
        mc = model_registry.model_config_dict_for_checkpoint(cfg)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model, mc


def run_inference(
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    inf = cfg["inference"]
    path_mrc = inf["input_mrc"]
    k = int(inf["K"])
    ckpt = inf["checkpoint"]
    prefix = inf.get("output_prefix", "outputs/infer/run")

    model, mc = load_model(ckpt, cfg, device)
    max_k = int(mc["max_K"])
    if k > max_k:
        raise ValueError(
            f"inference.K={k} is greater than the model's max_K={max_k} (trained slot count). "
            f"Lower inference.K or retrain with a larger data.K_max."
        )

    m = mrc_io.read_mrc(path_mrc)
    mask = m.data.astype(np.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.shape}")
    exp = int(mc["box_size"])
    if tuple(mask.shape) != (exp, exp, exp):
        raise ValueError(
            f"Mask shape {mask.shape} must match the training grid ({exp},{exp},{exp}) "
            f"from the checkpoint (not necessarily inference YAML data.box_size)."
        )

    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    kvec = torch.tensor([k], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_c, pred_d, pred_l = model(tensor, kvec)

    centers = pred_c[0, :k].detach().cpu().numpy()
    shift = shift_centered_lab_to_mrc_corner_frame(
        m.origin_corner_angstrom_zyx,
        (exp, exp, exp),
        m.voxel_size,
    )
    centers_aligned = centers + shift.reshape(1, 3)

    out: Dict[str, Any] = {
        "K": k,
        "centers": centers_aligned,
        "directions": pred_d[0, :k].detach().cpu().numpy(),
        "lengths": pred_l[0, :k].detach().cpu().numpy(),
    }

    o_can = centered_box_corner_origin_angstrom((exp, exp, exp), m.voxel_size)
    half_extent_zyx = tuple(0.5 * exp * float(m.voxel_size[i]) for i in range(3))
    frame_meta = {
        "convention": "centered_box",
        "box_size": exp,
        "voxel_size_angstrom_zyx": list(m.voxel_size),
        "origin_corner_angstrom_zyx": list(m.origin_corner_angstrom_zyx),
        "canonical_corner_angstrom_zyx": list(o_can),
        "half_extent_angstrom_zyx": list(half_extent_zyx),
        "shift_centered_to_mrc_corner_frame_zyx": shift.tolist(),
    }

    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    np.savez_compressed(prefix + "_pred.npz", **{key: val for key, val in out.items()})
    with open(prefix + "_pred.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "K": k,
                "centers": out["centers"].tolist(),
                "directions": out["directions"].tolist(),
                "lengths": out["lengths"].tolist(),
            },
            f,
            indent=2,
        )
    if inf.get("write_frame_json", True):
        with open(prefix + "_frame.json", "w", encoding="utf-8") as f:
            json.dump(frame_meta, f, indent=2)
    return out
