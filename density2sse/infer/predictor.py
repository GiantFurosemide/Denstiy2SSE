"""Run model on an MRC mask."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

from density2sse.io import mrc_io
from density2sse.model.baseline_cnn import BaselineHelixCNN


def load_model(
    checkpoint_path: str,
    cfg: Dict[str, Any],
    device: torch.device,
) -> BaselineHelixCNN:
    mcfg = cfg["model"]
    data_cfg = cfg["data"]
    max_K = int(data_cfg["K_max"])
    box = int(data_cfg["box_size"])
    vs = float(data_cfg["voxel_size"])
    box_extent = box * vs
    model = BaselineHelixCNN(
        max_K=max_K,
        box_size=box,
        in_channels=int(mcfg["in_channels"]),
        base_channels=int(mcfg["base_channels"]),
        hidden_dim=int(mcfg["hidden_dim"]),
        box_extent_angstrom=box_extent,
    )
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model


def run_inference(
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    inf = cfg["inference"]
    path_mrc = inf["input_mrc"]
    k = int(inf["K"])
    ckpt = inf["checkpoint"]
    prefix = inf.get("output_prefix", "outputs/infer/run")

    m = mrc_io.read_mrc(path_mrc)
    mask = m.data.astype(np.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.shape}")
    exp = int(cfg["data"]["box_size"])
    if tuple(mask.shape) != (exp, exp, exp):
        raise ValueError(
            f"Mask shape {mask.shape} must match data.box_size cubic grid ({exp},{exp},{exp}) "
            "for the loaded checkpoint architecture."
        )

    model = load_model(ckpt, cfg, device)
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    kvec = torch.tensor([k], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_c, pred_d, pred_l = model(tensor, kvec)

    out: Dict[str, Any] = {
        "K": k,
        "centers": pred_c[0, :k].detach().cpu().numpy(),
        "directions": pred_d[0, :k].detach().cpu().numpy(),
        "lengths": pred_l[0, :k].detach().cpu().numpy(),
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
    return out
