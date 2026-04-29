"""Save 2D slice overlays: mask, GT helix tubes, predicted tubes."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from density2sse.geometry.helix import HelixPrimitive, unit
from density2sse.render.cylinder_renderer import render_helices_binary


def _primitives_from_batch(
    centers: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    k: int,
) -> List[HelixPrimitive]:
    out: List[HelixPrimitive] = []
    for i in range(k):
        out.append(
            HelixPrimitive(
                center=centers[i],
                direction=unit(directions[i]),
                length=float(lengths[i]),
            )
        )
    return out


@torch.no_grad()
def save_example_overlays(
    run_dir: str,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    n_examples: int = 2,
) -> None:
    """Write ``example_*_overlay.png`` (central Z slice) for the first ``n_examples`` val samples."""
    if loader is None:
        return
    data_cfg = cfg["data"]
    syn = cfg.get("synthetic", {})
    box = int(data_cfg["box_size"])
    vs = float(data_cfg["voxel_size"])
    tube_r = float(syn.get("tube_radius", 2.5))
    model.eval()
    it = iter(loader)
    for ex in range(n_examples):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mask = batch["mask"]
        kvec = batch["K"]
        pred_c, pred_d, pred_l = model(mask, kvec)
        bj = 0
        kk = int(kvec[bj].item())
        gm = mask[bj, 0].detach().cpu().numpy()
        gc = batch["centers"][bj].detach().cpu().numpy()
        gd = batch["directions"][bj].detach().cpu().numpy()
        gl = batch["lengths"][bj].detach().cpu().numpy()
        pc = pred_c[bj].detach().cpu().numpy()
        pd = pred_d[bj].detach().cpu().numpy()
        pl = pred_l[bj].detach().cpu().numpy()
        gt_prims = _primitives_from_batch(gc, gd, gl, kk)
        pred_prims = _primitives_from_batch(pc, pd, pl, kk)
        gt_r = render_helices_binary(gt_prims, box, vs, tube_r)
        pr_r = render_helices_binary(pred_prims, box, vs, tube_r)
        z = box // 2
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(gm[z], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("mask slice")
        axes[0].axis("off")
        axes[1].imshow(gt_r[z], cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title("GT helices")
        axes[1].axis("off")
        axes[2].imshow(pr_r[z], cmap="Blues", vmin=0, vmax=1)
        axes[2].set_title("pred helices")
        axes[2].axis("off")
        plt.suptitle(f"epoch {epoch} example {ex} (z={z})")
        plt.tight_layout()
        path = os.path.join(run_dir, f"example_{ex}_overlay.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
