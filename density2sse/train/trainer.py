"""Training and validation loops."""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from density2sse.data.dataset import collate_batch
from density2sse.model.baseline_cnn import BaselineHelixCNN
from density2sse.train import losses as loss_mod
from density2sse.utils.logging_utils import setup_logging

LOG = setup_logging(name="density2sse.train")


def _prune_epoch_checkpoints(ckpt_dir: str, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return

    def _epoch_key(p: str) -> int:
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    paths = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")), key=_epoch_key)
    if len(paths) <= keep_last_k:
        return
    for p in paths[:-keep_last_k]:
        try:
            os.remove(p)
        except OSError:
            LOG.warning("Could not remove old epoch checkpoint %s", p)


def _checkpoint_payload(
    model_state: Dict[str, Any],
    epoch: int,
    max_K: int,
    box: int,
    vs: float,
    mcfg: Dict[str, Any],
    box_extent: float,
) -> Dict[str, Any]:
    """Bundle weights plus architecture hyperparameters for robust inference loading."""
    return {
        "model": model_state,
        "epoch": epoch,
        "model_config": {
            "max_K": int(max_K),
            "box_size": int(box),
            "in_channels": int(mcfg["in_channels"]),
            "base_channels": int(mcfg["base_channels"]),
            "hidden_dim": int(mcfg["hidden_dim"]),
            "box_extent_angstrom": float(box_extent),
        },
    }


def train_epoch(
    model: BaselineHelixCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
) -> float:
    model.train()
    total = 0.0
    n = 0
    w = cfg["loss"]
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mask = batch["mask"]
        k = batch["K"]
        optimizer.zero_grad(set_to_none=True)
        pred_c, pred_d, pred_l = model(mask, k)
        loss = loss_mod.batch_helix_loss(
            pred_c,
            pred_d,
            pred_l,
            batch,
            float(w["w_pos"]),
            float(w["w_dir"]),
            float(w["w_len"]),
        )
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: BaselineHelixCNN,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
) -> float:
    model.eval()
    total = 0.0
    n = 0
    w = cfg["loss"]
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mask = batch["mask"]
        k = batch["K"]
        pred_c, pred_d, pred_l = model(mask, k)
        loss = loss_mod.batch_helix_loss(
            pred_c,
            pred_d,
            pred_l,
            batch,
            float(w["w_pos"]),
            float(w["w_dir"]),
            float(w["w_len"]),
        )
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def run_training(
    resolved_cfg: Dict[str, Any],
    train_dir: str,
    val_dir: Optional[str],
    run_dir: str,
    device: torch.device,
) -> None:
    tcfg = resolved_cfg["training"]
    mcfg = resolved_cfg["model"]
    data_cfg = resolved_cfg["data"]
    max_K = int(data_cfg["K_max"])
    box = int(data_cfg["box_size"])
    vs = float(data_cfg["voxel_size"])
    box_extent = box * vs

    from density2sse.data.dataset import HelixNPZDataset

    train_ds = HelixNPZDataset(train_dir, max_K=max_K, box_size=box)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        collate_fn=collate_batch,
    )
    val_loader = None
    if val_dir and os.path.isdir(val_dir) and any(f.endswith(".npz") for f in os.listdir(val_dir)):
        val_ds = HelixNPZDataset(val_dir, max_K=max_K, box_size=box)
        val_loader = DataLoader(
            val_ds,
            batch_size=int(tcfg["batch_size"]),
            shuffle=False,
            num_workers=int(tcfg["num_workers"]),
            collate_fn=collate_batch,
        )

    model = BaselineHelixCNN(
        max_K=max_K,
        box_size=box,
        in_channels=int(mcfg["in_channels"]),
        base_channels=int(mcfg["base_channels"]),
        hidden_dim=int(mcfg["hidden_dim"]),
        box_extent_angstrom=box_extent,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    ex_dir = os.path.join(run_dir, "examples")
    for d in (ckpt_dir, plot_dir, ex_dir):
        os.makedirs(d, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    fields = ["epoch", "train_loss", "val_loss"]

    best_val = float("inf")
    epochs = int(tcfg["num_epochs"])
    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, opt, device, resolved_cfg)
        row: Dict[str, Any] = {"epoch": epoch, "train_loss": tr, "val_loss": ""}
        if val_loader is not None:
            va = validate_epoch(model, val_loader, device, resolved_cfg)
            row["val_loss"] = va
            LOG.info("epoch %s train=%.6f val=%.6f", epoch, tr, va)
            if va < best_val:
                best_val = va
                torch.save(
                    _checkpoint_payload(
                        model.state_dict(),
                        epoch,
                        max_K,
                        box,
                        vs,
                        mcfg,
                        box_extent,
                    ),
                    os.path.join(ckpt_dir, "best.pt"),
                )
        else:
            LOG.info("epoch %s train=%.6f", epoch, tr)
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        torch.save(
            _checkpoint_payload(
                model.state_dict(),
                epoch,
                max_K,
                box,
                vs,
                mcfg,
                box_extent,
            ),
            os.path.join(ckpt_dir, "last.pt"),
        )
        if tcfg.get("save_every_epoch"):
            pattern = str(tcfg.get("checkpoint_pattern", "epoch_{epoch:04d}.pt"))
            ep_name = pattern.format(epoch=epoch)
            torch.save(
                _checkpoint_payload(
                    model.state_dict(),
                    epoch,
                    max_K,
                    box,
                    vs,
                    mcfg,
                    box_extent,
                ),
                os.path.join(ckpt_dir, ep_name),
            )
            keep_k = int(tcfg.get("keep_last_k_epoch_checkpoints", 0))
            _prune_epoch_checkpoints(ckpt_dir, keep_k)

    if val_loader is None:
        shutil.copy(os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt"))
