"""Training and validation loops."""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from density2sse.data.dataset import collate_batch
from density2sse.model import registry as model_registry
from density2sse.train import losses as loss_mod
from density2sse.train import metrics as metrics_mod
from density2sse.train import viz_export
from density2sse.utils.logging_utils import setup_logging

LOG = setup_logging(name="density2sse.train")

METRICS_FIELDS = [
    "model_name",
    "run_id",
    "epoch",
    "split",
    "center_error",
    "angle_error",
    "length_error",
    "coverage_ratio",
    "clash_voxels",
    "loss_total",
]


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
    resolved_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Bundle weights plus architecture hyperparameters for robust inference loading."""
    mc = model_registry.model_config_dict_for_checkpoint(resolved_cfg)
    return {
        "model": model_state,
        "epoch": epoch,
        "model_config": mc,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mask = batch["mask"]
        k = batch["K"]
        optimizer.zero_grad(set_to_none=True)
        pred_c, pred_d, pred_l = model(mask, k)
        loss = loss_mod.batch_combined_loss(pred_c, pred_d, pred_l, batch, cfg)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mask = batch["mask"]
        k = batch["K"]
        pred_c, pred_d, pred_l = model(mask, k)
        loss = loss_mod.batch_combined_loss(pred_c, pred_d, pred_l, batch, cfg)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def run_training(
    resolved_cfg: Dict[str, Any],
    train_dir: str,
    val_dir: Optional[str],
    run_dir: str,
    device: torch.device,
    run_id: str,
) -> None:
    tcfg = resolved_cfg["training"]
    mcfg = resolved_cfg["model"]
    data_cfg = resolved_cfg["data"]
    model_name = str(mcfg.get("name", "baseline_cnn"))
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

    model = model_registry.build_model(resolved_cfg).to(device)

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
    train_metric_batches = int(tcfg.get("metrics_train_max_batches", 8))

    best_val = float("inf")
    epochs = int(tcfg["num_epochs"])
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device, resolved_cfg)
        LOG.info("epoch %s train_loss=%.6f", epoch, tr_loss)

        tm = metrics_mod.aggregate_metrics_loader(
            model,
            train_loader,
            device,
            resolved_cfg,
            max_batches=train_metric_batches,
        )
        row_train = {
            "model_name": model_name,
            "run_id": run_id,
            "epoch": epoch,
            "split": "train",
            "center_error": tm["center_error"],
            "angle_error": tm["angle_error"],
            "length_error": tm["length_error"],
            "coverage_ratio": tm["coverage_ratio"],
            "clash_voxels": tm["clash_voxels"],
            "loss_total": tm["loss_total"],
        }

        if val_loader is not None:
            va_loss = validate_epoch(model, val_loader, device, resolved_cfg)
            vm = metrics_mod.aggregate_metrics_loader(model, val_loader, device, resolved_cfg, max_batches=None)
            row_val = {
                "model_name": model_name,
                "run_id": run_id,
                "epoch": epoch,
                "split": "val",
                "center_error": vm["center_error"],
                "angle_error": vm["angle_error"],
                "length_error": vm["length_error"],
                "coverage_ratio": vm["coverage_ratio"],
                "clash_voxels": vm["clash_voxels"],
                "loss_total": vm["loss_total"],
            }
            LOG.info(
                "epoch %s val_loss=%.6f center=%.4f angle=%.4f cov=%.4f",
                epoch,
                va_loss,
                vm["center_error"],
                vm["angle_error"],
                vm["coverage_ratio"],
            )
            if vm["loss_total"] < best_val:
                best_val = vm["loss_total"]
                torch.save(
                    _checkpoint_payload(model.state_dict(), epoch, resolved_cfg),
                    os.path.join(ckpt_dir, "best.pt"),
                )
        else:
            row_val = None

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row_train)
            if row_val is not None:
                w.writerow(row_val)

        if val_loader is not None:
            viz_export.save_example_overlays(run_dir, model, val_loader, device, resolved_cfg, epoch, n_examples=2)

        torch.save(
            _checkpoint_payload(model.state_dict(), epoch, resolved_cfg),
            os.path.join(ckpt_dir, "last.pt"),
        )
        if tcfg.get("save_every_epoch", True):
            pattern = str(tcfg.get("checkpoint_pattern", "epoch_{epoch:04d}.pt"))
            ep_name = pattern.format(epoch=epoch)
            torch.save(
                _checkpoint_payload(model.state_dict(), epoch, resolved_cfg),
                os.path.join(ckpt_dir, ep_name),
            )
            keep_k = int(tcfg.get("keep_last_k_epoch_checkpoints", 0))
            _prune_epoch_checkpoints(ckpt_dir, keep_k)

    if val_loader is None:
        shutil.copy(os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt"))
