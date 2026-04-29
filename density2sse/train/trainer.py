"""Training and validation loops."""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
import time
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
    optimizer_state: Optional[Dict[str, Any]] = None,
    best_val: Optional[float] = None,
) -> Dict[str, Any]:
    """Bundle weights plus architecture hyperparameters for robust inference loading."""
    mc = model_registry.model_config_dict_for_checkpoint(resolved_cfg)
    return {
        "model": model_state,
        "epoch": epoch,
        "model_config": mc,
        "optimizer": optimizer_state,
        "best_val": best_val,
    }


def _state_dict_for_saving(model: nn.Module) -> Dict[str, Any]:
    """Return a clean state_dict even when wrapped in DataParallel."""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _model_for_loading(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _load_model_state(model: nn.Module, state: Dict[str, Any], strict: bool) -> None:
    target = _model_for_loading(model)
    try:
        target.load_state_dict(state, strict=strict)
        return
    except RuntimeError:
        # Compatibility bridge for module.-prefixed checkpoints.
        if any(k.startswith("module.") for k in state.keys()):
            stripped = {k[len("module.") :]: v for k, v in state.items()}
            target.load_state_dict(stripped, strict=strict)
            return
        prefixed = {"module." + k: v for k, v in state.items()}
        target.load_state_dict(prefixed, strict=strict)


def _apply_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    resolved_cfg: Dict[str, Any],
) -> tuple[int, float]:
    """
    Returns (start_epoch, best_val).
    """
    resume = resolved_cfg.get("training", {}).get("resume", {})
    if not bool(resume.get("enabled", False)):
        return 1, float("inf")
    ckpt_path = str(resume.get("checkpoint") or "").strip()
    if not ckpt_path:
        raise ValueError("training.resume.enabled=true requires training.resume.checkpoint")
    mode = str(resume.get("mode", "weights_only")).strip().lower()
    strict_load = bool(resume.get("strict_load", True))
    reset_lr = bool(resume.get("reset_lr", False))
    if mode not in {"weights_only", "full_resume"}:
        raise ValueError("training.resume.mode must be 'weights_only' or 'full_resume'")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model': {ckpt_path}")
    _load_model_state(model, ckpt["model"], strict=strict_load)
    if mode == "weights_only":
        LOG.info("Resume mode=weights_only loaded model from %s; start from epoch 1", ckpt_path)
        return 1, float("inf")

    # full_resume
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    opt_state = ckpt.get("optimizer", None)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
        if reset_lr:
            tcfg = resolved_cfg["training"]
            target_lr = float(tcfg["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = target_lr
    LOG.info(
        "Resume mode=full_resume loaded %s; start_epoch=%d best_val=%.6f optimizer_restored=%s",
        ckpt_path,
        start_epoch,
        best_val,
        opt_state is not None,
    )
    return start_epoch, best_val


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
    # DataParallel is CUDA only (PyTorch does not support multi-GPU MPS the same way).
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            # Use all visible GPUs on cluster nodes by default.
            model = nn.DataParallel(model)
            LOG.info("Using DataParallel on %d GPUs", gpu_count)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    start_epoch, best_val = _apply_resume(model, opt, device, resolved_cfg)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    ex_dir = os.path.join(run_dir, "examples")
    for d in (ckpt_dir, plot_dir, ex_dir):
        os.makedirs(d, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    train_metric_batches = int(tcfg.get("metrics_train_max_batches", 8))
    metrics_every_n_epochs = max(1, int(tcfg.get("metrics_every_n_epochs", 1)))
    val_metric_batches_raw = tcfg.get("val_metrics_max_batches", None)
    val_metric_batches = None if val_metric_batches_raw in (None, "", "none") else int(val_metric_batches_raw)
    metrics_compute_coverage = bool(tcfg.get("metrics_compute_coverage", True))
    metrics_compute_clash = bool(tcfg.get("metrics_compute_clash", True))
    metrics_heartbeat_batches = max(0, int(tcfg.get("metrics_log_every_n_batches", 0)))
    metrics_kernel_impl = str(tcfg.get("metrics_kernel_impl", "optimized")).strip().lower()
    metrics_backend = str(tcfg.get("metrics_backend", "auto")).strip().lower()
    metrics_profile_components = bool(tcfg.get("metrics_profile_components", False))
    viz_enabled = bool(tcfg.get("viz_enabled", True))
    viz_every_n_epochs = max(1, int(tcfg.get("viz_every_n_epochs", 1)))
    viz_n_examples = max(1, int(tcfg.get("viz_n_examples", 2)))
    metrics_target_seconds = float(tcfg.get("metrics_target_seconds", 0.0))
    adaptive_metrics_schedule = bool(tcfg.get("adaptive_metrics_schedule", False))
    final_exact_eval = bool(tcfg.get("final_exact_eval", True))
    dynamic_metrics_every = metrics_every_n_epochs

    epochs = int(tcfg["num_epochs"])
    if start_epoch > epochs:
        LOG.warning("Resume start_epoch=%d is greater than num_epochs=%d; no training iterations will run.", start_epoch, epochs)
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.perf_counter()
        do_epoch_metrics = (epoch % dynamic_metrics_every) == 0
        LOG.info("epoch %s/%s start", epoch, epochs)

        t0 = time.perf_counter()
        tr_loss = train_epoch(model, train_loader, opt, device, resolved_cfg)
        train_dt = time.perf_counter() - t0
        LOG.info("epoch %s stage=train_epoch done in %.2fs train_loss=%.6f", epoch, train_dt, tr_loss)

        if do_epoch_metrics:
            t0 = time.perf_counter()
            tm = metrics_mod.aggregate_metrics_loader(
                model,
                train_loader,
                device,
                resolved_cfg,
                max_batches=train_metric_batches,
                compute_coverage=metrics_compute_coverage,
                compute_clash=metrics_compute_clash,
                log_every_n_batches=metrics_heartbeat_batches,
                stage_label=f"epoch={epoch} split=train",
                kernel_impl=metrics_kernel_impl,
                backend=metrics_backend,
                profile_components=metrics_profile_components,
            )
            train_metrics_dt = time.perf_counter() - t0
            LOG.info("epoch %s stage=train_metrics done in %.2fs", epoch, train_metrics_dt)
        else:
            tm = {
                "center_error": 0.0,
                "angle_error": 0.0,
                "length_error": 0.0,
                "coverage_ratio": 0.0,
                "clash_voxels": 0.0,
                "loss_total": tr_loss,
            }
            LOG.info("epoch %s stage=train_metrics skipped (metrics_every_n_epochs=%d)", epoch, metrics_every_n_epochs)

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
            t0 = time.perf_counter()
            va_loss = validate_epoch(model, val_loader, device, resolved_cfg)
            val_dt = time.perf_counter() - t0
            LOG.info("epoch %s stage=validate_epoch done in %.2fs val_loss=%.6f", epoch, val_dt, va_loss)

            if do_epoch_metrics:
                t0 = time.perf_counter()
                vm = metrics_mod.aggregate_metrics_loader(
                    model,
                    val_loader,
                    device,
                    resolved_cfg,
                    max_batches=val_metric_batches,
                    compute_coverage=metrics_compute_coverage,
                    compute_clash=metrics_compute_clash,
                    log_every_n_batches=metrics_heartbeat_batches,
                    stage_label=f"epoch={epoch} split=val",
                    kernel_impl=metrics_kernel_impl,
                    backend=metrics_backend,
                    profile_components=metrics_profile_components,
                )
                val_metrics_dt = time.perf_counter() - t0
                LOG.info("epoch %s stage=val_metrics done in %.2fs", epoch, val_metrics_dt)
            else:
                vm = {
                    "center_error": 0.0,
                    "angle_error": 0.0,
                    "length_error": 0.0,
                    "coverage_ratio": 0.0,
                    "clash_voxels": 0.0,
                    "loss_total": va_loss,
                }
                LOG.info(
                    "epoch %s stage=val_metrics skipped (metrics_every_n_epochs=%d)",
                    epoch,
                    metrics_every_n_epochs,
                )

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
                t0 = time.perf_counter()
                torch.save(
                    _checkpoint_payload(
                        _state_dict_for_saving(model),
                        epoch,
                        resolved_cfg,
                        optimizer_state=opt.state_dict(),
                        best_val=best_val,
                    ),
                    os.path.join(ckpt_dir, "best.pt"),
                )
                LOG.info("epoch %s stage=save_best done in %.2fs", epoch, time.perf_counter() - t0)
        else:
            row_val = None

        t0 = time.perf_counter()
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row_train)
            if row_val is not None:
                w.writerow(row_val)
        LOG.info("epoch %s stage=write_metrics_csv done in %.2fs", epoch, time.perf_counter() - t0)

        if val_loader is not None and viz_enabled and (epoch % viz_every_n_epochs == 0):
            t0 = time.perf_counter()
            viz_export.save_example_overlays(
                run_dir,
                model,
                val_loader,
                device,
                resolved_cfg,
                epoch,
                n_examples=viz_n_examples,
            )
            LOG.info("epoch %s stage=viz_export done in %.2fs", epoch, time.perf_counter() - t0)
        elif val_loader is not None:
            LOG.info(
                "epoch %s stage=viz_export skipped (viz_enabled=%s, viz_every_n_epochs=%d)",
                epoch,
                viz_enabled,
                viz_every_n_epochs,
            )

        t0 = time.perf_counter()
        torch.save(
            _checkpoint_payload(
                _state_dict_for_saving(model),
                epoch,
                resolved_cfg,
                optimizer_state=opt.state_dict(),
                best_val=best_val,
            ),
            os.path.join(ckpt_dir, "last.pt"),
        )
        LOG.info("epoch %s stage=save_last done in %.2fs", epoch, time.perf_counter() - t0)
        if tcfg.get("save_every_epoch", True):
            pattern = str(tcfg.get("checkpoint_pattern", "epoch_{epoch:04d}.pt"))
            ep_name = pattern.format(epoch=epoch)
            t0 = time.perf_counter()
            torch.save(
                _checkpoint_payload(
                    _state_dict_for_saving(model),
                    epoch,
                    resolved_cfg,
                    optimizer_state=opt.state_dict(),
                    best_val=best_val,
                ),
                os.path.join(ckpt_dir, ep_name),
            )
            keep_k = int(tcfg.get("keep_last_k_epoch_checkpoints", 0))
            _prune_epoch_checkpoints(ckpt_dir, keep_k)
            LOG.info("epoch %s stage=save_epoch done in %.2fs", epoch, time.perf_counter() - t0)

        if adaptive_metrics_schedule and metrics_target_seconds > 0 and val_loader is not None and do_epoch_metrics:
            observed = 0.0
            if "train_metrics_dt" in locals():
                observed += float(train_metrics_dt)
            if "val_metrics_dt" in locals():
                observed += float(val_metrics_dt)
            if observed > metrics_target_seconds and dynamic_metrics_every < max(16, metrics_every_n_epochs):
                dynamic_metrics_every = min(dynamic_metrics_every * 2, max(16, metrics_every_n_epochs))
                LOG.info(
                    "epoch %s adaptive metrics schedule: observed %.2fs > target %.2fs, metrics_every_n_epochs -> %d",
                    epoch,
                    observed,
                    metrics_target_seconds,
                    dynamic_metrics_every,
                )
            elif observed < metrics_target_seconds * 0.5 and dynamic_metrics_every > metrics_every_n_epochs:
                dynamic_metrics_every = max(metrics_every_n_epochs, dynamic_metrics_every // 2)
                LOG.info(
                    "epoch %s adaptive metrics schedule: observed %.2fs < target %.2fs, metrics_every_n_epochs -> %d",
                    epoch,
                    observed,
                    metrics_target_seconds,
                    dynamic_metrics_every,
                )
        LOG.info("epoch %s total_time=%.2fs", epoch, time.perf_counter() - epoch_t0)

    # Final exact pass keeps benchmark semantics while allowing faster in-epoch schedules.
    if final_exact_eval and val_loader is not None:
        LOG.info("final_exact_eval start")
        t0 = time.perf_counter()
        vm_final = metrics_mod.aggregate_metrics_loader(
            model,
            val_loader,
            device,
            resolved_cfg,
            max_batches=None,
            compute_coverage=True,
            compute_clash=True,
            log_every_n_batches=metrics_heartbeat_batches,
            stage_label="final_exact_eval split=val",
            kernel_impl=metrics_kernel_impl,
            backend=metrics_backend,
            profile_components=metrics_profile_components,
        )
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            row_final = {
                "model_name": model_name,
                "run_id": run_id,
                "epoch": epochs,
                "split": "val",
                "center_error": vm_final["center_error"],
                "angle_error": vm_final["angle_error"],
                "length_error": vm_final["length_error"],
                "coverage_ratio": vm_final["coverage_ratio"],
                "clash_voxels": vm_final["clash_voxels"],
                "loss_total": vm_final["loss_total"],
            }
            w.writerow(row_final)
        LOG.info("final_exact_eval done in %.2fs", time.perf_counter() - t0)

    if val_loader is None:
        shutil.copy(os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt"))
