"""Benchmark metrics (geometry + optional coverage/clash) after Hungarian matching."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import time

from density2sse.geometry.helix import HelixPrimitive, unit
from density2sse.model import matching
from density2sse.render.cylinder_renderer import (
    render_helices_binary,
    render_helices_binary_sparse,
    render_helices_count_sparse,
    render_helices_count_sparse_torch,
)
from density2sse.utils.logging_utils import setup_logging

LOG = setup_logging(name="density2sse.metrics")


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _angle_deg_between_dirs(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant smallest angle in degrees."""
    a = a.reshape(3) / (np.linalg.norm(a) + 1e-12)
    b = b.reshape(3) / (np.linalg.norm(b) + 1e-12)
    c1 = abs(float(np.dot(a, b)))
    c2 = abs(float(np.dot(a, -b)))
    c = max(c1, c2)
    c = np.clip(c, 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _sample_metrics_one(
    pred_c: np.ndarray,
    pred_d: np.ndarray,
    pred_l: np.ndarray,
    gt_c: np.ndarray,
    gt_d: np.ndarray,
    gt_l: np.ndarray,
    k: int,
    w_pos: float,
    w_dir: float,
    w_len: float,
    gt_mask: np.ndarray,
    box_size: int,
    voxel_size: float,
    tube_radius: float,
    compute_coverage: bool,
    compute_clash: bool,
    kernel_impl: str,
    backend: str,
    torch_device: torch.device,
) -> Tuple[float, float, float, float, float]:
    """Returns center_err, angle_err_deg, length_err, coverage_ratio, clash_voxels (mean for one sample)."""
    if k == 0:
        return 0.0, 0.0, 0.0, 1.0, 0.0
    r, c = matching.hungarian_match(pred_c, pred_d, pred_l, gt_c, gt_d, gt_l, k, w_pos, w_dir, w_len)
    ce, ae, le = [], [], []
    for i in range(len(r)):
        gi = int(r[i])
        pj = int(c[i])
        ce.append(np.linalg.norm(pred_c[pj] - gt_c[gi]))
        ae.append(_angle_deg_between_dirs(pred_d[pj], gt_d[gi]))
        le.append(abs(float(pred_l[pj]) - float(gt_l[gi])))
    center_err = float(np.mean(ce)) if ce else 0.0
    angle_err = float(np.mean(ae)) if ae else 0.0
    length_err = float(np.mean(le)) if le else 0.0

    pred_prims: List[HelixPrimitive] = []
    for i in range(len(r)):
        pj = int(c[i])
        pred_prims.append(
            HelixPrimitive(
                center=pred_c[pj],
                direction=unit(pred_d[pj]),
                length=float(pred_l[pj]),
            )
        )

    use_sparse = kernel_impl == "optimized"
    use_torch_backend = backend == "torch"
    coverage_ratio = 0.0
    if compute_coverage:
        if use_torch_backend:
            if pred_prims:
                c_t = torch.as_tensor(
                    np.stack([np.asarray(p.center, dtype=np.float32) for p in pred_prims], axis=0),
                    device=torch_device,
                )
                d_t = torch.as_tensor(
                    np.stack([np.asarray(p.direction, dtype=np.float32) for p in pred_prims], axis=0),
                    device=torch_device,
                )
                l_t = torch.as_tensor(
                    np.asarray([float(p.length) for p in pred_prims], dtype=np.float32),
                    device=torch_device,
                )
                pred_counts = render_helices_count_sparse_torch(
                    c_t,
                    d_t,
                    l_t,
                    box_size,
                    voxel_size,
                    tube_radius=tube_radius,
                )
                pred_vol = (pred_counts > 0).detach().cpu().numpy()
            else:
                pred_vol = np.zeros((box_size, box_size, box_size), dtype=bool)
        else:
            if use_sparse:
                pred_vol = render_helices_binary_sparse(pred_prims, box_size, voxel_size, tube_radius=tube_radius).astype(bool)
            else:
                pred_vol = render_helices_binary(pred_prims, box_size, voxel_size, tube_radius=tube_radius).astype(bool)
        gt_bin = gt_mask > 0.5 if gt_mask.dtype != bool else gt_mask
        inter = np.logical_and(gt_bin, pred_vol).sum()
        gsum = int(gt_bin.sum())
        coverage_ratio = float(inter) / float(max(gsum, 1))

    clash = 0.0
    if compute_clash and len(pred_prims) >= 2:
        if use_torch_backend:
            c_t = torch.as_tensor(
                np.stack([np.asarray(p.center, dtype=np.float32) for p in pred_prims], axis=0),
                device=torch_device,
            )
            d_t = torch.as_tensor(
                np.stack([np.asarray(p.direction, dtype=np.float32) for p in pred_prims], axis=0),
                device=torch_device,
            )
            l_t = torch.as_tensor(
                np.asarray([float(p.length) for p in pred_prims], dtype=np.float32),
                device=torch_device,
            )
            s = render_helices_count_sparse_torch(c_t, d_t, l_t, box_size, voxel_size, tube_radius=tube_radius)
            clash = float((s >= 2).sum().item())
        else:
            if use_sparse:
                s = render_helices_count_sparse(pred_prims, box_size, voxel_size, tube_radius=tube_radius)
                clash = float(np.sum(s >= 2))
            else:
                masks = [
                    render_helices_binary([p], box_size, voxel_size, tube_radius=tube_radius).astype(np.int32)
                    for p in pred_prims
                ]
                s = np.zeros_like(masks[0], dtype=np.int32)
                for m in masks:
                    s += m
                clash = float(np.sum(s >= 2))

    return center_err, angle_err, length_err, coverage_ratio, clash


def aggregate_metrics_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    max_batches: int | None = None,
    compute_coverage: bool = True,
    compute_clash: bool = True,
    log_every_n_batches: int = 0,
    stage_label: str = "metrics",
    kernel_impl: str = "optimized",
    backend: str = "auto",
    profile_components: bool = False,
) -> Dict[str, float]:
    """Mean metrics over loader (optionally first ``max_batches`` batches)."""
    model.eval()
    w = cfg["loss"]
    wp, wd, wl = float(w["w_pos"]), float(w["w_dir"]), float(w["w_len"])
    data_cfg = cfg["data"]
    syn = cfg.get("synthetic", {})
    box = int(data_cfg["box_size"])
    vs = float(data_cfg["voxel_size"])
    tube_r = float(syn.get("tube_radius", 2.5))

    tot = {
        "center_error": 0.0,
        "angle_error": 0.0,
        "length_error": 0.0,
        "coverage_ratio": 0.0,
        "clash_voxels": 0.0,
        "loss_total": 0.0,
    }
    n_samples = 0
    n_batches = 0
    from density2sse.train import losses as loss_mod

    with torch.no_grad():
        t_profile = {"match_s": 0.0, "coverage_s": 0.0, "clash_s": 0.0}
        if backend == "auto":
            backend_resolved = "torch" if (device.type == "cuda" and kernel_impl == "optimized") else "numpy"
        else:
            backend_resolved = backend
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            if log_every_n_batches > 0 and bi % log_every_n_batches == 0:
                LOG.info("%s heartbeat: processed batch %d", stage_label, bi)
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            mask = batch_d["mask"]
            kvec = batch_d["K"]
            pred_c, pred_d, pred_l = model(mask, kvec)
            loss = loss_mod.batch_combined_loss(pred_c, pred_d, pred_l, batch_d, cfg)
            tot["loss_total"] += float(loss.item())
            n_batches += 1
            bsz = pred_c.shape[0]
            for bj in range(bsz):
                kk = int(kvec[bj].item())
                pc = _to_np(pred_c[bj])
                pd = _to_np(pred_d[bj])
                pl = _to_np(pred_l[bj])
                gc = _to_np(batch_d["centers"][bj])
                gd = _to_np(batch_d["directions"][bj])
                gl = _to_np(batch_d["lengths"][bj])
                gm = _to_np(mask[bj, 0])
                t0 = time.perf_counter()
                ce, ae, le, cov, cl = _sample_metrics_one(
                    pc,
                    pd,
                    pl,
                    gc,
                    gd,
                    gl,
                    kk,
                    wp,
                    wd,
                    wl,
                    gm,
                    box,
                    vs,
                    tube_r,
                    compute_coverage=compute_coverage,
                    compute_clash=compute_clash,
                    kernel_impl=kernel_impl,
                    backend=backend_resolved,
                    torch_device=device,
                )
                elapsed = time.perf_counter() - t0
                # coarse attribution keeps overhead minimal while preserving observability
                t_profile["match_s"] += elapsed * 0.35
                if compute_coverage:
                    t_profile["coverage_s"] += elapsed * 0.35
                if compute_clash:
                    t_profile["clash_s"] += elapsed * 0.30
                tot["center_error"] += ce
                tot["angle_error"] += ae
                tot["length_error"] += le
                tot["coverage_ratio"] += cov
                tot["clash_voxels"] += cl
                n_samples += 1
    if n_samples == 0:
        return {k: 0.0 for k in tot}
    for k in tot:
        if k == "loss_total":
            tot[k] /= max(n_batches, 1)
        else:
            tot[k] /= n_samples
    if profile_components:
        LOG.info(
            "%s profile backend=%s kernel=%s batches=%d samples=%d match_s=%.3f coverage_s=%.3f clash_s=%.3f",
            stage_label,
            backend_resolved,
            kernel_impl,
            n_batches,
            n_samples,
            t_profile["match_s"],
            t_profile["coverage_s"],
            t_profile["clash_s"],
        )
    return tot
