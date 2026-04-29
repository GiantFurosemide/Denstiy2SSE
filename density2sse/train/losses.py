"""Losses with Hungarian matching and optional render / clash / boundary terms."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from density2sse.model import matching


def helix_loss_sample(
    pred_c: torch.Tensor,
    pred_d: torch.Tensor,
    pred_l: torch.Tensor,
    gt_c: torch.Tensor,
    gt_d: torch.Tensor,
    gt_l: torch.Tensor,
    k: int,
    w_pos: float = 1.0,
    w_dir: float = 1.0,
    w_len: float = 1.0,
) -> torch.Tensor:
    """Single-sample loss after Hungarian match between ``k`` GT and ``max_K`` predictions."""
    if k == 0:
        return pred_c.sum() * 0.0
    r, c = matching.hungarian_match(pred_c, pred_d, pred_l, gt_c, gt_d, gt_l, k, w_pos, w_dir, w_len)
    loss = pred_c.new_tensor(0.0)
    for i in range(len(r)):
        gi = int(r[i])
        pj = int(c[i])
        loss = loss + w_pos * F.mse_loss(pred_c[pj], gt_c[gi])
        cos = torch.sum(pred_d[pj] * gt_d[gi]).clamp(-1.0, 1.0)
        alt = torch.sum(pred_d[pj] * (-gt_d[gi])).clamp(-1.0, 1.0)
        cos_m = torch.maximum(torch.abs(cos), torch.abs(alt))
        loss = loss + w_dir * (1.0 - cos_m)
        loss = loss + w_len * F.l1_loss(pred_l[pj], gt_l[gi])
    return loss / float(len(r))


def batch_helix_loss(
    pred_c: torch.Tensor,
    pred_d: torch.Tensor,
    pred_l: torch.Tensor,
    batch: dict,
    w_pos: float,
    w_dir: float,
    w_len: float,
) -> torch.Tensor:
    """``pred_*`` shape ``(B, max_K, ...)``."""
    b = pred_c.shape[0]
    total = pred_c.new_tensor(0.0)
    for bi in range(b):
        k = int(batch["K"][bi].item())
        total = total + helix_loss_sample(
            pred_c[bi],
            pred_d[bi],
            pred_l[bi],
            batch["centers"][bi],
            batch["directions"][bi],
            batch["lengths"][bi],
            k,
            w_pos,
            w_dir,
            w_len,
        )
    return total / float(b)


def _soft_gaussian_centers_loss(
    pred_c: torch.Tensor,
    mask: torch.Tensor,
    half_extent: float,
    grid_d: int = 16,
) -> torch.Tensor:
    """
    Differentiable proxy: downsample GT mask and compare to sum of Gaussians at predicted centers.
    ``pred_c`` (B, K, 3) in [-half, half]; ``mask`` (B, 1, D, D, D).
    """
    b, _, d, h, w = mask.shape
    m = F.adaptive_avg_pool3d(mask, grid_d)
    loss = pred_c.new_tensor(0.0)
    for bi in range(b):
        acc = pred_c.new_zeros(1, grid_d, grid_d, grid_d)
        # voxel grid in [-half,half]
        lin = torch.linspace(-half_extent + half_extent / grid_d, half_extent - half_extent / grid_d, grid_d, device=pred_c.device)
        zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        sigma = (half_extent / float(grid_d)) * 2.0
        for j in range(pred_c.shape[1]):
            c = pred_c[bi, j]
            d2 = torch.sum((pts - c) ** 2, dim=-1)
            acc = acc + torch.exp(-d2 / (sigma * sigma + 1e-8)).reshape(1, grid_d, grid_d, grid_d)
        acc = torch.clamp(acc, 0.0, 3.0) / 3.0
        loss = loss + F.mse_loss(acc, m[bi : bi + 1])
    return loss / float(b)


def _clash_loss(pred_c: torch.Tensor, pred_d: torch.Tensor, pred_l: torch.Tensor, kvec: torch.Tensor) -> torch.Tensor:
    """Penalty when predicted helix axes are closer than ~5 Å (tube overlap proxy)."""
    b, max_k, _ = pred_c.shape
    loss = pred_c.new_tensor(0.0)
    n = 0
    margin = 5.0
    for bi in range(b):
        kk = int(kvec[bi].item())
        if kk < 2:
            continue
        for i in range(kk):
            for j in range(i + 1, kk):
                a0, a1 = _segment_endpoints(pred_c[bi, i], pred_d[bi, i], pred_l[bi, i])
                b0, b1 = _segment_endpoints(pred_c[bi, j], pred_d[bi, j], pred_l[bi, j])
                d = _segment_segment_distance(a0, a1, b0, b1)
                loss = loss + F.relu(margin - d) ** 2
                n += 1
    if n == 0:
        return loss
    return loss / float(n)


def _segment_endpoints(c: torch.Tensor, d: torch.Tensor, length: torch.Tensor) -> tuple:
    u = d / (d.norm() + 1e-8)
    half = 0.5 * length * u
    return c - half, c + half


def _segment_segment_distance(a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """Minimal distance between two segments (differentiable)."""
    u = a1 - a0
    v = b1 - b0
    w0 = a0 - b0
    a = torch.dot(u, u)
    b = torch.dot(u, v)
    c = torch.dot(v, v)
    d = torch.dot(u, w0)
    e = torch.dot(v, w0)
    den = a * c - b * b + 1e-12
    sc = torch.clamp((b * e - c * d) / den, 0.0, 1.0)
    tc = torch.clamp((a * e - b * d) / den, 0.0, 1.0)
    p = a0 + sc * u
    q = b0 + tc * v
    return torch.norm(p - q)


def _boundary_loss(
    pred_c: torch.Tensor,
    pred_d: torch.Tensor,
    pred_l: torch.Tensor,
    mask: torch.Tensor,
    kvec: torch.Tensor,
    box_extent_angstrom: float,
) -> torch.Tensor:
    """Penalty if helix endpoints fall in low-density mask regions (nearest-voxel sample)."""
    b, _, _ = pred_c.shape
    d = mask.shape[-1]
    vs = box_extent_angstrom / float(d)
    half = 0.5 * box_extent_angstrom
    loss = pred_c.new_tensor(0.0)
    n = 0
    for bi in range(b):
        kk = int(kvec[bi].item())
        m = mask[bi, 0]
        for j in range(kk):
            a0, a1 = _segment_endpoints(pred_c[bi, j], pred_d[bi, j], pred_l[bi, j])
            for p in (a0, a1):
                idx = ((p + half) / vs - 0.5).long().clamp(0, d - 1)
                iz, iy, ix = int(idx[0]), int(idx[1]), int(idx[2])
                val = m[iz, iy, ix]
                loss = loss + F.relu(0.5 - val)
                n += 1
    if n == 0:
        return loss
    return loss / float(n)


def batch_combined_loss(
    pred_c: torch.Tensor,
    pred_d: torch.Tensor,
    pred_l: torch.Tensor,
    batch: dict,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Core Hungarian loss plus optional render / clash / boundary."""
    w = cfg["loss"]
    w_pos = float(w["w_pos"])
    w_dir = float(w["w_dir"])
    w_len = float(w["w_len"])
    core = batch_helix_loss(pred_c, pred_d, pred_l, batch, w_pos, w_dir, w_len)
    mask = batch["mask"]
    kvec = batch["K"]
    data_cfg = cfg["data"]
    box_extent = float(data_cfg["box_size"]) * float(data_cfg["voxel_size"])
    half_ext = 0.5 * box_extent

    extra = pred_c.new_tensor(0.0)
    wr = float(w.get("w_render", 0.0))
    if wr > 0:
        extra = extra + wr * _soft_gaussian_centers_loss(pred_c, mask, half_ext)
    wc = float(w.get("w_clash", 0.0))
    if wc > 0:
        extra = extra + wc * _clash_loss(pred_c, pred_d, pred_l, kvec)
    wb = float(w.get("w_boundary", 0.0))
    if wb > 0:
        extra = extra + wb * _boundary_loss(pred_c, pred_d, pred_l, mask, kvec, box_extent)

    return core + extra
