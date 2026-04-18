"""Losses with Hungarian matching."""

from __future__ import annotations

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
