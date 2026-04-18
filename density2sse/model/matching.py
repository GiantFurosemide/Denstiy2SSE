"""Hungarian matching for helix primitives."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def hungarian_match(
    pred_c: Any,
    pred_d: Any,
    pred_l: Any,
    gt_c: Any,
    gt_d: Any,
    gt_l: Any,
    k: int,
    w_pos: float = 1.0,
    w_dir: float = 1.0,
    w_len: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match ``k`` ground-truth helices to ``max_K`` prediction slots.

    Cost matrix is ``(k, Kp)``; scipy returns GT row index -> pred column index.
    """
    pc = _to_numpy(pred_c)
    pd = _to_numpy(pred_d)
    pl = _to_numpy(pred_l)
    gc = _to_numpy(gt_c[:k])
    gd = _to_numpy(gt_d[:k])
    gl = _to_numpy(gt_l[:k])
    # cost from gt[i] to pred[j]
    dc = np.sum((gc[:, None, :] - pc[None, :, :]) ** 2, axis=-1)
    cos = np.clip(np.abs(np.sum(gd[:, None, :] * pd[None, :, :], axis=-1)), 0.0, 1.0)
    dd = 1.0 - cos
    dl = np.abs(gl[:, None] - pl[None, :])
    cmat = w_pos * dc + w_dir * dd + w_len * dl
    r, c = linear_sum_assignment(cmat)
    return r, c
