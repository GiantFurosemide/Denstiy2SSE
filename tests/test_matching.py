import numpy as np

from density2sse.model import matching


def test_hungarian_runs():
    k = 3
    max_k = 5
    pred_c = np.random.randn(max_k, 3)
    pred_d = np.random.randn(max_k, 3)
    pred_d = pred_d / (np.linalg.norm(pred_d, axis=-1, keepdims=True) + 1e-8)
    pred_l = np.random.rand(max_k) * 10 + 10
    gt_c = np.random.randn(k, 3)
    gt_d = np.random.randn(k, 3)
    gt_d = gt_d / (np.linalg.norm(gt_d, axis=-1, keepdims=True) + 1e-8)
    gt_l = np.random.rand(k) * 10 + 10
    r, c = matching.hungarian_match(pred_c, pred_d, pred_l, gt_c, gt_d, gt_l, k)
    assert len(r) == k
    assert len(c) == k
