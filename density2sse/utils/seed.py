"""Deterministic seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_be = getattr(torch.backends, "mps", None)
    mps = getattr(torch, "mps", None)
    if mps_be is not None and mps_be.is_available() and mps is not None and hasattr(mps, "manual_seed"):
        mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
