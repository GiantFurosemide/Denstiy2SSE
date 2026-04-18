"""Soft density to binary mask."""

from __future__ import annotations

import numpy as np


def to_binary_mask(soft: np.ndarray, threshold: float) -> np.ndarray:
    """Values > ``threshold`` become 1, else 0; output ``uint8``."""
    return (np.asarray(soft) > float(threshold)).astype(np.uint8)
