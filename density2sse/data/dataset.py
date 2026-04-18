"""PyTorch datasets for NPZ samples."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from density2sse.data import sample_schema as S


class HelixNPZDataset(Dataset):
    """Load ``.npz`` samples; pad helix lists to ``max_K``."""

    def __init__(
        self,
        directory: str,
        max_K: int,
        box_size: Optional[int] = None,
    ) -> None:
        self.directory = directory
        self.max_K = max_K
        self.box_size = box_size
        self.files: List[str] = sorted(
            f for f in os.listdir(directory) if f.endswith(".npz")
        )
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = os.path.join(self.directory, self.files[idx])
        z = np.load(path)
        mask = z[S.MASK].astype(np.float32)
        if self.box_size is not None and mask.shape[0] != self.box_size:
            raise ValueError(f"Expected box {self.box_size}, got {mask.shape}")
        k = int(z[S.K])
        centers = z[S.CENTERS].astype(np.float32)
        dirs = z[S.DIRECTIONS].astype(np.float32)
        lens = z[S.LENGTHS].astype(np.float32)
        if k > self.max_K:
            raise ValueError(f"Sample K={k} exceeds max_K={self.max_K}")

        pad = self.max_K - k
        if pad > 0:
            centers = np.concatenate([centers, np.zeros((pad, 3), dtype=np.float32)], axis=0)
            dirs = np.concatenate([dirs, np.zeros((pad, 3), dtype=np.float32)], axis=0)
            lens = np.concatenate([lens, np.zeros((pad,), dtype=np.float32)], axis=0)
        valid = np.zeros((self.max_K,), dtype=np.float32)
        valid[:k] = 1.0

        return {
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "K": torch.tensor(k, dtype=torch.long),
            "centers": torch.from_numpy(centers),
            "directions": torch.from_numpy(dirs),
            "lengths": torch.from_numpy(lens),
            "valid": torch.from_numpy(valid),
        }


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {
        "mask": torch.stack([x["mask"] for x in items], dim=0),
        "K": torch.stack([x["K"] for x in items], dim=0),
        "centers": torch.stack([x["centers"] for x in items], dim=0),
        "directions": torch.stack([x["directions"] for x in items], dim=0),
        "lengths": torch.stack([x["lengths"] for x in items], dim=0),
        "valid": torch.stack([x["valid"] for x in items], dim=0),
    }
    return batch
