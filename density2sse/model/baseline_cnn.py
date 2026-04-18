"""3D CNN encoder + MLP head predicting K helices."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineHelixCNN(nn.Module):
    """
    Input: ``(B, 1, D, H, W)`` mask and scalar ``K`` per sample (used as conditioning).

    Output: per-slot predictions ``(B, max_K, 7)`` split into centers (3), directions (3), length (1).
    """

    def __init__(
        self,
        max_K: int,
        box_size: int,
        in_channels: int = 1,
        base_channels: int = 16,
        hidden_dim: int = 256,
        box_extent_angstrom: float = 144.0,
    ) -> None:
        super().__init__()
        self.max_K = max_K
        self.box_extent = float(box_extent_angstrom)
        self.half_extent = 0.5 * self.box_extent
        c = base_channels
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # global pool
        self.fc_k = nn.Linear(1, 32)
        self.fc = nn.Sequential(
            nn.Linear(c * 4 + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max_K * 7),
        )

    def forward(self, mask: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ``k``: ``(B,)`` integer number of helices (conditioning; use normalized embedding).
        """
        b = mask.shape[0]
        feat = self.enc(mask)
        feat = F.adaptive_avg_pool3d(feat, 1).flatten(1)
        kf = self.fc_k(k.float().unsqueeze(-1) / float(self.max_K))
        h = torch.cat([feat, kf], dim=1)
        raw = self.fc(h).view(b, self.max_K, 7)
        centers = torch.tanh(raw[..., :3]) * self.half_extent
        dirs = raw[..., 3:6]
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        lengths = F.softplus(raw[..., 6:7]).squeeze(-1) + 1e-3
        return centers, dirs, lengths
