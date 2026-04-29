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
        k_embed_dim: int = 32,
        mlp_hidden_dim: int = 256,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.max_K = max_K
        self.box_extent = float(box_extent_angstrom)
        self.half_extent = 0.5 * self.box_extent
        self.mlp_num_layers = max(1, int(mlp_num_layers))
        self.mlp_dropout = float(mlp_dropout)
        c = base_channels
        act = self._make_activation(activation)
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            act,
            nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1),
            self._make_activation(activation),
            nn.Conv3d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            self._make_activation(activation),
            nn.Conv3d(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            self._make_activation(activation),
        )
        # global pool
        self.fc_k = nn.Linear(1, int(k_embed_dim))
        layers = []
        in_dim = c * 4 + int(k_embed_dim)
        hid = int(mlp_hidden_dim or hidden_dim)
        for _ in range(self.mlp_num_layers):
            layers.append(nn.Linear(in_dim, hid))
            layers.append(self._make_activation(activation))
            if self.mlp_dropout > 0:
                layers.append(nn.Dropout(p=self.mlp_dropout))
            in_dim = hid
        layers.append(nn.Linear(in_dim, max_K * 7))
        self.fc = nn.Sequential(*layers)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        n = str(name).strip().lower()
        if n == "relu":
            return nn.ReLU(inplace=True)
        if n == "gelu":
            return nn.GELU()
        if n == "silu":
            return nn.SiLU(inplace=True)
        raise ValueError(f"Unsupported activation={name!r}; choose relu/gelu/silu")

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
