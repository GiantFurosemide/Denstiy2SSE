"""U-Net style encoder-decoder with global set head."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation={name!r}; choose relu/gelu/silu")


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
            _act(activation),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1),
            _act(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSetHead(nn.Module):
    """3D U-Net backbone + pooled set regression head."""

    def __init__(
        self,
        max_K: int,
        box_size: int,
        in_channels: int = 1,
        base_channels: int = 16,
        hidden_dim: int = 256,
        box_extent_angstrom: float = 144.0,
        activation: str = "relu",
        k_embed_dim: int = 32,
        mlp_hidden_dim: int = 256,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_K = int(max_K)
        self.half_extent = 0.5 * float(box_extent_angstrom)
        c = int(base_channels)
        self.enc1 = _ConvBlock(in_channels, c, activation)
        self.enc2 = _ConvBlock(c, c * 2, activation)
        self.enc3 = _ConvBlock(c * 2, c * 4, activation)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2, activation)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c, activation)

        self.fc_k = nn.Linear(1, int(k_embed_dim))
        pooled_dim = c + c * 2 + c * 4 + int(k_embed_dim)
        h = int(mlp_hidden_dim or hidden_dim)
        n_layers = max(1, int(mlp_num_layers))
        p = float(mlp_dropout)
        layers = []
        in_dim = pooled_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, h))
            layers.append(_act(activation))
            if p > 0:
                layers.append(nn.Dropout(p))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.max_K * 7))
        self.head = nn.Sequential(*layers)

    def forward(self, mask: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = mask.shape[0]
        e1 = self.enc1(mask)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Multi-scale pooled descriptors keep U-Net spatial context while producing set outputs.
        g1 = F.adaptive_avg_pool3d(d1, 1).flatten(1)
        g2 = F.adaptive_avg_pool3d(d2, 1).flatten(1)
        g3 = F.adaptive_avg_pool3d(e3, 1).flatten(1)
        kf = self.fc_k(k.float().unsqueeze(-1) / float(self.max_K))
        raw = self.head(torch.cat([g1, g2, g3, kf], dim=1)).view(b, self.max_K, 7)
        centers = torch.tanh(raw[..., :3]) * self.half_extent
        dirs = raw[..., 3:6]
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        lengths = F.softplus(raw[..., 6:7]).squeeze(-1) + 1e-3
        return centers, dirs, lengths
