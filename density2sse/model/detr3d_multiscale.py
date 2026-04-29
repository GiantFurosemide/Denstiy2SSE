"""Multi-scale DETR3D decoder over pyramid memory tokens."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Detr3DMultiScaleHelix(nn.Module):
    def __init__(
        self,
        max_K: int,
        box_size: int,
        in_channels: int = 1,
        base_channels: int = 16,
        hidden_dim: int = 256,
        box_extent_angstrom: float = 144.0,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        transformer_dropout: float = 0.1,
        transformer_norm_first: bool = True,
        transformer_activation: str = "relu",
        k_embed_mode: str = "add",
        multiscale_levels: int = 3,
    ) -> None:
        super().__init__()
        self.max_K = int(max_K)
        self.half_extent = 0.5 * float(box_extent_angstrom)
        c = int(base_channels)
        self.k_embed_mode = str(k_embed_mode).strip().lower()
        if self.k_embed_mode not in {"add", "none"}:
            raise ValueError("k_embed_mode must be 'add' or 'none'")
        self.multiscale_levels = max(1, min(3, int(multiscale_levels)))

        self.enc1 = nn.Sequential(nn.Conv3d(in_channels, c, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv3d(c, c * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv3d(c * 2, c * 4, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv3d(c * 4, c * 4, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.proj_l1 = nn.Linear(c * 2, d_model)
        self.proj_l2 = nn.Linear(c * 4, d_model)
        self.proj_l3 = nn.Linear(c * 4, d_model)
        self.fc_k = nn.Linear(1, d_model)
        self.scale_embed = nn.Embedding(3, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(transformer_dropout),
            batch_first=True,
            norm_first=bool(transformer_norm_first),
            activation=str(transformer_activation),
        )
        self.decoder = nn.TransformerDecoder(dec_layer, int(num_decoder_layers))
        self.query_embed = nn.Embedding(self.max_K, d_model)
        self.out = nn.Linear(d_model, 7)

    @staticmethod
    def _flatten_tokens(feat: torch.Tensor, proj: nn.Linear, scale_vec: torch.Tensor) -> torch.Tensor:
        t = feat.flatten(2).transpose(1, 2)
        return proj(t) + scale_vec.unsqueeze(1)

    def forward(self, mask: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = mask.shape[0]
        f1 = self.enc1(mask)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        lvl_feats = [f2, f3, f4]
        lvl_proj = [self.proj_l1, self.proj_l2, self.proj_l3]

        tokens = []
        for i in range(self.multiscale_levels):
            scale_vec = self.scale_embed.weight[i].unsqueeze(0).expand(b, -1)
            tokens.append(self._flatten_tokens(lvl_feats[i], lvl_proj[i], scale_vec))
        memory = torch.cat(tokens, dim=1)
        if self.k_embed_mode == "add":
            memory = memory + self.fc_k(k.float().unsqueeze(-1) / float(self.max_K)).unsqueeze(1)

        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = self.decoder(q, memory)
        raw = self.out(tgt)
        centers = torch.tanh(raw[..., :3]) * self.half_extent
        dirs = raw[..., 3:6]
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        lengths = F.softplus(raw[..., 6:7]).squeeze(-1) + 1e-3
        return centers, dirs, lengths
