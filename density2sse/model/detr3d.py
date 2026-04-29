"""CNN encoder + Transformer decoder with K learned queries (DETR-style helix set prediction)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Detr3DHelix(nn.Module):
    """
    Same I/O contract as ``BaselineHelixCNN``: ``forward(mask, k) -> centers, directions, lengths``.
    Encoder: 3D conv stack; memory = flattened spatial tokens. Decoder: ``max_K`` learned queries.
    """

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
        enc_c = c * 4
        self.enc_proj = nn.Linear(enc_c, d_model)
        self.fc_k = nn.Linear(1, d_model)
        self.k_embed_mode = str(k_embed_mode).strip().lower()
        if self.k_embed_mode not in {"add", "none"}:
            raise ValueError("k_embed_mode must be 'add' or 'none'")
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=float(transformer_dropout),
            batch_first=True,
            norm_first=bool(transformer_norm_first),
            activation=str(transformer_activation),
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.query_embed = nn.Embedding(max_K, d_model)
        self.out = nn.Linear(d_model, 7)

    def forward(self, mask: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = mask.shape[0]
        feat = self.enc(mask)
        # (B, C, Dz, Dy, Dx) -> tokens
        flat = feat.flatten(2).transpose(1, 2)
        memory = self.enc_proj(flat)
        if self.k_embed_mode == "add":
            kemb = self.fc_k(k.float().unsqueeze(-1) / float(self.max_K))
            memory = memory + kemb.unsqueeze(1)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = self.decoder(q, memory)
        raw = self.out(tgt)
        centers = torch.tanh(raw[..., :3]) * self.half_extent
        dirs = raw[..., 3:6]
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        lengths = F.softplus(raw[..., 6:7]).squeeze(-1) + 1e-3
        return centers, dirs, lengths
