"""3D CNN tokenization with Slot Attention set decoder."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention3D(nn.Module):
    def __init__(
        self,
        max_K: int,
        box_size: int,
        in_channels: int = 1,
        base_channels: int = 16,
        hidden_dim: int = 256,
        box_extent_angstrom: float = 144.0,
        slot_dim: int = 256,
        slot_iters: int = 3,
        token_proj_dim: int = 256,
        slot_mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.max_K = int(max_K)
        self.half_extent = 0.5 * float(box_extent_angstrom)
        c = int(base_channels)
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
        tdim = int(token_proj_dim or slot_dim)
        sdim = int(slot_dim)
        self.slot_iters = max(1, int(slot_iters))

        self.token_norm = nn.LayerNorm(c * 4)
        self.token_proj = nn.Linear(c * 4, tdim)
        self.fc_k = nn.Linear(1, tdim)
        self.in_proj = nn.Linear(tdim, sdim)
        self.k_proj = nn.Linear(sdim, sdim)
        self.v_proj = nn.Linear(sdim, sdim)
        self.q_proj = nn.Linear(sdim, sdim)
        self.gru = nn.GRUCell(sdim, sdim)
        self.mlp = nn.Sequential(
            nn.Linear(sdim, int(slot_mlp_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(slot_mlp_hidden), sdim),
        )
        self.slot_norm = nn.LayerNorm(sdim)
        self.out = nn.Linear(sdim, 7)
        self.slot_init = nn.Embedding(self.max_K, sdim)

    def forward(self, mask: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = mask.shape[0]
        feat = self.enc(mask)  # (B, C, Dz, Dy, Dx)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.token_proj(self.token_norm(tokens))
        tokens = tokens + self.fc_k(k.float().unsqueeze(-1) / float(self.max_K)).unsqueeze(1)
        inputs = self.in_proj(tokens)
        k_in = self.k_proj(inputs)
        v_in = self.v_proj(inputs)

        slots = self.slot_init.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        scale = float(slots.shape[-1]) ** -0.5
        for _ in range(self.slot_iters):
            slots_prev = slots
            q = self.q_proj(self.slot_norm(slots)) * scale
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k_in)
            attn = F.softmax(attn_logits, dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bsn,bnd->bsd", attn, v_in)
            slots = self.gru(updates.reshape(-1, updates.shape[-1]), slots_prev.reshape(-1, slots_prev.shape[-1]))
            slots = slots.view(b, self.max_K, -1)
            slots = slots + self.mlp(self.slot_norm(slots))

        raw = self.out(slots)
        centers = torch.tanh(raw[..., :3]) * self.half_extent
        dirs = raw[..., 3:6]
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        lengths = F.softplus(raw[..., 6:7]).squeeze(-1) + 1e-3
        return centers, dirs, lengths
