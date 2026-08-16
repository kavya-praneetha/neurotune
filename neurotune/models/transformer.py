"""BERT-style Transformer encoder over epoch embeddings.

Same CNN front-end as the CNN-LSTM, different temporal model. The encoder
blocks are hand-rolled rather than `nn.TransformerEncoderLayer` for one
reason: attention weights. `nn.TransformerEncoder` discards them, and the
interpretability claim -- "I can show which epochs the model attended to" --
needs them returned, not inferred.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import TrainConfig
from .blocks import MultiTaskHead, SpectroSpatialEncoder


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions. Sequences are short and fixed-length, so a
    learned table would only add parameters to overfit with."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        if steps > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {steps} exceeds positional table {self.encoding.shape[1]}"
            )
        return x + self.encoding[:, :steps]


class AttentionBlock(nn.Module):
    """Pre-norm self-attention block that hands back its attention weights."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embed dim {dim} must be divisible by n_heads {n_heads}")
        self.norm_attn = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm_attn(x)
        attended, weights = self.attention(
            normed, normed, normed, need_weights=True, average_attn_weights=True
        )
        x = x + attended
        x = x + self.feed_forward(self.norm_ff(x))
        return x, weights


class EEGTransformer(nn.Module):
    """CNN encoder -> [CLS] + positional -> stacked attention blocks -> heads."""

    name = "transformer"

    def __init__(self, in_channels: int, n_classes: int, cfg: TrainConfig) -> None:
        super().__init__()
        dim = cfg.cnn_embed_dim
        self.encoder = SpectroSpatialEncoder(in_channels, dim, cfg.dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.positions = PositionalEncoding(dim)
        self.blocks = nn.ModuleList(
            AttentionBlock(dim, cfg.transformer_heads, cfg.dropout)
            for _ in range(cfg.transformer_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = MultiTaskHead(dim, n_classes, cfg.dropout)

    def _encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        embedded = self.encoder(images)
        cls = self.cls_token.expand(embedded.shape[0], -1, -1)
        x = self.positions(torch.cat([cls, embedded], dim=1))
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x)
            attentions.append(weights)
        return self.norm(x), attentions

    def forward(
        self, images: torch.Tensor, return_internals: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        x, attentions = self._encode(images)
        pooled = x[:, 0]  # [CLS]
        logits, score = self.head(pooled)
        if return_internals:
            return logits, score, {
                "embedding": pooled.detach(),
                "attention": [a.detach() for a in attentions],
            }
        return logits, score

    @torch.no_grad()
    def attention_over_epochs(self, images: torch.Tensor) -> torch.Tensor:
        """How much the [CLS] token attends to each epoch, averaged over layers.

        Returns (B, seq_len) -- the column that answers "which part of the
        session drove this prediction". The leading [CLS]->[CLS] weight is
        dropped so the values line up with input epochs.
        """
        self.eval()
        _, attentions = self._encode(images)
        stacked = torch.stack([a[:, 0, 1:] for a in attentions], dim=0)
        return stacked.mean(dim=0)

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        x, _ = self._encode(images)
        return x[:, 0]
