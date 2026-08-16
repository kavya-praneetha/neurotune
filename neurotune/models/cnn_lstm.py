"""CNN-LSTM: the primary stress-detection architecture.

The CNN reads one epoch at a time and answers "what does this 2-second window
look like spectrally"; the LSTM reads that sequence of answers and tracks how
the state evolves across the 40-second window. Attention-free by design -- it
is the baseline the Transformer has to beat.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import TrainConfig
from .blocks import MultiTaskHead, SpectroSpatialEncoder


class CNNLSTM(nn.Module):
    """Spectro-spatial CNN encoder followed by a bidirectional LSTM."""

    name = "cnn_lstm"

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        cfg: TrainConfig,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SpectroSpatialEncoder(in_channels, cfg.cnn_embed_dim, cfg.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.cnn_embed_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=bidirectional,
        )
        out_dim = cfg.lstm_hidden * (2 if bidirectional else 1)
        self.head = MultiTaskHead(out_dim, n_classes, cfg.dropout)

    def forward(
        self, images: torch.Tensor, return_internals: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        embedded = self.encoder(images)          # (B, T, E)
        sequence, _ = self.lstm(embedded)        # (B, T, H*dirs)
        # Mean-pool over time rather than taking the last state: with a
        # bidirectional LSTM the "last" step is only last in one direction.
        pooled = sequence.mean(dim=1)
        logits, score = self.head(pooled)
        if return_internals:
            return logits, score, {"embedding": pooled.detach(), "sequence": sequence.detach()}
        return logits, score

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Stress embedding used by the recommender's re-ranking stage."""
        self.eval()
        sequence, _ = self.lstm(self.encoder(images))
        return sequence.mean(dim=1)
