"""Shared building blocks: the spectro-spatial CNN encoder and the task heads.

Both architectures use the *same* encoder and the *same* heads, so a CNN-LSTM
vs Transformer comparison isolates the temporal model rather than measuring an
accidental difference in capacity somewhere else.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectroSpatialEncoder(nn.Module):
    """Per-epoch encoder: (C, F, T) spectrogram -> a single embedding vector.

    Channels-as-input-planes means the first convolution mixes Fp1/Fz/Fp2 at
    every time-frequency point, which is the spatial half of "spatial-spectral
    patterns". Depth is modest on purpose: this trains on CPU.
    """

    def __init__(self, in_channels: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, T, C, F, Tt) -> (B, T, embed_dim).

        The time axis is folded into the batch so one convolution pass covers
        every epoch in every sequence.
        """
        if images.ndim != 5:
            raise ValueError(f"expected (B, T, C, F, Tt), got shape {tuple(images.shape)}")
        batch, steps = images.shape[:2]
        flat = images.reshape(batch * steps, *images.shape[2:])
        embedded = self.project(self.features(flat))
        return embedded.reshape(batch, steps, -1)


class MultiTaskHead(nn.Module):
    """Two outputs from one representation: phase class and 0-10 stress score.

    The shared trunk is the point -- the regression target regularises the
    classifier, and the classifier gives the regressor a coarse anchor.
    """

    def __init__(self, embed_dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.dropout(embedding)
        logits = self.classifier(hidden)
        # Bounded to the VAS range so the head cannot emit an impossible score.
        score = torch.sigmoid(self.regressor(hidden)).squeeze(-1) * 10.0
        return logits, score


class MultiTaskLoss(nn.Module):
    """Weighted cross-entropy + Huber regression.

    Huber rather than MSE: self-reported stress has occasional wild ratings,
    and a squared penalty lets one of them dominate a batch.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None,
        regression_weight: float,
    ) -> None:
        super().__init__()
        if regression_weight < 0:
            raise ValueError("regression_weight must be non-negative")
        self.classification = nn.CrossEntropyLoss(weight=class_weights)
        self.regression = nn.SmoothL1Loss()
        self.regression_weight = regression_weight

    def forward(
        self,
        logits: torch.Tensor,
        scores: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = self.classification(logits, target_labels)
        reg_loss = self.regression(scores, target_scores)
        total = cls_loss + self.regression_weight * reg_loss
        return total, {
            "loss": float(total.detach()),
            "loss_cls": float(cls_loss.detach()),
            "loss_reg": float(reg_loss.detach()),
        }
