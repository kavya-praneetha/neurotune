"""Torch dataset over a SequenceSet, with training-time augmentation.

Augmentation is applied to the spectrogram, not the raw trace, because that is
where the model actually looks. Two transforms, both label-preserving:

  * additive Gaussian noise -- simulates recording-quality variation
  * circular shift along the spectrogram time axis -- the phase of a 2-second
    window relative to the epoch grid is arbitrary, so the model should not
    depend on it

Frequency content is never altered: shifting or masking along the frequency
axis would move alpha into beta and change the label.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import TrainConfig
from ..types import SequenceSet


class SequenceDataset(Dataset):
    """Gathers a window from the shared spectrogram bank on demand."""

    def __init__(self, sequences: SequenceSet, cfg: TrainConfig, train: bool) -> None:
        self.sequences = sequences
        self.cfg = cfg
        self.train = train
        self._rng = np.random.default_rng(cfg.seed + (0 if train else 1))

    def __len__(self) -> int:
        return len(self.sequences)

    def _augment(self, window: np.ndarray) -> np.ndarray:
        if self.cfg.augment_time_shift > 0:
            shift = int(self._rng.integers(-self.cfg.augment_time_shift, self.cfg.augment_time_shift + 1))
            if shift:
                window = np.roll(window, shift, axis=-1)
        if self.cfg.augment_noise_std > 0:
            window = window + self._rng.normal(
                0.0, self.cfg.augment_noise_std, size=window.shape
            ).astype(np.float32)
        return window

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.sequences.window(index)
        if self.train:
            window = self._augment(window)
        return {
            "images": torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)),
            "label": torch.tensor(int(self.sequences.labels[index]), dtype=torch.long),
            "score": torch.tensor(float(self.sequences.scores[index]), dtype=torch.float32),
        }


def make_loader(
    sequences: SequenceSet,
    cfg: TrainConfig,
    train: bool,
    batch_size: int | None = None,
) -> DataLoader:
    """Build a DataLoader. Shuffles only when training."""
    if len(sequences) == 0:
        raise ValueError("cannot build a DataLoader over an empty SequenceSet")
    return DataLoader(
        SequenceDataset(sequences, cfg, train),
        batch_size=batch_size or cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        drop_last=False,
    )


def split_by_session(
    sequences: SequenceSet, holdout_fraction: float = 0.2, seed: int = 0
) -> tuple[SequenceSet, SequenceSet]:
    """Carve a validation split out of the training subjects, by session.

    Splitting by session rather than by sequence matters: overlapping windows
    from the same session share epochs, so a random sequence split leaks the
    validation data straight into training and makes early stopping useless.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    keys = np.stack([sequences.subject_ids, sequences.session_ids], axis=1)
    unique = np.unique(keys, axis=0)
    if unique.shape[0] < 2:
        raise ValueError("need at least 2 subject-sessions to form a validation split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.shape[0])
    n_val = max(1, int(round(unique.shape[0] * holdout_fraction)))
    val_keys = {tuple(int(v) for v in unique[i]) for i in order[:n_val]}

    is_val = np.array([tuple(int(v) for v in row) in val_keys for row in keys])
    train_set, val_set = sequences.select(~is_val), sequences.select(is_val)
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("session split produced an empty side; lower holdout_fraction")
    return train_set, val_set
