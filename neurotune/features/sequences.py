"""Assemble epochs into the temporal sequences the recurrent models consume.

A sequence is `sequence_length` consecutive epochs from one subject, one
session, one phase. Windows never straddle a phase boundary -- a window half
in "stress" and half in "post_music" has no defensible label, and letting one
through is a quiet way to cap your own accuracy.
"""

from __future__ import annotations

import numpy as np

from ..config import TrainConfig
from ..types import EpochSet, SequenceSet


def _contiguous_runs(epochs: EpochSet) -> list[np.ndarray]:
    """Index runs sharing (subject, session, label), in acquisition order.

    Relies on epochs being stored in acquisition order, which
    `preprocess.pipeline` guarantees. Runs are split wherever any key changes,
    so a phase revisited later in a session starts a fresh run rather than
    silently joining the earlier one.
    """
    n = len(epochs)
    if n == 0:
        return []
    keys = np.stack([epochs.subject_ids, epochs.session_ids, epochs.labels], axis=1)
    boundaries = np.any(keys[1:] != keys[:-1], axis=1)
    split_points = np.flatnonzero(boundaries) + 1
    return np.split(np.arange(n), split_points)


def build_sequences(
    epochs: EpochSet,
    bank: np.ndarray,
    cfg: TrainConfig,
) -> SequenceSet:
    """Slide windows over each contiguous run.

    Epochs whose label is -1 (the music block, excluded from classification)
    are skipped. The regression target is the mean self-report score across
    the window.
    """
    if bank.shape[0] != len(epochs):
        raise ValueError(
            f"bank has {bank.shape[0]} rows but EpochSet has {len(epochs)} epochs"
        )

    length, stride = cfg.sequence_length, cfg.sequence_stride
    indices: list[np.ndarray] = []
    labels: list[int] = []
    scores: list[float] = []
    subjects: list[int] = []
    sessions: list[int] = []

    for run in _contiguous_runs(epochs):
        label = int(epochs.labels[run[0]])
        if label < 0:  # music block: intervention, not a classification target
            continue
        if run.size < length:
            continue
        for start in range(0, run.size - length + 1, stride):
            window = run[start : start + length]
            indices.append(window)
            labels.append(label)
            scores.append(float(np.mean(epochs.stress_score[window])))
            subjects.append(int(epochs.subject_ids[window[0]]))
            sessions.append(int(epochs.session_ids[window[0]]))

    if not indices:
        raise ValueError(
            f"no sequences produced: every run is shorter than sequence_length="
            f"{length}. Reduce sequence_length or lengthen the phase blocks."
        )

    return SequenceSet(
        bank=bank,
        indices=np.stack(indices).astype(np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        subject_ids=np.asarray(subjects, dtype=np.int64),
        session_ids=np.asarray(sessions, dtype=np.int64),
    )


def class_weights(sequences: SequenceSet) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.

    Stress detection is imbalanced by construction -- the stress block is
    longer than baseline -- and an unweighted model happily learns the prior.
    """
    counts = np.bincount(sequences.labels, minlength=sequences.n_classes).astype(np.float64)
    if (counts == 0).any():
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(
            f"classes {missing} have no sequences; cannot form class weights. "
            "Check that every phase survived preprocessing."
        )
    weights = counts.sum() / (counts.size * counts)
    return (weights / weights.mean()).astype(np.float32)
