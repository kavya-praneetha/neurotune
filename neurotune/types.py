"""Immutable data carriers passed between pipeline stages.

Every container validates its own invariants on construction, so a malformed
array is caught at the boundary it crossed rather than three stages later in a
shape mismatch nobody can trace.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class Phase(str, Enum):
    """The four blocks of a 15-minute session, in order."""

    BASELINE = "baseline"
    STRESS = "stress"
    MUSIC = "music"
    POST_MUSIC = "post_music"

    @classmethod
    def ordered(cls) -> tuple["Phase", ...]:
        return (cls.BASELINE, cls.STRESS, cls.MUSIC, cls.POST_MUSIC)


#: Phases used for the 3-way classification head. The MUSIC block is the
#: intervention itself -- it is analysed for physiological change but excluded
#: from the classifier, which distinguishes baseline / stress / post-music.
CLASSIFICATION_PHASES: tuple[Phase, ...] = (
    Phase.BASELINE,
    Phase.STRESS,
    Phase.POST_MUSIC,
)

PHASE_TO_LABEL: dict[Phase, int] = {p: i for i, p in enumerate(CLASSIFICATION_PHASES)}
LABEL_TO_PHASE: dict[int, Phase] = {i: p for p, i in PHASE_TO_LABEL.items()}


@dataclass(frozen=True)
class TrackFeatures:
    """One music track: curated metadata plus computed audio descriptors."""

    track_id: str
    raga: str
    scale: str
    tempo_bpm: float
    rhythmic_intensity: float  # 0-1
    drone: bool
    spectral_centroid: float
    zero_crossing_rate: float
    mfcc_mean: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.tempo_bpm <= 0:
            raise ValueError(f"{self.track_id}: tempo must be positive")
        if not 0.0 <= self.rhythmic_intensity <= 1.0:
            raise ValueError(f"{self.track_id}: rhythmic_intensity must be in [0, 1]")
        if not self.mfcc_mean:
            raise ValueError(f"{self.track_id}: mfcc_mean must not be empty")


@dataclass(frozen=True)
class SessionSignal:
    """Continuous multi-channel recording for one subject-session.

    `data` is (n_channels, n_samples) in volts. `phase_bounds` maps each phase
    to a half-open [start, stop) sample range covering the whole recording.
    """

    subject_id: int
    session_idx: int
    data: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    phase_bounds: dict[Phase, tuple[int, int]]
    track_id: str | None
    stress_ratings: dict[Phase, float]

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D (channels, samples), got {self.data.shape}")
        if self.data.shape[0] != len(self.channel_names):
            raise ValueError(
                f"channel count mismatch: data has {self.data.shape[0]}, "
                f"names has {len(self.channel_names)}"
            )
        for phase, (start, stop) in self.phase_bounds.items():
            if not 0 <= start < stop <= self.data.shape[1]:
                raise ValueError(f"phase {phase.value} bounds {(start, stop)} out of range")
        missing = set(Phase.ordered()) - set(self.stress_ratings)
        if missing:
            raise ValueError(
                f"stress_ratings missing phases {sorted(p.value for p in missing)} -- "
                "the protocol collects a VAS rating after every block"
            )
        for phase, value in self.stress_ratings.items():
            if not 0.0 <= value <= 10.0:
                raise ValueError(f"rating for {phase.value} must be 0-10, got {value}")

    @property
    def key(self) -> tuple[int, int]:
        return (self.subject_id, self.session_idx)

    @property
    def rating_pre_music(self) -> float:
        """VAS taken after stress induction, immediately before the music block."""
        return self.stress_ratings[Phase.STRESS]

    @property
    def rating_post_music(self) -> float:
        return self.stress_ratings[Phase.POST_MUSIC]

    def with_data(self, data: np.ndarray, channel_names: tuple[str, ...] | None = None) -> "SessionSignal":
        """Return a new SessionSignal carrying replacement data. Never mutates."""
        return replace(
            self,
            data=data,
            channel_names=self.channel_names if channel_names is None else channel_names,
        )


@dataclass(frozen=True)
class EpochSet:
    """Epoched ROI signals with aligned metadata.

    Shapes, all first-axis aligned on n_epochs:
        signals      (n_epochs, n_roi_channels, n_samples)
        phases       (n_epochs,) int, index into Phase.ordered()
        labels       (n_epochs,) int, index into CLASSIFICATION_PHASES, -1 if excluded
        stress_score (n_epochs,) float, 0-10
        subject_ids  (n_epochs,) int
        session_ids  (n_epochs,) int
    """

    signals: np.ndarray
    phases: np.ndarray
    labels: np.ndarray
    stress_score: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError(f"signals must be 3-D, got {self.signals.shape}")
        n = self.signals.shape[0]
        aligned = {
            "phases": self.phases,
            "labels": self.labels,
            "stress_score": self.stress_score,
            "subject_ids": self.subject_ids,
            "session_ids": self.session_ids,
        }
        for name, arr in aligned.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.signals.shape[1] != len(self.channel_names):
            raise ValueError("signals channel axis does not match channel_names")

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    @property
    def subjects(self) -> tuple[int, ...]:
        return tuple(int(s) for s in np.unique(self.subject_ids))

    def select(self, mask: np.ndarray) -> "EpochSet":
        """Return a new EpochSet restricted to `mask`. Never mutates."""
        if mask.shape != (len(self),):
            raise ValueError(f"mask must have shape ({len(self)},), got {mask.shape}")
        return EpochSet(
            signals=self.signals[mask],
            phases=self.phases[mask],
            labels=self.labels[mask],
            stress_score=self.stress_score[mask],
            subject_ids=self.subject_ids[mask],
            session_ids=self.session_ids[mask],
            channel_names=self.channel_names,
            sfreq=self.sfreq,
        )


@dataclass(frozen=True)
class SequenceSet:
    """Sequences of consecutive spectrogram epochs, the model's input unit.

    Sequences overlap (stride < length), so materialising every window would
    duplicate most of the data. Instead this holds one shared spectrogram
    `bank` and an `indices` table; the torch Dataset gathers a window on
    demand. For the full 20-subject study that is ~1 GB saved.

    bank    (n_epochs, n_channels, n_freqs, n_times) float32 -- shared
    indices (n_seq, seq_len) int -- rows into `bank`
    labels  (n_seq,) int   -- classification target
    scores  (n_seq,) float -- regression target, 0-10
    """

    bank: np.ndarray
    indices: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.bank.ndim != 4:
            raise ValueError(f"bank must be 4-D (epochs, ch, freq, time), got {self.bank.shape}")
        if self.indices.ndim != 2:
            raise ValueError(f"indices must be 2-D (n_seq, seq_len), got {self.indices.shape}")
        n = self.indices.shape[0]
        for name, arr in (
            ("labels", self.labels),
            ("scores", self.scores),
            ("subject_ids", self.subject_ids),
            ("session_ids", self.session_ids),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if n and int(self.indices.max()) >= self.bank.shape[0]:
            raise ValueError(
                f"indices reference epoch {int(self.indices.max())} but bank holds "
                f"only {self.bank.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_classes(self) -> int:
        return len(CLASSIFICATION_PHASES)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(int(d) for d in self.bank.shape[1:])  # type: ignore[return-value]

    def window(self, i: int) -> np.ndarray:
        """Materialise one sequence: (seq_len, n_channels, n_freqs, n_times)."""
        return self.bank[self.indices[i]]

    def select(self, mask: np.ndarray) -> "SequenceSet":
        """Restrict to a subset of sequences. The bank is shared, not copied."""
        return SequenceSet(
            bank=self.bank,
            indices=self.indices[mask],
            labels=self.labels[mask],
            scores=self.scores[mask],
            subject_ids=self.subject_ids[mask],
            session_ids=self.session_ids[mask],
        )


@dataclass(frozen=True)
class InterventionOutcome:
    """Pre/post music change for one session -- the closed-loop unit of analysis."""

    subject_id: int
    session_idx: int
    track_id: str
    delta_alpha: float
    delta_beta_alpha: float
    delta_theta: float
    delta_stress_rating: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.subject_id, self.session_idx)
