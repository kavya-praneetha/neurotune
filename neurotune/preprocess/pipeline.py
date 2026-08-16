"""Artifact-robust preprocessing with MNE-Python.

Order matters and is deliberate:

    band-pass (1-45 Hz) -> ICA artifact removal -> epoching -> z-scoring

The z-score step is the subtle one. Per-subject normalisation is right for the
network -- it removes inter-subject amplitude scale, which is nuisance variance
under LOSO -- but it *destroys the microvolt units* that make an "alpha rose by
1.3 uV^2" claim meaningful. So this module emits two aligned epoch sets from
the same cuts:

    physio : cleaned, in microvolts, for band power and all statistics
    model  : the same epochs, per-subject z-scored, for the network

Reporting effect sizes off z-scored data is a real and easy mistake; keeping
the two explicitly separate is how this pipeline avoids it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EEGConfig, PreprocessConfig
from ..types import CLASSIFICATION_PHASES, PHASE_TO_LABEL, EpochSet, Phase, SessionSignal


@dataclass(frozen=True)
class ProcessedEpochs:
    """Aligned pair: microvolt epochs for stats, z-scored epochs for the model."""

    physio: EpochSet
    model: EpochSet

    def __post_init__(self) -> None:
        if len(self.physio) != len(self.model):
            raise ValueError(
                f"physio ({len(self.physio)}) and model ({len(self.model)}) "
                "epoch counts must match -- they must come from the same cuts"
            )


@dataclass(frozen=True)
class ArtifactReport:
    """What ICA actually removed, per session. Worth logging, not discarding."""

    subject_id: int
    session_idx: int
    n_components: int
    excluded_eog: tuple[int, ...]
    excluded_muscle: tuple[int, ...]
    n_epochs_rejected: int
    detector_failures: tuple[str, ...] = ()

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_eog) + len(self.excluded_muscle)


def _build_raw(session: SessionSignal, eeg: EEGConfig):
    import mne

    info = mne.create_info(
        ch_names=list(session.channel_names),
        sfreq=session.sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(session.data.astype(np.float64), info, verbose="ERROR")
    try:
        raw.set_montage(eeg.montage, on_missing="warn", verbose="ERROR")
    except ValueError as exc:  # a non-standard cap should not kill the run
        raise ValueError(
            f"channel names are not compatible with montage {eeg.montage!r}: {exc}"
        ) from exc
    return raw


def _filter(raw, eeg: EEGConfig):
    low, high = eeg.bandpass
    raw = raw.copy().filter(l_freq=low, h_freq=high, verbose="ERROR")
    # A 50 Hz notch below a 45 Hz low-pass is a no-op; only apply it when the
    # passband actually reaches line frequency.
    if eeg.notch_hz is not None and eeg.notch_hz < high:
        raw = raw.notch_filter(freqs=[eeg.notch_hz], verbose="ERROR")
    return raw


def _run_ica(raw, session: SessionSignal, pre: PreprocessConfig) -> tuple[object, ArtifactReport]:
    """Fit ICA, identify ocular and (optionally) muscle components, remove them."""
    import mne
    from mne.preprocessing import ICA

    n_components = min(pre.ica_components, raw.info["nchan"] - 1)
    ica = ICA(
        n_components=n_components,
        method=pre.ica_method,
        max_iter=pre.ica_max_iter,
        random_state=pre.ica_seed,
        verbose="ERROR",
    )
    ica.fit(raw, verbose="ERROR")

    # Detector failures are RECORDED, not swallowed. "Zero components removed"
    # is ambiguous on its own -- it can mean a clean recording or a detector
    # that errored on every channel, and those demand opposite responses.
    eog_idx: list[int] = []
    failures: list[str] = []
    for channel in pre.eog_proxy_channels:
        if channel not in raw.ch_names:
            failures.append(f"eog proxy {channel!r} absent from montage")
            continue
        try:
            found, _ = ica.find_bads_eog(
                raw, ch_name=channel, threshold=pre.eog_z_threshold, verbose="ERROR"
            )
            eog_idx.extend(found)
        except (RuntimeError, ValueError) as exc:
            failures.append(f"eog detection on {channel} failed: {type(exc).__name__}: {exc}")

    muscle_idx: list[int] = []
    if pre.reject_muscle:
        if not hasattr(ica, "find_bads_muscle"):
            failures.append("mne ICA has no find_bads_muscle; muscle rejection skipped")
        else:
            try:
                found, _ = ica.find_bads_muscle(
                    raw, threshold=pre.muscle_z_threshold, verbose="ERROR"
                )
                muscle_idx.extend(found)
            except (RuntimeError, ValueError) as exc:
                failures.append(f"muscle detection failed: {type(exc).__name__}: {exc}")

    eog_unique = tuple(sorted(set(eog_idx)))
    muscle_unique = tuple(sorted(set(muscle_idx) - set(eog_unique)))
    ica.exclude = list(eog_unique) + list(muscle_unique)
    cleaned = ica.apply(raw.copy(), verbose="ERROR")

    report = ArtifactReport(
        subject_id=session.subject_id,
        session_idx=session.session_idx,
        n_components=n_components,
        excluded_eog=eog_unique,
        excluded_muscle=muscle_unique,
        n_epochs_rejected=0,
        detector_failures=tuple(failures),
    )
    return cleaned, report


def _roi_indices(channel_names: tuple[str, ...], roi: tuple[str, ...]) -> tuple[int, ...]:
    missing = set(roi) - set(channel_names)
    if missing:
        raise ValueError(
            f"ROI channels {sorted(missing)} absent from montage {list(channel_names)}"
        )
    lookup = {name: i for i, name in enumerate(channel_names)}
    return tuple(lookup[name] for name in roi)


def _cut_epochs(
    roi_data: np.ndarray,
    session: SessionSignal,
    eeg: EEGConfig,
    pre: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a session into fixed epochs. Returns (signals, phases, labels, scores)."""
    width = eeg.epoch_samples
    signals: list[np.ndarray] = []
    phases: list[int] = []
    labels: list[int] = []
    scores: list[float] = []
    order = Phase.ordered()

    for phase in order:
        start, stop = session.phase_bounds[phase]
        rating = session.stress_ratings[phase]
        label = PHASE_TO_LABEL.get(phase, -1) if phase in CLASSIFICATION_PHASES else -1
        for begin in range(start, stop - width + 1, width):
            chunk = roi_data[:, begin : begin + width]
            if pre.epoch_reject_peak_to_peak_uv is not None:
                ptp = float(np.max(chunk.max(axis=1) - chunk.min(axis=1)))
                if ptp > pre.epoch_reject_peak_to_peak_uv:
                    continue
            signals.append(chunk)
            phases.append(order.index(phase))
            labels.append(label)
            scores.append(rating)

    if not signals:
        raise ValueError(
            f"subject {session.subject_id} session {session.session_idx}: "
            "every epoch was rejected -- check epoch_reject_peak_to_peak_uv"
        )
    return (
        np.stack(signals).astype(np.float32),
        np.asarray(phases, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
    )


def process_subject(
    sessions: list[SessionSignal],
    eeg: EEGConfig,
    pre: PreprocessConfig,
) -> tuple[ProcessedEpochs, tuple[ArtifactReport, ...], tuple[SessionSignal, ...]]:
    """Filter, de-artifact, epoch and normalise all sessions of ONE subject.

    Normalisation is per-subject by design, so every session for that subject
    must be present here. Returns the epoch pair, the per-session artifact
    reports, and the cleaned continuous signals (microvolts) for downstream
    phase-level physiology.
    """
    if not sessions:
        raise ValueError("process_subject requires at least one session")
    subject_ids = {s.subject_id for s in sessions}
    if len(subject_ids) != 1:
        raise ValueError(
            f"process_subject expects exactly one subject, got {sorted(subject_ids)} -- "
            "per-subject z-scoring is invalid across subjects"
        )

    roi = eeg.roi_channels
    per_session_signals: list[np.ndarray] = []
    per_session_meta: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    reports: list[ArtifactReport] = []
    cleaned_sessions: list[SessionSignal] = []

    for session in sessions:
        raw = _filter(_build_raw(session, eeg), eeg)
        cleaned, report = _run_ica(raw, session, pre)
        full = cleaned.get_data() * 1e6  # volts -> microvolts
        roi_data = full[list(_roi_indices(session.channel_names, roi))]

        signals, phases, labels, scores = _cut_epochs(roi_data, session, eeg, pre)
        expected = sum(
            (stop - start) // eeg.epoch_samples for start, stop in session.phase_bounds.values()
        )
        reports.append(
            ArtifactReport(
                subject_id=report.subject_id,
                session_idx=report.session_idx,
                n_components=report.n_components,
                excluded_eog=report.excluded_eog,
                excluded_muscle=report.excluded_muscle,
                n_epochs_rejected=int(expected - signals.shape[0]),
                detector_failures=report.detector_failures,
            )
        )
        per_session_signals.append(signals)
        per_session_meta.append((phases, labels, scores, session.subject_id, session.session_idx))
        cleaned_sessions.append(session.with_data(full))

    physio_signals = np.concatenate(per_session_signals, axis=0)
    phases = np.concatenate([m[0] for m in per_session_meta])
    labels = np.concatenate([m[1] for m in per_session_meta])
    scores = np.concatenate([m[2] for m in per_session_meta])
    subject_ids_arr = np.concatenate(
        [np.full(sig.shape[0], meta[3], dtype=np.int64)
         for sig, meta in zip(per_session_signals, per_session_meta)]
    )
    session_ids_arr = np.concatenate(
        [np.full(sig.shape[0], meta[4], dtype=np.int64)
         for sig, meta in zip(per_session_signals, per_session_meta)]
    )

    # Per-subject, per-channel z-score over every epoch this subject produced.
    mean = physio_signals.mean(axis=(0, 2), keepdims=True)
    sd = physio_signals.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    model_signals = ((physio_signals - mean) / sd).astype(np.float32)

    common = dict(
        phases=phases,
        labels=labels,
        stress_score=scores,
        subject_ids=subject_ids_arr,
        session_ids=session_ids_arr,
        channel_names=roi,
        sfreq=eeg.sfreq,
    )
    return (
        ProcessedEpochs(
            physio=EpochSet(signals=physio_signals, **common),
            model=EpochSet(signals=model_signals, **common),
        ),
        tuple(reports),
        tuple(cleaned_sessions),
    )


def concat_epochs(parts: list[EpochSet]) -> EpochSet:
    """Join per-subject EpochSets into one cohort-level set."""
    if not parts:
        raise ValueError("concat_epochs requires at least one EpochSet")
    first = parts[0]
    for part in parts[1:]:
        if part.channel_names != first.channel_names:
            raise ValueError("cannot concatenate EpochSets with different channels")
        if part.sfreq != first.sfreq:
            raise ValueError("cannot concatenate EpochSets with different sample rates")
    return EpochSet(
        signals=np.concatenate([p.signals for p in parts], axis=0),
        phases=np.concatenate([p.phases for p in parts]),
        labels=np.concatenate([p.labels for p in parts]),
        stress_score=np.concatenate([p.stress_score for p in parts]),
        subject_ids=np.concatenate([p.subject_ids for p in parts]),
        session_ids=np.concatenate([p.session_ids for p in parts]),
        channel_names=first.channel_names,
        sfreq=first.sfreq,
    )
