"""Stage orchestration: the functions the CLI and notebooks both call.

Kept separate from `cli.py` so the pipeline is usable from a notebook without
going through argparse, and separate from the stage modules so none of them
has to know about the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig
from .preprocess.pipeline import ArtifactReport, ProcessedEpochs, concat_epochs, process_subject
from .types import EpochSet, SequenceSet, SessionSignal, TrackFeatures


@dataclass(frozen=True)
class PreparedData:
    """Everything the downstream stages need, computed once."""

    catalog: tuple[TrackFeatures, ...]
    epochs_physio: EpochSet
    epochs_model: EpochSet
    sequences: SequenceSet
    sessions: tuple[SessionSignal, ...]
    artifact_reports: tuple[ArtifactReport, ...]

    def summary(self) -> str:
        excluded = [r.n_excluded for r in self.artifact_reports]
        rejected = sum(r.n_epochs_rejected for r in self.artifact_reports)
        return "\n".join([
            f"  catalogue      {len(self.catalog)} tracks",
            f"  sessions       {len(self.sessions)}",
            f"  epochs         {len(self.epochs_model)} "
            f"({rejected} rejected on amplitude)",
            f"  sequences      {len(self.sequences)} "
            f"of {self.sequences.indices.shape[1]} epochs",
            f"  spectrograms   {self.sequences.image_shape} per epoch "
            f"(channels x freqs x times)",
            f"  ICA removed    {np.mean(excluded):.1f} components/session "
            f"(min {min(excluded)}, max {max(excluded)})",
        ])


def build_catalog(cfg: PipelineConfig, seed: int = 7) -> tuple[TrackFeatures, ...]:
    from .data.raga_catalog import build_synthetic_catalog

    return build_synthetic_catalog(cfg.music, seed=seed)


def prepare(cfg: PipelineConfig, catalog: tuple[TrackFeatures, ...], verbose: bool = True) -> PreparedData:
    """Simulate, preprocess, epoch, transform and sequence the whole cohort.

    Processes one subject at a time: per-subject z-scoring needs all of a
    subject's sessions together, and holding all 80 raw recordings at once
    would cost several gigabytes for no benefit.
    """
    from .data.eeg_simulator import simulate_cohort
    from .features.sequences import build_sequences
    from .features.timefreq import compute_images

    by_subject: dict[int, list[SessionSignal]] = {}
    for session in simulate_cohort(catalog, cfg.eeg, cfg.study, cfg.synthetic):
        by_subject.setdefault(session.subject_id, []).append(session)

    physio_parts: list[EpochSet] = []
    model_parts: list[EpochSet] = []
    reports: list[ArtifactReport] = []
    cleaned: list[SessionSignal] = []

    for subject_id in sorted(by_subject):
        if verbose:
            print(f"  preprocessing subject {subject_id + 1}/{len(by_subject)}", flush=True)
        processed, subject_reports, subject_sessions = process_subject(
            by_subject[subject_id], cfg.eeg, cfg.preprocess
        )
        physio_parts.append(processed.physio)
        model_parts.append(processed.model)
        reports.extend(subject_reports)
        cleaned.extend(subject_sessions)

    epochs_physio = concat_epochs(physio_parts)
    epochs_model = concat_epochs(model_parts)
    if verbose:
        print(f"  computing {cfg.timefreq.method.upper()} images for "
              f"{len(epochs_model)} epochs", flush=True)
    bank = compute_images(epochs_model, cfg.timefreq)
    sequences = build_sequences(epochs_model, bank, cfg.train)

    return PreparedData(
        catalog=catalog,
        epochs_physio=epochs_physio,
        epochs_model=epochs_model,
        sequences=sequences,
        sessions=tuple(cleaned),
        artifact_reports=tuple(reports),
    )


def response_function(cfg: PipelineConfig, catalog: tuple[TrackFeatures, ...]):
    """Ground-truth alpha response, for evaluating the recommender offline.

    Synthetic only. Against real participants this is replaced by playing the
    track and measuring the change -- which is why `run_personalisation_
    experiment` takes it as an argument instead of importing the simulator.
    """
    from .data.eeg_simulator import subject_traits_table, track_alpha_effect

    traits = subject_traits_table(cfg.study, cfg.synthetic)
    centroids = np.array([t.spectral_centroid for t in catalog], dtype=float)
    mean, sd = float(centroids.mean()), float(centroids.std() or 1.0)

    def response(subject_id: int, track: TrackFeatures) -> float:
        if subject_id not in traits:
            raise ValueError(f"no traits for subject {subject_id}")
        return track_alpha_effect(
            traits[subject_id], track, cfg.synthetic, (track.spectral_centroid - mean) / sd
        )

    return response
