"""Stage 2: personalised re-ranking with a gradient-boosted regressor.

Stage 1 answered "is this track appropriate for this stress level". Stage 2
answers "which of these will work for *this person*", by predicting the alpha
change each candidate would produce and sorting on it.

Personalisation lives in the history features, not in a per-subject model.
Fitting one model per participant would be hopeless at four sessions each;
instead a single model sees the subject's running summary alongside the track
properties and learns the interaction. That is what lets a new subject with an
empty history still get a sensible population-level ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import RecommendConfig
from ..types import InterventionOutcome, TrackFeatures

#: Track columns fed to the ranker, in a fixed order.
TRACK_FEATURES = ("tempo_bpm", "rhythmic_intensity", "drone", "spectral_centroid", "zero_crossing_rate")
#: Per-subject running summary, computed only from sessions already observed.
HISTORY_FEATURES = ("hist_n", "hist_mean_alpha", "hist_best_tempo", "hist_drone_benefit", "hist_tempo_gap")


@dataclass(frozen=True)
class SubjectHistory:
    """Immutable running summary of what has worked for one participant."""

    subject_id: int
    outcomes: tuple[InterventionOutcome, ...] = ()

    def extended(self, outcome: InterventionOutcome) -> "SubjectHistory":
        if outcome.subject_id != self.subject_id:
            raise ValueError(
                f"outcome belongs to subject {outcome.subject_id}, not {self.subject_id}"
            )
        return replace(self, outcomes=self.outcomes + (outcome,))

    def summary(self, catalog: dict[str, TrackFeatures]) -> dict[str, float]:
        """Aggregate prior sessions. All-zeros for a cold-start subject."""
        if not self.outcomes:
            return {name: 0.0 for name in HISTORY_FEATURES}

        deltas = np.array([o.delta_alpha for o in self.outcomes])
        tempos = np.array([catalog[o.track_id].tempo_bpm for o in self.outcomes])
        drones = np.array([1.0 if catalog[o.track_id].drone else 0.0 for o in self.outcomes])

        with_drone = deltas[drones == 1.0]
        without_drone = deltas[drones == 0.0]
        benefit = (
            float(with_drone.mean() - without_drone.mean())
            if with_drone.size and without_drone.size
            else 0.0
        )
        return {
            "hist_n": float(deltas.size),
            "hist_mean_alpha": float(deltas.mean()),
            "hist_best_tempo": float(tempos[int(np.argmax(deltas))]),
            "hist_drone_benefit": benefit,
            "hist_tempo_gap": 0.0,  # filled per-candidate in build_row
        }


def build_row(
    track: TrackFeatures,
    stress: float,
    history_summary: dict[str, float],
) -> list[float]:
    """One feature vector: track properties + stress + subject history."""
    best_tempo = history_summary.get("hist_best_tempo", 0.0)
    # Distance from the tempo that has worked best so far is the single most
    # useful personalisation signal, and a tree cannot construct a difference
    # of two features on its own -- so it is supplied directly.
    tempo_gap = abs(track.tempo_bpm - best_tempo) if history_summary.get("hist_n", 0.0) > 0 else 0.0
    summary = {**history_summary, "hist_tempo_gap": tempo_gap}
    return [
        track.tempo_bpm,
        track.rhythmic_intensity,
        1.0 if track.drone else 0.0,
        track.spectral_centroid,
        track.zero_crossing_rate,
        stress,
        *[summary[name] for name in HISTORY_FEATURES],
    ]


FEATURE_NAMES = TRACK_FEATURES + ("stress",) + HISTORY_FEATURES


class PersonalRanker:
    """Predicts the alpha change a track will produce for a given subject."""

    def __init__(self, cfg: RecommendConfig) -> None:
        self.cfg = cfg
        self._model = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        observations: list[tuple[InterventionOutcome, float, dict[str, float]]],
        catalog: dict[str, TrackFeatures],
    ) -> "PersonalRanker":
        """Fit on (outcome, stress at the time, history summary at the time).

        The history summary must be the one that existed *before* that session,
        or the model trains on information it will not have at prediction time
        and its offline scores become fiction.
        """
        from xgboost import XGBRegressor

        if len(observations) < 4:
            raise ValueError(
                f"need at least 4 observations to fit a ranker, got {len(observations)}"
            )
        rows, targets = [], []
        for outcome, stress, summary in observations:
            if outcome.track_id not in catalog:
                raise ValueError(f"observation references unknown track {outcome.track_id!r}")
            rows.append(build_row(catalog[outcome.track_id], stress, summary))
            targets.append(outcome.delta_alpha)

        model = XGBRegressor(
            n_estimators=self.cfg.ranker_estimators,
            max_depth=self.cfg.ranker_max_depth,
            learning_rate=self.cfg.ranker_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=0,
        )
        model.fit(np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32))
        self._model = model
        return self

    def predict(
        self,
        candidates: list[TrackFeatures],
        stress: float,
        history_summary: dict[str, float],
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ranker is not fitted; call fit() first")
        if not candidates:
            raise ValueError("no candidates to score")
        rows = np.asarray(
            [build_row(track, stress, history_summary) for track in candidates],
            dtype=np.float32,
        )
        return np.asarray(self._model.predict(rows), dtype=np.float64)

    def feature_importance(self) -> dict[str, float]:
        """Gain-based importance, for explaining what the ranker keyed on."""
        if self._model is None:
            raise RuntimeError("ranker is not fitted; call fit() first")
        scores = self._model.feature_importances_
        return dict(zip(FEATURE_NAMES, (float(s) for s in scores)))
