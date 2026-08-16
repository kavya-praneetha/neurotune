"""The two-stage recommender and the closed-loop personalisation experiment.

`Recommender` composes Stage 1 and Stage 2 and carries per-subject history.
`run_personalisation_experiment` is the honest way to measure whether
personalisation works in simulation: it plays out four sessions per subject,
recommending with only the history available at that point, and scores the
choice against a ground-truth response function supplied by the caller.

That callable is the seam. Here it comes from the EEG simulator; against real
participants it would be replaced by actually playing the track and measuring
the response. Nothing else in this module changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import RecommendConfig
from ..types import InterventionOutcome, TrackFeatures

#: (subject_id, track) -> true alpha change. Simulator in dev, measurement live.
ResponseFn = Callable[[int, TrackFeatures], float]


@dataclass(frozen=True)
class Recommendation:
    track: TrackFeatures
    predicted_delta_alpha: float
    rank: int
    personalised: bool
    explored: bool
    constraints: str
    pool_size: int


class Recommender:
    """Stage 1 retrieval + Stage 2 re-ranking, with per-subject memory."""

    def __init__(
        self, catalog: tuple[TrackFeatures, ...], cfg: RecommendConfig, seed: int = 0
    ) -> None:
        from ..data.raga_catalog import catalog_to_frame
        from .stage2 import PersonalRanker, SubjectHistory

        if not catalog:
            raise ValueError("catalogue must not be empty")
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self.catalog = catalog
        self.by_id = {track.track_id: track for track in catalog}
        self.frame = catalog_to_frame(catalog)
        self.ranker = PersonalRanker(cfg)
        self._history: dict[int, SubjectHistory] = {}
        self._training_rows: list[tuple[InterventionOutcome, float, dict[str, float]]] = []
        self._SubjectHistory = SubjectHistory

    def history_for(self, subject_id: int):
        return self._history.get(subject_id, self._SubjectHistory(subject_id))

    def recommend(
        self,
        subject_id: int,
        stress: float,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> tuple[Recommendation, ...]:
        """Return the top-k tracks for this subject at this stress level.

        `greedy=True` suppresses exploration -- it asks "what would the system
        deploy right now", which is the policy worth reporting, as opposed to
        the exploratory choice actually played to gather data.
        """
        from .stage1 import retrieve

        k = top_k or self.cfg.top_k
        pool = retrieve(self.frame, stress, self.cfg)
        candidates = [self.by_id[tid] for tid in pool.track_ids]

        history = self.history_for(subject_id)
        summary = history.summary(self.by_id)
        personalised = (
            self.ranker.is_fitted
            and len(history.outcomes) >= self.cfg.min_history_for_personalisation
        )

        # Exploration is not decoration. A greedy policy on an unfitted ranker
        # hands every subject the same "calmest" track forever, so the history
        # has no variation in what was tried and the ranker can never learn an
        # individual preference. Explore always at cold start, then with
        # probability epsilon.
        explore = (not self.ranker.is_fitted) or (
            not greedy and float(self._rng.random()) < self.cfg.exploration_epsilon
        )

        if explore:
            scores = self._rng.permutation(len(candidates)).astype(np.float64)
        else:
            scores = self.ranker.predict(candidates, stress, summary)

        order = np.argsort(-scores)[:k]
        return tuple(
            Recommendation(
                track=candidates[i],
                predicted_delta_alpha=float(scores[i]),
                rank=rank,
                personalised=personalised and not explore,
                explored=explore,
                constraints=pool.constraints.describe(),
                pool_size=len(pool),
            )
            for rank, i in enumerate(order)
        )

    def observe(self, outcome: InterventionOutcome, stress_at_time: float) -> None:
        """Record a played session. The history summary is snapshotted *before*
        the update, which is what the ranker will have at prediction time."""
        if outcome.track_id not in self.by_id:
            raise ValueError(f"unknown track {outcome.track_id!r}")
        history = self.history_for(outcome.subject_id)
        self._training_rows.append((outcome, stress_at_time, history.summary(self.by_id)))
        self._history[outcome.subject_id] = history.extended(outcome)

    def refit(self) -> bool:
        """Refit the ranker on everything observed so far. False if too little."""
        try:
            self.ranker.fit(self._training_rows, self.by_id)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class PersonalisationReport:
    per_session_mean_alpha: tuple[float, float]
    per_session_all: dict[int, float]
    improvement_pct: float
    n_subjects: int
    n_sessions: int
    random_policy_mean: float

    def format(self) -> str:
        first, last = self.per_session_mean_alpha
        lines = [
            f"Personalisation across {self.n_sessions} sessions x {self.n_subjects} subjects",
            f"  random-policy reference   mean delta-alpha = {self.random_policy_mean:+.3f}",
        ]
        for session, value in sorted(self.per_session_all.items()):
            lines.append(f"  session {session + 1}                 mean delta-alpha = {value:+.3f}")
        lines.append(
            f"  session 1 -> {self.n_sessions}: {first:+.3f} -> {last:+.3f} "
            f"= {self.improvement_pct:+.1f}%"
        )
        return "\n".join(lines)


def run_personalisation_experiment(
    catalog: tuple[TrackFeatures, ...],
    cfg: RecommendConfig,
    response: ResponseFn,
    subject_ids: tuple[int, ...],
    n_sessions: int,
    stress_level: float = 7.5,
    noise_sd: float = 0.15,
    seed: int = 11,
) -> PersonalisationReport:
    """Play out the closed loop and measure whether it improves with use.

    Session 1 is a genuine cold start -- no ranker, no history, Stage 1 order
    only. The model is refit after every session, so any gain is attributable
    to accumulated observations rather than to a model that saw them upfront.
    """
    if n_sessions < 2:
        raise ValueError("need at least 2 sessions to measure improvement")
    if len(subject_ids) < 2:
        raise ValueError("need at least 2 subjects")

    rng = np.random.default_rng(seed)
    engine = Recommender(catalog, cfg, seed=seed)
    realised: dict[int, list[float]] = {s: [] for s in range(n_sessions)}

    for session in range(n_sessions):
        for subject_id in subject_ids:
            # Two different questions, deliberately kept apart:
            #   greedy  -- what the system would deploy now. This is the metric.
            #   played  -- the possibly-exploratory choice actually delivered,
            #              which is what generates the next observation.
            # Scoring the exploratory choice would penalise the system for
            # gathering data and understate the personalisation it achieved.
            greedy_choice = engine.recommend(subject_id, stress_level, top_k=1, greedy=True)[0]
            realised[session].append(response(subject_id, greedy_choice.track))

            played = engine.recommend(subject_id, stress_level, top_k=1)[0]
            engine.observe(
                InterventionOutcome(
                    subject_id=subject_id,
                    session_idx=session,
                    track_id=played.track.track_id,
                    delta_alpha=response(subject_id, played.track) + float(rng.normal(0, noise_sd)),
                    delta_beta_alpha=float("nan"),
                    delta_theta=float("nan"),
                    delta_stress_rating=float("nan"),
                ),
                stress_at_time=stress_level,
            )
        engine.refit()

    # Reference: what uniformly random selection would have delivered.
    random_mean = float(
        np.mean([
            response(subject_id, catalog[int(rng.integers(0, len(catalog)))])
            for subject_id in subject_ids
            for _ in range(n_sessions)
        ])
    )

    means = {session: float(np.mean(values)) for session, values in realised.items()}
    first, last = means[0], means[n_sessions - 1]
    if abs(first) < 1e-9:
        raise ValueError("session-1 mean is zero; percentage improvement is undefined")
    return PersonalisationReport(
        per_session_mean_alpha=(first, last),
        per_session_all=means,
        improvement_pct=100.0 * (last - first) / abs(first),
        n_subjects=len(subject_ids),
        n_sessions=n_sessions,
        random_policy_mean=random_mean,
    )
