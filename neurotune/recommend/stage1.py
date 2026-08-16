"""Stage 1: fast metadata retrieval.

A pandas filter over the catalogue, keyed on the detected stress level. Its
only job is musical appropriateness -- narrow a few hundred tracks to a
candidate pool the ranker can afford to score. It knows nothing about the
individual.

The constraints are deliberately relaxed rather than abandoned when they
return too few tracks: an empty pool means no recommendation at all, which is
worse than a slightly-off one. Every relaxation is recorded so the caller can
see the pool was not built as specified.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RecommendConfig


@dataclass(frozen=True)
class RetrievalConstraints:
    """The filter actually applied, after any relaxation."""

    max_tempo: float
    max_rhythmic_intensity: float | None
    require_drone: bool
    relaxations: tuple[str, ...] = ()

    def describe(self) -> str:
        ceiling = "no limit" if self.max_tempo == float("inf") else f"<= {self.max_tempo:.0f} BPM"
        parts = [f"tempo {ceiling}"]
        if self.max_rhythmic_intensity is not None:
            parts.append(f"rhythmic intensity <= {self.max_rhythmic_intensity:.2f}")
        if self.require_drone:
            parts.append("drone present")
        described = ", ".join(parts)
        # Surface relaxations: a caller that cannot tell the pool was widened
        # will read a fallback selection as an intentional one.
        if self.relaxations:
            described += f"  [RELAXED: {'; '.join(self.relaxations)}]"
        return described


@dataclass(frozen=True)
class CandidatePool:
    frame: object  # pandas.DataFrame
    constraints: RetrievalConstraints
    stress_level: float

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(self.frame["track_id"])

    def __len__(self) -> int:
        return int(len(self.frame))  # type: ignore[arg-type]


def constraints_for(stress: float, cfg: RecommendConfig) -> RetrievalConstraints:
    """Map a detected stress level onto musical constraints."""
    if not 0.0 <= stress <= 10.0:
        raise ValueError(f"stress must be on a 0-10 scale, got {stress}")
    if stress >= cfg.high_stress_threshold:
        return RetrievalConstraints(
            max_tempo=cfg.high_stress_max_tempo,
            max_rhythmic_intensity=cfg.high_stress_max_rhythm,
            require_drone=cfg.require_drone_when_high,
        )
    if stress >= cfg.moderate_stress_threshold:
        return RetrievalConstraints(
            max_tempo=cfg.moderate_stress_max_tempo,
            max_rhythmic_intensity=None,
            require_drone=False,
        )
    return RetrievalConstraints(max_tempo=float("inf"), max_rhythmic_intensity=None, require_drone=False)


def _apply(frame, constraints: RetrievalConstraints):
    mask = frame["tempo_bpm"] <= constraints.max_tempo
    if constraints.max_rhythmic_intensity is not None:
        mask &= frame["rhythmic_intensity"] <= constraints.max_rhythmic_intensity
    if constraints.require_drone:
        mask &= frame["drone"].astype(bool)
    return frame[mask]


def retrieve(catalog_frame, stress: float, cfg: RecommendConfig) -> CandidatePool:
    """Filter the catalogue down to a stress-appropriate candidate pool.

    Relaxes in a fixed order -- drone requirement, then rhythm ceiling, then
    tempo ceiling -- because that is least-to-most damaging to the acoustic
    intent. Sorting by tempo before truncating keeps the calmest candidates.
    """
    required = {"track_id", "tempo_bpm", "rhythmic_intensity", "drone"}
    missing = required - set(catalog_frame.columns)
    if missing:
        raise ValueError(f"catalogue frame is missing columns: {sorted(missing)}")
    if len(catalog_frame) == 0:
        raise ValueError("catalogue is empty; nothing can be recommended")

    constraints = constraints_for(stress, cfg)
    relaxations: list[str] = []
    pool = _apply(catalog_frame, constraints)

    if len(pool) < cfg.top_k and constraints.require_drone:
        relaxations.append("dropped drone requirement")
        constraints = RetrievalConstraints(
            constraints.max_tempo, constraints.max_rhythmic_intensity, False, tuple(relaxations)
        )
        pool = _apply(catalog_frame, constraints)

    if len(pool) < cfg.top_k and constraints.max_rhythmic_intensity is not None:
        relaxations.append("dropped rhythmic-intensity ceiling")
        constraints = RetrievalConstraints(
            constraints.max_tempo, None, constraints.require_drone, tuple(relaxations)
        )
        pool = _apply(catalog_frame, constraints)

    if len(pool) < cfg.top_k:
        relaxations.append("widened tempo ceiling to the whole catalogue")
        constraints = RetrievalConstraints(
            float("inf"), None, False, tuple(relaxations)
        )
        pool = catalog_frame

    pool = pool.sort_values("tempo_bpm").head(cfg.candidate_pool_size)
    return CandidatePool(frame=pool.reset_index(drop=True), constraints=constraints, stress_level=stress)
