"""Raga-to-physiology mapping: which musical properties predict stress relief?

The design is repeated-measures -- each subject contributes several sessions --
so an ordinary least-squares fit understates the standard errors by treating
correlated observations as independent. Both are fitted here: OLS for the R²
that people expect to see quoted, and a mixed-effects model with a random
intercept per subject for the coefficients that should actually be trusted.

Predictors are centred, not standardised, so the coefficients stay in
interpretable units: per BPM, per unit of rhythmic intensity, and drone as a
plain presence contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import InterventionOutcome, TrackFeatures

#: Reference tempo (BPM) that tempo is centred on, so the intercept means
#: "a mid-tempo track" rather than "a 0 BPM track".
TEMPO_REFERENCE = 105.0

OUTCOME_COLUMNS = ("delta_alpha", "delta_beta_alpha", "delta_theta", "delta_stress_rating")
PREDICTORS = ("tempo_centered", "drone", "rhythm_centered", "centroid_z")


@dataclass(frozen=True)
class MappingResult:
    """Fitted coefficients for one outcome."""

    outcome: str
    coefficients: dict[str, float]
    p_values: dict[str, float]
    ols_r2: float
    ols_r2_adjusted: float
    n_observations: int
    n_subjects: int
    mixed_converged: bool

    def significant(self, alpha: float = 0.05) -> tuple[str, ...]:
        return tuple(k for k, p in self.p_values.items() if np.isfinite(p) and p < alpha)

    def format(self) -> str:
        lines = [
            f"{self.outcome}: OLS R2={self.ols_r2:.3f} (adj {self.ols_r2_adjusted:.3f}), "
            f"n={self.n_observations} across {self.n_subjects} subjects"
            + ("" if self.mixed_converged else "  [mixed model did NOT converge]")
        ]
        for name in PREDICTORS:
            beta = self.coefficients.get(name, float("nan"))
            p = self.p_values.get(name, float("nan"))
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            lines.append(f"    {name:<16} beta={beta:+.4f}  p={p:.2e} {stars}")
        return "\n".join(lines)


def build_design_frame(
    outcomes: tuple[InterventionOutcome, ...],
    catalog: tuple[TrackFeatures, ...],
):
    """Join session outcomes to the musical properties of the track played."""
    import pandas as pd

    if not outcomes:
        raise ValueError("no intervention outcomes to model")
    by_id = {track.track_id: track for track in catalog}
    missing = {o.track_id for o in outcomes} - set(by_id)
    if missing:
        raise ValueError(f"outcomes reference tracks absent from the catalogue: {sorted(missing)}")

    centroids = np.array([t.spectral_centroid for t in catalog], dtype=float)
    centroid_mean = float(centroids.mean())
    centroid_sd = float(centroids.std() or 1.0)

    rows = []
    for outcome in outcomes:
        track = by_id[outcome.track_id]
        rows.append(
            {
                "subject": outcome.subject_id,
                "session": outcome.session_idx,
                "track_id": track.track_id,
                "raga": track.raga,
                "tempo_centered": track.tempo_bpm - TEMPO_REFERENCE,
                "drone": 1.0 if track.drone else 0.0,
                "rhythm_centered": track.rhythmic_intensity - 0.5,
                "centroid_z": (track.spectral_centroid - centroid_mean) / centroid_sd,
                "delta_alpha": outcome.delta_alpha,
                "delta_beta_alpha": outcome.delta_beta_alpha,
                "delta_theta": outcome.delta_theta,
                "delta_stress_rating": outcome.delta_stress_rating,
            }
        )
    return pd.DataFrame(rows)


def fit_mapping(frame, outcome: str) -> MappingResult:
    """Fit OLS and a subject-random-intercept mixed model for one outcome."""
    import statsmodels.formula.api as smf

    if outcome not in frame.columns:
        raise ValueError(f"unknown outcome {outcome!r}; available: {OUTCOME_COLUMNS}")
    formula = f"{outcome} ~ " + " + ".join(PREDICTORS)

    n_subjects = int(frame["subject"].nunique())
    if len(frame) <= len(PREDICTORS) + 1:
        raise ValueError(
            f"{len(frame)} observations cannot support {len(PREDICTORS)} predictors"
        )

    ols = smf.ols(formula, frame).fit()

    # The mixed model is the one to trust, but it can fail to converge on small
    # or degenerate samples. Fall back to OLS inference rather than reporting
    # coefficients from a model that did not actually fit.
    converged = False
    coefficients, p_values = dict(ols.params), dict(ols.pvalues)
    if n_subjects >= 3:
        try:
            mixed = smf.mixedlm(formula, frame, groups=frame["subject"]).fit(method="lbfgs")
            if getattr(mixed, "converged", False):
                converged = True
                coefficients = {k: float(v) for k, v in mixed.params.items()}
                p_values = {k: float(v) for k, v in mixed.pvalues.items()}
        except (ValueError, np.linalg.LinAlgError):
            converged = False

    return MappingResult(
        outcome=outcome,
        coefficients={k: float(v) for k, v in coefficients.items()},
        p_values={k: float(v) for k, v in p_values.items()},
        ols_r2=float(ols.rsquared),
        ols_r2_adjusted=float(ols.rsquared_adj),
        n_observations=int(len(frame)),
        n_subjects=n_subjects,
        mixed_converged=converged,
    )


def fit_all(
    outcomes: tuple[InterventionOutcome, ...],
    catalog: tuple[TrackFeatures, ...],
) -> tuple[dict[str, MappingResult], object]:
    """Fit every outcome. Returns (results by outcome name, design frame)."""
    frame = build_design_frame(outcomes, catalog)
    return {name: fit_mapping(frame, name) for name in OUTCOME_COLUMNS}, frame


def recovery_check(
    result: MappingResult,
    truth: dict[str, float],
    tolerance: float = 0.5,
) -> dict[str, dict[str, float | bool]]:
    """Compare fitted coefficients against the simulator's ground truth.

    Only meaningful on synthetic data -- it verifies that the estimation
    machinery recovers coefficients that were genuinely written in. On real
    recordings there is no truth to compare against and this must not be run.
    """
    # With 4 predictors, a handful of sessions leaves almost no residual
    # degrees of freedom and the estimates are noise. Reporting "recovery
    # failed" from an under-powered fit would blame the estimator for a
    # sample-size problem, so say which it is.
    residual_df = result.n_observations - len(PREDICTORS) - 1
    if residual_df < 10:
        print(
            f"  WARNING: only {residual_df} residual df ({result.n_observations} "
            f"observations, {len(PREDICTORS)} predictors). Coefficient estimates "
            "are too noisy for a meaningful recovery check -- use more "
            "subjects/sessions before reading anything into the result below."
        )

    report: dict[str, dict[str, float | bool]] = {}
    for name, true_value in truth.items():
        estimate = result.coefficients.get(name, float("nan"))
        error = abs(estimate - true_value)
        scale = max(abs(true_value), 1e-6)
        report[name] = {
            "true": true_value,
            "estimated": estimate,
            "absolute_error": error,
            "within_tolerance": bool(error <= tolerance * scale),
        }
    return report
