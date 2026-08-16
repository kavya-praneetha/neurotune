"""Closed-loop validation: did the intervention actually change anything?

Within-session paired contrasts (pre-music versus post-music), repeated
measures across the four sessions, effect sizes, and the physiology-versus-
self-report correlation that says whether the two agree.

Two things this module is careful about:

  * Cohen's d for paired data uses the standard deviation of the *differences*,
    not the pooled SD. The pooled version answers a different question and
    generally inflates d for within-subject designs.
  * Every test reports n. A p-value without a sample size is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..config import BandConfig, EEGConfig
from ..features.bandpower import phase_band_summary
from ..types import InterventionOutcome, Phase, SessionSignal


@dataclass(frozen=True)
class PairedTest:
    """One pre/post contrast."""

    name: str
    mean_change: float
    ci95: tuple[float, float]
    t_statistic: float
    p_value: float
    cohens_d: float
    n: int

    @property
    def effect_label(self) -> str:
        magnitude = abs(self.cohens_d)
        if magnitude < 0.2:
            return "negligible"
        if magnitude < 0.5:
            return "small"
        if magnitude < 0.8:
            return "medium"
        return "large"

    def format(self) -> str:
        low, high = self.ci95
        return (
            f"{self.name:<22} delta={self.mean_change:+.3f} "
            f"[95% CI {low:+.3f}, {high:+.3f}] "
            f"t({self.n - 1})={self.t_statistic:+.2f} p={self.p_value:.2e} "
            f"d={self.cohens_d:+.2f} ({self.effect_label}) n={self.n}"
        )


def paired_change(name: str, before: np.ndarray, after: np.ndarray) -> PairedTest:
    """Paired t-test with a within-subject Cohen's d and a CI on the change."""
    if before.shape != after.shape:
        raise ValueError(f"{name}: before {before.shape} and after {after.shape} must match")
    differences = after - before
    n = differences.size
    if n < 2:
        raise ValueError(f"{name}: need at least 2 pairs, got {n}")

    sd = float(np.std(differences, ddof=1))
    mean = float(np.mean(differences))
    if sd < 1e-12:
        # Identical every time: a real possibility with degenerate inputs, and
        # reporting p=0 / d=inf would be a lie about certainty.
        return PairedTest(name, mean, (mean, mean), float("nan"), float("nan"), float("nan"), n)

    test = stats.ttest_rel(after, before)
    margin = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return PairedTest(
        name=name,
        mean_change=mean,
        ci95=(mean - margin, mean + margin),
        t_statistic=float(test.statistic),
        p_value=float(test.pvalue),
        cohens_d=mean / sd,
        n=n,
    )


def session_outcomes(
    sessions: list[SessionSignal],
    eeg: EEGConfig,
    bands: BandConfig,
    channel: str = "Fz",
) -> tuple[InterventionOutcome, ...]:
    """Per-session pre/post deltas at one channel, from cleaned uV recordings.

    "Pre" is the stress block (immediately before music) and "post" is the
    post-music block -- the contrast the design was built to support.
    """
    outcomes: list[InterventionOutcome] = []
    for session in sessions:
        summary = phase_band_summary(session, eeg, bands, channel)
        pre, post = summary[Phase.STRESS], summary[Phase.POST_MUSIC]
        if session.track_id is None:
            raise ValueError(
                f"subject {session.subject_id} session {session.session_idx} has no "
                "track_id; cannot attribute an intervention outcome"
            )
        outcomes.append(
            InterventionOutcome(
                subject_id=session.subject_id,
                session_idx=session.session_idx,
                track_id=session.track_id,
                delta_alpha=post["alpha"] - pre["alpha"],
                delta_beta_alpha=post["ratio_beta_alpha"] - pre["ratio_beta_alpha"],
                delta_theta=post["theta"] - pre["theta"],
                delta_stress_rating=session.rating_post_music - session.rating_pre_music,
            )
        )
    return tuple(outcomes)


def intervention_tests(
    sessions: list[SessionSignal],
    eeg: EEGConfig,
    bands: BandConfig,
    channel: str = "Fz",
) -> tuple[dict[str, PairedTest], tuple[InterventionOutcome, ...]]:
    """Run every pre/post contrast at once."""
    outcomes = session_outcomes(sessions, eeg, bands, channel)
    summaries = [phase_band_summary(s, eeg, bands, channel) for s in sessions]

    def column(phase: Phase, key: str) -> np.ndarray:
        return np.array([s[phase][key] for s in summaries], dtype=float)

    tests = {
        "alpha_power": paired_change(
            f"alpha power ({channel})", column(Phase.STRESS, "alpha"), column(Phase.POST_MUSIC, "alpha")
        ),
        "beta_alpha_ratio": paired_change(
            "beta/alpha ratio",
            column(Phase.STRESS, "ratio_beta_alpha"),
            column(Phase.POST_MUSIC, "ratio_beta_alpha"),
        ),
        "theta_power": paired_change(
            f"theta power ({channel})", column(Phase.STRESS, "theta"), column(Phase.POST_MUSIC, "theta")
        ),
        "stress_rating": paired_change(
            "self-reported stress",
            np.array([s.rating_pre_music for s in sessions]),
            np.array([s.rating_post_music for s in sessions]),
        ),
    }
    return tests, outcomes


def physiology_subjective_correlation(
    outcomes: tuple[InterventionOutcome, ...]
) -> tuple[float, float, int]:
    """Correlate the alpha change with the self-report change.

    Alpha rises as stress falls, so the raw correlation is negative; it is
    sign-flipped here so that a positive r means "the two measures agree",
    which is what the number is used to claim.
    """
    if len(outcomes) < 3:
        raise ValueError(f"need at least 3 sessions to correlate, got {len(outcomes)}")
    alpha = np.array([o.delta_alpha for o in outcomes])
    rating = np.array([o.delta_stress_rating for o in outcomes])
    if np.std(alpha) < 1e-12 or np.std(rating) < 1e-12:
        return float("nan"), float("nan"), len(outcomes)
    result = stats.pearsonr(alpha, rating)
    return -float(result.statistic), float(result.pvalue), len(outcomes)


def across_session_model(outcomes: tuple[InterventionOutcome, ...]):
    """Mixed-effects model of alpha response over sessions, random intercept
    per subject. Tests whether the intervention improves with repetition."""
    import pandas as pd
    import statsmodels.formula.api as smf

    if len({o.subject_id for o in outcomes}) < 3:
        raise ValueError("mixed-effects across sessions needs at least 3 subjects")
    frame = pd.DataFrame(
        {
            "subject": [o.subject_id for o in outcomes],
            "session": [o.session_idx for o in outcomes],
            "delta_alpha": [o.delta_alpha for o in outcomes],
        }
    )
    return smf.mixedlm("delta_alpha ~ session", frame, groups=frame["subject"]).fit()


def repeated_measures_anova(outcomes: tuple[InterventionOutcome, ...]):
    """Repeated-measures ANOVA of alpha change across the four sessions.

    Requires a balanced design -- every subject present in every session --
    which is exactly what the within-subject protocol produces when nobody
    drops out. Unbalanced input raises rather than silently dropping subjects.
    """
    import pandas as pd
    from statsmodels.stats.anova import AnovaRM

    frame = pd.DataFrame(
        {
            "subject": [o.subject_id for o in outcomes],
            "session": [o.session_idx for o in outcomes],
            "delta_alpha": [o.delta_alpha for o in outcomes],
        }
    )
    counts = frame.groupby("subject")["session"].nunique()
    if counts.nunique() != 1:
        raise ValueError(
            "AnovaRM requires a balanced design; subjects have differing session "
            f"counts: {counts.to_dict()}"
        )
    if int(counts.iloc[0]) < 2:
        raise ValueError("AnovaRM needs at least 2 sessions per subject")
    return AnovaRM(frame, depvar="delta_alpha", subject="subject", within=["session"]).fit()
