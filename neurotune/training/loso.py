"""Leave-one-subject-out cross-validation.

LOSO is the only split that answers the question that matters clinically: does
this work on a person the model has never seen? A random split over epochs
would let windows from the same subject -- often the same *session* -- sit on
both sides, and the resulting number measures memorisation.

Every fold retrains from scratch. Nothing subject-specific, including the
normalisation statistics, crosses the boundary: per-subject z-scoring happens
in `preprocess.pipeline` using only that subject's own data, so a test
subject's statistics were never pooled with training subjects' in the first
place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrainConfig
from ..features.sequences import class_weights
from ..types import SequenceSet
from .dataset import split_by_session
from .metrics import EvalResult, aggregate
from .trainer import build_model, evaluate_model, train_model


@dataclass(frozen=True)
class FoldResult:
    held_out_subject: int
    result: EvalResult
    best_epoch: int
    stopped_early: bool
    n_train: int
    n_test: int


@dataclass(frozen=True)
class LOSOReport:
    model_name: str
    folds: tuple[FoldResult, ...]

    @property
    def summary(self) -> dict[str, tuple[float, float]]:
        return aggregate([f.result for f in self.folds])

    def metric(self, name: str) -> tuple[float, float]:
        summary = self.summary
        if name not in summary:
            raise KeyError(f"unknown metric {name!r}; available: {sorted(summary)}")
        return summary[name]

    def format_table(self) -> str:
        lines = [f"LOSO results -- {self.model_name} ({len(self.folds)} folds)"]
        headline = ("pr_auc_macro", "pr_auc_stress", "recall_macro", "f1_macro",
                    "accuracy", "mae", "pearson_r")
        for name in headline:
            mean, sd = self.metric(name)
            lines.append(f"  {name:<16} {mean:6.3f} +/- {sd:.3f}")
        return "\n".join(lines)


def run_loso(
    sequences: SequenceSet,
    model_name: str,
    cfg: TrainConfig,
    subjects: tuple[int, ...] | None = None,
    progress=None,
) -> LOSOReport:
    """Train and evaluate one fold per held-out subject."""
    all_subjects = tuple(int(s) for s in np.unique(sequences.subject_ids))
    targets = subjects if subjects is not None else all_subjects
    unknown = set(targets) - set(all_subjects)
    if unknown:
        raise ValueError(f"subjects {sorted(unknown)} are not present in the data")
    if len(all_subjects) < 2:
        raise ValueError("LOSO needs at least 2 subjects")

    in_channels, _, _ = sequences.image_shape
    folds: list[FoldResult] = []

    for subject in targets:
        is_test = sequences.subject_ids == subject
        test_set = sequences.select(is_test)
        train_pool = sequences.select(~is_test)
        if len(test_set) == 0:
            raise ValueError(f"subject {subject} has no sequences")

        train_set, val_set = split_by_session(train_pool, seed=cfg.seed + subject)
        model = build_model(model_name, in_channels, sequences.n_classes, cfg)
        model, history = train_model(
            model, train_set, val_set, cfg, class_weights=class_weights(train_set)
        )
        result = evaluate_model(model, test_set, cfg)
        folds.append(
            FoldResult(
                held_out_subject=subject,
                result=result,
                best_epoch=history.best_epoch,
                stopped_early=history.stopped_early,
                n_train=len(train_set),
                n_test=len(test_set),
            )
        )
        if progress is not None:
            progress(subject, result)

    return LOSOReport(model_name=model_name, folds=tuple(folds))


def compare(primary: LOSOReport, baseline: LOSOReport, metric: str = "pr_auc_macro") -> dict[str, float]:
    """Relative improvement of `primary` over `baseline` on one metric.

    Reported as a percentage of the baseline value, which is the convention
    behind claims like "+15% over Random Forest". A raw difference in AUC
    points is a different, smaller number -- both are returned so the two are
    never confused.
    """
    primary_mean, _ = primary.metric(metric)
    baseline_mean, _ = baseline.metric(metric)
    if not np.isfinite(primary_mean) or not np.isfinite(baseline_mean):
        raise ValueError(f"metric {metric!r} is undefined in one of the reports")
    if abs(baseline_mean) < 1e-12:
        raise ValueError(f"baseline {metric!r} is zero; relative improvement undefined")
    return {
        "primary": primary_mean,
        "baseline": baseline_mean,
        "absolute_gain": primary_mean - baseline_mean,
        "relative_gain_pct": 100.0 * (primary_mean - baseline_mean) / baseline_mean,
    }
