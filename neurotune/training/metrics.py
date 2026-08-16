"""Evaluation metrics for the multi-task head.

PR-AUC leads, not accuracy. The phase blocks are unequal by design (4 minutes
of stress against 3 of baseline), and a model that always answers "stress"
scores respectably on accuracy while being clinically worthless. Average
precision measures how well the positive class is actually retrieved.

Both a macro average and the stress-class figure are reported, because they
answer different questions and quoting one as the other is an easy way to
overstate a result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..types import CLASSIFICATION_PHASES, Phase


@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc_macro: float
    pr_auc_stress: float
    roc_auc_macro: float
    recall_macro: float
    recall_stress: float
    precision_macro: float
    f1_macro: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    pearson_r: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EvalResult:
    classification: ClassificationMetrics
    regression: RegressionMetrics
    n_samples: int

    def as_dict(self) -> dict[str, float]:
        return {
            **self.classification.as_dict(),
            **self.regression.as_dict(),
            "n_samples": float(self.n_samples),
        }


def _stress_index() -> int:
    return CLASSIFICATION_PHASES.index(Phase.STRESS)


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((labels.size, n_classes), dtype=np.float64)
    out[np.arange(labels.size), labels] = 1.0
    return out


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> ClassificationMetrics:
    """`probabilities` is (n_samples, n_classes) and must sum to 1 per row."""
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got {probabilities.shape}")
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("labels and probabilities have different lengths")

    n_classes = probabilities.shape[1]
    present = np.unique(labels)
    predictions = probabilities.argmax(axis=1)
    stress = _stress_index()

    # A fold where a class never appears cannot support a one-vs-rest AUC for
    # it; restrict the macro average to classes actually present rather than
    # letting sklearn substitute a misleading value.
    if present.size < 2:
        pr_macro = roc_macro = float("nan")
    else:
        targets = _one_hot(labels, n_classes)[:, present]
        scores = probabilities[:, present]
        pr_macro = float(average_precision_score(targets, scores, average="macro"))
        roc_macro = float(roc_auc_score(targets, scores, average="macro"))

    if stress in present:
        pr_stress = float(average_precision_score((labels == stress).astype(int), probabilities[:, stress]))
        recall_stress = float(recall_score(labels, predictions, labels=[stress], average="macro", zero_division=0))
    else:
        pr_stress = recall_stress = float("nan")

    return ClassificationMetrics(
        pr_auc_macro=pr_macro,
        pr_auc_stress=pr_stress,
        roc_auc_macro=roc_macro,
        recall_macro=float(recall_score(labels, predictions, average="macro", zero_division=0)),
        recall_stress=recall_stress,
        precision_macro=float(precision_score(labels, predictions, average="macro", zero_division=0)),
        f1_macro=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        accuracy=float(np.mean(predictions == labels)),
    )


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> RegressionMetrics:
    if targets.shape != predictions.shape:
        raise ValueError(
            f"shape mismatch: targets {targets.shape} vs predictions {predictions.shape}"
        )
    residual = predictions - targets
    if targets.size < 3 or np.std(targets) < 1e-9 or np.std(predictions) < 1e-9:
        correlation = float("nan")  # undefined, not zero
    else:
        correlation = float(pearsonr(targets, predictions).statistic)
    return RegressionMetrics(
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual**2))),
        pearson_r=correlation,
    )


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    score_targets: np.ndarray,
    score_predictions: np.ndarray,
) -> EvalResult:
    return EvalResult(
        classification=classification_metrics(labels, probabilities),
        regression=regression_metrics(score_targets, score_predictions),
        n_samples=int(labels.size),
    )


def aggregate(results: list[EvalResult]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each metric across folds.

    NaNs are skipped per metric rather than poisoning the average -- a fold
    missing a class should not erase the metric for every other fold.
    """
    if not results:
        raise ValueError("aggregate() requires at least one EvalResult")
    keys = results[0].as_dict().keys()
    summary: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = np.array([r.as_dict()[key] for r in results], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = (float("nan"), float("nan"))
        else:
            summary[key] = (float(finite.mean()), float(finite.std()))
    return summary
