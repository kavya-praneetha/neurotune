"""Random Forest on hand-designed band features -- the comparison baseline.

For the "+15% over traditional approaches" claim to mean anything, the
baseline has to be handicapped in exactly one way: its features. Same
sequences, same LOSO folds, same metrics, same validation discipline. Only the
representation differs -- log band powers and ratios instead of a learned
time-frequency encoding.

Anything else (fewer folds, a different split, a different metric) would make
the comparison a statement about the protocol rather than the model.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..config import BandConfig, TrainConfig
from ..features.bandpower import epoch_feature_matrix
from ..types import EpochSet, SequenceSet
from .loso import FoldResult, LOSOReport
from .metrics import evaluate

MODEL_NAME = "random_forest"


def sequence_feature_matrix(
    epochs: EpochSet,
    sequences: SequenceSet,
    bands: BandConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Average per-epoch band features across each sequence window.

    Mean and standard deviation are both kept: the mean is the level, the
    standard deviation is how much it moved over the window, which is the only
    temporal information this baseline gets. Withholding it entirely would
    make the comparison unfair in the other direction.
    """
    per_epoch, names = epoch_feature_matrix(epochs, bands)
    if per_epoch.shape[0] != len(epochs):
        raise ValueError("feature matrix rows do not match epoch count")

    windows = per_epoch[sequences.indices]  # (n_seq, seq_len, n_features)
    matrix = np.concatenate([windows.mean(axis=1), windows.std(axis=1)], axis=1)
    columns = tuple(f"mean_{n}" for n in names) + tuple(f"sd_{n}" for n in names)
    return matrix.astype(np.float64), columns


def run_loso_baseline(
    epochs: EpochSet,
    sequences: SequenceSet,
    bands: BandConfig,
    cfg: TrainConfig,
    n_estimators: int = 400,
    progress=None,
) -> LOSOReport:
    """Same LOSO protocol as the deep models, classical features."""
    features, _ = sequence_feature_matrix(epochs, sequences, bands)
    subjects = tuple(int(s) for s in np.unique(sequences.subject_ids))
    if len(subjects) < 2:
        raise ValueError("LOSO needs at least 2 subjects")

    folds: list[FoldResult] = []
    for subject in subjects:
        is_test = sequences.subject_ids == subject
        x_train, x_test = features[~is_test], features[is_test]
        y_train, y_test = sequences.labels[~is_test], sequences.labels[is_test]
        s_train, s_test = sequences.scores[~is_test], sequences.scores[is_test]

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced" if cfg.use_class_weights else None,
            random_state=cfg.seed,
            n_jobs=-1,
        ).fit(x_train, y_train)
        regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=cfg.seed, n_jobs=-1
        ).fit(x_train, s_train)

        # RandomForest drops columns for classes absent from the training
        # split; re-expand so the probability matrix always has n_classes
        # columns and the metrics line up across folds.
        probabilities = np.zeros((x_test.shape[0], sequences.n_classes))
        probabilities[:, classifier.classes_] = classifier.predict_proba(x_test)

        folds.append(
            FoldResult(
                held_out_subject=subject,
                result=evaluate(y_test, probabilities, s_test, regressor.predict(x_test)),
                best_epoch=-1,
                stopped_early=False,
                n_train=int(x_train.shape[0]),
                n_test=int(x_test.shape[0]),
            )
        )
        if progress is not None:
            progress(subject, folds[-1].result)

    return LOSOReport(model_name=MODEL_NAME, folds=tuple(folds))
