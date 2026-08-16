"""Training loop: multi-task optimisation with early stopping on CPU.

Regularisation is the whole game here -- 20 subjects is a small cohort and the
model will memorise individuals given any chance. Four defences, all active:
dropout inside the encoder and heads, L2 via the optimiser's weight decay,
augmentation in the Dataset, and class weighting in the loss.

Early stopping watches macro PR-AUC on a *session-disjoint* validation split,
never a random one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..config import TrainConfig
from ..types import SequenceSet
from .dataset import make_loader
from .metrics import EvalResult, evaluate


@dataclass
class TrainHistory:
    """Per-epoch record. Mutable by design -- it is an append-only log."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False


def configure_threads(n_threads: int | None = None) -> None:
    """Pin torch to a sensible CPU thread count.

    Left alone, torch grabs every core; with a DataLoader also running that
    oversubscribes and gets slower, not faster.
    """
    import os

    threads = n_threads or max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(threads)


@torch.no_grad()
def predict(
    model: nn.Module, sequences: SequenceSet, cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, class probabilities, score targets, score predictions)."""
    model.eval()
    loader = make_loader(sequences, cfg, train=False)
    labels, probs, targets, preds = [], [], [], []
    for batch in loader:
        logits, score = model(batch["images"])
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(score.cpu().numpy())
        labels.append(batch["label"].numpy())
        targets.append(batch["score"].numpy())
    return (
        np.concatenate(labels),
        np.concatenate(probs),
        np.concatenate(targets),
        np.concatenate(preds),
    )


def evaluate_model(model: nn.Module, sequences: SequenceSet, cfg: TrainConfig) -> EvalResult:
    return evaluate(*predict(model, sequences, cfg))


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass. `optimizer=None` means evaluation."""
    training = optimizer is not None
    model.train(training)
    total, seen = 0.0, 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            logits, score = model(batch["images"])
            loss, _ = criterion(logits, score, batch["label"], batch["score"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Recurrent stacks blow up without this; cheap insurance.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = batch["label"].shape[0]
            total += float(loss.detach()) * batch_size
            seen += batch_size
    return total / max(seen, 1)


def train_model(
    model: nn.Module,
    train_sequences: SequenceSet,
    val_sequences: SequenceSet,
    cfg: TrainConfig,
    class_weights: np.ndarray | None = None,
    logger=None,
) -> tuple[nn.Module, TrainHistory]:
    """Fit `model`, restoring the best-validation weights before returning."""
    from ..models.blocks import MultiTaskLoss

    torch.manual_seed(cfg.seed)
    weights = (
        torch.tensor(class_weights, dtype=torch.float32)
        if cfg.use_class_weights and class_weights is not None
        else None
    )
    criterion = MultiTaskLoss(weights, cfg.regression_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, cfg.early_stop_patience // 2)
    )

    train_loader = make_loader(train_sequences, cfg, train=True)
    val_loader = make_loader(val_sequences, cfg, train=False)

    history = TrainHistory()
    best_score = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience = 0

    for epoch in range(cfg.max_epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, None)
        result = evaluate_model(model, val_sequences, cfg)

        # A fold can leave macro PR-AUC undefined; fall back to negative loss
        # so early stopping still has a signal rather than silently freezing.
        monitored = result.classification.pr_auc_macro
        if not np.isfinite(monitored):
            monitored = -val_loss

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_pr_auc.append(result.classification.pr_auc_macro)
        scheduler.step(monitored)

        if logger is not None:
            logger(epoch, {"train_loss": train_loss, "val_loss": val_loss,
                           **result.classification.as_dict()})

        if monitored > best_score + 1e-5:
            best_score = monitored
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                history.stopped_early = True
                break

    model.load_state_dict(best_state)
    return model, history


def build_model(name: str, in_channels: int, n_classes: int, cfg: TrainConfig) -> nn.Module:
    """Factory so the CLI and LOSO runner share one place that knows the names."""
    from ..models.cnn_lstm import CNNLSTM
    from ..models.transformer import EEGTransformer

    builders = {
        CNNLSTM.name: lambda: CNNLSTM(in_channels, n_classes, cfg),
        EEGTransformer.name: lambda: EEGTransformer(in_channels, n_classes, cfg),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; available: {sorted(builders)}")
    return builders[name]()
