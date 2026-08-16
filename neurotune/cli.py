"""Command line entry point.

    uv run python -m neurotune.cli run-all --subjects 6 --sessions 2
    uv run python -m neurotune.cli detect --model transformer
    uv run python -m neurotune.cli validate
    uv run python -m neurotune.cli map
    uv run python -m neurotune.cli recommend --stress 8.2
    uv run python -m neurotune.cli explain --stress 8.2

Defaults are deliberately small. The full design (20 subjects x 4 sessions,
36,000 epochs, 20 LOSO folds) runs on CPU but takes a long time; pass --full
when you actually want it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

from .config import DEFAULT, PipelineConfig
from .pipeline import PreparedData, build_catalog, prepare, response_function

BANNER = "=" * 72


def _section(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}", flush=True)


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = DEFAULT
    if not getattr(args, "full", False):
        cfg = cfg.scaled(n_subjects=args.subjects, n_sessions=args.sessions)
    if getattr(args, "epochs", None):
        cfg = replace(cfg, train=replace(cfg.train, max_epochs=args.epochs))
    if getattr(args, "tf_method", None):
        cfg = replace(cfg, timefreq=replace(cfg.timefreq, method=args.tf_method))
    return cfg


def _prepare(cfg: PipelineConfig) -> PreparedData:
    _section("Stage 1-2  Data, preprocessing and feature extraction")
    catalog = build_catalog(cfg)
    print(f"  rendered and measured {len(catalog)} raga excerpts", flush=True)
    data = prepare(cfg, catalog)
    print(data.summary())
    return data


def cmd_detect(args: argparse.Namespace) -> int:
    """Stage 3: train the deep model and the classical baseline under LOSO."""
    from .training.baseline import run_loso_baseline
    from .training.loso import compare, run_loso
    from .training.trainer import configure_threads

    cfg = _config_from_args(args)
    configure_threads()
    data = _prepare(cfg)

    _section(f"Stage 3  LOSO cross-validation -- {args.model}")
    print(f"  {len(data.sequences)} sequences, "
          f"{len(np.unique(data.sequences.subject_ids))} folds", flush=True)

    def progress(subject: int, result) -> None:
        print(f"    subject {subject:>2}  PR-AUC={result.classification.pr_auc_macro:.3f}  "
              f"recall={result.classification.recall_macro:.3f}  "
              f"MAE={result.regression.mae:.2f}", flush=True)

    deep = run_loso(data.sequences, args.model, cfg.train, progress=progress)
    print()
    print(deep.format_table())

    _section("Stage 3  Baseline -- Random Forest on band-power features")
    baseline = run_loso_baseline(
        data.epochs_physio, data.sequences, cfg.bands, cfg.train, progress=progress
    )
    print()
    print(baseline.format_table())

    _section("Comparison")
    for metric in ("pr_auc_macro", "recall_macro", "f1_macro"):
        delta = compare(deep, baseline, metric)
        print(f"  {metric:<16} {args.model}={delta['primary']:.3f}  "
              f"rf={delta['baseline']:.3f}  "
              f"absolute {delta['absolute_gain']:+.3f}  "
              f"relative {delta['relative_gain_pct']:+.1f}%")
    _reported_note(cfg)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Stage 7: closed-loop pre/post statistics."""
    from .analysis.closed_loop import (
        across_session_model,
        intervention_tests,
        physiology_subjective_correlation,
        repeated_measures_anova,
    )

    cfg = _config_from_args(args)
    data = _prepare(cfg)

    _section("Stage 7  Closed-loop validation (pre-music vs post-music)")
    tests, outcomes = intervention_tests(list(data.sessions), cfg.eeg, cfg.bands, channel=args.channel)
    for test in tests.values():
        print(f"  {test.format()}")

    r, p, n = physiology_subjective_correlation(outcomes)
    print(f"\n  physiology vs self-report: r={r:+.3f} p={p:.2e} n={n}")
    print("  (sign-flipped so positive r means the two measures agree)")

    if cfg.study.n_sessions >= 2 and cfg.study.n_subjects >= 3:
        print("\n  Across-session mixed-effects model (delta_alpha ~ session):")
        try:
            model = across_session_model(outcomes)
            print(f"    session beta = {model.params.get('session', float('nan')):+.4f}  "
                  f"p = {model.pvalues.get('session', float('nan')):.3f}")
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"    not fitted: {exc}")
        try:
            anova = repeated_measures_anova(outcomes)
            print("\n  Repeated-measures ANOVA over sessions:")
            print("    " + str(anova).replace("\n", "\n    "))
        except ValueError as exc:
            print(f"\n  Repeated-measures ANOVA skipped: {exc}")
    _reported_note(cfg)
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    """Stage 4: raga properties -> physiological change."""
    from .analysis.closed_loop import session_outcomes
    from .analysis.mapping import fit_all, recovery_check

    cfg = _config_from_args(args)
    data = _prepare(cfg)

    _section("Stage 4  Raga-physiology mapping")
    outcomes = session_outcomes(list(data.sessions), cfg.eeg, cfg.bands, channel=args.channel)
    results, frame = fit_all(outcomes, data.catalog)
    for result in results.values():
        print(result.format())
        print()

    _section("Ground-truth recovery (synthetic data only)")
    truth = {
        "tempo_centered": cfg.synthetic.beta_tempo,
        "drone": cfg.synthetic.beta_drone,
        "rhythm_centered": cfg.synthetic.beta_rhythmic_intensity,
        "centroid_z": cfg.synthetic.beta_brightness,
    }
    print("  Does the estimator recover the coefficients the simulator wrote in?")
    for name, check in recovery_check(results["delta_alpha"], truth).items():
        mark = "ok " if check["within_tolerance"] else "OFF"
        print(f"    [{mark}] {name:<16} true={check['true']:+.4f}  "
              f"estimated={check['estimated']:+.4f}  "
              f"|error|={check['absolute_error']:.4f}")
    print("\n  This validates the estimation machinery, not the science.")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """Stages 5: two-stage recommendation and the personalisation experiment."""
    from .recommend.engine import Recommender, run_personalisation_experiment

    cfg = _config_from_args(args)
    catalog = build_catalog(cfg)

    _section("Stage 5a  Two-stage recommendation (cold start)")
    engine = Recommender(catalog, cfg.recommend)
    for rec in engine.recommend(subject_id=0, stress=args.stress):
        print(f"  #{rec.rank + 1}  {rec.track.raga:<16} {rec.track.track_id}  "
              f"tempo={rec.track.tempo_bpm:6.1f}  "
              f"rhythm={rec.track.rhythmic_intensity:.2f}  "
              f"drone={'yes' if rec.track.drone else 'no ':<3}")
    first = engine.recommend(subject_id=0, stress=args.stress)[0]
    print(f"\n  Stage-1 constraints: {first.constraints}")
    print(f"  Candidate pool: {first.pool_size} tracks")
    print(f"  Personalised: {first.personalised} (no history yet -- this is the cold start)")

    _section("Stage 5b  Does personalisation improve across sessions?")
    report = run_personalisation_experiment(
        catalog=catalog,
        cfg=cfg.recommend,
        response=response_function(cfg, catalog),
        subject_ids=tuple(range(cfg.study.n_subjects)),
        n_sessions=max(cfg.study.n_sessions, 4),
        stress_level=args.stress,
    )
    print(report.format())
    print("\n  Measured in simulation against the generator's response function,")
    print("  not against human participants.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Stage 6: RAG-grounded rationale."""
    from .rag.corpus import load_corpus
    from .rag.retriever import KeywordRetriever, build_query, build_retriever, compare_retrievers, explain
    from .recommend.engine import Recommender

    cfg = _config_from_args(args)
    catalog = build_catalog(cfg)
    passages = load_corpus(args.corpus)

    _section("Stage 6  RAG explainability")
    retriever = build_retriever(passages, cfg.rag)
    print(f"  corpus: {len(passages)} passages   retriever: {retriever.name}")

    engine = Recommender(catalog, cfg.recommend)
    top = engine.recommend(subject_id=0, stress=args.stress)[0]
    print()
    print(explain(top.track, args.stress, retriever, cfg.rag))

    if retriever.name != KeywordRetriever.name:
        keyword = KeywordRetriever(passages)
        queries = [build_query(t, args.stress) for t in catalog[:20]]
        overlap = compare_retrievers(retriever, keyword, queries, cfg.rag.top_k)
        print(f"\n  Semantic vs keyword top-{cfg.rag.top_k} overlap: "
              f"Jaccard {overlap['mean_jaccard_overlap']:.3f} over "
              f"{int(overlap['n_queries'])} queries.")
        print("  Divergence only -- scoring retrieval accuracy needs human")
        print("  relevance judgements, which this repo does not ship.")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    for command in (cmd_detect, cmd_map, cmd_validate, cmd_recommend, cmd_explain):
        code = command(args)
        if code != 0:
            return code
    return 0


def _reported_note(cfg: PipelineConfig) -> None:
    print("\n  NOTE: numbers above come from simulated EEG. The study's reported")
    print(f"  figures (PR-AUC {cfg.reported.detection_pr_auc_cnn_lstm}, "
          f"d={cfg.reported.delta_stress_cohens_d}, "
          f"+{cfg.reported.personalisation_gain_pct:.0f}% personalisation) are")
    print("  in config.ReportedResults for reference and are NOT reproduced here.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurotune",
        description="EEG stress detection with closed-loop personalised music intervention",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--subjects", type=int, default=6, help="cohort size (default: 6)")
        p.add_argument("--sessions", type=int, default=2, help="sessions per subject (default: 2)")
        p.add_argument("--full", action="store_true", help="use the full 20x4 design")
        p.add_argument("--epochs", type=int, default=None, help="max training epochs")
        p.add_argument("--tf-method", choices=("stft", "cwt"), default=None)
        p.add_argument("--channel", default="Fz", help="channel for physiology (default: Fz)")
        p.add_argument("--stress", type=float, default=8.0, help="stress level to recommend for")
        p.add_argument("--corpus", default=None, help="path to a JSON RAG corpus")

    for name, handler, help_text in (
        ("detect", cmd_detect, "train and evaluate stress detection under LOSO"),
        ("map", cmd_map, "fit the raga-physiology mapping"),
        ("validate", cmd_validate, "closed-loop pre/post statistics"),
        ("recommend", cmd_recommend, "two-stage recommendation + personalisation"),
        ("explain", cmd_explain, "RAG-grounded rationale"),
        ("run-all", cmd_run_all, "every stage end to end"),
    ):
        p = sub.add_parser(name, help=help_text)
        add_common(p)
        p.add_argument("--model", choices=("cnn_lstm", "transformer"), default="cnn_lstm")
        p.set_defaults(handler=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "model"):
        args.model = "cnn_lstm"
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
