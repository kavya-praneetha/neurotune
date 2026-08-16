# neurotune

EEG stress detection with a closed-loop, personalised raga-music intervention.
Implements the seven-stage research pipeline: preprocessing → dual feature
extraction → CNN-LSTM / Transformer under LOSO → raga-physiology mapping →
two-stage recommendation → RAG explainability → closed-loop validation.

> **New to the project?** Read [`GUIDE.md`](GUIDE.md) first — a plain-English
> walkthrough of the flow, the components, and the results, with no EEG or ML
> background assumed. This file is the technical reference.

## Read this first

**There is no EEG data in this repository.** The pipeline runs end to end on a
simulator that generates 32-channel recordings with known, configurable ground
truth. That makes every stage executable and testable today, and it makes one
thing possible that real data cannot: checking whether the analysis recovers
effects that were definitely there.

**The numbers this produces are not the study's numbers.** The reported
figures (PR-AUC 0.71, Cohen's d = 0.89, +89% personalisation) live in
`config.ReportedResults` purely for side-by-side reference. Nothing here
reproduces them, and any agreement is coincidence. Swap in real recordings
before comparing anything.

**The RAG citations are unverified.** `rag/corpus.py` ships ~13 paraphrased
passages as a development seed. Every one carries `verified=False`, explanations
print an `[UNVERIFIED CITATION]` marker, and `assert_verified_corpus()` refuses
an unchecked corpus. Resolve each to a DOI before this reaches anyone.

## Run it

```bash
cd ~/ai-lab
uv run python -m neurotune.cli run-all --subjects 6 --sessions 2   # ~10 min, CPU
uv run python -m neurotune.cli detect --model transformer
uv run python -m neurotune.cli validate
uv run python -m neurotune.cli map
uv run python -m neurotune.cli recommend --stress 8.2
uv run python -m neurotune.cli explain --stress 8.2
uv run python -m neurotune.cli run-all --full                      # 20x4, hours
```

Defaults are small on purpose. `--full` is the real 20 subjects × 4 sessions ×
15 minutes = 36,000 epochs across 80 sessions, which is CPU-feasible but slow.

## Layout

| Path | Stage |
|---|---|
| `config.py` | every tunable, as frozen dataclasses. No magic numbers elsewhere |
| `types.py` | immutable carriers with boundary validation |
| `data/eeg_simulator.py` | 32-ch BioSemi simulator: independent sources, real topographic mixing |
| `data/raga_catalog.py` | renders raga audio, then measures it with librosa |
| `preprocess/pipeline.py` | 1–45 Hz band-pass → ICA → epoching → per-subject z-score |
| `features/timefreq.py` | STFT / Morlet CWT → `[3 × 45 × 50]` spectrograms |
| `features/bandpower.py` | band powers, β/α and θ/β ratios (µV²) |
| `features/sequences.py` | 20-epoch windows, never crossing a phase boundary |
| `models/` | shared CNN encoder + LSTM / Transformer + multi-task heads |
| `training/loso.py` | leave-one-subject-out cross-validation |
| `training/baseline.py` | Random Forest on band features, identical protocol |
| `analysis/mapping.py` | OLS + subject-random-intercept mixed model |
| `analysis/closed_loop.py` | paired t, RM-ANOVA, paired Cohen's d, physio↔subjective r |
| `recommend/` | Stage 1 metadata filter, Stage 2 XGBoost ranker, closed loop |
| `rag/` | Sentence-BERT + FAISS, with TF-IDF as fallback *and* as the baseline |

## Decisions worth knowing about

**Two epoch sets, not one.** Per-subject z-scoring is correct for the network
but destroys the microvolt units that make "alpha rose 1.3 µV²" mean anything.
`preprocess` emits `physio` (µV, for all statistics) and `model` (z-scored, for
the network) from the same cuts. Reporting effect sizes off z-scored data is an
easy and invisible mistake.

**PR-AUC, not accuracy.** The stress block is longer than baseline, so a model
that always answers "stress" scores well on accuracy and is useless. Macro and
stress-class PR-AUC are reported separately, because quoting one as the other
overstates the result.

**Validation splits by session, never by sequence.** Windows overlap and share
epochs; a random sequence split leaks validation data into training and makes
early stopping meaningless.

**Cohen's d uses the SD of differences.** The pooled-SD version answers a
different question and inflates d for within-subject designs.

**Exploration is load-bearing.** A greedy cold-start policy hands every subject
the same "calmest" track, so the history contains no variation and the ranker
can never learn an individual preference — personalisation then measures as
zero no matter how good the model is. `exploration_epsilon` fixes that.
The personalisation metric scores the *greedy* policy while the *exploratory*
choice is what gets played, so the system isn't penalised for gathering data.

**Alpha and blinks need different topographies.** Both project frontally; if
they project *identically*, ICA's EOG detector cannot separate them and deletes
the neural signal along with the artifact. Blinks are prefrontal (Fp/AF),
frontal alpha peaks a row back (F3/Fz/F4). This was a real bug found by
measuring retention, not by reading the code — see below.

**`scipy.signal.cwt` no longer exists** (removed in SciPy 1.15). The CWT path
uses `mne.time_frequency.tfr_array_morlet`. Most tutorials online are stale.

## Verified vs not

Verified by running it:

- Every stage executes end to end; catalogue features are genuinely measured
  from rendered audio by librosa.
- ICA removes blinks while preserving the signal: **99.8%** of the simulated
  ΔAlpha survives preprocessing, Fp1 peak-to-peak drops 177 µV → 15 µV, and
  exactly one component is excluded.
- The simulator produces the intended phase ordering of alpha power
  (baseline 4.37 → stress 3.49 → music 5.12 → post 6.73 µV²).
- Both architectures forward-pass and train; the Transformer returns real
  per-epoch attention weights.
- LOSO, the RF baseline, all statistics, both recommender stages, and both
  retrievers run and produce finite numbers.

**Not** verified:

- That CNN-LSTM beats Random Forest. On small smoke configs it does not — both
  sit near chance, which is what an under-trained model on ~136 sequences
  should do. Whether the architecture wins at full scale is an open question
  this repo has not answered.
- That the mapping recovers the ground-truth coefficients at realistic n. With
  8 sessions and 4 predictors there are 3 residual degrees of freedom and the
  estimates are noise; `recovery_check` now warns below 10 residual df.
- Anything about real EEG, real ragas, or real people.

## Using real data

Three seams, nothing else changes:

1. **Recordings** — replace `data/eeg_simulator.simulate_cohort` with a loader
   producing `SessionSignal` objects (BioSemi `.bdf` reads directly via
   `mne.io.read_raw_bdf`). Ratings go in `stress_ratings`, one per phase.
2. **Audio** — `data/raga_catalog.load_audio_catalog(audio_dir, metadata_csv, cfg)`
   already exists; the feature code is shared with the synthetic path.
3. **Response** — `run_personalisation_experiment` takes the response function
   as an argument precisely so it can be replaced by actually playing a track
   and measuring the result. Delete `pipeline.response_function` when you do;
   it depends on simulator internals that will not exist.

Once real data is in, delete `recovery_check` calls — there is no ground truth
to recover, and running it would compare a measurement against a fiction.
