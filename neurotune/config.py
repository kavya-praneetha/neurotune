"""Central configuration. No magic numbers anywhere else in the package.

Every value the pipeline depends on lives here as a frozen dataclass, so a run
is fully described by one `PipelineConfig` object. Swap a field, re-run, and
the change is traceable -- nothing is buried in a function body.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

# --- EEG acquisition -------------------------------------------------------


@dataclass(frozen=True)
class EEGConfig:
    """Acquisition and montage parameters (BioSemi 32-channel, 512 Hz)."""

    sfreq: float = 512.0
    n_channels: int = 32
    roi_channels: tuple[str, ...] = ("Fp1", "Fz", "Fp2")
    bandpass: tuple[float, float] = (1.0, 45.0)
    notch_hz: float | None = 50.0
    epoch_seconds: float = 2.0
    montage: str = "standard_1020"

    def __post_init__(self) -> None:
        lo, hi = self.bandpass
        if not 0 < lo < hi:
            raise ValueError(f"bandpass must satisfy 0 < low < high, got {self.bandpass}")
        if hi >= self.sfreq / 2:
            raise ValueError(f"bandpass high {hi} exceeds Nyquist {self.sfreq / 2}")
        if not self.roi_channels:
            raise ValueError("roi_channels must not be empty")
        if self.epoch_seconds <= 0:
            raise ValueError(f"epoch_seconds must be positive, got {self.epoch_seconds}")

    @property
    def epoch_samples(self) -> int:
        return int(round(self.epoch_seconds * self.sfreq))


# --- Frequency bands -------------------------------------------------------


@dataclass(frozen=True)
class BandConfig:
    """Canonical EEG band edges in Hz (inclusive low, exclusive high)."""

    bands: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
    )
    ratios: tuple[tuple[str, str], ...] = (("beta", "alpha"), ("theta", "beta"))

    def __post_init__(self) -> None:
        for name, (lo, hi) in self.bands.items():
            if not 0 < lo < hi:
                raise ValueError(f"band {name!r} has invalid edges {(lo, hi)}")
        for num, den in self.ratios:
            missing = {num, den} - set(self.bands)
            if missing:
                raise ValueError(f"ratio {num}/{den} references unknown band(s) {missing}")


# --- Preprocessing ---------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Artifact handling. ICA runs on the full montage, not just the ROI --
    three channels carry far too little spatial information to unmix."""

    ica_components: int = 20
    ica_method: str = "fastica"
    ica_max_iter: int = 500
    ica_seed: int = 97
    eog_proxy_channels: tuple[str, ...] = ("Fp1", "Fp2")
    eog_z_threshold: float = 3.0
    reject_muscle: bool = True
    muscle_z_threshold: float = 3.0
    #: Peak-to-peak ceiling in MICROVOLTS, above which an epoch is dropped.
    #: Units are in the name on purpose: epochs are handed to the rejection
    #: check in uV, and a volts-scaled threshold here silently rejects
    #: everything rather than failing loudly.
    epoch_reject_peak_to_peak_uv: float | None = 400.0

    def __post_init__(self) -> None:
        if self.ica_components < 2:
            raise ValueError("ica_components must be >= 2")
        if not self.eog_proxy_channels:
            raise ValueError("eog_proxy_channels must not be empty")


# --- Study design ----------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """Within-subject design: 20 participants x 4 sessions x 15 minutes.

    Phase durations sum to `session_minutes`. With 2 s epochs this yields
    450 epochs/session and 450 x 20 x 4 = 36,000 epochs across 80 sessions,
    matching the target corpus size.
    """

    n_subjects: int = 20
    n_sessions: int = 4
    session_minutes: float = 15.0
    phase_minutes: Mapping[str, float] = field(
        default_factory=lambda: {
            "baseline": 3.0,
            "stress": 4.0,
            "music": 5.0,
            "post_music": 3.0,
        }
    )

    def __post_init__(self) -> None:
        if self.n_subjects < 2:
            raise ValueError("LOSO requires at least 2 subjects")
        total = sum(self.phase_minutes.values())
        if abs(total - self.session_minutes) > 1e-6:
            raise ValueError(
                f"phase_minutes sum to {total}, expected {self.session_minutes}"
            )

    @property
    def n_sessions_total(self) -> int:
        return self.n_subjects * self.n_sessions


# --- Time-frequency representation ----------------------------------------


@dataclass(frozen=True)
class TimeFrequencyConfig:
    """Spectrogram geometry: [n_roi_channels x n_freqs x n_times] per epoch."""

    method: str = "stft"  # "stft" | "cwt"
    n_freqs: int = 45
    n_times: int = 50
    fmin: float = 1.0
    fmax: float = 45.0
    stft_nperseg: int = 128
    stft_overlap_ratio: float = 0.85
    log_power: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"stft", "cwt"}:
            raise ValueError(f"method must be 'stft' or 'cwt', got {self.method!r}")
        if self.n_freqs < 2 or self.n_times < 2:
            raise ValueError("n_freqs and n_times must both be >= 2")
        if not 0.0 <= self.stft_overlap_ratio < 1.0:
            raise ValueError("stft_overlap_ratio must be in [0, 1)")


# --- Model + training ------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Shared training hyper-parameters for CNN-LSTM and Transformer."""

    sequence_length: int = 20
    sequence_stride: int = 10
    batch_size: int = 16
    max_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2
    dropout: float = 0.3
    cnn_embed_dim: int = 128
    lstm_hidden: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    regression_weight: float = 0.3  # multi-task loss balance
    early_stop_patience: int = 6
    augment_noise_std: float = 0.05
    augment_time_shift: int = 3
    use_class_weights: bool = True
    seed: int = 1337
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.sequence_stride < 1 or self.sequence_stride > self.sequence_length:
            raise ValueError("sequence_stride must be in [1, sequence_length]")
        if self.regression_weight < 0:
            raise ValueError("regression_weight must be non-negative")


# --- Music + recommendation -----------------------------------------------


@dataclass(frozen=True)
class MusicConfig:
    """Raga catalogue synthesis and audio feature extraction."""

    n_tracks: int = 120
    audio_sr: int = 22050
    audio_seconds: float = 8.0
    n_mfcc: int = 13
    tempo_range: tuple[float, float] = (50.0, 160.0)
    drone_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.n_tracks < 10:
            raise ValueError("n_tracks must be >= 10 for a usable candidate pool")
        lo, hi = self.tempo_range
        if not 0 < lo < hi:
            raise ValueError(f"invalid tempo_range {self.tempo_range}")


@dataclass(frozen=True)
class RecommendConfig:
    """Two-stage recommender: metadata retrieval then personalised re-ranking."""

    high_stress_threshold: float = 7.0
    moderate_stress_threshold: float = 4.0
    high_stress_max_tempo: float = 80.0
    moderate_stress_max_tempo: float = 110.0
    high_stress_max_rhythm: float = 0.4
    require_drone_when_high: bool = True
    candidate_pool_size: int = 25
    top_k: int = 5
    ranker_estimators: int = 300
    ranker_max_depth: int = 4
    ranker_learning_rate: float = 0.05
    min_history_for_personalisation: int = 1
    #: Probability of returning an exploratory (randomised) ordering instead of
    #: the ranker's. Without exploration the cold-start policy hands every
    #: subject the same "calmest" track, the history contains no variation in
    #: what was tried, and the ranker cannot learn an individual preference --
    #: personalisation then measures as zero however good the model is.
    exploration_epsilon: float = 0.3

    def __post_init__(self) -> None:
        if not 0 <= self.moderate_stress_threshold < self.high_stress_threshold <= 10:
            raise ValueError("stress thresholds must satisfy 0 <= mod < high <= 10")
        if self.top_k > self.candidate_pool_size:
            raise ValueError("top_k cannot exceed candidate_pool_size")
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError(
                f"exploration_epsilon must be in [0, 1], got {self.exploration_epsilon}"
            )


# --- RAG -------------------------------------------------------------------


@dataclass(frozen=True)
class RAGConfig:
    """Retrieval-augmented explanation of a recommendation."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k: int = 5
    allow_download: bool = True
    fallback_to_tfidf: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("RAG top_k must be >= 1")


# --- Synthetic ground truth ------------------------------------------------


@dataclass(frozen=True)
class SyntheticConfig:
    """Ground-truth effect sizes baked into the simulator.

    These are the values the downstream analysis is supposed to *recover*.
    They are deliberately modest and fully configurable -- they are NOT the
    study's reported findings, and recovering them validates the pipeline's
    machinery, not the scientific claim. See `ReportedResults`.
    """

    # Latent stress in arbitrary units, mapped to a 0-10 self-report scale.
    baseline_stress: float = 2.5
    stress_induction_delta: float = 4.5
    subject_stress_sd: float = 1.0
    session_habituation: float = -0.25  # stress reactivity drops per session
    #: Stress that subsides on its own once the stressor stops, independent of
    #: any music. Without it the simulated intervention effect is the ONLY
    #: source of recovery, so a randomly-assigned catalogue (half of it
    #: activating) averages to exactly zero change and the pre/post contrast
    #: measures nothing. Real post-stressor recovery is partial and automatic;
    #: it is also why an uncontrolled pre/post design overstates efficacy.
    natural_recovery: float = 1.5

    # How musical properties drive the alpha response (the effect to recover).
    beta_tempo: float = -0.031  # per BPM
    beta_drone: float = 0.18
    beta_rhythmic_intensity: float = -0.12
    beta_brightness: float = -0.05  # spectral centroid, standardised

    # Personalisation: each subject has a latent preferred tempo/scale.
    personal_effect_scale: float = 0.9
    personal_tempo_sd: float = 18.0

    # Signal construction.
    alpha_peak_hz: float = 10.0
    beta_peak_hz: float = 20.0
    theta_peak_hz: float = 6.0
    pink_noise_amplitude: float = 1.0
    blink_rate_hz: float = 0.25
    muscle_burst_rate_hz: float = 0.1
    observation_noise: float = 0.35
    seed: int = 20240517


# --- Reported study results (reference only) -------------------------------


@dataclass(frozen=True)
class ReportedResults:
    """Numbers reported in the write-up, kept for side-by-side comparison.

    IMPORTANT: these are transcribed from the project description. The
    synthetic pipeline in this package does NOT reproduce them and is not
    intended to -- it runs on simulated data. Treat any agreement as
    coincidence. Replace the synthetic loader with real recordings before
    comparing anything here to a pipeline output.
    """

    detection_pr_auc_cnn_lstm: float = 0.71
    detection_recall_cnn_lstm: float = 0.67
    detection_f1_cnn_lstm: float = 0.70
    detection_pr_auc_transformer: float = 0.72
    detection_pr_auc_random_forest: float = 0.62
    detection_recall_random_forest: float = 0.58

    delta_alpha_uv2: float = 1.3
    delta_alpha_cohens_d: float = 0.62
    delta_beta_alpha_ratio: float = -0.5
    delta_beta_alpha_cohens_d: float = 0.51
    delta_theta_uv2: float = -0.5
    delta_theta_cohens_d: float = 0.38

    delta_stress_rating: float = -2.5
    delta_stress_cohens_d: float = 0.89
    physio_subjective_r: float = 0.61

    personalisation_gain_pct: float = 89.0
    playlist_relevance_gain_pct: float = 17.0
    mapping_r2_range: tuple[float, float] = (0.47, 0.54)
    beta_tempo: float = -0.031
    beta_drone: float = 0.18


# --- Root ------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """The whole run, in one object."""

    eeg: EEGConfig = field(default_factory=EEGConfig)
    bands: BandConfig = field(default_factory=BandConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    timefreq: TimeFrequencyConfig = field(default_factory=TimeFrequencyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    recommend: RecommendConfig = field(default_factory=RecommendConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    reported: ReportedResults = field(default_factory=ReportedResults)
    artifacts_dir: str = "artifacts"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "neurotune"

    def scaled(self, *, n_subjects: int | None = None, n_sessions: int | None = None) -> "PipelineConfig":
        """Return a copy with a smaller cohort -- for fast smoke runs.

        Immutable: the original config is untouched.
        """
        study = self.study
        if n_subjects is not None:
            study = replace(study, n_subjects=n_subjects)
        if n_sessions is not None:
            study = replace(study, n_sessions=n_sessions)
        return replace(self, study=study)


DEFAULT = PipelineConfig()
