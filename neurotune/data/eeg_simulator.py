"""Synthetic 32-channel EEG with known ground-truth stress and music effects.

This exists so the whole pipeline is runnable without the BioSemi recordings.
It is a *simulator*, not a stand-in for data: it generates independent neural
sources (alpha/beta/theta oscillations, pink background, blink and muscle
artifacts), mixes them onto a real BioSemi-32 layout with plausible
topographies, and only then hands over channel-space signals. That mixing is
what makes ICA a genuine unmixing problem downstream rather than theatre.

The effects the analysis stages are meant to recover -- the tempo/drone/rhythm
coefficients on alpha response, and per-subject musical preference -- are
written in from `SyntheticConfig`. Recovering them validates the machinery.
It says nothing about whether the effect is real in humans.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..config import EEGConfig, StudyConfig, SyntheticConfig
from ..types import Phase, SessionSignal, TrackFeatures

#: The real BioSemi 32-channel cap layout, in standard 10-20 nomenclature.
BIOSEMI_32: tuple[str, ...] = (
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
)

#: Track selector signature: (subject_id, session_idx, catalog, stress) -> track.
TrackSelector = Callable[[int, int, Sequence[TrackFeatures], float], TrackFeatures]


def _scalp_zone(channel: str) -> str:
    """Coarse zone for a 10-20 electrode label.

    Prefrontal is split out from frontal deliberately. Blinks are a prefrontal
    phenomenon (Fp1/Fp2/AF) while frontal alpha peaks a row back at F3/Fz/F4.
    Collapsing the two gives the ocular and alpha sources identical
    topographies, and then ICA's EOG detector -- which correlates components
    against an Fp proxy -- cannot tell them apart and removes the neural
    signal along with the artifact.
    """
    if channel in {"T7", "T8", "F7", "F8", "P7", "P8", "FC5", "FC6", "CP5", "CP6"}:
        return "temporal"
    if channel.startswith(("Fp", "AF")):
        return "prefrontal"
    if channel in {"F3", "F4", "Fz"}:
        return "frontal"
    if channel.startswith(("P", "PO", "O")):
        return "posterior"
    return "central"


#: Source topographies as weight-by-zone, keyed by source name.
#:
#: Alpha's weight at the *frontal* zone is exactly 1.0 on purpose: band power
#: scales with the SQUARE of the mixing weight, so any other value silently
#: rescales every coefficient the raga-physiology regression estimates.
#: Keeping it at unity at Fz -- the channel the physiology is measured on --
#: means the betas written into SyntheticConfig are directly comparable to the
#: ones recovered from the measured signal.
_TOPOGRAPHY: dict[str, dict[str, float]] = {
    #                prefrontal  frontal  central  posterior  temporal
    "alpha":     {"prefrontal": 0.45, "frontal": 1.00, "central": 0.60, "posterior": 1.00, "temporal": 0.30},
    "beta":      {"prefrontal": 0.70, "frontal": 1.00, "central": 0.50, "posterior": 0.20, "temporal": 0.30},
    "theta":     {"prefrontal": 0.45, "frontal": 0.85, "central": 1.00, "posterior": 0.40, "temporal": 0.15},
    "ocular":    {"prefrontal": 1.00, "frontal": 0.25, "central": 0.06, "posterior": 0.02, "temporal": 0.10},
    "muscle":    {"prefrontal": 0.15, "frontal": 0.12, "central": 0.12, "posterior": 0.12, "temporal": 1.00},
}


def _region_weight(channel: str, region: str) -> float:
    """Topographic weight for a source `region` at `channel`.

    Enough structure for ICA to have something real to separate; not a head
    model, and not a claim about volume conduction.
    """
    if region not in _TOPOGRAPHY:
        raise ValueError(f"unknown region {region!r}; known: {sorted(_TOPOGRAPHY)}")
    return _TOPOGRAPHY[region][_scalp_zone(channel)]


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise: EEG background is not white, and whitening it is a tell."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n_samples)
    sd = float(np.std(shaped))
    return shaped / sd if sd > 0 else shaped


def _oscillation(
    n_samples: int,
    sfreq: float,
    peak_hz: float,
    envelope: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """A narrowband rhythm with a wandering phase and time-varying amplitude."""
    t = np.arange(n_samples) / sfreq
    # Slow random-walk phase makes the rhythm narrowband rather than a pure tone.
    drift = np.cumsum(rng.standard_normal(n_samples)) / sfreq * 0.6
    return envelope * np.sin(2 * np.pi * peak_hz * t + drift)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width < 2:
        return x
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


class SubjectTraits:
    """Latent per-subject parameters. Fixed across a subject's four sessions."""

    __slots__ = ("subject_id", "stress_reactivity", "alpha_peak", "preferred_tempo",
                 "prefers_drone", "effect_scale", "baseline_offset")

    def __init__(self, subject_id: int, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
        self.subject_id = subject_id
        self.stress_reactivity = float(np.clip(rng.normal(1.0, 0.25), 0.4, 1.8))
        self.alpha_peak = float(cfg.alpha_peak_hz + rng.normal(0.0, 0.8))
        self.preferred_tempo = float(rng.uniform(55.0, 120.0))
        self.prefers_drone = bool(rng.random() < 0.65)
        self.effect_scale = float(np.clip(rng.normal(1.0, 0.3), 0.3, 1.9))
        self.baseline_offset = float(rng.normal(0.0, cfg.subject_stress_sd))


def personal_bonus(traits: SubjectTraits, track: TrackFeatures, cfg: SyntheticConfig) -> float:
    """Subject-specific alpha bonus -- the signal personalisation must learn."""
    tempo_fit = np.exp(
        -((track.tempo_bpm - traits.preferred_tempo) ** 2) / (2 * cfg.personal_tempo_sd**2)
    )
    drone_fit = 1.0 if (track.drone and traits.prefers_drone) else 0.0
    return float(
        cfg.personal_effect_scale
        * traits.effect_scale
        * (0.75 * tempo_fit + 0.25 * drone_fit)
    )


def track_alpha_effect(
    traits: SubjectTraits,
    track: TrackFeatures,
    cfg: SyntheticConfig,
    centroid_z: float,
) -> float:
    """Ground-truth alpha response to a track, in the same units the regression
    will estimate. The betas here are exactly what `analysis.mapping` recovers.
    """
    return float(
        cfg.beta_tempo * (track.tempo_bpm - 105.0)
        + cfg.beta_drone * (1.0 if track.drone else 0.0)
        + cfg.beta_rhythmic_intensity * (track.rhythmic_intensity - 0.5)
        + cfg.beta_brightness * centroid_z
        + personal_bonus(traits, track, cfg)
    )


def _stress_trajectory(
    traits: SubjectTraits,
    session_idx: int,
    phase_samples: dict[Phase, int],
    music_effect: float,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Latent stress on a 0-10 scale, sample by sample across the session."""
    base = cfg.baseline_stress + traits.baseline_offset
    induction = (
        cfg.stress_induction_delta * traits.stress_reactivity
        + cfg.session_habituation * session_idx
    )
    peak = base + max(induction, 0.5)

    segments: list[np.ndarray] = []
    for phase in Phase.ordered():
        n = phase_samples[phase]
        if phase is Phase.BASELINE:
            seg = np.full(n, base)
        elif phase is Phase.STRESS:
            seg = base + (peak - base) * np.clip(np.linspace(0.0, 1.4, n), 0, 1)
        elif phase is Phase.MUSIC:
            # Two components: recovery that would happen anyway once the
            # stressor stops, plus a music-driven drop scaled by track fit.
            drop = cfg.natural_recovery + np.clip(music_effect, 0.0, None) * 1.6
            seg = peak - drop * np.clip(np.linspace(0.0, 1.2, n), 0, 1)
        else:
            drop = cfg.natural_recovery + np.clip(music_effect, 0.0, None) * 1.6
            seg = np.full(n, peak - drop) + rng.normal(0, 0.05, n)
        segments.append(seg)

    trajectory = np.concatenate(segments)
    trajectory = trajectory + _smooth(rng.normal(0, 0.35, trajectory.size), 4096)
    return np.clip(trajectory, 0.0, 10.0)


def _band_envelopes(stress: np.ndarray, alpha_boost: np.ndarray) -> dict[str, np.ndarray]:
    """Map latent stress onto band amplitudes.

    Alpha falls with arousal, beta rises, theta rises mildly -- the standard
    frontal stress signature the classifier is expected to pick up.

    Everything is defined in POWER (uV^2) and converted to amplitude at the
    end, because a sinusoid of amplitude A carries power A^2/2. Defining the
    music effect as an additive power term is what makes the tempo/drone
    coefficients recoverable in the same units the regression reports: a
    multiplicative or amplitude-domain effect would come back through the
    squaring transformed and un-comparable.
    """
    s = stress / 10.0
    powers = {
        "alpha": np.clip(6.0 * np.exp(-1.5 * s) + alpha_boost, 0.05, None),
        "beta": 1.2 * (0.6 + 1.6 * s),
        "theta": 3.0 * (0.9 + 0.4 * s),
    }
    return {name: np.sqrt(2.0 * power) for name, power in powers.items()}


def _alpha_boost_profile(
    music_effect: float,
    phase_samples: dict[Phase, int],
    bounds: dict[Phase, tuple[int, int]],
    n_samples: int,
) -> np.ndarray:
    """Additive alpha power (uV^2) contributed by the music, over time.

    Zero through baseline and stress, ramping across the music block, held
    through post-music -- which is exactly the pre/post contrast the closed-
    loop analysis measures.
    """
    boost = np.zeros(n_samples)
    music_start, music_stop = bounds[Phase.MUSIC]
    post_start, post_stop = bounds[Phase.POST_MUSIC]
    ramp = np.clip(np.linspace(0.0, 1.3, phase_samples[Phase.MUSIC]), 0.0, 1.0)
    boost[music_start:music_stop] = music_effect * ramp
    boost[post_start:post_stop] = music_effect
    return boost


def _blink_train(n_samples: int, sfreq: float, cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Eye blinks: large, slow, frontally dominant. ICA's job to remove."""
    signal = np.zeros(n_samples)
    n_blinks = rng.poisson(cfg.blink_rate_hz * n_samples / sfreq)
    width = int(0.3 * sfreq)
    shape = np.hanning(width) if width > 1 else np.ones(1)
    for _ in range(int(n_blinks)):
        start = int(rng.integers(0, max(1, n_samples - width)))
        signal[start : start + width] += shape * rng.uniform(60.0, 130.0)
    return signal


def _muscle_bursts(n_samples: int, sfreq: float, cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """EMG: short high-frequency bursts, temporally dominant."""
    signal = np.zeros(n_samples)
    n_bursts = rng.poisson(cfg.muscle_burst_rate_hz * n_samples / sfreq)
    width = int(0.5 * sfreq)
    for _ in range(int(n_bursts)):
        start = int(rng.integers(0, max(1, n_samples - width)))
        t = np.arange(width) / sfreq
        carrier = np.sin(2 * np.pi * rng.uniform(28, 45) * t)
        signal[start : start + width] += carrier * np.hanning(width) * rng.uniform(8, 20)
    return signal


def simulate_session(
    traits: SubjectTraits,
    session_idx: int,
    track: TrackFeatures,
    centroid_z: float,
    eeg: EEGConfig,
    study: StudyConfig,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> SessionSignal:
    """Generate one 15-minute subject-session recording in volts."""
    phase_samples = {
        phase: int(round(study.phase_minutes[phase.value] * 60.0 * eeg.sfreq))
        for phase in Phase.ordered()
    }
    n_samples = sum(phase_samples.values())

    bounds: dict[Phase, tuple[int, int]] = {}
    cursor = 0
    for phase in Phase.ordered():
        bounds[phase] = (cursor, cursor + phase_samples[phase])
        cursor += phase_samples[phase]

    music_effect = track_alpha_effect(traits, track, cfg, centroid_z)
    stress = _stress_trajectory(traits, session_idx, phase_samples, music_effect, cfg, rng)
    alpha_boost = _alpha_boost_profile(music_effect, phase_samples, bounds, n_samples)
    envelopes = _band_envelopes(stress, alpha_boost)

    # --- independent sources ------------------------------------------------
    sources = {
        "alpha": (_oscillation(n_samples, eeg.sfreq, traits.alpha_peak, envelopes["alpha"], rng), "alpha"),
        "beta": (_oscillation(n_samples, eeg.sfreq, cfg.beta_peak_hz, envelopes["beta"], rng), "beta"),
        "theta": (_oscillation(n_samples, eeg.sfreq, cfg.theta_peak_hz, envelopes["theta"], rng), "theta"),
        "blink": (_blink_train(n_samples, eeg.sfreq, cfg, rng), "ocular"),
        "muscle": (_muscle_bursts(n_samples, eeg.sfreq, cfg, rng), "muscle"),
    }

    # --- mix onto the cap ---------------------------------------------------
    channels = BIOSEMI_32[: eeg.n_channels]
    data = np.empty((len(channels), n_samples), dtype=np.float32)
    for ch_idx, channel in enumerate(channels):
        mixed = cfg.pink_noise_amplitude * 3.0 * _pink_noise(n_samples, rng)
        for source, region in sources.values():
            # Tight jitter: band power scales with weight^2, so a wide spread
            # here becomes a wide multiplicative bias on every recovered beta.
            weight = _region_weight(channel, region) * rng.uniform(0.95, 1.05)
            mixed = mixed + weight * source
        mixed = mixed + rng.normal(0, cfg.observation_noise, n_samples)
        data[ch_idx] = mixed.astype(np.float32)

    # A VAS rating is collected after every block; it tracks the latent state
    # with reporting noise, which is what makes physiology-vs-subjective
    # correlation a non-trivial thing to measure.
    ratings = {
        phase: float(np.clip(stress[slice(*bounds[phase])].mean() + rng.normal(0, 0.5), 0, 10))
        for phase in Phase.ordered()
    }

    return SessionSignal(
        subject_id=traits.subject_id,
        session_idx=session_idx,
        data=data * 1e-6,  # volts, as MNE expects
        channel_names=channels,
        sfreq=eeg.sfreq,
        phase_bounds=bounds,
        track_id=track.track_id,
        stress_ratings=ratings,
    )


def random_selector(rng: np.random.Generator) -> TrackSelector:
    """Baseline policy: pick a track uniformly at random, ignoring stress."""

    def select(subject_id: int, session_idx: int, catalog: Sequence[TrackFeatures], stress: float) -> TrackFeatures:
        return catalog[int(rng.integers(0, len(catalog)))]

    return select


def simulate_cohort(
    catalog: Sequence[TrackFeatures],
    eeg: EEGConfig,
    study: StudyConfig,
    cfg: SyntheticConfig,
    selector: TrackSelector | None = None,
):
    """Yield every subject-session, streaming.

    Streams rather than returning a list: 32 channels x 15 min x 80 sessions is
    ~4.7 GB held at once, and every downstream stage only needs one session in
    memory at a time.
    """
    if not catalog:
        raise ValueError("catalog must contain at least one track")

    rng = np.random.default_rng(cfg.seed)
    centroids = np.array([t.spectral_centroid for t in catalog], dtype=float)
    centroid_mean, centroid_sd = float(centroids.mean()), float(centroids.std() or 1.0)
    choose = selector if selector is not None else random_selector(rng)

    for subject_id in range(study.n_subjects):
        traits = SubjectTraits(subject_id, cfg, np.random.default_rng(cfg.seed + subject_id))
        for session_idx in range(study.n_sessions):
            expected_stress = cfg.baseline_stress + cfg.stress_induction_delta * traits.stress_reactivity
            track = choose(subject_id, session_idx, catalog, expected_stress)
            centroid_z = (track.spectral_centroid - centroid_mean) / centroid_sd
            yield simulate_session(
                traits, session_idx, track, centroid_z, eeg, study, cfg, rng
            )


def subject_traits_table(study: StudyConfig, cfg: SyntheticConfig) -> dict[int, SubjectTraits]:
    """Rebuild the latent traits -- used by the closed-loop replay to score
    what a recommender's choice *would* have produced for a given subject."""
    return {
        sid: SubjectTraits(sid, cfg, np.random.default_rng(cfg.seed + sid))
        for sid in range(study.n_subjects)
    }
