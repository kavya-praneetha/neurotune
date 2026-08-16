"""Classic spectral features: band powers and the ratios used as stress markers.

These are the interpretable half of the dual feature approach. They feed the
Random Forest baseline, every statistical test, and the raga-physiology
regression -- so they are computed on the *microvolt* epochs, never the
z-scored ones. Units are uV^2 per band, which is what "alpha rose by
1.3 uV^2" actually means.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..config import BandConfig, EEGConfig
from ..types import EpochSet, Phase, SessionSignal

#: Guard against divide-by-zero when a denominator band is silent.
_EPS = 1e-12


def _welch_psd(signals: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD over the last axis. Returns (freqs, psd) with psd in uV^2/Hz."""
    nperseg = min(signals.shape[-1], int(sfreq))
    freqs, psd = welch(signals, fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd


def band_powers(
    signals: np.ndarray,
    sfreq: float,
    bands: BandConfig,
) -> dict[str, np.ndarray]:
    """Absolute band power per epoch per channel.

    `signals` is (..., n_samples) in microvolts. Each returned array has the
    leading shape of `signals` minus the sample axis.
    """
    if signals.shape[-1] < 8:
        raise ValueError(f"need at least 8 samples to estimate a PSD, got {signals.shape[-1]}")
    freqs, psd = _welch_psd(signals, sfreq)
    out: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.bands.items():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            raise ValueError(
                f"band {name!r} ({low}-{high} Hz) contains no PSD bins at "
                f"sfreq={sfreq}; frequency resolution is {freqs[1] - freqs[0]:.3f} Hz"
            )
        # Integrate the PSD across the band -> power in uV^2.
        out[name] = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
    return out


def band_ratios(powers: dict[str, np.ndarray], bands: BandConfig) -> dict[str, np.ndarray]:
    """Ratios such as beta/alpha (arousal) and theta/beta (inattention)."""
    return {
        f"{num}_{den}": powers[num] / (powers[den] + _EPS)
        for num, den in bands.ratios
    }


def epoch_feature_matrix(
    epochs: EpochSet,
    bands: BandConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Flatten band powers and ratios into a (n_epochs, n_features) design matrix.

    This is the Random Forest baseline's entire input -- deliberately so, since
    the comparison being made is "learned time-frequency representation" versus
    "hand-designed band features".
    """
    powers = band_powers(epochs.signals, epochs.sfreq, bands)
    ratios = band_ratios(powers, bands)

    columns: list[np.ndarray] = []
    names: list[str] = []
    for name, values in powers.items():  # (n_epochs, n_channels)
        for ch_idx, channel in enumerate(epochs.channel_names):
            columns.append(values[:, ch_idx])
            names.append(f"{name}_{channel}")
    for name, values in ratios.items():
        for ch_idx, channel in enumerate(epochs.channel_names):
            columns.append(values[:, ch_idx])
            names.append(f"ratio_{name}_{channel}")

    # Log-transform powers: band power is heavily right-skewed, and the tree
    # ensemble is scale-free but the downstream linear models are not.
    matrix = np.column_stack(columns).astype(np.float64)
    matrix = np.log1p(np.clip(matrix, 0.0, None))
    return matrix, tuple(names)


def phase_band_summary(
    session: SessionSignal,
    eeg: EEGConfig,
    bands: BandConfig,
    channel: str,
) -> dict[Phase, dict[str, float]]:
    """Mean band power per phase at one channel, from a cleaned uV recording.

    Used for the pre/post intervention contrast: `Fz` is the conventional
    frontal-midline site for theta and for the beta/alpha arousal ratio.
    """
    if channel not in session.channel_names:
        raise ValueError(
            f"channel {channel!r} not in recording; available: {list(session.channel_names)}"
        )
    ch_idx = session.channel_names.index(channel)
    width = eeg.epoch_samples
    summary: dict[Phase, dict[str, float]] = {}

    for phase, (start, stop) in session.phase_bounds.items():
        segment = session.data[ch_idx, start:stop]
        n_epochs = segment.size // width
        if n_epochs == 0:
            raise ValueError(f"phase {phase.value} is shorter than one epoch")
        trimmed = segment[: n_epochs * width].reshape(n_epochs, width)
        powers = band_powers(trimmed, session.sfreq, bands)
        ratios = band_ratios(powers, bands)
        summary[phase] = {
            **{name: float(np.mean(value)) for name, value in powers.items()},
            **{f"ratio_{name}": float(np.mean(value)) for name, value in ratios.items()},
        }
    return summary
