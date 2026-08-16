"""Time-frequency images: EEG epochs -> [n_roi x n_freqs x n_times] spectrograms.

Two transforms, one output geometry. STFT is fast and uniform-resolution; the
Morlet CWT gives better low-frequency resolution, which matters for theta and
alpha. Both are resampled onto the same fixed (n_freqs, n_times) grid so the
CNN input shape is a config value rather than a consequence of window sizes.

Note on CWT: `scipy.signal.cwt` was removed in SciPy 1.15, so this uses
`mne.time_frequency.tfr_array_morlet` instead of the recipe found in most
older tutorials.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from ..config import TimeFrequencyConfig
from ..types import EpochSet


def _interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Linear-interpolation weights mapping `src` samples onto `dst` positions.

    Returns (len(dst), len(src)). Built once and applied with einsum, which is
    far cheaper than per-epoch interpolation calls.
    """
    if src.size < 2:
        raise ValueError("source grid needs at least 2 points to interpolate")
    weights = np.zeros((dst.size, src.size), dtype=np.float32)
    positions = np.clip(np.interp(dst, src, np.arange(src.size)), 0, src.size - 1)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, src.size - 1)
    frac = (positions - lower).astype(np.float32)
    rows = np.arange(dst.size)
    weights[rows, lower] += 1.0 - frac
    weights[rows, upper] += frac
    return weights


def _resample_grid(
    power: np.ndarray,
    src_freqs: np.ndarray,
    src_times: np.ndarray,
    cfg: TimeFrequencyConfig,
) -> np.ndarray:
    """Map (..., n_src_freqs, n_src_times) onto the configured fixed grid."""
    dst_freqs = np.linspace(cfg.fmin, cfg.fmax, cfg.n_freqs)
    dst_times = np.linspace(src_times[0], src_times[-1], cfg.n_times)
    w_freq = _interp_matrix(src_freqs, dst_freqs)
    w_time = _interp_matrix(src_times, dst_times)
    resampled = np.einsum("af,...ft->...at", w_freq, power, optimize=True)
    return np.einsum("bt,...at->...ab", w_time, resampled, optimize=True)


def stft_images(signals: np.ndarray, sfreq: float, cfg: TimeFrequencyConfig) -> np.ndarray:
    """Short-time Fourier magnitude-squared, on the configured grid.

    `signals` is (n_epochs, n_channels, n_samples).
    """
    nperseg = min(cfg.stft_nperseg, signals.shape[-1])
    noverlap = int(nperseg * cfg.stft_overlap_ratio)
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    freqs, times, spectrum = stft(
        signals, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1
    )
    power = np.abs(spectrum) ** 2
    band = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
    if band.sum() < 2:
        raise ValueError(
            f"STFT produced {band.sum()} bins in {cfg.fmin}-{cfg.fmax} Hz; "
            f"increase stft_nperseg (currently {nperseg})"
        )
    return _resample_grid(power[..., band, :], freqs[band], times, cfg)


def cwt_images(signals: np.ndarray, sfreq: float, cfg: TimeFrequencyConfig) -> np.ndarray:
    """Morlet wavelet power via MNE, on the configured grid."""
    from mne.time_frequency import tfr_array_morlet

    freqs = np.linspace(cfg.fmin, cfg.fmax, cfg.n_freqs)
    # Fewer cycles at low frequencies, or the wavelet outgrows a 2 s epoch.
    n_cycles = np.clip(freqs / 2.0, 1.0, 7.0)
    power = tfr_array_morlet(
        signals.astype(np.float64),
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        verbose="ERROR",
    )
    times = np.arange(power.shape[-1]) / sfreq
    return _resample_grid(power, freqs, times, cfg)


def compute_images(
    epochs: EpochSet,
    cfg: TimeFrequencyConfig,
    batch_size: int = 512,
) -> np.ndarray:
    """Spectrogram bank for an EpochSet: (n_epochs, n_channels, n_freqs, n_times).

    Batched because the intermediate STFT of 36,000 epochs at once is several
    gigabytes even though the output is under one.
    """
    if len(epochs) == 0:
        raise ValueError("cannot compute images for an empty EpochSet")
    transform = {"stft": stft_images, "cwt": cwt_images}[cfg.method]

    chunks: list[np.ndarray] = []
    for start in range(0, len(epochs), batch_size):
        block = epochs.signals[start : start + batch_size]
        images = transform(block, epochs.sfreq, cfg)
        if cfg.log_power:
            images = np.log1p(np.clip(images, 0.0, None))
        chunks.append(images.astype(np.float32))

    bank = np.concatenate(chunks, axis=0)
    expected = (len(epochs), len(epochs.channel_names), cfg.n_freqs, cfg.n_times)
    if bank.shape != expected:
        raise ValueError(f"spectrogram bank has shape {bank.shape}, expected {expected}")

    # Standardise per channel across the bank: log-power scales differ by an
    # order of magnitude between channels and the CNN converges poorly without.
    mean = bank.mean(axis=(0, 2, 3), keepdims=True)
    sd = bank.std(axis=(0, 2, 3), keepdims=True)
    return ((bank - mean) / np.where(sd < 1e-12, 1.0, sd)).astype(np.float32)
