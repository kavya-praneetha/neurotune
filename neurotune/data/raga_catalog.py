"""Raga track catalogue: synthesised audio, genuinely measured features.

Rather than inventing numbers for spectral centroid / MFCC / ZCR, this module
*renders actual audio* for each track -- a tanpura-style drone plus a melodic
line drawn from the raga's scale at the track's tempo -- and then runs librosa
over the waveform. The descriptors are therefore real measurements of a real
signal, and swapping in a directory of genuine recordings changes only the
loader, not the feature code.

Scale degrees are semitone offsets from the tonic (Sa). The raga set below is
a small, deliberately simple sample for pipeline development; a musicologist
should review it before any of it is used as a claim about the ragas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import MusicConfig
from ..types import TrackFeatures


@dataclass(frozen=True)
class RagaSpec:
    """Name, ascending scale in semitones from Sa, and its conventional mood."""

    name: str
    degrees: tuple[int, ...]
    traditional_mood: str


#: Semitone sets are the common aroha forms; see module docstring caveat.
RAGA_SPECS: tuple[RagaSpec, ...] = (
    RagaSpec("Malkauns", (0, 3, 5, 8, 10), "calming, introspective"),
    RagaSpec("Bhairavi", (0, 1, 3, 5, 7, 8, 10), "devotional, soothing"),
    RagaSpec("Yaman", (0, 2, 4, 6, 7, 9, 11), "serene, uplifting"),
    RagaSpec("Darbari Kanada", (0, 2, 3, 5, 7, 8, 10), "deep, meditative"),
    RagaSpec("Bhupali", (0, 2, 4, 7, 9), "tranquil, spacious"),
    RagaSpec("Todi", (0, 1, 3, 6, 7, 8, 11), "pensive"),
    RagaSpec("Desh", (0, 2, 4, 5, 7, 9, 10), "gentle, romantic"),
    RagaSpec("Marwa", (0, 1, 4, 6, 7, 9, 11), "contemplative"),
    RagaSpec("Kafi", (0, 2, 3, 5, 7, 9, 10), "warm, pastoral"),
    RagaSpec("Charukeshi", (0, 2, 4, 5, 7, 8, 10), "wistful"),
)

_TONIC_HZ = 146.83  # D3, a common tanpura Sa


def _render_track(
    spec: RagaSpec,
    tempo_bpm: float,
    rhythmic_intensity: float,
    drone: bool,
    cfg: MusicConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive synthesis of a short excerpt. Returns mono float32 in [-1, 1]."""
    n = int(cfg.audio_seconds * cfg.audio_sr)
    t = np.arange(n) / cfg.audio_sr
    audio = np.zeros(n, dtype=np.float64)

    if drone:
        # Tanpura-ish: tonic + fifth, several harmonics, very slow beating.
        for ratio, gain in ((1.0, 0.5), (1.5, 0.35), (2.0, 0.2), (3.0, 0.08)):
            freq = _TONIC_HZ * ratio
            beat = 1.0 + 0.004 * np.sin(2 * np.pi * 0.3 * t)
            audio += gain * np.sin(2 * np.pi * freq * beat * t)

    # Melodic line: one note per beat, drawn from the raga's scale.
    seconds_per_beat = 60.0 / tempo_bpm
    n_notes = max(1, int(cfg.audio_seconds / seconds_per_beat))
    note_len = int(seconds_per_beat * cfg.audio_sr)
    envelope_shape = np.hanning(max(note_len, 2))

    for i in range(n_notes):
        start = i * note_len
        stop = min(start + note_len, n)
        if stop <= start:
            break
        degree = spec.degrees[int(rng.integers(0, len(spec.degrees)))]
        octave = int(rng.integers(0, 2))
        freq = _TONIC_HZ * 2 ** ((degree + 12 * octave) / 12.0)
        local_t = np.arange(stop - start) / cfg.audio_sr
        note = np.sin(2 * np.pi * freq * local_t)
        note += 0.3 * np.sin(2 * np.pi * 2 * freq * local_t)  # timbre
        audio[start:stop] += 0.45 * note * envelope_shape[: stop - start]

    # Percussion proxy: rhythmic intensity controls transient density and bite.
    if rhythmic_intensity > 0.05:
        stride = max(1, int(note_len / (1 + 3 * rhythmic_intensity)))
        click = np.exp(-np.linspace(0, 12, max(2, int(0.03 * cfg.audio_sr))))
        for start in range(0, n - click.size, stride):
            noise = rng.standard_normal(click.size) * click
            audio[start : start + click.size] += rhythmic_intensity * 0.5 * noise

    peak = float(np.max(np.abs(audio)) or 1.0)
    return (audio / peak * 0.95).astype(np.float32)


def _measure(audio: np.ndarray, cfg: MusicConfig) -> dict[str, object]:
    """Compute librosa descriptors on a rendered or loaded waveform."""
    import librosa  # imported lazily: heavy, and only needed when building a catalogue

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=cfg.audio_sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    mfcc = librosa.feature.mfcc(y=audio, sr=cfg.audio_sr, n_mfcc=cfg.n_mfcc)
    return {
        "spectral_centroid": centroid,
        "zero_crossing_rate": zcr,
        "mfcc_mean": tuple(float(v) for v in np.mean(mfcc, axis=1)),
    }


def build_synthetic_catalog(cfg: MusicConfig, seed: int = 7) -> tuple[TrackFeatures, ...]:
    """Render `cfg.n_tracks` excerpts and measure each one."""
    rng = np.random.default_rng(seed)
    tracks: list[TrackFeatures] = []

    for idx in range(cfg.n_tracks):
        spec = RAGA_SPECS[idx % len(RAGA_SPECS)]
        tempo = float(rng.uniform(*cfg.tempo_range))
        rhythm = float(np.clip(rng.beta(2.0, 2.5), 0.0, 1.0))
        drone = bool(rng.random() < cfg.drone_probability)
        audio = _render_track(spec, tempo, rhythm, drone, cfg, rng)
        measured = _measure(audio, cfg)
        tracks.append(
            TrackFeatures(
                track_id=f"trk_{idx:03d}",
                raga=spec.name,
                scale=",".join(str(d) for d in spec.degrees),
                tempo_bpm=tempo,
                rhythmic_intensity=rhythm,
                drone=drone,
                spectral_centroid=float(measured["spectral_centroid"]),
                zero_crossing_rate=float(measured["zero_crossing_rate"]),
                mfcc_mean=measured["mfcc_mean"],  # type: ignore[arg-type]
            )
        )
    return tuple(tracks)


def load_audio_catalog(
    audio_dir: Path,
    metadata_csv: Path,
    cfg: MusicConfig,
) -> tuple[TrackFeatures, ...]:
    """Build a catalogue from real recordings plus a metadata table.

    `metadata_csv` must have columns: track_id, filename, raga, scale,
    tempo_bpm, rhythmic_intensity, drone. Audio features are measured from the
    files; tempo is taken from metadata when present and estimated otherwise.
    """
    import librosa
    import pandas as pd

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio_dir does not exist: {audio_dir}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata_csv does not exist: {metadata_csv}")

    frame = pd.read_csv(metadata_csv)
    required = {"track_id", "filename", "raga", "scale", "rhythmic_intensity", "drone"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{metadata_csv} is missing required columns: {sorted(missing)}")

    tracks: list[TrackFeatures] = []
    for row in frame.itertuples(index=False):
        path = audio_dir / str(row.filename)
        if not path.is_file():
            raise FileNotFoundError(f"track {row.track_id}: audio file not found at {path}")
        audio, _ = librosa.load(path, sr=cfg.audio_sr, mono=True, duration=cfg.audio_seconds)
        measured = _measure(audio, cfg)
        tempo = getattr(row, "tempo_bpm", None)
        if tempo is None or not np.isfinite(float(tempo)):
            estimated, _ = librosa.beat.beat_track(y=audio, sr=cfg.audio_sr)
            tempo = float(np.atleast_1d(estimated)[0])
        tracks.append(
            TrackFeatures(
                track_id=str(row.track_id),
                raga=str(row.raga),
                scale=str(row.scale),
                tempo_bpm=float(tempo),
                rhythmic_intensity=float(row.rhythmic_intensity),
                drone=bool(row.drone),
                spectral_centroid=float(measured["spectral_centroid"]),
                zero_crossing_rate=float(measured["zero_crossing_rate"]),
                mfcc_mean=measured["mfcc_mean"],  # type: ignore[arg-type]
            )
        )
    if not tracks:
        raise ValueError(f"{metadata_csv} produced no tracks")
    return tuple(tracks)


def catalog_to_frame(catalog: tuple[TrackFeatures, ...]):
    """Flatten a catalogue into a DataFrame for Stage-1 metadata filtering."""
    import pandas as pd

    rows = []
    for track in catalog:
        row = {
            "track_id": track.track_id,
            "raga": track.raga,
            "scale": track.scale,
            "tempo_bpm": track.tempo_bpm,
            "rhythmic_intensity": track.rhythmic_intensity,
            "drone": track.drone,
            "spectral_centroid": track.spectral_centroid,
            "zero_crossing_rate": track.zero_crossing_rate,
        }
        row.update({f"mfcc_{i}": v for i, v in enumerate(track.mfcc_mean)})
        rows.append(row)
    return pd.DataFrame(rows)
