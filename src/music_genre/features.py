"""MFCC feature extraction aligned with the trained CNN input shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import (
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MFCC,
    SAMPLE_RATE,
    SEGMENT_DURATION_SEC,
)


class AudioTooShortError(ValueError):
    """Raised when audio cannot fill a minimum analysis window."""


def expected_frame_count(
    samples_per_segment: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> int:
    """Match original notebook formula for MFCC time frames."""
    return 1 + (samples_per_segment - n_fft) // hop_length


def pad_or_truncate_mfcc(mfcc: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """Force MFCC to (n_frames, n_mfcc)."""
    if mfcc.ndim != 2:
        raise ValueError(f"Expected 2D MFCC, got shape {mfcc.shape}")
    if len(mfcc) > n_frames:
        return mfcc[:n_frames, :]
    if len(mfcc) < n_frames:
        pad = n_frames - len(mfcc)
        return np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    return mfcc


def load_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at target sample rate (requires librosa)."""
    import librosa

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")
    signal, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return signal.astype(np.float32, copy=False)


def segment_signal(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION_SEC,
    max_segments: int | None = None,
) -> list[np.ndarray]:
    """Split signal into fixed-length segments (pad last if partial but non-empty)."""
    samples_per_segment = int(sample_rate * segment_duration)
    if samples_per_segment <= 0:
        raise ValueError("Invalid segment duration")
    if len(signal) < sample_rate * 0.5:
        raise AudioTooShortError("Audio shorter than 0.5s")

    segments: list[np.ndarray] = []
    n_full = max(1, int(np.ceil(len(signal) / samples_per_segment)))
    if max_segments is not None:
        n_full = min(n_full, max_segments)

    for i in range(n_full):
        start = i * samples_per_segment
        finish = start + samples_per_segment
        if start >= len(signal):
            break
        chunk = signal[start:finish]
        if len(chunk) < samples_per_segment:
            chunk = np.pad(chunk, (0, samples_per_segment - len(chunk)), mode="constant")
        segments.append(chunk)
    if not segments:
        raise AudioTooShortError("No segments extracted")
    return segments


def mfcc_from_segment(
    segment: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_frames: int = N_FRAMES,
) -> np.ndarray:
    """Return MFCC array shaped (n_frames, n_mfcc)."""
    import librosa

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sample_rate,
        n_fft=n_fft,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )
    mfcc = mfcc.T  # (time, n_mfcc)
    return pad_or_truncate_mfcc(mfcc, n_frames=n_frames)


def mfcc_batch_from_path(
    path: str | Path,
    max_segments: int = 10,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio and return batch of MFCCs ready for CNN:
    shape (n_segments, n_frames, n_mfcc, 1)
    """
    signal = load_mono(path, sample_rate=sample_rate)
    segments = segment_signal(signal, sample_rate=sample_rate, max_segments=max_segments)
    frames = [mfcc_from_segment(seg, sample_rate=sample_rate) for seg in segments]
    batch = np.stack(frames, axis=0)[..., np.newaxis]
    return batch.astype(np.float32)
