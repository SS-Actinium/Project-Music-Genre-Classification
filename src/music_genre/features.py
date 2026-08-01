"""MFCC feature extraction aligned with the trained CNN input shape.

Prefers soundfile + scipy (no numba) so Windows Application Control hosts
that block numba DLLs can still run inference. Falls back to librosa if available.
"""

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


def _resample_mono(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    # Polyphase resample (scipy) — no numba
    from scipy import signal as sps

    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    out = sps.resample_poly(signal, up, down)
    return out.astype(np.float32)


def load_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at target sample rate (soundfile-first)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")

    # 1) soundfile (handles wav/flac/ogg/mp3 when backend supports it)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        mono = data.mean(axis=1)
        return _resample_mono(mono, int(sr), sample_rate)
    except Exception:
        pass

    # 2) librosa (may fail if numba DLL is blocked by Application Control)
    try:
        import librosa

        signal, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        return signal.astype(np.float32, copy=False)
    except Exception as e:
        raise RuntimeError(
            f"Could not load audio {path}. Install soundfile (recommended) "
            f"or fix librosa/numba. Last error: {e}"
        ) from e


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


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    n_fft: int,
    sample_rate: int,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular mel filterbank (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < n_freqs:
                fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < n_freqs:
                fb[i, j] = (right - j) / max(right - center, 1)
    # Slaney-style normalize
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


def mfcc_numpy(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Librosa-compatible-ish MFCC via numpy/scipy (no numba).
    Returns shape (n_mfcc, n_frames) like librosa.feature.mfcc.
    """
    from scipy.fft import dct, rfft

    y = np.asarray(y, dtype=np.float64)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    # Reflect pad like librosa center=True
    pad = n_fft // 2
    y_pad = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (len(y_pad) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1

    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.as_strided(
        y_pad,
        shape=(n_frames, n_fft),
        strides=(y_pad.strides[0] * hop_length, y_pad.strides[0]),
        writeable=False,
    ).copy()
    frames *= window
    # Power spectrum
    spec = np.abs(rfft(frames, n=n_fft, axis=1)) ** 2
    fb = _mel_filterbank(n_fft, sample_rate, n_mels=n_mels)
    mel = spec @ fb.T
    mel = np.maximum(mel, 1e-10)
    log_mel = np.log(mel)
    # DCT type-II, orthonormal — librosa default for MFCC
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfcc.T.astype(np.float32)  # (n_mfcc, time)


def mfcc_from_segment(
    segment: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_frames: int = N_FRAMES,
) -> np.ndarray:
    """Return MFCC array shaped (n_frames, n_mfcc)."""
    # Prefer pure numpy path (Windows Application Control safe)
    try:
        mfcc = mfcc_numpy(
            segment,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mfcc = mfcc.T  # (time, n_mfcc)
        return pad_or_truncate_mfcc(mfcc, n_frames=n_frames)
    except Exception:
        pass

    import librosa

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sample_rate,
        n_fft=n_fft,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )
    mfcc = mfcc.T
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
