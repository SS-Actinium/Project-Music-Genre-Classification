"""Unit tests — no TensorFlow required."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from music_genre.config import GENRES, N_FRAMES, N_MFCC, SAMPLE_RATE
from music_genre.features import (
    AudioTooShortError,
    expected_frame_count,
    pad_or_truncate_mfcc,
    segment_signal,
)
from music_genre.mapping import load_mapping, save_mapping
from music_genre.predict import aggregate_mean_proba, top_k_from_proba


def test_pad_or_truncate_short():
    x = np.ones((10, N_MFCC), dtype=np.float32)
    y = pad_or_truncate_mfcc(x, n_frames=N_FRAMES)
    assert y.shape == (N_FRAMES, N_MFCC)
    assert np.allclose(y[:10], 1.0)
    assert np.allclose(y[10:], 0.0)


def test_pad_or_truncate_long():
    x = np.arange(N_FRAMES + 50, dtype=np.float32).reshape(-1, 1)
    x = np.repeat(x, N_MFCC, axis=1)
    y = pad_or_truncate_mfcc(x, n_frames=N_FRAMES)
    assert y.shape == (N_FRAMES, N_MFCC)


def test_expected_frame_count_3s():
    samples = int(SAMPLE_RATE * 3)
    assert expected_frame_count(samples) == N_FRAMES


def test_segment_signal_count():
    sig = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    segs = segment_signal(sig, max_segments=4)
    assert len(segs) == 4
    assert all(len(s) == int(SAMPLE_RATE * 3) for s in segs)


def test_segment_too_short():
    sig = np.zeros(100, dtype=np.float32)
    with pytest.raises(AudioTooShortError):
        segment_signal(sig)


def test_load_mapping_fallback(tmp_path: Path):
    missing = tmp_path / "nope.json"
    m = load_mapping(missing)
    assert m == GENRES


def test_save_and_load_mapping(tmp_path: Path):
    path = tmp_path / "m.json"
    save_mapping(["a", "b", "c"], path)
    assert load_mapping(path) == ["a", "b", "c"]


def test_aggregate_mean_proba():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mean = aggregate_mean_proba(probs)
    assert np.allclose(mean, [0.5, 0.5])


def test_top_k_from_proba():
    mean = np.array([0.1, 0.7, 0.2])
    rows = top_k_from_proba(mean, ["a", "b", "c"], k=2)
    assert rows[0]["genre"] == "b"
    assert rows[1]["genre"] == "c"


def _write_silence_wav(path: Path, seconds: float = 3.0, sr: int = SAMPLE_RATE) -> None:
    n = int(sr * seconds)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def test_mfcc_batch_shape_optional_librosa(tmp_path: Path):
    pytest.importorskip("librosa")
    wav = tmp_path / "silence.wav"
    _write_silence_wav(wav, seconds=6.0)
    from music_genre.features import mfcc_batch_from_path

    try:
        batch = mfcc_batch_from_path(wav, max_segments=2)
    except ImportError as e:
        # e.g. numba DLL blocked by host Application Control policy
        pytest.skip(f"librosa audio backend unavailable: {e}")
    assert batch.shape == (2, N_FRAMES, N_MFCC, 1)
