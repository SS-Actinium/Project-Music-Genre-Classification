"""Shared configuration for training and inference (matches trained model)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_VERSION = "0.2.0"

# Canonical GTZAN genres — alphabetical (matches training remap in original notebook)
GENRES: list[str] = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

SAMPLE_RATE = 22_050
TRACK_DURATION_SEC = 30
NUM_SEGMENTS = 10
SEGMENT_DURATION_SEC = TRACK_DURATION_SEC / NUM_SEGMENTS  # 3.0
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
# Model InputLayer batch_shape: [null, 126, 13, 1]
N_FRAMES = 126
N_CHANNELS = 1

# Known corrupt GTZAN sample
SKIP_FILES = frozenset({"jazz.00054.wav"})

# API / CLI safety
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MiB
ALLOWED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    epochs: int = 30
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42
    num_segments: int = NUM_SEGMENTS
    n_mfcc: int = N_MFCC
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    dropout: float = 0.3
    early_stopping_patience: int = 5


def package_root() -> Path:
    """Repo root: .../project-music-genre-classification."""
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    root = package_root()
    for candidate in (
        root / "models" / "genre_classifier.keras",
        root / "genre_classifier.keras",
        root / "models" / "genre_classifier.h5",
        root / "genre_classifier.h5",
    ):
        if candidate.is_file():
            return candidate
    return root / "models" / "genre_classifier.keras"


def default_mapping_path() -> Path:
    root = package_root()
    for candidate in (
        root / "models" / "genre_mapping.json",
        root / "genre_mapping.json",
    ):
        if candidate.is_file():
            return candidate
    return root / "models" / "genre_mapping.json"
