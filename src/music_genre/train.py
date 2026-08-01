"""Reproducible training on GTZAN-style folder layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    GENRES,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MFCC,
    SAMPLE_RATE,
    SKIP_FILES,
    TrainConfig,
)
from .features import mfcc_from_segment, pad_or_truncate_mfcc, segment_signal
from .mapping import save_mapping


def set_seeds(seed: int = 42) -> None:
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def extract_dataset_mfcc(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk dataset_path/<genre>/*.wav and build X, y with sorted genre labels."""
    import librosa

    dataset = Path(cfg.dataset_path)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    # Prefer fixed canonical list when folders match; else sorted folder names
    present = sorted(
        [
            p.name
            for p in dataset.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name != "genres"
        ]
    )
    if set(present) >= set(GENRES):
        genres = list(GENRES)
    else:
        genres = present
    genre_to_id = {g: i for i, g in enumerate(genres)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for genre in genres:
        gdir = dataset / genre
        if not gdir.is_dir():
            continue
        for wav in sorted(gdir.glob("*.wav")):
            if wav.name in SKIP_FILES or wav.name.startswith("._"):
                continue
            signal, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
            for seg in segment_signal(
                signal,
                sample_rate=SAMPLE_RATE,
                max_segments=cfg.num_segments,
            ):
                mfcc = mfcc_from_segment(
                    seg,
                    sample_rate=SAMPLE_RATE,
                    n_mfcc=cfg.n_mfcc,
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    n_frames=N_FRAMES,
                )
                mfcc = pad_or_truncate_mfcc(mfcc, n_frames=N_FRAMES)
                X_list.append(mfcc)
                y_list.append(genre_to_id[genre])

    if not X_list:
        raise RuntimeError("No training samples found — check dataset path")
    X = np.stack(X_list, axis=0)[..., np.newaxis].astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y, genres


def build_cnn(input_shape: tuple[int, int, int], n_classes: int, dropout: float = 0.3):
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            Conv2D(64, (2, 2), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(dropout),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(cfg: TrainConfig) -> dict:
    """Train model; write keras weights + mapping under cfg.output_dir."""
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    set_seeds(cfg.random_state)
    X, y, genres = extract_dataset_mfcc(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    model = build_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2], 1),
        n_classes=len(genres),
        dropout=cfg.dropout,
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    keras_path = out / "genre_classifier.keras"
    map_path = out / "genre_mapping.json"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(str(keras_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Ensure final save
    model.save(keras_path)
    save_mapping(genres, map_path)

    val_acc = float(history.history.get("val_accuracy", [0])[-1])
    metrics = {
        "val_accuracy_last": val_acc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "genres": genres,
        "model_path": str(keras_path),
        "mapping_path": str(map_path),
    }
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
