"""Inference: multi-segment MFCC + CNN with mean-probability voting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import default_mapping_path, default_model_path
from .features import mfcc_batch_from_path
from .mapping import load_mapping


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass
class PredictionResult:
    genre: str
    confidence: float
    top_k: list[dict[str, Any]]
    n_segments: int
    audio_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "confidence": self.confidence,
            "top_k": self.top_k,
            "n_segments": self.n_segments,
            "audio_path": self.audio_path,
        }


_model_cache: dict[str, Any] = {}
_tf_quiet_done = False


def configure_tensorflow_quiet() -> None:
    """Reduce TensorFlow/oneDNN/absl console noise for CLI use."""
    global _tf_quiet_done
    if _tf_quiet_done:
        return
    import os
    import warnings

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=UserWarning, module="keras")
    warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
    warnings.filterwarnings("ignore", message=".*TensorFlow GPU support is not available.*")
    _tf_quiet_done = True


def load_keras_model(model_path: str | Path):
    """Load and cache Keras model (requires tensorflow)."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise ModelNotFoundError(f"Model not found: {path}")
    key = str(path)
    if key not in _model_cache:
        configure_tensorflow_quiet()
        import tensorflow as tf

        try:
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
        _model_cache[key] = tf.keras.models.load_model(path)
    return _model_cache[key]


def predict_proba_batch(model, batch: np.ndarray) -> np.ndarray:
    """Return probabilities shape (n_segments, n_classes)."""
    preds = model.predict(batch, verbose=0)
    return np.asarray(preds)


def aggregate_mean_proba(probs: np.ndarray) -> np.ndarray:
    """Mean probability across segments."""
    if probs.ndim != 2:
        raise ValueError("probs must be 2D")
    return probs.mean(axis=0)


def top_k_from_proba(
    mean_proba: np.ndarray,
    mapping: list[str],
    k: int = 3,
) -> list[dict[str, Any]]:
    k = max(1, min(k, len(mapping)))
    order = np.argsort(mean_proba)[::-1][:k]
    return [
        {"genre": mapping[int(i)], "confidence": float(mean_proba[int(i)])}
        for i in order
    ]


def predict_file(
    audio_path: str | Path,
    model_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    max_segments: int = 10,
    top_k: int = 3,
) -> PredictionResult:
    """
    Predict genre for an audio file using multi-segment mean-prob voting.
    """
    audio_path = Path(audio_path)
    model_path = Path(model_path) if model_path else default_model_path()
    mapping_path = Path(mapping_path) if mapping_path else default_mapping_path()

    mapping = load_mapping(mapping_path)
    model = load_keras_model(model_path)
    batch = mfcc_batch_from_path(audio_path, max_segments=max_segments)
    probs = predict_proba_batch(model, batch)
    mean_p = aggregate_mean_proba(probs)
    ranking = top_k_from_proba(mean_p, mapping, k=top_k)
    best = ranking[0]
    return PredictionResult(
        genre=best["genre"],
        confidence=float(best["confidence"]),
        top_k=ranking,
        n_segments=int(batch.shape[0]),
        audio_path=str(audio_path),
    )
