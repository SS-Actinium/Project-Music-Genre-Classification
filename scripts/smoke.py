#!/usr/bin/env python3
"""Offline smoke checks for package integrity (no GTZAN, TF optional)."""

from __future__ import annotations

import struct
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    errors: list[str] = []
    try:
        from music_genre.config import (
            GENRES,
            N_FRAMES,
            N_MFCC,
            PACKAGE_VERSION,
            default_mapping_path,
            default_model_path,
        )
        from music_genre.features import pad_or_truncate_mfcc
        from music_genre.mapping import load_mapping
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"FAIL import: {e}")
        return 1

    print(f"version={PACKAGE_VERSION}")
    if len(GENRES) != 10:
        errors.append(f"GENRES len {len(GENRES)} != 10")

    mapping = load_mapping()
    if len(mapping) != 10:
        errors.append(f"mapping len {len(mapping)} != 10")

    x = pad_or_truncate_mfcc(np.zeros((5, N_MFCC), dtype=np.float32))
    if x.shape != (N_FRAMES, N_MFCC):
        errors.append(f"pad shape {x.shape}")

    model = default_model_path()
    if not model.is_file():
        errors.append(f"model missing: {model}")
    else:
        print(f"model_ok={model}")

    mp = default_mapping_path()
    if not mp.is_file():
        errors.append(f"mapping file missing: {mp}")
    else:
        print(f"mapping_ok={mp}")

    # Optional librosa MFCC shape
    try:
        import librosa  # noqa: F401
        from music_genre.features import mfcc_batch_from_path

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "s.wav"
            n = 22050 * 3
            with wave.open(str(wav), "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
            batch = mfcc_batch_from_path(wav, max_segments=1)
            if batch.shape != (1, N_FRAMES, N_MFCC, 1):
                errors.append(f"mfcc batch {batch.shape}")
            else:
                print("mfcc_shape_ok")
    except ImportError:
        print("librosa_skipped")

    if errors:
        for e in errors:
            print(f"FAIL {e}")
        return 1
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
