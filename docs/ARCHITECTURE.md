# Architecture

## Pipeline

```
Audio file
  → mono resample 22050 Hz (librosa)
  → split into 3 s segments (max N)
  → MFCC (n_mfcc=13, n_fft=2048, hop=512)
  → pad/truncate to 126 frames
  → shape (batch, 126, 13, 1)
  → Keras CNN → softmax(10)
  → mean probability across segments
  → top-k genres
```

## Why multi-segment voting

The original notebook inferred on the **first 3 seconds only**. That biases toward intros. Mean-prob voting over up to 10 segments is more stable for full tracks.

## Package map

| Module | Role |
|--------|------|
| `config.py` | constants, paths, TrainConfig |
| `features.py` | load audio, segments, MFCC |
| `mapping.py` | genre JSON |
| `predict.py` | model cache + inference |
| `train.py` | stratified train + early stopping |
| `cli.py` | argparse entrypoint |
| `api/main.py` | FastAPI |

## Compatibility

Trained weights expect **exactly** `126×13×1`. Changing MFCC hyperparameters requires full retrain.
