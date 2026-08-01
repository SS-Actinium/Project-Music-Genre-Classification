# Changelog

## 0.2.3 — 2026-08-02

### Fixed
- Quieter CLI: suppress TensorFlow / oneDNN / optimizer-load noise on `predict`

## 0.2.2 — 2026-08-02

### Fixed
- Inference when Windows Application Control blocks **numba** (librosa path)
- Load audio via **soundfile** + **scipy/numpy MFCC** fallback
- Allow `.mpeg` / `.mpga` suffixes for uploads/CLI

## 0.2.1 — 2026-08-02

### Fixed
- TensorFlow pin for **Python 3.13** (`>=2.20,<2.23`; old `<2.20` blocked install)
- TF moved to optional extra `[ml]` so `pip install -e .` succeeds without TF
- `python -m music_genre` entry (`__main__.py`) when `music-genre` script not on PATH

## 0.2.0 — 2026-08-02

### Added
- Installable package `src/music_genre` (config, features, train, predict, CLI)
- FastAPI server (`api/main.py`) with upload limits and in-memory rate limit
- `models/genre_mapping.json` + models under `models/`
- Multi-segment mean-probability voting for inference
- Stratified train split, early stopping, metrics JSON
- pytest suite + `scripts/smoke.py`
- GitHub Actions CI (lightweight deps)
- README, AGENTS.md, docs (architecture, model card, team audit)
- MIT LICENSE, `.gitignore`, `pyproject.toml`

### Fixed
- Windows-safe paths via pathlib in package code
- Deterministic alphabetical genre mapping
- Missing mapping artifact for inference

### Notes
- Original Colab notebook retained for research history
- Not a SOTA system; portfolio pilot
