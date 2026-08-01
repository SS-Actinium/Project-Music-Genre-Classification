# Changelog

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
