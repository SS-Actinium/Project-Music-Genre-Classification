# AGENTS.md — Music Genre Classification

**Product OS** for this repo. Shared workspace standards live in `../agency-memory/` when present (local pack only — not part of this product brand).

## Product

| Field | Value |
|-------|--------|
| Name | Project Music Genre Classification |
| Version | **0.2.3** |
| Stack | Python 3.10+, librosa, TensorFlow/Keras, FastAPI |
| Dataset | GTZAN (not vendored) |
| Models | `models/genre_classifier.keras`, `models/genre_mapping.json` |
| GitHub | `SS-Actinium/Project-Music-Genre-Classification` |

## Standing orders

1. **Honest claims** — portfolio pilot, not SOTA or multi-genre production system.
2. **Do not commit** `Data/`, `data.json`, raw GTZAN tarballs, `.env`.
3. **Mapping SSOT** — 10 alphabetical genres in `models/genre_mapping.json` / `config.GENRES`.
4. **Input shape** — MFCC `(126, 13, 1)` must stay compatible with saved weights unless retrain.
5. **Inference** — multi-segment mean-prob voting via `predict.predict_file`.
6. **Windows-safe paths** — use `pathlib`, never `dirpath.split("/")` alone.
7. **API security** — max upload 20MB, suffix allow-list, temp file cleanup, rate limit.
8. **Skip** corrupt `jazz.00054.wav` on train.
9. **Tests** — offline unit tests must not require GTZAN or GPU.
10. **Notebook** — keep for history; new features go in `src/music_genre/`.

## Departments (auto-select)

Software Eng · Research/ML · Security · Web/API · DevOps · QA · Documentation

## Quality gates

- `pytest -q` green  
- `python scripts/smoke.py` → `SMOKE_OK`  
- No secrets in git  
- README limitations section intact  
- **Versioning:** bump semver + CHANGELOG + annotated tag `vX.Y.Z` + push on every meaningful ship  

## Git versioning (standing)

See global rule `~/.grok/rules/project-git-workflow.md` — always maintain git log, CHANGELOG, and tags.
