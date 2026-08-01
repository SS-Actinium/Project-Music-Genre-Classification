# Music Genre Classification

**Version 0.2.0** · Portfolio / pilot package for [SS-Actinium](https://github.com/SS-Actinium)

CNN classifier over **MFCCs** trained on the **GTZAN** 10-genre dataset (research notebook → production-style Python package + CLI + optional API).

> **Honest scope:** This is a **student / portfolio pilot**, not a state-of-the-art production music AI. Genre labels are coarse; real-world music is multi-label and culturally contextual.

## Features

- **Package** `src/music_genre/` — config, MFCC features, train, multi-segment predict
- **CLI** `music-genre predict|train|serve`
- **FastAPI** upload endpoint with size limits + rate limit
- **Models** in `models/` (`.keras` + mapping JSON)
- **Original Colab notebook** retained for research history
- **CI** unit tests + smoke (no GTZAN download)

## Genres

`blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock`

## Quickstart

```bash
# clone
git clone https://github.com/SS-Actinium/Project-Music-Genre-Classification.git
cd Project-Music-Genre-Classification

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# smoke (no full TF needed for core tests)
pip install numpy librosa soundfile scikit-learn pytest
pytest -q
python scripts/smoke.py
```

### Predict

```bash
music-genre predict path/to/song.wav --top-k 3
music-genre predict path/to/song.mp3 --json --segments 10
```

Uses multi-segment **mean probability** voting (not first-3-seconds only).

### Train (needs GTZAN on disk)

```bash
# Place GTZAN as Data/genres_original/<genre>/*.wav
music-genre train --data Data/genres_original --output models --epochs 30
```

### API

```bash
music-genre serve --host 127.0.0.1 --port 8080
# POST /predict multipart file=
# GET  /health
```

## Project layout

```
├── Project_Music_Genre_Classification.ipynb  # original Colab research
├── src/music_genre/                          # library
├── api/main.py                               # FastAPI
├── models/                                   # genre_classifier.keras + mapping
├── tests/                                    # offline unit tests
├── scripts/smoke.py
├── docs/                                     # architecture, model card
└── .github/workflows/ci.yml
```

## Model

| Item | Value |
|------|--------|
| Input | MFCC `(126, 13, 1)` ≈ 3 s @ 22.05 kHz |
| Architecture | 3× Conv2D + Dense + Dropout + Softmax(10) |
| Artifacts | `models/genre_classifier.keras`, `models/genre_mapping.json` |
| Saved | 2026-07-29 (Keras 3.x) |

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Dataset note (GTZAN)

GTZAN is a classic MIR research set (~1000×30 s clips, 10 genres). It has known label noise and limited diversity. Redistribute and use according to the **dataset’s own terms** and your institution’s rules — this repo does **not** re-host the full audio. Prefer the official / Hugging Face research mirrors for local training only.

## Limitations

- Single-label 10-way taxonomy (not multi-genre / mood / language)
- MFCC+small CNN; no transformers / large audio foundation models
- Metrics from the original Colab run are not fully logged in-repo — re-train to regenerate `train_metrics.json`
- API rate limit is in-memory (single process)

## License

MIT for **this code**. Dataset and any third-party audio remain under their respective licenses.

## Agency / development

Improved under **AETHER** multi-department workflow (Security, ML, Web, DevOps, QA, Docs). Product OS: [AGENTS.md](AGENTS.md).
