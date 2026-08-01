"""FastAPI inference server — fail-closed uploads, rate-limited predict."""

from __future__ import annotations

import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from music_genre.config import (  # noqa: E402
    ALLOWED_AUDIO_SUFFIXES,
    MAX_UPLOAD_BYTES,
    PACKAGE_VERSION,
    default_mapping_path,
    default_model_path,
)
from music_genre.features import AudioTooShortError  # noqa: E402
from music_genre.predict import ModelNotFoundError, predict_file  # noqa: E402

app = FastAPI(
    title="Music Genre Classifier",
    version=PACKAGE_VERSION,
    description="GTZAN-trained MFCC CNN — portfolio pilot, not production SOTA.",
)

# Simple in-memory rate limit: 20 requests / 60s / client host
_RATE_MAX = 20
_RATE_WINDOW = 60.0
_hits: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > _RATE_WINDOW:
        q.popleft()
    if len(q) >= _RATE_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (20/min)")
    q.append(now)


@app.get("/health")
def health() -> dict:
    model = default_model_path()
    mapping = default_mapping_path()
    return {
        "ok": True,
        "version": PACKAGE_VERSION,
        "model_present": model.is_file(),
        "mapping_present": mapping.is_file(),
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    segments: int = 10,
    top_k: int = 3,
) -> JSONResponse:
    _rate_limit(request)

    filename = file.filename or "upload.bin"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suffix!r}. Allowed: {sorted(ALLOWED_AUDIO_SUFFIXES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes)",
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        result = predict_file(
            tmp_path,
            max_segments=max(1, min(segments, 20)),
            top_k=max(1, min(top_k, 10)),
        )
        payload = result.to_dict()
        payload["audio_path"] = filename  # do not leak server temp path
        return JSONResponse(payload)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except AudioTooShortError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
