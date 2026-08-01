"""CLI: music-genre predict | train | serve | version."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PACKAGE_VERSION, TrainConfig, default_mapping_path, default_model_path


def _cmd_predict(args: argparse.Namespace) -> int:
    from .features import AudioTooShortError
    from .mapping import MappingError
    from .predict import ModelNotFoundError, predict_file

    try:
        result = predict_file(
            args.audio,
            model_path=args.model,
            mapping_path=args.mapping,
            max_segments=args.segments,
            top_k=args.top_k,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ModelNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except MappingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except AudioTooShortError as e:
        print(f"error: {e}", file=sys.stderr)
        return 5
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"error: prediction failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"file:       {result.audio_path}")
        print(f"genre:      {result.genre}")
        print(f"confidence: {result.confidence:.2%}")
        print(f"segments:   {result.n_segments}")
        print("top:")
        for row in result.top_k:
            print(f"  - {row['genre']}: {row['confidence']:.2%}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    cfg = TrainConfig(
        dataset_path=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    metrics = train(cfg)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="music-genre",
        description="GTZAN music genre classifier (MFCC + CNN)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {PACKAGE_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    pred = sub.add_parser("predict", help="Predict genre for an audio file")
    pred.add_argument("audio", type=str, help="Path to wav/mp3/flac/ogg")
    pred.add_argument("--model", type=str, default=None, help="Path to .keras/.h5 model")
    pred.add_argument("--mapping", type=str, default=None, help="Path to genre_mapping.json")
    pred.add_argument("--segments", type=int, default=10, help="Max segments to analyze")
    pred.add_argument("--top-k", type=int, default=3, dest="top_k")
    pred.add_argument("--json", action="store_true", help="JSON output")
    pred.set_defaults(func=_cmd_predict)

    tr = sub.add_parser("train", help="Train on GTZAN-style folder dataset")
    tr.add_argument("--data", required=True, help="Path to genres_original (folder per genre)")
    tr.add_argument("--output", default="models", help="Output directory for model + mapping")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    tr.set_defaults(func=_cmd_train)

    srv = sub.add_parser("serve", help="Run FastAPI inference server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8080)
    srv.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Defaults printed for transparency when missing
    if getattr(args, "model", None) is None and args.command == "predict":
        args.model = str(default_model_path())
    if getattr(args, "mapping", None) is None and args.command == "predict":
        args.mapping = str(default_mapping_path())
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
