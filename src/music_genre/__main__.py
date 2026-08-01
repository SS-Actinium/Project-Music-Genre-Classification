"""Allow: python -m music_genre predict ..."""

import os

# Must run before TensorFlow is imported anywhere
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

from .cli import main

raise SystemExit(main())
