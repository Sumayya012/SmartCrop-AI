"""Central configuration. Values here were read from the saved model, not guessed."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.environ.get("SMARTCROP_MODEL_PATH", ROOT / "models" / "model.keras"))
CLASS_MAP_PATH = Path(os.environ.get("SMARTCROP_CLASS_MAP", ROOT / "class_map.json"))
HISTORY_DB_PATH = Path(os.environ.get("SMARTCROP_HISTORY_DB", ROOT / "data" / "history.sqlite3"))
# Default: history lives in memory, private to one browser session. Set to "1" ONLY for single-user local use;
# the on-disk file is shared by everyone who can open the app.
PERSIST_HISTORY = os.environ.get("SMARTCROP_PERSIST_HISTORY", "0") == "1"

# Verified from Week3_Final/model.keras config.json: InputLayer batch_shape [None, 128, 128, 3].
# (The Week 3 notebook uses 180, but that notebook builds a different model; see docs/ARCHITECTURE.md.)
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 15

# The training generators (Keras ImageDataGenerator.flow_from_directory) resized with 'nearest' by default.
# Kept as the default for train/inference parity; scripts/verify_model.py can compare alternatives.
RESIZE_METHOD = os.environ.get("SMARTCROP_RESIZE", "nearest").lower()  # nearest | bilinear

# Upload limits (security / resource protection).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 64

# Heuristic display threshold. NOT a calibrated probability; used only to tell users to be extra careful.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Loading a .keras file that contains a Lambda layer executes Python bytecode stored in the file.
# Disabled unless you explicitly opt in (only do this for files you created yourself).
ALLOW_UNSAFE_LOAD = os.environ.get("SMARTCROP_ALLOW_UNSAFE_LOAD", "0") == "1"
