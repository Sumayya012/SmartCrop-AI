"""Read-only diagnostic for the saved model. Never touches model.inputs / model.input_shape.

Usage:  python scripts/inspect_model.py            # static inspection of the .keras archive (no TensorFlow needed)
        python scripts/inspect_model.py --load     # also try to load it and report what survives
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartcrop import config  # noqa: E402


def safe(label, fn):
    try:
        print(f"  {label}: {fn()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label}: !! {type(exc).__name__}: {exc}")


def static(path):
    print(f"== Saved archive: {path}")
    with zipfile.ZipFile(path) as z:
        print("  files:", z.namelist())
        print("  metadata:", z.read("metadata.json").decode())
        cfg = json.loads(z.read("config.json"))
    top = cfg["config"]
    print("  top class:", cfg["class_name"], "| build_input_shape:", top.get("build_input_shape"))
    for layer in top["layers"]:
        c = layer["config"]
        print("   -", layer["class_name"], c.get("name"), c.get("batch_shape") or "", "| build_config:", layer.get("build_config"))


def loaded(path):
    import keras
    import tensorflow as tf
    print(f"== Environment: python {sys.version.split()[0]}, tensorflow {tf.__version__}, keras {keras.__version__}")
    print("== Legacy load (safe_mode=False), each probe isolated")
    try:
        model = keras.models.load_model(str(path), compile=False, safe_mode=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  load_model raised {type(exc).__name__}: {exc}")
        model = None
    if model is not None:
        safe("type", lambda: type(model).__name__)
        safe("built", lambda: model.built)
        safe("layers", lambda: [(l.name, l.built) for l in model.layers])
        safe("weight tensors", lambda: len(model.weights))
    print("== Rebuilt architecture + weights (what the app uses)")
    from smartcrop.model import build_architecture, smoke_test
    m = build_architecture()
    safe("load_weights", lambda: m.load_weights(str(path)))
    safe("built", lambda: m.built)
    safe("weight tensors", lambda: len(m.weights))
    safe("smoke test", lambda: smoke_test(m, config.NUM_CLASSES) or "passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()
    static(a.model)
    if a.load:
        loaded(a.model)
