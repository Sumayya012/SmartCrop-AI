"""Model loading.

Root cause being addressed: Week3_Final/model.keras stores a Keras Sequential whose first layer is a
Lambda wrapping mobilenet_v2.preprocess_input, serialized as Python bytecode (saved on Colab, Python
3.12, Keras 3.10.0). Deserializing that bytecode on a different Python/Keras version can leave the
Sequential unbuilt, which surfaces as "Sequential model 'sequential_4' has no defined inputs yet".

Strategy (no retraining, weights untouched):
  1. Rebuild the identical architecture in code, replacing the Lambda with an equivalent Rescaling layer
     (preprocess_input for MobileNetV2 is x / 127.5 - 1), then load only the weights from the .keras file.
  2. Optional fallback (explicit opt-in): the original load_model(..., safe_mode=False).
Every path ends with a smoke test so an unusable model fails loudly at startup, not at prediction time.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


def build_architecture(num_classes: int = config.NUM_CLASSES):
    """Identical layer structure to the saved model (see docs/ARCHITECTURE.md)."""
    import keras
    from keras import layers

    h, w = config.IMAGE_SIZE
    base = keras.applications.MobileNetV2(input_shape=(h, w, 3), include_top=False, weights=None)
    return keras.Sequential(
        [
            layers.Input(shape=(h, w, 3)),
            layers.Rescaling(1.0 / 127.5, offset=-1.0),  # == mobilenet_v2.preprocess_input
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="smartcrop_mobilenetv2",
    )


def smoke_test(model, num_classes: int) -> None:
    """Fail fast if the model cannot run a forward pass or returns an invalid distribution."""
    h, w = config.IMAGE_SIZE
    out = np.asarray(model.predict(np.full((1, h, w, 3), 127.0, dtype=np.float32), verbose=0))
    if out.shape != (1, num_classes):
        raise ModelLoadError(f"Smoke test: expected output shape (1, {num_classes}), got {out.shape}.")
    if not np.all(np.isfinite(out)) or abs(float(out.sum()) - 1.0) > 1e-3:
        raise ModelLoadError("Smoke test: model output is not a valid probability vector.")


def load_rebuilt(path: Path = config.MODEL_PATH, num_classes: int = config.NUM_CLASSES):
    """Rebuilt architecture + weights only (the safe path the app prefers)."""
    model = build_architecture(num_classes)
    model.load_weights(str(Path(path)))
    smoke_test(model, num_classes)
    return model


def load_legacy(path: Path = config.MODEL_PATH, num_classes: int = config.NUM_CLASSES):
    """Original deserialization, which executes the Lambda bytecode stored in the file. Trusted files only."""
    import keras

    model = keras.models.load_model(str(Path(path)), compile=False, safe_mode=False)
    smoke_test(model, num_classes)
    return model


def compare_models(model_a, model_b, batches, atol: float = 1e-4) -> dict:
    """Compare two models' probability outputs on the same input batches (each shaped (n, H, W, 3))."""
    a = np.concatenate([np.asarray(model_a.predict(b, verbose=0), dtype=np.float64) for b in batches])
    b = np.concatenate([np.asarray(model_b.predict(x, verbose=0), dtype=np.float64) for x in batches])
    if a.shape != b.shape:
        raise ModelLoadError(f"Output shapes differ: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    agree = a.argmax(axis=1) == b.argmax(axis=1)
    return {
        "n_images": int(a.shape[0]),
        "top1_agreement": float(agree.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "atol": atol,
        "equivalent": bool(agree.all() and diff.max() <= atol),
    }


def load_model(path: Path = config.MODEL_PATH, num_classes: int = config.NUM_CLASSES):
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")
    errors: list[str] = []

    try:
        model = load_rebuilt(path, num_classes)
        log.info("Loaded model via rebuilt architecture + weights.")
        return model
    except Exception as exc:  # noqa: BLE001 - collect and try the next strategy
        errors.append(f"rebuild+load_weights: {type(exc).__name__}: {exc}")

    if config.ALLOW_UNSAFE_LOAD:
        try:
            model = load_legacy(path, num_classes)
            log.warning("Loaded model via unsafe Lambda deserialization (opt-in).")
            return model
        except Exception as exc:  # noqa: BLE001
            errors.append(f"load_model(safe_mode=False): {type(exc).__name__}: {exc}")
    else:
        errors.append("legacy load_model skipped (set SMARTCROP_ALLOW_UNSAFE_LOAD=1 to opt in for your own files)")

    raise ModelLoadError("Could not load the model.\n- " + "\n- ".join(errors))
