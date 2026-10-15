"""Class-label loading and validation."""
from __future__ import annotations

import json
from pathlib import Path


class LabelError(ValueError):
    """Raised when the class map is missing, malformed, or inconsistent."""


def load_labels(path: Path, expected: int | None = None) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise LabelError(f"Class map not found: {path}")
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelError(f"Class map is not valid JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise LabelError("Class map must be a non-empty JSON object of label -> index.")
    try:
        indices = {label: int(idx) for label, idx in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise LabelError("Class-map indices must be integers.") from exc
    if sorted(indices.values()) != list(range(len(indices))):
        raise LabelError("Class-map indices must be unique and contiguous from 0.")
    labels = [label for label, _ in sorted(indices.items(), key=lambda kv: kv[1])]
    if expected is not None and len(labels) != expected:
        raise LabelError(f"Class map has {len(labels)} labels but {expected} were expected.")
    return labels


def readable_label(label: str) -> str:
    """'Tomato__Tomato_mosaic_virus' -> 'Tomato mosaic virus' (crop prefix removed for display)."""
    crop, _, condition = label.partition("___") if "___" in label else (None, None, None)
    if crop is None:
        crop, _, condition = label.partition("__") if "__" in label else label.partition("_")
    condition = condition.replace("__", " ").replace("_", " ").strip()
    crop_name = crop.replace("__", " ").replace("_", " ").strip()
    if condition.lower().startswith(crop_name.lower()):
        condition = condition[len(crop_name):].strip()
    return f"{crop_name.capitalize()} — {condition}".strip(" —")


def crop_of(label: str) -> str:
    return label.replace("___", "_").split("_")[0].capitalize()


def is_healthy(label: str) -> bool:
    return label.lower().endswith("healthy")
