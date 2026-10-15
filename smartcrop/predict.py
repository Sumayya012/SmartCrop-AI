"""Prediction on top of any object exposing .predict(batch) -> probabilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import config
from .preprocess import to_model_batch


class PredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top: list[tuple[str, float]]
    low_confidence: bool


def predict_image(model, labels: list[str], image: Image.Image, top_k: int = 3) -> Prediction:
    scores = np.asarray(model.predict(to_model_batch(image), verbose=0), dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] != len(labels):
        raise PredictionError(f"Unexpected model output shape {scores.shape} for {len(labels)} labels.")
    scores = scores[0]
    if not np.all(np.isfinite(scores)) or abs(scores.sum() - 1.0) > 1e-2:
        raise PredictionError("The model returned an invalid probability vector.")
    order = np.argsort(scores)[::-1][: max(1, top_k)]
    top = [(labels[int(i)], float(scores[int(i)])) for i in order]
    return Prediction(top[0][0], top[0][1], top, top[0][1] < config.LOW_CONFIDENCE_THRESHOLD)
