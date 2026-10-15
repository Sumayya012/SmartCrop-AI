"""UI text/markup kept free of Streamlit so it can be unit-tested.

HTML passed to st.markdown must have no indented lines and no blank lines: Markdown treats a
4-space-indented line after an HTML block as a code block and prints the tags as text.
"""
from __future__ import annotations

from .knowledge import HEALTHY_WARNING
from .labels import is_healthy

HERO_HTML = "\n".join([
    "<style>",
    ".hero{padding:1.2rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#e9f8ec,#f6fbf4);margin-bottom:1rem;}",
    ".hero h1{color:#14532d;margin:0 0 .2rem 0;}",
    ".hero p{color:#52665a;margin:0;}",
    "</style>",
    '<div class="hero">',
    "<h1>🌱 SmartCrop AI</h1>",
    "<p>AI-assisted leaf disease classification for potato, pepper and tomato · educational prototype</p>",
    "</div>",
])

SCORE_NOTE = (
    "**How to read these scores.** They are the model's raw output values for each of its 15 classes and add up to 100%. "
    "They are **not calibrated probabilities**: a 40% score does not mean a 40% chance of being right, and a 95% score "
    "does not guarantee a correct answer. The model also gives one of its 15 classes for *any* image, even a non-leaf. "
    "A single result says nothing about overall accuracy."
)

ABOUT_EVALUATION = (
    "**Evaluation.** No accuracy figure is claimed for the model in this app. The repository's notebooks record results for "
    "models that cannot be confirmed to match the saved weights, and performance on field photos has not been measured. "
    "See `docs/ARCHITECTURE.md` and run `scripts/verify_model.py` to measure it on your own images."
)


def healthy_warning_for(label: str) -> str | None:
    """Caution shown whenever the top prediction is a 'healthy' class."""
    return HEALTHY_WARNING if is_healthy(label) else None
