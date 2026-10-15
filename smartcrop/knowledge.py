"""Educational disease information for the 15 supported classes.

General, non-prescriptive summaries of widely documented symptoms and cultural practices. No pesticide
products or doses are given on purpose. Users should confirm with a local agricultural extension service.
Content should be reviewed by an agronomist before any public release.
"""
from __future__ import annotations

from dataclasses import dataclass

from .labels import crop_of, is_healthy, readable_label

SOURCE_NOTE = (
    "General educational summary written for this prototype; it has not been reviewed by an agronomist. "
    "Confirm with a local agricultural extension officer before taking action."
)


@dataclass(frozen=True)
class DiseaseInfo:
    cause_type: str
    summary: str
    symptoms: tuple[str, ...]
    prevention: tuple[str, ...]


_BLIGHT_CULTURE = (
    "Rotate crops; avoid planting the same family in the same bed in consecutive seasons.",
    "Water at the base and early in the day so leaves dry quickly.",
    "Remove heavily affected plant debris and dispose of it following local guidance; home composting may not reliably destroy some disease-causing organisms.",
    "Use certified disease-free seed or seedlings.",
)
_BACTERIAL_CULTURE = (
    "Use certified clean seed or transplants.",
    "Avoid overhead irrigation and avoid working among wet plants.",
    "Clean tools between plants and remove severely affected material.",
    "Rotate away from related crops for a season or more.",
)

KNOWLEDGE: dict[str, DiseaseInfo] = {
    "Pepper__bell___Bacterial_spot": DiseaseInfo(
        "Bacterial", "A bacterial disease that causes spotting on pepper leaves and fruit.",
        ("Small water-soaked spots that darken and may have yellow halos", "Spots can merge and leaves may drop"),
        _BACTERIAL_CULTURE),
    "Pepper__bell___healthy": DiseaseInfo(
        "Healthy", "No supported disease class was detected.", ("No obvious disease pattern for this model",),
        ("Keep monitoring leaves for spots, curling, or discolouration.",)),
    "Potato___Early_blight": DiseaseInfo(
        "Fungal", "A fungal leaf disease, often seen first on older leaves.",
        ("Brown spots with concentric ring patterns", "Yellowing around lesions"), _BLIGHT_CULTURE),
    "Potato___Late_blight": DiseaseInfo(
        "Oomycete (water mould)", "A fast-spreading disease favoured by cool, wet, humid weather.",
        ("Dark, water-soaked lesions on leaves and stems", "White growth on leaf undersides in humid conditions"),
        _BLIGHT_CULTURE + ("Because it can spread quickly, seek local expert advice promptly if suspected.",)),
    "Potato___healthy": DiseaseInfo(
        "Healthy", "No supported disease class was detected.", ("No obvious disease pattern for this model",),
        ("Keep monitoring leaves for spots, curling, or discolouration.",)),
    "Tomato_Bacterial_spot": DiseaseInfo(
        "Bacterial", "A bacterial disease causing leaf and fruit spotting in tomato.",
        ("Small dark, greasy-looking spots on leaves", "Spots may have yellow halos; fruit can show raised scabby spots"),
        _BACTERIAL_CULTURE),
    "Tomato_Early_blight": DiseaseInfo(
        "Fungal", "A fungal disease that usually starts on lower, older leaves.",
        ("Dark spots with concentric rings (target-like)", "Yellowing and leaf drop from the bottom up"), _BLIGHT_CULTURE),
    "Tomato_Late_blight": DiseaseInfo(
        "Oomycete (water mould)", "A fast-spreading disease favoured by cool, wet, humid weather.",
        ("Large, greasy-looking grey-green to brown patches", "Rapid collapse of foliage in humid conditions"),
        _BLIGHT_CULTURE + ("Because it can spread quickly, seek local expert advice promptly if suspected.",)),
    "Tomato_Leaf_Mold": DiseaseInfo(
        "Fungal", "A fungal disease associated with high humidity and poor air movement.",
        ("Pale yellow patches on upper leaf surface", "Olive-green to grey mould on the underside"),
        ("Improve ventilation and spacing; reduce humidity where possible.", "Avoid wetting leaves.",
         "Remove badly affected leaves and clean up debris.")),
    "Tomato_Septoria_leaf_spot": DiseaseInfo(
        "Fungal", "A fungal leaf-spot disease that usually begins on lower leaves.",
        ("Many small circular spots with dark borders and lighter centres", "Lower leaves yellow and drop"), _BLIGHT_CULTURE),
    "Tomato_Spider_mites_Two_spotted_spider_mite": DiseaseInfo(
        "Pest (mite)", "Tiny mites feed on leaf tissue; damage is common in hot, dry conditions.",
        ("Fine pale stippling on leaves", "Fine webbing and bronzing in heavier infestations"),
        ("Inspect leaf undersides with a magnifier.", "Reduce plant stress and dust; keep plants adequately watered.",
         "Isolate or remove heavily infested leaves.", "Ask an extension officer about suitable control options.")),
    "Tomato__Target_Spot": DiseaseInfo(
        "Fungal", "A fungal leaf-spot disease favoured by warm, humid conditions.",
        ("Brown spots with concentric rings and a yellow margin", "Spots can appear on leaves and fruit"), _BLIGHT_CULTURE),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": DiseaseInfo(
        "Viral (whitefly-transmitted)", "A virus spread by whiteflies; it cannot be cured once a plant is infected.",
        ("Upward curling and yellowing of leaf edges", "Stunted growth and reduced fruit set"),
        ("Control and monitor whiteflies (e.g., with screens or sticky traps).", "Use resistant or tolerant varieties where available.",
         "Remove and dispose of severely affected plants following local guidance.")),
    "Tomato__Tomato_mosaic_virus": DiseaseInfo(
        "Viral (mechanically transmitted)", "A virus that spreads easily on hands, tools and contaminated material.",
        ("Mottled light and dark green mosaic pattern", "Distorted or fern-like leaves and stunted growth"),
        ("Wash hands and sanitize tools between plants.", "Use certified clean seed and resistant varieties where available.",
         "Remove and dispose of affected plants following local guidance.")),
    "Tomato_healthy": DiseaseInfo(
        "Healthy", "No supported disease class was detected.", ("No obvious disease pattern for this model",),
        ("Keep monitoring leaves for spots, curling, or discolouration.",)),
}

HEALTHY_WARNING = (
    "A \"healthy\" result does not guarantee that the plant is disease-free. The model only knows 15 classes, can miss "
    "diseases and early symptoms, and may label a diseased leaf as healthy. Check the whole plant, and if you notice spots, "
    "curling, wilting or other changes, confirm with a local agricultural extension officer."
)

GENERIC_STEPS = (
    "Check several leaves and the whole plant; one photo may not show the full situation.",
    "Confirm the diagnosis with a local agricultural extension officer or agronomist before treating.",
    "Do not apply pesticides or other chemicals based only on this model's prediction.",
)


def info_for(label: str) -> DiseaseInfo | None:
    return KNOWLEDGE.get(label)


def next_steps(label: str) -> tuple[str, ...]:
    if is_healthy(label):
        return (
            "Continue monitoring the plant for changes in colour, spots, wilting, or unusual growth.",
            GENERIC_STEPS[0],
            "If any symptoms appear, confirm with a local agricultural extension officer or agronomist.",
        )
    return GENERIC_STEPS


def search(query: str = "", crop: str | None = None, cause_type: str | None = None) -> list[str]:
    """Labels matching a case-insensitive free-text query (all words must match) and optional filters."""
    words = query.lower().split()
    out = []
    for label, info in KNOWLEDGE.items():
        if crop and crop_of(label) != crop:
            continue
        if cause_type and info.cause_type != cause_type:
            continue
        haystack = " ".join((readable_label(label), label.replace("_", " "), info.cause_type, info.summary,
                             *info.symptoms, *info.prevention)).lower()
        if all(w in haystack for w in words):
            out.append(label)
    return out


def cause_types() -> list[str]:
    return sorted({info.cause_type for info in KNOWLEDGE.values()})
