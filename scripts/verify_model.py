"""Verify prediction behaviour with the app's exact pipeline.

Accuracy on your own images (measured, never hard-coded):
  python scripts/verify_model.py --data-dir PATH/TO/PlantVillage [--limit 50] [--resize nearest|bilinear]
  PATH needs one sub-folder per class, named exactly as in Week2_Progress/class_map.json.

Equivalence of the rebuilt model vs the ORIGINAL model (Lambda layer) on identical inputs:
  python scripts/verify_model.py --compare-legacy [--data-dir PATH] [--limit 5]
  The original loader executes bytecode stored in the .keras file: use it only on a file you created.
  It may fail outside the environment that saved the file (Python 3.12 / Keras 3.10.0, e.g. Colab);
  that failure is exactly the bug the rebuilt loader works around, and is reported, not hidden.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", help="folder with one sub-folder per class")
ap.add_argument("--limit", type=int, default=50, help="max images per class")
ap.add_argument("--resize", choices=["nearest", "bilinear"], default=None)
ap.add_argument("--compare-legacy", action="store_true", help="compare rebuilt vs original model outputs")
a = ap.parse_args()
if not a.data_dir and not a.compare_legacy:
    ap.error("provide --data-dir and/or --compare-legacy")
if a.resize:
    os.environ["SMARTCROP_RESIZE"] = a.resize

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from smartcrop import config  # noqa: E402
from smartcrop.labels import load_labels  # noqa: E402
from smartcrop.model import compare_models, load_legacy, load_model, load_rebuilt  # noqa: E402
from smartcrop.predict import predict_image  # noqa: E402
from smartcrop.preprocess import to_model_batch  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
labels = load_labels(config.CLASS_MAP_PATH, expected=config.NUM_CLASSES)


def class_files(root: Path, limit: int):
    for label in labels:
        d = root / label
        if d.is_dir():
            yield label, sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)[:limit]


def synthetic_batches():
    """Deterministic non-photo inputs that exercise the full 0-255 range."""
    rng = np.random.default_rng(0)
    h, w = config.IMAGE_SIZE
    imgs = [rng.integers(0, 256, (h, w, 3)).astype(np.float32) for _ in range(12)]
    imgs += [np.full((h, w, 3), v, np.float32) for v in (0, 64, 127, 128, 192, 255)]
    ramp = np.tile(np.linspace(0, 255, w, dtype=np.float32)[None, :, None], (h, 1, 3))
    imgs += [ramp, ramp[::-1, ::-1].copy()]
    return [im[None, ...] for im in imgs]


def report(title, result):
    print(f"[{title}] images={result['n_images']}  top-1 agreement={result['top1_agreement']:.4f}  "
          f"max|diff|={result['max_abs_diff']:.3e}  mean|diff|={result['mean_abs_diff']:.3e}  "
          f"-> {'EQUIVALENT' if result['equivalent'] else 'NOT EQUIVALENT'} (tolerance {result['atol']:g})")


exit_code = 0
if a.compare_legacy:
    print("WARNING: the original loader runs code stored in the model file. Trusted files only.")
    try:
        rebuilt = load_rebuilt()
        print("rebuilt model: loaded")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"rebuilt model failed to load, nothing to compare: {type(exc).__name__}: {exc}")
    try:
        legacy = load_legacy()
        print("original model: loaded")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"original model failed to load in THIS environment ({type(exc).__name__}: {exc}).\n"
                 "Run the comparison in an environment matching how the file was saved (Python 3.12, Keras 3.10.0).")
    results = [("synthetic", compare_models(rebuilt, legacy, synthetic_batches()))]
    if a.data_dir:
        real = [to_model_batch(Image.open(p).convert("RGB")) for _, fs in class_files(Path(a.data_dir), a.limit) for p in fs]
        if real:
            results.append(("real images", compare_models(rebuilt, legacy, real)))
        else:
            print("No images found under --data-dir; real-image comparison skipped.")
    for title, res in results:
        report(title, res)
    exit_code = 0 if all(r["equivalent"] for _, r in results) else 1
    print("Interpretation: agreement shows the Lambda replacement is numerically equivalent on THESE inputs; "
          "it says nothing about accuracy.")

if a.data_dir:
    model = rebuilt if a.compare_legacy else load_model()
    root = Path(a.data_dir)
    missing = [l for l in labels if not (root / l).is_dir()]
    if missing:
        sys.exit(f"Missing class folders: {missing}")
    hits, total, per = 0, 0, defaultdict(lambda: [0, 0])
    for label, files in class_files(root, a.limit):
        for p in files:
            ok = predict_image(model, labels, Image.open(p).convert("RGB")).label == label
            per[label][0] += ok
            per[label][1] += 1
            hits += ok
            total += 1
    print(f"resize={config.RESIZE_METHOD}  images={total}  accuracy={hits / max(total, 1):.4f}")
    for label, (h, n) in per.items():
        print(f"  {label:48s} {h}/{n}")
    print("Note: PlantVillage validation images may overlap with training data unless you use a held-out split.")
sys.exit(exit_code)
