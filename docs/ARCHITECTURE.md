# Architecture and audit notes

## Pipeline
```
upload (JPG/PNG/WEBP, ≤10 MB) → smartcrop.preprocess.load_image   (validate, EXIF-rotate, RGB, quality warnings)
  → to_model_batch  (resize to 128×128, float32, raw 0–255)
  → model: Rescaling(x/127.5−1) → MobileNetV2 → GlobalAvgPool → Dropout(0.3) → Dense(15, softmax)
  → smartcrop.predict (validates output, top-3, low-score flag)
  → UI: knowledge.py guidance · history.py (SQLite, no images stored) · report.py (PDF)
```
Modules: `smartcrop/{config,labels,preprocess,model,predict,knowledge,history,report}.py`, UI in `app.py`.

## Verified facts about the shipped model (read from `Week3_Final/model.keras/config.json`)
| Item | Value |
|---|---|
| Saved with | Keras 3.10.0, 2025-11-09 |
| Top-level class / name | `Sequential` / `sequential_4` |
| Input | `[None, 128, 128, 3]` (**not** 180×180) |
| Layers | InputLayer → Lambda(`mobilenet_v2.preprocess_input`) → MobileNetV2 (154 layers) → GAP → Dropout 0.3 → Dense 15 softmax |
| Weights file | present (`model.weights.h5`, ~21.9 MB) |

## Audit findings
1. **Input-size mismatch.** `app.py` resized to 180×180; the saved model expects 128×128.
2. **Lambda layer is serialized as Python bytecode** (Colab, Python 3.12). Loading it on another Python/Keras can leave the Sequential unbuilt, which matches the reported error *"Sequential model 'sequential_4' has no defined inputs yet"*. This is the most likely cause but **was not reproduced here** (no TensorFlow in the audit environment); `scripts/inspect_model.py --load` produces the evidence.
3. **Environment drift.** `requirements-app.txt` pinned TensorFlow 2.19.0, but the VS Code status bar in the supplied screenshot shows a Python 3.14 venv; TF 2.19 does not publish wheels for it, so the running TF is not the pinned one.
4. **`safe_mode=False`** executes code stored in a model file. The app now avoids this by default (see below).
5. **Notebook ↔ model provenance gap.** `Week3_Final/SmartCropAI.ipynb` builds a *different* model (Functional, 180×180, augmentation layers) and logs 90.72% / loss 0.2802 on 4,127 validation images. `Week2_Progress/SmartCropAI.ipynb` builds the Sequential 128×128 Lambda architecture that matches `model.keras`, and logs a best fine-tuning epoch of 90.66% val-accuracy (4,122 validation images). **Which run produced the exact weights in `model.keras` is not recorded**, so the 90.72% figure and the Week 3 confusion matrix/classification report cannot be attributed to the shipped model without re-evaluation (`scripts/verify_model.py`).
6. Week 3 notebook: the history-merging cell checks `fine_history`, but the fine-tuning cell defines `fine_tune_history`, so training graphs show only the first phase. The final-summary cell refers to an undefined `MODEL_PATH`.
7. Resize interpolation: training generators used Keras's default (`nearest`); the default here is `nearest` for parity. Compare with `--resize bilinear` on real images.

## Fix applied (no retraining; weights untouched)
`smartcrop/model.py` rebuilds the identical architecture in code, replacing the Lambda with `Rescaling(1/127.5, offset=-1)` (mathematically what MobileNetV2 `preprocess_input` does), loads only the weights from the `.keras` archive, and runs a smoke test. The legacy `load_model(safe_mode=False)` path is available only with `SMARTCROP_ALLOW_UNSAFE_LOAD=1`.
**Status: written and unit-tested around, but the TensorFlow-dependent load has not been executed yet** — run the verification steps in the README.

## Interpretation and wording safeguards
| Safeguard | Where | Automated test |
|---|---|---|
| Scores are described as raw model outputs, not calibrated probabilities; one result says nothing about accuracy | `smartcrop/ui.py` `SCORE_NOTE`, shown under the top-3 list | wording constant only (`tests/test_review_fixes.py`) |
| The About tab makes **no accuracy claim**. The notebooks record results for models that cannot be matched to the saved weights, so no figure is quoted anywhere in the app | `smartcrop/ui.py` `ABOUT_EVALUATION`, used by `app.py` | text and `app.py` source (`tests/test_wording.py`) |
| A "healthy" top prediction triggers a warning that it does not guarantee a disease-free plant; the PDF report carries the same text | `smartcrop/knowledge.py` `HEALTHY_WARNING`, `ui.healthy_warning_for`, `report.py` | all 3 healthy classes get it, no disease class does; PDF text checked with `pdftotext` when installed |
| Plant-debris guidance defers to local guidance and says home composting may not reliably destroy some pathogens (no universal "do not compost") | `smartcrop/knowledge.py` | no banned phrases; every disposal or composting line is qualified |
| Disease and prevention text is general education, unreviewed by an agronomist, with no pesticide products or doses | `smartcrop/knowledge.py` `SOURCE_NOTE` | keyword check only (weak evidence) |

## Privacy
Uploaded images are never written to disk. History is held per browser session in memory by default; `SMARTCROP_PERSIST_HISTORY=1` writes a SQLite file shared by everyone who can open the app (single-user local use only). The same image (by content hash) is stored once per history. Filenames are stored, so avoid personal information in them.

## Verification status (what has and has not been run)
| Item | Status |
|---|---|
| Unit tests (labels, preprocessing, prediction checks with a fake model, history, search, PDF, wording, model-comparison logic) | **Run and passing** in the audit environment without TensorFlow or Streamlit |
| PDF layout: one page for every class across six image shapes | **Run**; the page counter was cross-checked with `pdfinfo` and the worst case inspected visually |
| Hero markup: no Markdown code-block trap | **Run** with a CommonMark parser; Streamlit's own renderer not run in the audit environment |
| App launches locally and one real pepper-leaf prediction succeeded | **Reported by the project owner**; not reproduced in the audit environment. One image says nothing about accuracy |
| Rebuilt model vs original model equivalence (`scripts/verify_model.py --compare-legacy`) | **Not yet run**; the original loader may only work where the file was saved (Python 3.12, Keras 3.10.0) |
| Accuracy of the shipped model | **Not measured** (`scripts/verify_model.py --data-dir ...`); no figure is claimed |
| Streamlit widgets (top-3 layout, library search, history across sessions, healthy warning display) | **Manual check needed** |

## Not claimed
No field-image accuracy, calibration, or agronomic validation exists. Disease text is general education, unreviewed by an agronomist.
