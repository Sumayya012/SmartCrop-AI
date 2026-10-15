# 🌱 SmartCrop AI — Plant Disease Classification

SmartCrop AI is an educational computer-vision prototype that classifies uploaded leaf images into one of 15 supported pepper, potato, and tomato classes. It uses a MobileNetV2 transfer-learning model trained with the PlantVillage dataset.

## Features

- Upload a JPG, JPEG, PNG, or WEBP leaf image
- Predict one of 15 supported classes
- Display the model's top prediction, confidence score, and top-three results
- Show general, non-prescriptive follow-up guidance
- Disease library (symptoms and general prevention practices for all 15 classes)
- Local prediction history (results only; images are not stored) with CSV export
- Downloadable PDF report per analysis
- Image validation (format, size, dark/small-image warnings) and a low-score warning
- Streamlit web interface for a local demo

> **Important limitation:** This is an educational prototype, not a definitive agricultural diagnosis. Confidence is not a guarantee of correctness. Model performance on real-world field images may be lower than on PlantVillage-style images. Confirm suspected disease with a qualified agricultural professional before taking treatment action.

## Repository structure

```text
SmartCrop-AI-main/
├── app.py                  # Streamlit UI
├── smartcrop/              # tested pipeline: preprocess, model loading, predict, knowledge, history, report
├── tests/                  # unit tests (no TensorFlow needed)
├── scripts/                # inspect_model.py, verify_model.py
├── docs/ARCHITECTURE.md    # audit notes and verified model facts
├── requirements-app.txt
├── requirements-dev.txt
├── README.md
├── Week1_Setup/
│   ├── README.md
│   └── SmartCrop-AI-Problem-Statement.pdf
├── Week2_Progress/
│   ├── SmartCropAI.ipynb
│   ├── SmartCropAI_MobileNet_Final.keras
│   └── class_map.json
└── Week3_Final/
    ├── SmartCropAI.ipynb
    ├── model.keras
    ├── ConfusionMatrix.png
    ├── TrainingGraphs.png
    └── README.md
```

## Run locally

Use **Python 3.11** (TensorFlow 2.19 has no wheels for Python 3.13/3.14). The model was saved with Keras 3.10.0, which is pinned.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-app.txt
streamlit run app.py
```

Open the local URL shown in the terminal (normally `http://localhost:8501`). The app loads `Week3_Final/model.keras` and `Week2_Progress/class_map.json` from the repository. Keep those files in their current locations.

If TensorFlow installation fails, confirm your Python version first. Do not install a random TensorFlow version without checking compatibility.

## Model and evaluation

The shipped model (`Week3_Final/model.keras`) is a MobileNetV2 transfer-learning classifier with **128 × 128 RGB input** and MobileNetV2 preprocessing inside the graph, so the app passes raw 0–255 pixels.

Accuracy figures recorded in the notebooks (not independently re-run):

| Notebook | Architecture | Validation images | Logged result |
|---|---|---|---|
| `Week2_Progress/SmartCropAI.ipynb` | Sequential, 128 px (matches `model.keras`) | 4,122 | best fine-tuning epoch 90.66% val-accuracy |
| `Week3_Final/SmartCropAI.ipynb` | Functional, 180 px (differs from `model.keras`) | 4,127 | 90.72% accuracy, loss 0.2802; class-wise report and confusion matrix |

The repository does not record which run produced the exact weights in `model.keras`, so these figures are **not attributed to the shipped model**. To measure it with the app's exact pipeline on your own images: `python scripts/verify_model.py --data-dir <PlantVillage folder>`. All results come from the PlantVillage distribution; field-image performance has not been measured. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tests and verification

```powershell
pip install -r requirements-dev.txt
python -m pytest            # or: python -m unittest discover -s tests -v
python scripts/inspect_model.py --load      # model diagnostics (needs TensorFlow)
python scripts/verify_model.py --data-dir <folder with one sub-folder per class>
python scripts/verify_model.py --compare-legacy      # rebuilt vs original model on identical inputs (trusted file only;
                                                     # run where the file was saved: Python 3.12 / Keras 3.10.0, e.g. Colab)
```

## Privacy

Uploaded images are never saved. Analysis history is private to the browser session and kept in memory by default. Setting
`SMARTCROP_PERSIST_HISTORY=1` writes it to a local SQLite file that **everyone who can open the app shares**, so use that only for
single-user local runs. History stores filenames, so avoid personal information in them.

## Troubleshooting

- *"Sequential model has no defined inputs yet"*: usually a Python/Keras mismatch when deserializing the saved Lambda layer. Use the Python 3.11 environment above and run `scripts/inspect_model.py --load`. The app rebuilds the architecture and loads only the weights; the legacy loader is opt-in via `SMARTCROP_ALLOW_UNSAFE_LOAD=1` and should only be used for files you created.
- `use_container_width` deprecation warnings: fixed in this version (`width="stretch"`).

## Dataset

The notebooks reference the PlantVillage dataset. Follow the dataset provider's terms when obtaining or redistributing data. The full dataset is not included in this repository.

## SDG connection

The project is aligned with **UN SDG 2: Zero Hunger**, through an educational demonstration of AI-assisted crop-health awareness. The project has not yet demonstrated measured improvements in yield or farmer outcomes.

## Current limitations

- Supports only the 15 classes represented by the trained model.
- Predictions can be unreliable for unfamiliar crops, diseases, field conditions, or low-quality images.
- Guidance is general and does not prescribe pesticides or guarantee treatment outcomes.
- Notebook validation results have not been independently reproduced, and are not attributed to the shipped model file (see above).
- Disease and prevention text is general education that has not been reviewed by an agronomist.

## Future improvements

- Test on carefully sourced real-world leaf images
- Re-evaluate the shipped model on a held-out split and on real field photos
- Improve accessibility and multilingual presentation
- Deploy only after validating model loading, prediction consistency, privacy, and hosting requirements

## Acknowledgements

Built as a student AI/ML project using TensorFlow/Keras, MobileNetV2, and the PlantVillage dataset.
