"""Unit tests that run without TensorFlow/Streamlit (stdlib unittest; also works under pytest)."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from smartcrop import config
from smartcrop.history import History
from smartcrop.knowledge import KNOWLEDGE, next_steps
from smartcrop.labels import LabelError, crop_of, is_healthy, load_labels, readable_label
from smartcrop.predict import PredictionError, predict_image
from smartcrop.preprocess import ImageError, load_image, to_model_batch
from smartcrop.report import build_report

ROOT = Path(__file__).resolve().parent.parent


def png_bytes(size=(200, 150), color=(40, 140, 60), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


class FakeModel:
    def __init__(self, n=15, hot=3):
        self.n, self.hot, self.last_batch = n, hot, None

    def predict(self, batch, verbose=0):
        self.last_batch = batch
        out = np.full((1, self.n), 0.02 / (self.n - 1))
        out[0, self.hot] = 0.98
        return out


class ArtifactTests(unittest.TestCase):
    """Guards the shipped model file and class map against the mismatches found in the audit."""

    def setUp(self):
        with zipfile.ZipFile(config.MODEL_PATH) as z:
            self.cfg = json.loads(z.read("config.json"))["config"]

    def test_saved_input_shape_matches_app(self):
        self.assertEqual(tuple(self.cfg["build_input_shape"][1:3]), config.IMAGE_SIZE)
        self.assertEqual(self.cfg["layers"][0]["class_name"], "InputLayer")

    def test_architecture_matches_rebuild(self):
        self.assertEqual([l["class_name"] for l in self.cfg["layers"]],
                         ["InputLayer", "Lambda", "Functional", "GlobalAveragePooling2D", "Dropout", "Dense"])
        self.assertEqual(self.cfg["layers"][-1]["config"]["units"], config.NUM_CLASSES)
        self.assertEqual(self.cfg["layers"][-1]["config"]["activation"], "softmax")
        self.assertEqual(self.cfg["layers"][4]["config"]["rate"], 0.3)

    def test_class_map_matches_output_units(self):
        labels = load_labels(config.CLASS_MAP_PATH)
        self.assertEqual(len(labels), self.cfg["layers"][-1]["config"]["units"])

    def test_every_label_has_knowledge(self):
        self.assertEqual(set(load_labels(config.CLASS_MAP_PATH)), set(KNOWLEDGE))


class LabelTests(unittest.TestCase):
    def test_load_and_order(self):
        labels = load_labels(config.CLASS_MAP_PATH, expected=15)
        self.assertEqual(labels[0], "Pepper__bell___Bacterial_spot")
        self.assertEqual(labels[14], "Tomato_healthy")

    def test_bad_maps(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            for bad in ('{"a":0,"b":2}', '{"a":0,"b":0}', "not json", "[]", '{"a":"x"}'):
                p.write_text(bad)
                with self.assertRaises(LabelError):
                    load_labels(p)
            with self.assertRaises(LabelError):
                load_labels(Path(d) / "missing.json")

    def test_display_helpers(self):
        self.assertTrue(is_healthy("Tomato_healthy"))
        self.assertFalse(is_healthy("Tomato_Early_blight"))
        self.assertEqual(crop_of("Pepper__bell___healthy"), "Pepper")
        self.assertEqual(crop_of("Tomato__Target_Spot"), "Tomato")
        for lbl in KNOWLEDGE:
            self.assertTrue(readable_label(lbl))


class PreprocessTests(unittest.TestCase):
    def test_batch_shape_dtype_range(self):
        b = to_model_batch(Image.new("RGB", (300, 200), (255, 0, 128)))
        self.assertEqual(b.shape, (1, *config.IMAGE_SIZE, 3))
        self.assertEqual(b.dtype, np.float32)
        self.assertEqual(float(b.max()), 255.0)  # raw 0-255: the model rescales internally

    def test_valid_formats_and_rgba(self):
        for fmt in ("PNG", "JPEG", "WEBP"):
            self.assertEqual(load_image(png_bytes(fmt=fmt)).image.mode, "RGB")
        buf = io.BytesIO()
        Image.new("RGBA", (100, 100), (0, 255, 0, 128)).save(buf, "PNG")
        self.assertEqual(load_image(buf.getvalue()).image.mode, "RGB")

    def test_rejections(self):
        for bad in (b"", b"not an image", b"GIF89a" + b"\x00" * 20):
            with self.assertRaises(ImageError):
                load_image(bad)
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, "BMP")
        with self.assertRaises(ImageError):
            load_image(buf.getvalue())
        with self.assertRaises(ImageError):
            load_image(b"\x00" * (config.MAX_UPLOAD_BYTES + 1))

    def test_quality_warnings(self):
        self.assertTrue(load_image(png_bytes(size=(32, 32))).warnings)
        self.assertTrue(any("dark" in w for w in load_image(png_bytes(color=(5, 5, 5))).warnings))
        self.assertFalse(load_image(png_bytes()).warnings)


class PredictTests(unittest.TestCase):
    labels = load_labels(config.CLASS_MAP_PATH)

    def test_prediction(self):
        m = FakeModel(hot=7)
        p = predict_image(m, self.labels, Image.new("RGB", (64, 64)))
        self.assertEqual(p.label, "Tomato_Late_blight")
        self.assertAlmostEqual(p.confidence, 0.98)
        self.assertEqual(len(p.top), 3)
        self.assertFalse(p.low_confidence)
        self.assertEqual(m.last_batch.shape, (1, 128, 128, 3))

    def test_low_confidence_flag(self):
        class Flat:
            def predict(self, b, verbose=0):
                return np.full((1, 15), 1 / 15)
        self.assertTrue(predict_image(Flat(), self.labels, Image.new("RGB", (64, 64))).low_confidence)

    def test_invalid_outputs(self):
        class Bad:
            def __init__(self, out): self.out = out
            def predict(self, b, verbose=0): return self.out
        for out in (np.zeros((1, 14)), np.full((1, 15), np.nan), np.ones((1, 15))):
            with self.assertRaises(PredictionError):
                predict_image(Bad(out), self.labels, Image.new("RGB", (64, 64)))


class KnowledgeTests(unittest.TestCase):
    def test_no_prescriptive_chemical_advice(self):
        banned = ("mg", "ml/", "per litre", "dosage", "spray with", "fungicide", "pesticide:")
        for info in KNOWLEDGE.values():
            text = " ".join((info.summary, *info.symptoms, *info.prevention)).lower()
            self.assertFalse(any(b in text for b in banned), text)

    def test_next_steps_always_recommend_expert(self):
        for lbl in KNOWLEDGE:
            if not is_healthy(lbl):
                self.assertTrue(any("extension" in s or "agronomist" in s for s in next_steps(lbl)))


class HistoryAndReportTests(unittest.TestCase):
    def test_history_roundtrip_and_csv_safety(self):
        h = History(Path(":memory:"))
        p = predict_image(FakeModel(), load_labels(config.CLASS_MAP_PATH), Image.new("RGB", (64, 64)))
        h.add(p, "../../etc/=cmd.png")
        rows = h.recent()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].filename, "=cmd.png")  # path stripped
        self.assertIn("'=cmd.png", h.to_csv())          # formula neutralised
        h.clear()
        self.assertEqual(h.recent(), [])

    def test_pdf_report(self):
        for hot in (4, 7, 12):
            p = predict_image(FakeModel(hot=hot), load_labels(config.CLASS_MAP_PATH), Image.new("RGB", (64, 64)))
            pdf = build_report(p, Image.new("RGB", (400, 300), (30, 120, 40)), "leaf & <b>.jpg")
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertGreater(len(pdf), 2000)


if __name__ == "__main__":
    unittest.main()
