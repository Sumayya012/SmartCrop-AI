"""Focused tests for the three wording corrections: About accuracy claim, composting guidance, 'healthy' warning."""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from smartcrop import config
from smartcrop.knowledge import HEALTHY_WARNING, KNOWLEDGE, next_steps
from smartcrop.labels import is_healthy, load_labels
from smartcrop.predict import Prediction
from smartcrop.report import build_report
from smartcrop.ui import ABOUT_EVALUATION, healthy_warning_for

ROOT = Path(__file__).resolve().parent.parent
LABELS = load_labels(config.CLASS_MAP_PATH)
HEALTHY = [l for l in LABELS if is_healthy(l)]
DISEASED = [l for l in LABELS if not is_healthy(l)]


def knowledge_text(info) -> str:
    return " ".join((info.summary, *info.symptoms, *info.prevention)).lower()


class AboutTabTests(unittest.TestCase):
    def test_no_accuracy_figures_in_about_text(self):
        self.assertIsNone(re.search(r"\d\s*%", ABOUT_EVALUATION))
        self.assertNotIn("90", ABOUT_EVALUATION)
        self.assertNotIn("91", ABOUT_EVALUATION)

    def test_about_text_is_explicit_about_limits(self):
        self.assertIn("No accuracy figure is claimed", ABOUT_EVALUATION)
        self.assertIn("cannot be confirmed to match the saved weights", ABOUT_EVALUATION)
        self.assertIn("has not been measured", ABOUT_EVALUATION)

    def test_app_source_no_longer_combines_notebook_results(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("90–91", source)
        self.assertNotIn("around 90", source)
        self.assertIn("ABOUT_EVALUATION", source)


class CompostingTests(unittest.TestCase):
    def test_no_universal_composting_instruction_anywhere(self):
        for label, info in KNOWLEDGE.items():
            text = knowledge_text(info)
            for banned in ("do not compost", "don't compost", "never compost"):
                self.assertNotIn(banned, text, label)

    def test_disposal_guidance_defers_to_local_guidance(self):
        checked = 0
        for label, info in KNOWLEDGE.items():
            for item in info.prevention:
                if "dispose" in item.lower():
                    checked += 1
                    self.assertIn("local guidance", item.lower(), label)
        self.assertGreaterEqual(checked, 8)  # six blight/spot classes + two viral classes

    def test_composting_mentions_are_qualified(self):
        checked = 0
        for label, info in KNOWLEDGE.items():
            for item in info.prevention:
                if "compost" in item.lower():
                    checked += 1
                    self.assertIn("may not reliably", item.lower(), label)
        self.assertEqual(checked, 6)


class HealthyWarningTests(unittest.TestCase):
    def test_warning_text(self):
        self.assertIn("does not guarantee", HEALTHY_WARNING)
        self.assertIn("disease-free", HEALTHY_WARNING)
        self.assertIn("may label a diseased leaf as healthy", HEALTHY_WARNING)
        self.assertIn("extension officer", HEALTHY_WARNING)

    def test_shown_for_every_healthy_class_and_only_those(self):
        self.assertEqual(len(HEALTHY), 3)
        for label in HEALTHY:
            self.assertEqual(healthy_warning_for(label), HEALTHY_WARNING, label)
        for label in DISEASED:
            self.assertIsNone(healthy_warning_for(label), label)

    def test_next_steps_for_healthy_mention_symptoms_and_expert(self):
        for label in HEALTHY:
            steps = " ".join(next_steps(label)).lower()
            self.assertIn("symptoms", steps, label)
            self.assertIn("extension officer", steps, label)

    def test_app_wires_the_warning(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("healthy_warning_for(pred.label)", source)
        self.assertIn("st.warning(healthy_warning)", source)


@unittest.skipUnless(shutil.which("pdftotext"), "pdftotext (poppler) not installed")
class HealthyWarningInPdfTests(unittest.TestCase):
    @staticmethod
    def pdf_text(label: str) -> str:
        others = [l for l in LABELS if l != label][:2]
        pred = Prediction(label, 0.7, [(label, 0.7), (others[0], 0.2), (others[1], 0.1)], False)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.pdf"
            path.write_bytes(build_report(pred, Image.new("RGB", (900, 1200), (40, 120, 60)), "leaf.jpg"))
            out = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, check=True).stdout
        return re.sub(r"\s+", " ", out)

    def test_healthy_report_contains_warning(self):
        for label in HEALTHY:
            text = self.pdf_text(label)
            self.assertIn("Important:", text, label)
            self.assertIn("does not guarantee that the plant is disease-free", text, label)

    def test_disease_report_has_no_healthy_warning_and_qualified_debris_text(self):
        for label in ("Potato___Late_blight", "Tomato_Early_blight"):
            text = self.pdf_text(label)
            self.assertNotIn("does not guarantee that the plant is disease-free", text, label)
            self.assertNotIn("do not compost", text.lower(), label)
            self.assertIn("following local guidance", text, label)


if __name__ == "__main__":
    unittest.main()
