"""Tests for the hero markup, PDF pagination, top-3/score wording, library search, history privacy and model comparison."""
import contextlib
import io
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from smartcrop import config
from smartcrop.history import History, open_history
from smartcrop.knowledge import KNOWLEDGE, cause_types, search
from smartcrop.labels import crop_of, load_labels
from smartcrop.model import ModelLoadError, compare_models
from smartcrop.predict import Prediction
from smartcrop.report import build_report
from smartcrop.ui import HERO_HTML, SCORE_NOTE

ROOT = Path(__file__).resolve().parent.parent
LABELS = load_labels(config.CLASS_MAP_PATH)


def markdown_pitfalls(html: str) -> list[str]:
    """Things that make Markdown print HTML as text: indented lines or blank lines inside the block."""
    problems = []
    for n, line in enumerate(html.split("\n"), 1):
        if not line.strip():
            problems.append(f"blank line {n}")
        elif line[0] in " \t":
            problems.append(f"indented line {n}")
    return problems


class HeroTests(unittest.TestCase):
    def test_no_markdown_pitfalls(self):
        self.assertEqual(markdown_pitfalls(HERO_HTML), [])

    def test_detector_catches_the_original_bug(self):
        original = "<style>\n    .hero {color:red}\n    </style>\n    <div class=\"hero\"><h1>x</h1></div>"
        self.assertTrue(markdown_pitfalls(original))

    def test_tags_balanced(self):
        for tag in ("div", "h1", "p", "style"):
            self.assertEqual(HERO_HTML.count(f"<{tag}"), HERO_HTML.count(f"</{tag}>"), tag)

    @unittest.skipUnless(__import__("importlib").util.find_spec("markdown_it"), "markdown-it-py not installed")
    def test_commonmark_render_has_no_code_block(self):
        from markdown_it import MarkdownIt
        rendered = MarkdownIt("commonmark", {"html": True}).render(HERO_HTML)
        self.assertNotIn("<pre", rendered)
        self.assertNotIn("&lt;div", rendered)
        self.assertIn('<div class="hero">', rendered)


class ScoreWordingTests(unittest.TestCase):
    def test_note_says_not_calibrated(self):
        self.assertIn("not calibrated probabilities", SCORE_NOTE)
        self.assertIn("single result says nothing about overall accuracy", SCORE_NOTE)


def page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))  # '\b' excludes '/Pages'


class PdfPaginationTests(unittest.TestCase):
    SHAPES = {"portrait 3:4": (900, 1200), "tall 1:3": (600, 1800), "landscape 4:3": (1600, 1200),
              "wide 4:1": (2400, 600), "square": (800, 800), "tiny": (40, 40)}

    @staticmethod
    def pred(label, low=True):
        others = [l for l in LABELS if l != label][:2]
        top = [(label, 0.52), (others[0], 0.30), (others[1], 0.10)]
        return Prediction(label, 0.52, top, low)

    def test_page_counter_agrees_with_multi_page_pdf(self):
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(10, 10, "a"); c.showPage(); c.drawString(10, 10, "b"); c.showPage(); c.save()
        self.assertEqual(page_count(buf.getvalue()), 2)

    def test_single_page_for_every_class_and_shape(self):
        name = "IMG_20261012_a_very_long_farm_photo_name_final_v2_copy.jpg"
        for label in LABELS:
            for shape, (w, h) in self.SHAPES.items():
                pdf = build_report(self.pred(label), Image.new("RGB", (w, h), (30, 120, 40)), name)
                self.assertEqual(page_count(pdf), 1, f"{label} / {shape}")

    def test_single_page_without_low_score_banner(self):
        for label in LABELS:
            pdf = build_report(self.pred(label, low=False), Image.new("RGB", (900, 1200)), "")
            self.assertEqual(page_count(pdf), 1, label)


class SearchTests(unittest.TestCase):
    def test_all_and_crop_counts(self):
        self.assertEqual(len(search()), 15)
        self.assertEqual(len(search(crop="Potato")), 3)
        self.assertEqual(len(search(crop="Pepper")), 2)
        self.assertEqual(len(search(crop="Tomato")), 10)
        self.assertEqual({crop_of(l) for l in search(crop="Potato")}, {"Potato"})

    def test_query_behaviour(self):
        self.assertEqual(search("whitefly"), ["Tomato__Tomato_YellowLeaf__Curl_Virus"])
        self.assertEqual(search("WHITEFLY"), search("whitefly"))
        self.assertTrue(len(search("blight")) >= 4)
        self.assertEqual(search("zzzz-no-such-thing"), [])
        self.assertEqual(search("blight potato"), ["Potato___Early_blight", "Potato___Late_blight"])
        self.assertEqual(search("  "), search())

    def test_type_filter(self):
        self.assertIn("Fungal", cause_types())
        for label in search(cause_type="Fungal"):
            self.assertEqual(KNOWLEDGE[label].cause_type, "Fungal")
        self.assertEqual(search("blight", crop="Pepper"), [])


def make_pred(label="Tomato_healthy"):
    return Prediction(label, 0.9, [(label, 0.9), ("Potato___healthy", 0.05), ("Tomato_Early_blight", 0.05)], False)


def open_files_under(directory) -> list[str]:
    """Open file handles whose path is inside `directory` (Linux /proc check; Windows refuses to delete such files)."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        return []
    found = []
    for fd in os.listdir(fd_dir):
        try:
            target = os.readlink(fd_dir / fd)
        except OSError:
            continue
        if target.startswith(str(directory)):
            found.append(target)
    return found


class HistoryPrivacyTests(unittest.TestCase):
    def track(self, history: History) -> History:
        """Register a History for closing at the end of the test, even if an assertion fails."""
        self.addCleanup(history.close)
        return history

    def test_default_is_private_in_memory(self):
        self.assertFalse(config.PERSIST_HISTORY, "SMARTCROP_PERSIST_HISTORY must be unset for this test")
        a = self.track(open_history(False, Path("/nonexistent/should_not_be_created.db")))
        b = self.track(open_history(False))
        a.add(make_pred(), "mine.jpg")
        self.assertEqual(len(a.recent()), 1)
        self.assertEqual(b.recent(), [])  # a second session cannot see the first
        a.clear()
        self.assertEqual(len(b.recent()), 0)
        self.assertFalse(Path("/nonexistent/should_not_be_created.db").exists())

    def test_one_session_clear_does_not_affect_another(self):
        a, b = self.track(open_history(False)), self.track(open_history(False))
        a.add(make_pred(), "a.jpg"); b.add(make_pred(), "b.jpg")
        a.clear()
        self.assertEqual([r.filename for r in b.recent()], ["b.jpg"])

    def test_persistence_is_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "h.sqlite3"
            try:
                h = open_history(True, path)
                try:
                    h.add(make_pred(), "x.jpg")
                    self.assertTrue(path.exists())
                finally:
                    h.close()
                with contextlib.closing(History(path)) as reopened:  # a second connection to the same file
                    self.assertEqual(len(reopened.recent()), 1)
            finally:
                # every connection is closed before the directory is removed (Windows cannot delete open files)
                self.assertEqual(open_files_under(d), [])

    def test_close_is_idempotent_and_blocks_further_use(self):
        h = open_history(False)
        h.close()
        h.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            h.recent()

    def test_dedupe(self):
        h = self.track(open_history(False))
        first = h.add(make_pred(), "leaf.jpg", dedupe_key="abc")
        again = h.add(make_pred(), "leaf.jpg", dedupe_key="abc")
        self.assertEqual(first, again)
        self.assertEqual(len(h.recent()), 1)
        h.add(make_pred(), "other.jpg", dedupe_key="def")
        self.assertEqual(len(h.recent()), 2)
        h.add(make_pred(), "nokey.jpg"); h.add(make_pred(), "nokey.jpg")  # no key: no dedupe
        self.assertEqual(len(h.recent()), 4)

    def test_migrates_v020_database(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.sqlite3"
            try:
                legacy = sqlite3.connect(path)
                try:
                    legacy.execute("""CREATE TABLE predictions (
                      id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL, filename TEXT,
                      label TEXT NOT NULL, confidence REAL NOT NULL, low_confidence INTEGER NOT NULL, top_json TEXT NOT NULL)""")
                    legacy.execute("INSERT INTO predictions (created_utc, filename, label, confidence, low_confidence, top_json) "
                                   "VALUES ('2026-10-12T10:00:00+00:00','old.jpg','Tomato_healthy',0.8,0,'[]')")
                    legacy.commit()
                finally:
                    legacy.close()
                h = History(path)
                try:
                    self.assertEqual([r.filename for r in h.recent()], ["old.jpg"])
                    h.add(make_pred(), "new.jpg", dedupe_key="k")
                    self.assertEqual(len(h.recent()), 2)
                finally:
                    h.close()
            finally:
                self.assertEqual(open_files_under(d), [])


class _Fake:
    def __init__(self, fn): self.fn = fn
    def predict(self, batch, verbose=0): return self.fn(batch)


def _softmax_model(shift=0.0):
    def fn(batch):
        z = batch.reshape(len(batch), -1)[:, :15].astype(np.float64) / 255.0
        z = z + shift * np.arange(15)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return _Fake(fn)


class CompareModelsTests(unittest.TestCase):
    batches = [np.random.default_rng(i).integers(0, 256, (1, 128, 128, 3)).astype(np.float32) for i in range(8)]

    def test_identical_models_equivalent(self):
        r = compare_models(_softmax_model(), _softmax_model(), self.batches)
        self.assertTrue(r["equivalent"])
        self.assertEqual((r["n_images"], r["top1_agreement"], r["max_abs_diff"]), (8, 1.0, 0.0))

    def test_small_numeric_drift_within_tolerance(self):
        noisy = _Fake(lambda b: np.clip(_softmax_model().predict(b) + 1e-6, 0, 1))
        self.assertTrue(compare_models(_softmax_model(), noisy, self.batches)["equivalent"])

    def test_different_models_flagged(self):
        r = compare_models(_softmax_model(), _softmax_model(shift=0.5), self.batches)
        self.assertFalse(r["equivalent"])
        self.assertGreater(r["max_abs_diff"], 1e-4)

    def test_top1_disagreement_flagged_even_if_diff_small(self):
        a = _Fake(lambda b: np.tile(np.r_[0.5001, 0.4999, np.zeros(13)], (len(b), 1)))
        b = _Fake(lambda b: np.tile(np.r_[0.4999, 0.5001, np.zeros(13)], (len(b), 1)))
        r = compare_models(a, b, self.batches, atol=1e-3)
        self.assertEqual(r["top1_agreement"], 0.0)
        self.assertFalse(r["equivalent"])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ModelLoadError):
            compare_models(_softmax_model(), _Fake(lambda b: np.full((len(b), 14), 1 / 14)), self.batches)


class ScriptCliTests(unittest.TestCase):
    def test_requires_an_option_and_does_not_need_tensorflow(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_model.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--compare-legacy", r.stderr)


if __name__ == "__main__":
    unittest.main()
