"""Regression tests for 'app.py and the smartcrop package are out of sync' (e.g. ImportError: open_history).

1. Static check: every name app.py imports from smartcrop.*, and every `config.X` / `ui.X`-style attribute it uses on an
   imported smartcrop module, must exist. Catches a new app.py paired with an old history.py/config.py.
2. Smoke run: execute app.py end-to-end against a small fake Streamlit (no Streamlit/TensorFlow needed) with a fake model,
   covering startup, an analysis, a repeat click (no duplicate history row), the healthy warning, and the library tab.
"""
import ast
import importlib
import io
import pkgutil
import runpy
import sys
import types
import unittest
from pathlib import Path

from PIL import Image

import smartcrop
from smartcrop import config
from smartcrop.knowledge import HEALTHY_WARNING, KNOWLEDGE
from smartcrop.labels import load_labels
from smartcrop.ui import SCORE_NOTE

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def missing_app_dependencies(source: str) -> list[str]:
    """Names imported from smartcrop.* (or attributes used on imported smartcrop modules) that do not exist."""
    tree = ast.parse(source)
    problems, module_aliases = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "smartcrop":
            mod = importlib.import_module(node.module)
            for alias in node.names:
                bound = alias.asname or alias.name
                if hasattr(mod, alias.name):
                    value = getattr(mod, alias.name)
                    if isinstance(value, types.ModuleType):
                        module_aliases[bound] = value
                    continue
                try:  # `from smartcrop import config` style submodule import
                    module_aliases[bound] = importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    problems.append(f"cannot import name '{alias.name}' from '{node.module}'")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in module_aliases:
            if not hasattr(module_aliases[node.value.id], node.attr):
                problems.append(f"module '{node.value.id}' has no attribute '{node.attr}'")
    return sorted(set(problems))


class StaticImportTests(unittest.TestCase):
    def test_app_imports_resolve(self):
        self.assertEqual(missing_app_dependencies(APP.read_text(encoding="utf-8")), [])

    def test_detector_flags_missing_names(self):
        bad = "from smartcrop.history import History, no_such_function\nfrom smartcrop import config\nprint(config.NO_SUCH_SETTING)"
        found = missing_app_dependencies(bad)
        self.assertIn("cannot import name 'no_such_function' from 'smartcrop.history'", found)
        self.assertIn("module 'config' has no attribute 'NO_SUCH_SETTING'", found)

    def test_every_package_module_imports_without_streamlit_or_tensorflow(self):
        for info in pkgutil.iter_modules(smartcrop.__path__):
            importlib.import_module(f"smartcrop.{info.name}")


class FakeModel:
    def __init__(self, hot):
        self.hot = hot

    def predict(self, batch, verbose=0):
        import numpy as np
        out = np.full((1, 15), 0.02 / 14)
        out[0, self.hot] = 0.98
        return out


RESOURCE_CACHE: dict = {}


class _Ctx:
    """Stands in for st.tabs/st.columns/st.expander objects: a context manager exposing the same widgets."""
    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def __getattr__(self, name): return getattr(self._st, name)


class FakeStreamlit(types.ModuleType):
    """Just enough of the Streamlit API for app.py; records what the app displays."""

    def __init__(self, upload=None, analyse=False):
        super().__init__("streamlit")
        self.session_state = {}
        self.upload, self.analyse = upload, analyse
        self.shown = {"warning": [], "info": [], "error": [], "markdown": [], "write": [], "download": [], "caption": []}

    def cache_resource(self, *a, **k):
        """Process-wide cache shared by every fake session, like the real st.cache_resource."""
        def decorate(fn):
            def cached(*args, **kwargs):
                key = (fn.__module__, fn.__qualname__, args, tuple(sorted(kwargs.items())))
                if key not in RESOURCE_CACHE:
                    RESOURCE_CACHE[key] = fn(*args, **kwargs)
                return RESOURCE_CACHE[key]
            return cached
        return decorate(a[0]) if a and callable(a[0]) else decorate

    def tabs(self, labels): return [_Ctx(self) for _ in labels]
    def columns(self, spec, **k): return [_Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def expander(self, *a, **k): return _Ctx(self)
    def file_uploader(self, *a, **k): return self.upload
    def button(self, label, **k): return self.analyse and label == "Analyse leaf"
    def text_input(self, *a, **k): return ""
    def selectbox(self, label, options, **k): return options[0]
    def metric(self, *a, **k): pass
    def stop(self): raise AssertionError("st.stop() reached: app reported an error to the user")
    def rerun(self): raise AssertionError("unexpected rerun")
    def download_button(self, label, data, *a, **k): self.shown["download"].append((label, data))
    def error(self, msg, **k): self.shown["error"].append(msg)
    def warning(self, msg, **k): self.shown["warning"].append(msg)
    def info(self, msg, **k): self.shown["info"].append(msg)
    def markdown(self, msg, **k): self.shown["markdown"].append(msg)
    def write(self, msg, **k): self.shown["write"].append(str(msg))
    def caption(self, msg, **k): self.shown["caption"].append(str(msg))
    def __getattr__(self, name):  # set_page_config, image, caption, progress, subheader, code, dataframe, ...
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


class FakeUpload:
    def __init__(self, data, name="leaf.png"):
        self._data, self.name, self.size = data, name, len(data)

    def getvalue(self): return self._data


def png(color=(40, 140, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 150), color).save(buf, "PNG")
    return buf.getvalue()


def run_app(fake: FakeStreamlit, hot: int = 9):
    import smartcrop.model as model_module
    original_loader, original_streamlit = model_module.load_model, sys.modules.get("streamlit")
    model_module.load_model = lambda *a, **k: FakeModel(hot)
    sys.modules["streamlit"] = fake
    try:
        runpy.run_path(str(APP), run_name="__main__")
    finally:
        model_module.load_model = original_loader
        if original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = original_streamlit


@unittest.skipIf(getattr(config, "PERSIST_HISTORY", False), "smoke test expects the default in-memory history")
class AppSmokeTests(unittest.TestCase):
    labels = load_labels(config.CLASS_MAP_PATH)

    def setUp(self):
        RESOURCE_CACHE.clear()

    def test_startup_without_upload(self):
        fake = FakeStreamlit()
        run_app(fake)
        self.assertEqual(fake.shown["error"], [])
        self.assertTrue(any("Upload a leaf image" in m for m in fake.shown["info"]))
        self.assertIn("history", fake.session_state)
        self.assertEqual(fake.session_state["history"].recent(), [])
        self.assertTrue(any("<div class=\"hero\">" in m for m in fake.shown["markdown"]))

    def test_analysis_top3_notes_and_pdf(self):
        fake = FakeStreamlit(FakeUpload(png()), analyse=True)
        run_app(fake, hot=9)
        self.assertEqual(fake.shown["error"], [])
        self.assertIn(SCORE_NOTE, fake.shown["info"])
        self.assertEqual(sum("%" in w and "—" in w for w in fake.shown["write"]), 3)  # top-3 lines
        pdfs = [d for label, d in fake.shown["download"] if "PDF" in label]
        self.assertTrue(pdfs and pdfs[0].startswith(b"%PDF"))
        self.assertNotIn(HEALTHY_WARNING, fake.shown["warning"])  # a disease class shows no healthy warning
        self.assertEqual(len(fake.session_state["history"].recent()), 1)

    def test_repeat_click_does_not_duplicate_history(self):
        fake = FakeStreamlit(FakeUpload(png()), analyse=True)
        run_app(fake)
        run_app_again = FakeStreamlit(FakeUpload(png()), analyse=True)
        run_app_again.session_state = fake.session_state  # same browser session, second click
        run_app(run_app_again)
        self.assertEqual(len(fake.session_state["history"].recent()), 1)
        other = FakeStreamlit(FakeUpload(png((10, 10, 200))), analyse=True)
        other.session_state = fake.session_state
        run_app(other)
        self.assertEqual(len(fake.session_state["history"].recent()), 2)

    def test_two_sessions_do_not_share_history(self):
        a, b = FakeStreamlit(FakeUpload(png()), analyse=True), FakeStreamlit()
        run_app(a)
        run_app(b)
        self.assertEqual(len(a.session_state["history"].recent()), 1)
        self.assertEqual(b.session_state["history"].recent(), [])

    def test_healthy_prediction_shows_warning(self):
        fake = FakeStreamlit(FakeUpload(png()), analyse=True)
        run_app(fake, hot=self.labels.index("Tomato_healthy"))
        self.assertIn(HEALTHY_WARNING, fake.shown["warning"])

    def test_library_tab_lists_all_classes_by_default(self):
        fake = FakeStreamlit()
        run_app(fake)
        self.assertIn(f"{len(KNOWLEDGE)} of {len(KNOWLEDGE)} classes shown", fake.shown["caption"])


if __name__ == "__main__":
    unittest.main()
