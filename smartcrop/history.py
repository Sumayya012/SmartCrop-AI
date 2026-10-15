"""Local prediction history (SQLite). Stores results only, never the uploaded images."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .predict import Prediction

_SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL, filename TEXT,
  label TEXT NOT NULL, confidence REAL NOT NULL, low_confidence INTEGER NOT NULL, top_json TEXT NOT NULL,
  dedupe_key TEXT)"""


@dataclass(frozen=True)
class HistoryRow:
    id: int
    created_utc: str
    filename: str
    label: str
    confidence: float
    low_confidence: bool


class History:
    def __init__(self, path: Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(predictions)")}
        if "dedupe_key" not in columns:  # migrate databases created by version 0.2.0
            self._conn.execute("ALTER TABLE predictions ADD COLUMN dedupe_key TEXT")
        self._conn.commit()

    def add(self, pred: Prediction, filename: str = "", dedupe_key: str | None = None) -> int:
        """Store a result. With a dedupe_key (e.g. a hash of the image bytes), the same image is stored once."""
        if dedupe_key:
            existing = self._conn.execute("SELECT id FROM predictions WHERE dedupe_key = ?", (dedupe_key,)).fetchone()
            if existing:
                return int(existing[0])
        safe_name = Path(filename).name[:120]  # strip any path components
        cur = self._conn.execute(
            "INSERT INTO predictions (created_utc, filename, label, confidence, low_confidence, top_json, dedupe_key) VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), safe_name, pred.label, pred.confidence,
             int(pred.low_confidence), json.dumps(pred.top), dedupe_key),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> list[HistoryRow]:
        rows = self._conn.execute(
            "SELECT id, created_utc, filename, label, confidence, low_confidence FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [HistoryRow(r[0], r[1], r[2] or "", r[3], r[4], bool(r[5])) for r in rows]

    def close(self) -> None:
        """Release the SQLite connection (needed on Windows before the database file can be deleted). Safe to call twice."""
        self._conn.close()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM predictions")
        self._conn.commit()

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "created_utc", "filename", "label", "confidence", "low_confidence"])
        for r in reversed(self.recent(limit=100000)):
            w.writerow([r.id, r.created_utc, _csv_safe(r.filename), r.label, f"{r.confidence:.4f}", int(r.low_confidence)])
        return buf.getvalue()


def open_history(persist: bool, path: Path | None = None) -> History:
    """Private in-memory history by default; an on-disk file only when persistence is explicitly enabled."""
    return History(Path(path) if persist and path else Path(":memory:"))


def _csv_safe(value: str) -> str:
    """Neutralise spreadsheet formula injection from user-controlled filenames."""
    return "'" + value if value[:1] in ("=", "+", "-", "@") else value
