"""Downloadable PDF report for a single prediction."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import __version__, config
from .knowledge import HEALTHY_WARNING, SOURCE_NOTE, info_for, next_steps
from .labels import is_healthy, readable_label
from .predict import Prediction

DISCLAIMER = (
    "SmartCrop AI is an educational prototype. This report is a model output, not a diagnosis. Confidence is the "
    "model's score, not a probability of being correct, and has not been validated on field images."
)


def build_report(pred: Prediction, image: Image.Image, filename: str = "") -> bytes:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=colors.HexColor("#14532d"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#14532d"))
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="SmartCrop AI report")
    s = [Paragraph("SmartCrop AI — Leaf Analysis Report", h1)]
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    s.append(Paragraph(escape(f"Generated {when}" + (f" · File: {filename[:80]}" if filename else "")), small))
    s.append(Spacer(1, 10))

    thumb = image.convert("RGB").copy()
    thumb.thumbnail((600, 600))
    tbuf = io.BytesIO()
    thumb.save(tbuf, format="JPEG", quality=85)
    tbuf.seek(0)
    # Fit inside a bounding box so tall portrait photos cannot push the report onto a second page.
    box_w, box_h = 7 * cm, 4.5 * cm
    scale = min(box_w / thumb.width, box_h / thumb.height)
    s.append(RLImage(tbuf, width=thumb.width * scale, height=thumb.height * scale))
    s.append(Spacer(1, 10))

    s.append(Paragraph("Model prediction", h2))
    s.append(Paragraph(escape(readable_label(pred.label)) + f" — model score {pred.confidence:.1%}", body))
    if is_healthy(pred.label):
        s.append(Paragraph("<b>Important:</b> " + escape(HEALTHY_WARNING), body))
    if pred.low_confidence:
        s.append(Paragraph(f"<b>Low score</b> (below {config.LOW_CONFIDENCE_THRESHOLD:.0%}): treat this result with extra caution.", body))

    rows = [["Rank", "Class", "Model score"]] + [[str(i + 1), readable_label(lbl), f"{p:.1%}"] for i, (lbl, p) in enumerate(pred.top)]
    t = Table(rows, hAlign="LEFT", colWidths=[1.5 * cm, 10 * cm, 3 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f8ec")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    s += [Spacer(1, 6), t]

    info = info_for(pred.label)
    if info:
        s.append(Paragraph("About this class", h2))
        s.append(Paragraph(escape(f"{info.cause_type}. {info.summary}"), body))
        s.append(Paragraph("Commonly described symptoms: " + escape("; ".join(info.symptoms)) + ".", body))
        s.append(Paragraph("General prevention practices", h2))
        for item in info.prevention:
            s.append(Paragraph("• " + escape(item), body))
    s.append(Paragraph("Suggested next steps", h2))
    for item in next_steps(pred.label):
        s.append(Paragraph("• " + escape(item), body))

    s += [Spacer(1, 12), KeepTogether([Paragraph(escape(DISCLAIMER), small), Paragraph(escape(SOURCE_NOTE), small),
                                       Paragraph(f"SmartCrop AI v{__version__}", small)])]
    doc.build(s)
    return buf.getvalue()
