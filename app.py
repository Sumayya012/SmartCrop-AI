"""SmartCrop AI - Streamlit app for plant-leaf disease classification (educational prototype)."""
from __future__ import annotations

import hashlib

import streamlit as st

from smartcrop import __version__, config
from smartcrop.history import History, open_history
from smartcrop.knowledge import KNOWLEDGE, SOURCE_NOTE, cause_types, info_for, next_steps, search
from smartcrop.labels import LabelError, load_labels, readable_label
from smartcrop.model import ModelLoadError, load_model
from smartcrop.predict import PredictionError, predict_image
from smartcrop.preprocess import ImageError, load_image
from smartcrop.report import build_report
from smartcrop.ui import ABOUT_EVALUATION, HERO_HTML, SCORE_NOTE, healthy_warning_for

st.set_page_config(page_title="SmartCrop AI", page_icon="🌱", layout="wide")
st.markdown(HERO_HTML, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained model…")
def get_artifacts():
    labels = load_labels(config.CLASS_MAP_PATH, expected=config.NUM_CLASSES)
    return load_model(config.MODEL_PATH, config.NUM_CLASSES), labels


def get_history() -> History:
    """One private history per browser session (in memory) unless local persistence is explicitly enabled."""
    if "history" not in st.session_state:
        st.session_state["history"] = open_history(config.PERSIST_HISTORY, config.HISTORY_DB_PATH)
    return st.session_state["history"]


tab_scan, tab_history, tab_library, tab_about = st.tabs(["🔍 Analyse leaf", "🕘 History", "📚 Disease library", "ℹ️ About"])

with tab_scan:
    st.write("Upload a clear, well-lit photo of one leaf filling most of the frame. Supported crops: potato, pepper, tomato.")
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload a leaf image to begin.")
    else:
        try:
            loaded = load_image(uploaded.getvalue())
        except ImageError as exc:
            st.error(str(exc))
            st.stop()
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.image(loaded.image, caption=f"{loaded.image.width} × {loaded.image.height} px", width="stretch")
            for w in loaded.warnings:
                st.warning(w)
        # Key by file so each upload is analysed once and stays visible across reruns.
        key = f"{uploaded.name}:{uploaded.size}"
        if st.button("Analyse leaf", type="primary", width="stretch"):
            try:
                model, labels = get_artifacts()
                pred = predict_image(model, labels, loaded.image)
                get_history().add(pred, uploaded.name, dedupe_key=hashlib.sha256(uploaded.getvalue()).hexdigest())
                st.session_state["result"] = (key, pred)
            except (ModelLoadError, LabelError) as exc:
                st.error("The model could not be loaded. See the details below and README → Troubleshooting.")
                st.code(str(exc))
            except PredictionError as exc:
                st.error(f"Prediction failed: {exc}")
        result = st.session_state.get("result")
        if result and result[0] == key:
            pred = result[1]
            with right:
                st.subheader(readable_label(pred.label))
                st.metric("Model score", f"{pred.confidence:.1%}")
                st.progress(max(0.0, min(1.0, pred.confidence)))
                healthy_warning = healthy_warning_for(pred.label)
                if healthy_warning:
                    st.warning(healthy_warning)
                if pred.low_confidence:
                    st.warning(f"Low score (below {config.LOW_CONFIDENCE_THRESHOLD:.0%}, an uncalibrated rule of thumb). "
                               "Try a clearer photo, and treat this result with extra caution.")
                st.markdown("**Top 3 model scores**")
                for lbl, p in pred.top:
                    st.write(f"{readable_label(lbl)} — {p:.1%}")
                    st.progress(max(0.0, min(1.0, p)))
                st.info(SCORE_NOTE)
                info = info_for(pred.label)
                if info:
                    st.markdown(f"**Type:** {info.cause_type}  \n{info.summary}")
                    st.markdown("**Commonly described symptoms**\n" + "\n".join(f"- {s}" for s in info.symptoms))
                    st.markdown("**General prevention practices**\n" + "\n".join(f"- {p}" for p in info.prevention))
                st.markdown("**Suggested next steps**\n" + "\n".join(f"- {s}" for s in next_steps(pred.label)))
                st.download_button("Download PDF report", build_report(pred, loaded.image, uploaded.name),
                                   file_name="smartcrop_report.pdf", mime="application/pdf", width="stretch")
                st.caption(SOURCE_NOTE)

with tab_history:
    if config.PERSIST_HISTORY:
        st.caption("History is saved to a local file shared by everyone who can open this app. Intended for single-user local use.")
    else:
        st.caption("History is private to this browser session and disappears when the session ends. Uploaded images are never saved.")
    rows = get_history().recent(100)
    if not rows:
        st.info("No analyses yet. Uploaded images are never saved.")
    else:
        st.dataframe(
            [{"Time (UTC)": r.created_utc, "File": r.filename, "Prediction": readable_label(r.label),
              "Score": f"{r.confidence:.1%}", "Low score": "yes" if r.low_confidence else ""} for r in rows],
            width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Export CSV", get_history().to_csv(), "smartcrop_history.csv", "text/csv", width="stretch")
        if c2.button("Clear history", width="stretch"):
            get_history().clear()
            st.rerun()

with tab_library:
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search symptoms, diseases or practices", placeholder="e.g. yellow, whitefly, blight")
    crop = f2.selectbox("Crop", ["All", "Potato", "Pepper", "Tomato"])
    kind = f3.selectbox("Type", ["All"] + cause_types())
    matches = search(query, None if crop == "All" else crop, None if kind == "All" else kind)
    st.caption(f"{len(matches)} of {len(KNOWLEDGE)} classes shown")
    if not matches:
        st.info("No classes match. Try fewer words or clear the filters.")
    for label in matches:
        info = KNOWLEDGE[label]
        with st.expander(readable_label(label)):
            st.markdown(f"**Type:** {info.cause_type}  \n{info.summary}")
            st.markdown("**Commonly described symptoms**\n" + "\n".join(f"- {s}" for s in info.symptoms))
            st.markdown("**General prevention practices**\n" + "\n".join(f"- {p}" for p in info.prevention))
    st.caption(SOURCE_NOTE)

with tab_about:
    st.markdown(
        f"""**SmartCrop AI v{__version__}** uses a MobileNetV2 transfer-learning classifier trained on 15 PlantVillage classes
(potato, pepper, tomato). Input: {config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]} RGB.

This tool does not replace expert agricultural advice."""
    )
    st.markdown(ABOUT_EVALUATION)
