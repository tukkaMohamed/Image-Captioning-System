"""
app.py  —  Image Captioning System
Streamlit UI for LSTM / GRU / Transformer caption generation.

Run:  streamlit run app.py
"""

import os
import time

import streamlit as st
from PIL import Image

from model_loader import load_model
from predict import predict

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioning System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --ink:     #0a0e13;
    --panel:   #111820;
    --panel2:  #171f29;
    --line:    rgba(255,255,255,0.08);
    --cyan:    #5eeacb;
    --amber:   #f5a623;
    --rose:    #f5667a;
    --text:    #e7edf2;
    --muted:   #8a99a8;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--ink) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--line);
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

.metric-box {
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-box .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--cyan);
}
.metric-box .lbl {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 4px;
}

.caption-card {
    background: var(--panel2);
    border: 1px solid var(--line);
    border-left: 3px solid var(--cyan);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.caption-card .model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.caption-card .caption-text {
    font-size: 1.05rem;
    color: var(--text);
    line-height: 1.6;
}
.caption-card .time-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 8px;
}

.compare-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 16px;
}

.info-chip {
    display: inline-block;
    background: rgba(94,234,203,0.1);
    border: 1px solid rgba(94,234,203,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--cyan);
    margin: 4px;
}

.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.stButton > button {
    background: linear-gradient(135deg, #5eeacb, #2dd4bf) !important;
    color: #0a0e13 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: var(--panel2) !important;
    border: 1px dashed rgba(94,234,203,0.35) !important;
    border-radius: 12px !important;
}

hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def get_model(name):
    return load_model(name)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖼️ Image Captioning")
    st.markdown("<div style='color:#8a99a8;font-size:0.85rem;margin-bottom:24px;'>Multimodal AI — Flickr8k + EfficientNetB0</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🚀 Generate Caption", "📊 Project Info"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#8a99a8;'>"
        "Dataset · Flickr8k<br>"
        "CNN · EfficientNetB0<br>"
        "Models · LSTM · GRU · Transformer<br>"
        "Metrics · BLEU · METEOR · ROUGE"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Generate Caption
# ──────────────────────────────────────────────────────────────────────────────
if "Generate" in page:

    st.markdown("<div class='section-eyebrow'>Multimodal Vision & Language</div>", unsafe_allow_html=True)
    st.title("Image Captioning System")
    st.markdown("<div style='color:#8a99a8;margin-bottom:32px;'>Upload an image, pick a model, and generate a natural-language description.</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.3], gap="large")

    # ── Left: upload + model select ──────────────────────────────────────────
    with col_left:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader(
            "Choose an image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True, caption="Preview")

            # save to disk
            os.makedirs("uploaded_images", exist_ok=True)
            save_path = os.path.join("uploaded_images", uploaded.name)
            img.save(save_path)

        st.markdown("#### Select Model")
        model_choice = st.radio(
            "model",
            ["LSTM", "GRU", "Transformer", "⚡ Compare All Models"],
            label_visibility="collapsed",
        )

        generate_btn = st.button("Generate Caption ✨")

    # ── Right: results ────────────────────────────────────────────────────────
    with col_right:
        if generate_btn:
            if not uploaded:
                st.warning("Please upload an image first.")
            else:
                if model_choice == "⚡ Compare All Models":
                    st.markdown("### Compare All Models")

                    results = {}
                    for name in ["LSTM", "GRU", "Transformer"]:
                        with st.spinner(f"Running {name}…"):
                            model, vocab, idx2word = get_model(name)
                            cap, t = predict(img, model, vocab, idx2word, name)
                            results[name] = {"caption": cap, "time": t}

                    # caption cards
                    colors = {"LSTM": "#5eeacb", "GRU": "#f5a623", "Transformer": "#f5667a"}
                    for name, res in results.items():
                        st.markdown(
                            f"""
                            <div class='caption-card' style='border-left-color:{colors[name]}'>
                                <div class='model-tag'>{name}</div>
                                <div class='caption-text'>{res['caption'].capitalize()}</div>
                                <div class='time-badge'>⏱ {res['time']:.3f} s</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # inference time comparison
                    st.markdown("---")
                    st.markdown("#### Inference Time")
                    tc1, tc2, tc3 = st.columns(3)
                    for col, name in zip([tc1, tc2, tc3], ["LSTM", "GRU", "Transformer"]):
                        with col:
                            st.markdown(
                                f"<div class='metric-box'>"
                                f"<div class='num'>{results[name]['time']:.3f}s</div>"
                                f"<div class='lbl'>{name}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                else:
                    # single model
                    name = model_choice
                    with st.spinner(f"Loading {name} and generating caption…"):
                        model, vocab, idx2word = get_model(name)
                        cap, elapsed = predict(img, model, vocab, idx2word, name)

                    st.markdown("### Generated Caption")
                    st.write("**The generated description is:**")
                    st.markdown(
                        f"<div class='caption-card'>"
                        f"<div class='model-tag'>{name}</div>"
                        f"<div class='caption-text'>{cap.capitalize()}</div>"
                        f"<div class='time-badge'>⏱ {elapsed:.3f} s</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # copy button workaround
                    st.code(cap, language=None)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Project Info
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("<div class='section-eyebrow'>About This Project</div>", unsafe_allow_html=True)
    st.title("Project Information")
    st.markdown("<div style='color:#8a99a8;margin-bottom:32px;'>Multimodal Image Captioning — comparing LSTM, GRU, and Transformer decoders on Flickr8k.</div>", unsafe_allow_html=True)

    # ── Dataset & Architecture ────────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("#### Dataset")
        st.markdown("""
        <span class='info-chip'>Flickr8k</span>
        <span class='info-chip'>8,000 images</span>
        <span class='info-chip'>5 captions / image</span>
        <span class='info-chip'>Train / Val / Test split</span>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Feature Extractor")
        st.markdown("""
        <span class='info-chip'>EfficientNetB0</span>
        <span class='info-chip'>ImageNet pretrained</span>
        <span class='info-chip'>Frozen weights</span>
        <span class='info-chip'>Output: 49 × 1280</span>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("#### Decoder Models")
        st.markdown("""
        <span class='info-chip'>LSTM + MultiHead Attention</span>
        <span class='info-chip'>GRU + MultiHead Attention</span>
        <span class='info-chip'>Transformer Decoder (4-layer)</span>
        <span class='info-chip'>Beam Search (width=3)</span>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Evaluation Metrics")
        st.markdown("""
        <span class='info-chip'>BLEU-1 / BLEU-2 / BLEU-4</span>
        <span class='info-chip'>ROUGE-L</span>
        <span class='info-chip'>METEOR</span>
        <span class='info-chip'>CIDEr</span>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Performance Table ─────────────────────────────────────────────────────
    st.markdown("#### Performance Comparison (Test Set)")

    import pandas as pd
    perf = pd.DataFrame({
    "Model": ["LSTM", "GRU", "Transformer"],
    "BLEU-1": [0.3119, 0.3119, 0.3288],
    "BLEU-2": [0.1764, 0.1764, 0.1897],
    "BLEU-4": [0.0630, 0.0630, 0.0738],
    "METEOR": [0.2441, 0.2441, 0.2600],
    "ROUGE-L": [0.3242, 0.3242, 0.3262],
})
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.caption("Evaluation on Flickr8k Test Set.")

    st.markdown("---")

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.markdown("#### System Pipeline")
    st.markdown("""
    ```
    Image (any size)
        ↓
    EfficientNetB0  →  Feature Map (49 × 1280)
        ↓
    ┌──────────────────────────────────────────┐
    │   LSTM / GRU  →  Greedy Decoding         │
    │   Transformer →  Beam Search (width=3)   │
    └──────────────────────────────────────────┘
        ↓
    Natural Language Caption
    ```
    """)
