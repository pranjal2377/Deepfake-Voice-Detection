"""
Streamlit Dashboard — Full interactive UI.

Features:
  - File upload & analysis with visual results
  - Real-time simulation mode (processes uploaded audio frame-by-frame)
  - Risk gauge, probability timeline, feature charts
  - Per-frame breakdown table
  - Model info and system status
  - Demo mode with pre-generated data

Launch:
    streamlit run src/ui/dashboard.py
"""

import streamlit as st
import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.scoring.risk_scorer import RISK_COLORS, RiskLevel
from src.detection.detector import RealtimeDetector
from src.utils.config import MODELS_DIR, RAW_DATA_DIR, SAMPLE_RATE

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Voice Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .risk-badge {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.5em;
        text-align: center;
        color: white;
    }
    .risk-LOW { background-color: #28a745; }
    .risk-MEDIUM { background-color: #ffc107; color: #333; }
    .risk-HIGH { background-color: #dc3545; }
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load model (cached) ─────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model (cached across reruns)."""
    model_path = os.path.join(MODELS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        return None
    try:
        from src.evaluation.evaluator import load_model_for_evaluation
        return load_model_for_evaluation(model_path)
    except Exception as e:
        st.warning(f"Failed to load model: {e}")
        return None


def _save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to a temp directory and return its path."""
    tmpdir = tempfile.mkdtemp()
    fpath = os.path.join(tmpdir, uploaded_file.name)
    with open(fpath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return fpath


def _risk_color(level: RiskLevel) -> str:
    """Get color hex for a risk level."""
    return {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}.get(
        level.name, "#6c757d"
    )


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🎙️ Deepfake Voice Detector")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Analysis Mode",
    ["📁 Upload Audio", "🎬 Demo Mode", "📊 Call History & Metrics"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
model = load_model()
if model is not None:
    st.sidebar.success("🧠 Model loaded")
else:
    st.sidebar.warning("⚠️ No trained model")
    st.sidebar.caption("Run `python scripts/train.py` first")

# ── Main area ────────────────────────────────────────────────
st.title("🎙️ Deepfake Voice Scam Detection")
st.markdown(
    "Analyze phone-call audio to determine whether the voice is "
    "**real (human)** or **AI-generated (deepfake)**."
)
st.markdown("---")

# ── Helper: Demo analysis ────────────────────────────────────
def _run_demo_analysis(fpath, _model, _threshold):
    """Run analysis on a demo file and display results."""
    st.audio(fpath, format="audio/wav")

    with st.spinner("🔍 Analyzing..."):
        detector = RealtimeDetector(model=_model)
        result = detector.analyze_file(fpath)

    assessment = result["assessment"]
    alert = result["alert"]
    level = assessment["level"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Risk Score", f"{assessment['score']:.1f}%")
    with c2:
        st.metric("Confidence", f"{assessment['confidence']:.1f}%")
    with c3:
        st.markdown(
            f'<div class="risk-badge risk-{level.name}">{level.name}</div>',
            unsafe_allow_html=True,
        )

    if level == RiskLevel.HIGH:
        st.error(f"⚠️ {alert['message']}")
    elif level == RiskLevel.MEDIUM:
        st.warning(f"🔶 {alert['message']}")
    else:
        st.success(f"✅ {alert['message']}")

# ═══════════════════════════════════════════════════════════════
#  UPLOAD MODE
# ═══════════════════════════════════════════════════════════════
if mode == "📁 Upload Audio":
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        help="Supported: WAV, FLAC, MP3, OGG, M4A",
    )

    if uploaded is not None:
        # Play audio
        st.audio(uploaded, format="audio/wav")

        # Save and analyze
        fpath = _save_uploaded_file(uploaded)

        with st.spinner("� Analyzing audio..."):
            detector = RealtimeDetector(model=model)
            result = detector.analyze_file(fpath)

        assessment = result["assessment"]
        alert = result["alert"]
        level = assessment["level"]

        # ── Result header ──
        st.markdown("### 📊 Analysis Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Score", f"{assessment['score']:.1f}%")
        with col2:
            st.metric("Deepfake Prob", f"{assessment.get('voice_score', 0):.1f}%")
        with col3:
            st.metric("Scam Prob", f"{assessment.get('nlp_score', 0):.1f}%")
        with col4:
            color = _risk_color(level)
            st.markdown(
                f'<div class="risk-badge risk-{level.name}">{level.name}</div>',
                unsafe_allow_html=True,
            )

        # ── Alert box ──
        if level == RiskLevel.HIGH:
            st.error(f"⚠️ **{alert['title']}**: {alert['message']}")
        elif level == RiskLevel.MEDIUM:
            st.warning(f"🔶 **{alert['title']}**: {alert['message']}")
        else:
            st.success(f"✅ **{alert['title']}**: {alert['message']}")

        # ── Explanations ──
        if alert.get("explanation"):
            with st.expander("🔍 Detection Details", expanded=True):
                for exp in alert["explanation"]:
                    st.markdown(f"- {exp}")
                if alert.get("action"):
                    st.info(f"**Recommended action:** {alert['action']}")

        # ── NLP & Transcript Analysis ──
        if result["frame_results"]:
            with st.expander("📝 Transcript & Scam Intent Analysis", expanded=True):
                # Combine transcripts if multiple frames
                transcripts = []
                detected_phrases = set()
                intent = "Normal conversation"
                nlp_score = 0.0

                for fr in result["frame_results"]:
                    if "nlp_result" in fr:
                        nlp = fr["nlp_result"]
                        if nlp["transcript"]:
                            transcripts.append(nlp["transcript"])
                        detected_phrases.update(nlp["detected_phrases"])
                        if nlp["nlp_probability"] > nlp_score:
                            nlp_score = nlp["nlp_probability"]
                            intent = nlp["intent"]

                full_transcript = " ".join(transcripts) if transcripts else "No speech detected."
                
                st.markdown(f"**Transcript:**\n> {full_transcript}")
                
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.metric("Scam Content Likelihood", f"{nlp_score * 100:.1f}%")
                with col_i2:
                    st.markdown(f"**Detected Intent:** `{intent}`")
                
                if detected_phrases:
                    st.warning(f"**Suspicious Phrases Detected:** {', '.join(detected_phrases)}")

        # ── Probability Timeline ──
        if result["frame_results"]:
            st.markdown("### 📈 Probability Timeline")
            probs = [fr["probability"] for fr in result["frame_results"]]

            chart_df = pd.DataFrame({
                "Frame": range(len(probs)),
                "Deepfake Probability": probs,
                "Threshold": [threshold] * len(probs),
            })

            st.line_chart(
                chart_df.set_index("Frame"),
                use_container_width=True,
            )

        # ── Feature breakdown ──
        if result["frame_results"] and result["frame_results"][-1].get("features"):
            with st.expander("🔬 Audio Feature Analysis"):
                features = result["frame_results"][-1]["features"]

                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown("**Spectral Features**")
                    if "mfcc_mean" in features:
                        st.write(f"MFCC Mean: `{features['mfcc_mean']:.4f}`")
                    if "spectral_centroid_mean" in features:
                        st.write(f"Spectral Centroid: `{features['spectral_centroid_mean']:.1f}` Hz")
                    if "spectral_centroid_std" in features:
                        st.write(f"Spectral Centroid Std: `{features['spectral_centroid_std']:.1f}` Hz")

                with fcol2:
                    st.markdown("**Voice Quality Features**")
                    if "pitch_mean" in features:
                        st.write(f"Pitch: `{features['pitch_mean']:.1f}` Hz")
                    if "pitch_stability" in features:
                        st.write(f"Pitch Stability: `{features['pitch_stability']:.4f}`")
                    if "hnr" in features:
                        st.write(f"Harmonics-to-Noise: `{features['hnr']:.2f}` dB")
                    if "energy_mean" in features:
                        st.write(f"Energy: `{features['energy_mean']:.6f}`")

        # ── Per-frame table ──
        if len(result["frame_results"]) > 1:
            with st.expander(f"📋 Per-Frame Details ({result['num_frames']} frames)"):
                rows = []
                for i, fr in enumerate(result["frame_results"]):
                    rows.append({
                        "Frame": i,
                        "Probability": f"{fr['probability']:.4f}",
                        "Verdict": "FAKE" if fr["probability"] >= threshold else "REAL",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    else:
        st.info("👆 Upload an audio file to get started.")

# ═══════════════════════════════════════════════════════════════
#  DEMO MODE
# ═══════════════════════════════════════════════════════════════
elif mode == "🎬 Demo Mode":
    st.markdown("### 🎬 Demo Mode")
    st.markdown("Run analysis on pre-generated sample audio to see the system in action.")

    # Find demo files
    demo_files = []
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for f in sorted(files):
            if f.endswith(".wav"):
                demo_files.append(os.path.join(root, f))

    if not demo_files:
        st.warning(
            "No demo data found. Generate it by running:\n\n"
            "```bash\npython scripts/prepare_dataset.py --generate-demo\n```"
        )
    else:
        real_files = [f for f in demo_files if "/real/" in f][:5]
        fake_files = [f for f in demo_files if "/fake/" in f][:5]

        col_r, col_f = st.columns(2)

        with col_r:
            st.markdown("#### 🟢 Real Voice Samples")
            for fpath in real_files:
                fname = os.path.basename(fpath)
                if st.button(f"Analyze {fname}", key=f"real_{fname}"):
                    _run_demo_analysis(fpath, model, threshold)

        with col_f:
            st.markdown("#### 🔴 Fake Voice Samples")
            for fpath in fake_files:
                fname = os.path.basename(fpath)
                if st.button(f"Analyze {fname}", key=f"fake_{fname}"):
                    _run_demo_analysis(fpath, model, threshold)

# ═══════════════════════════════════════════════════════════════
#  HISTORY AND METRICS (NEW)
# ═══════════════════════════════════════════════════════════════
elif mode == "📊 Call History & Metrics":
    st.markdown("### 📊 Historical Logs & System Metrics")
    import json
    from src.utils.config import RESULTS_DIR, LOG_PATH
    
    tab1, tab2 = st.tabs(["📞 Call History", "📈 Evaluation Metrics"])
    
    with tab1:
        st.markdown("#### recent Call Analysis Logs")
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, 'r') as f:
                    logs = json.load(f)
                if logs:
                    df_logs = pd.DataFrame(logs)
                    st.dataframe(df_logs, use_container_width=True)
                else:
                    st.info("No calls logged yet.")
            except Exception as e:
                st.error("Error reading logs.")
        else:
            st.info(f"Log path `{LOG_PATH}` not found.")
            
    with tab2:
        st.markdown("#### System Performance visualisations")
        viz_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.png')] if os.path.exists(RESULTS_DIR) else []
        
        if not viz_files:
            st.info("Run `python run_system.py --mode evaluate` or use the `MetricsVisualizer` to generate metric plots.")
        else:
            cols = st.columns(2)
            for i, viz in enumerate(viz_files):
                viz_path = os.path.join(RESULTS_DIR, viz)
                with cols[i % 2]:
                    st.markdown(f"**{viz}**")
                    st.image(viz_path, use_column_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Deepfake Voice Detection System • "
    "CNN-based spectrogram analysis • "
    "Built for real-time phone call protection"
)
