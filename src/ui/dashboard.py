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
import base64
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.scoring.risk_scorer import RISK_COLORS, RiskLevel
from src.detection.detector import RealtimeDetector
from src.utils.config import MODELS_DIR, RAW_DATA_DIR, SAMPLE_RATE

# ── Icon paths ───────────────────────────────────────────────
_ICONS_DIR = os.path.join(_project_root, "docs", "icons")

def _img_to_base64(path: str) -> str:
    """Convert an image file to a base64 data URI for inline HTML."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

# Pre-load icon data URIs
_ICON_SHIELD = _img_to_base64(os.path.join(_ICONS_DIR, "shield_waveform.png"))
_ICON_PADLOCK = _img_to_base64(os.path.join(_ICONS_DIR, "padlock_circuit.png"))
_ICON_RADAR = _img_to_base64(os.path.join(_ICONS_DIR, "radar_scan.png"))
_ICON_CHART = _img_to_base64(os.path.join(_ICONS_DIR, "chart_analytics.png"))
_ICON_MIC = _img_to_base64(os.path.join(_ICONS_DIR, "cyber_mic.png"))

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Voice Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Make the sidebar collapse/expand arrows more noticeable */
    [data-testid="collapsedControl"] {
        background-color: rgba(59, 130, 246, 0.2);
        border: 2px solid #3b82f6;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        z-index: 100;
        margin-top: 10px;
        margin-left: 10px;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(59, 130, 246, 0.4);
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.6);
    }
    
    [data-testid="collapsedControl"] svg {
        fill: #3b82f6;
        color: #3b82f6;
        height: 1.8rem;
        width: 1.8rem;
    }
    
    /* Global dark gradient background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #0b1120 45%, #020617 100%);
        color: #e5e7eb;
    }

    /* Main content width tweaks */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* Glassmorphism-style cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.75);
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 1.25rem;
    }

    .glass-card h3 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        font-weight: 600;
        color: #e5e7eb;
    }

    .sub-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #9ca3af;
    }

    /* Risk badges */
    .risk-badge {
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .risk-LOW { background: linear-gradient(135deg, #16a34a, #22c55e); color: #ecfdf3; }
    .risk-MEDIUM { background: linear-gradient(135deg, #f97316, #facc15); color: #111827; }
    .risk-HIGH { background: linear-gradient(135deg, #dc2626, #fb7185); color: #fef2f2; }

    /* Metric centering */
    .stMetric {
        text-align: center;
    }

    /* Sidebar header with icon */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .sidebar-icon {
        width: 22px;
        height: 22px;
        vertical-align: middle;
        margin-right: 6px;
        border-radius: 4px;
    }

    /* Cyber mic hero section */
    .cyber-mic-hero {
        text-align: center;
        padding: 1.2rem 0 0.5rem 0;
    }
    .cyber-mic-hero img {
        width: 120px;
        height: 120px;
        border-radius: 18px;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.25), 0 0 60px rgba(56, 189, 248, 0.08);
        animation: mic-pulse 3s ease-in-out infinite;
    }
    @keyframes mic-pulse {
        0%, 100% { box-shadow: 0 0 30px rgba(56, 189, 248, 0.25), 0 0 60px rgba(56, 189, 248, 0.08); transform: scale(1); }
        50% { box-shadow: 0 0 40px rgba(56, 189, 248, 0.45), 0 0 80px rgba(56, 189, 248, 0.18); transform: scale(1.03); }
    }
    .cyber-mic-hero .mic-label {
        margin-top: 0.6rem;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #38bdf8;
        font-weight: 600;
    }

    /* Nav icon inline */
    .nav-icon {
        width: 18px;
        height: 18px;
        vertical-align: middle;
        margin-right: 5px;
        border-radius: 3px;
    }
</style>
""",
    unsafe_allow_html=True,
)


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
    return {"LOW": "#16a34a", "MEDIUM": "#f97316", "HIGH": "#dc2626"}.get(
        level.name, "#6b7280"
    )


def _new_user_panel():
    """Beginner-friendly description of the system."""
    with st.expander("What Does This System Do?", expanded=True):
        st.markdown(
            """
            This dashboard analyzes phone-call audio to help you spot **AI-generated voices** and
            **scam conversations** in real time.

            - Detects fake (AI-generated) voices using a CNN-based deepfake detector.
            - Transcribes speech with **Whisper** and analyzes the text using a **BERT classifier**.
            - Identifies **scam patterns, social-engineering cues, and risky language**.
            - Combines voice + language signals into a clear **risk score** and threat summary.
            """
        )


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown(
    f"<div class='sidebar-title'><img src='{_ICON_SHIELD}' class='sidebar-icon'/>Deepfake Voice Scam Detection</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Real-time audio + NLP threat analytics")
st.sidebar.markdown("---")

from streamlit_option_menu import option_menu

# Build navigation labels with inline icons
with st.sidebar:
    mode = option_menu(
        menu_title="Navigation",
        options=["Upload Audio", "Live Microphone", "Analyze Demo", "Call History & Reports"],
        icons=["cloud-upload", "mic", "play-circle", "clock-history"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#444"},
            "nav-link-selected": {"background-color": "#1f2937"},
        }
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    0.0,
    1.0,
    0.5,
    0.05,
    help="Threshold separating REAL vs FAKE verdicts for individual frames.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")
model = load_model()
if model is not None:
    st.sidebar.success("Model Operational")
else:
    st.sidebar.error("Model Not Loaded")
    st.sidebar.caption("Run `python scripts/train.py` to train the deepfake model.")


# ── Main area ────────────────────────────────────────────────
st.title("Deepfake Voice Scam Detection Console")
st.caption("Cybersecurity-grade analytics for voice spoofing and scam calls.")

_new_user_panel()

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


def _aggregate_nlp_results(frame_results):
    """Aggregate NLP (BERT) outputs across frames into a single summary."""
    transcripts = []
    detected_phrases = set()
    best_probs = {"normal": 1.0, "suspicious": 0.0, "scam": 0.0}
    nlp_score = 0.0
    intent = "Normal"

    for fr in frame_results:
        if "nlp_result" not in fr:
            continue
        nlp = fr["nlp_result"]
        text = nlp.get("transcript", "") or ""
        if text.strip():
            transcripts.append(text)
        detected_phrases.update(nlp.get("detected_phrases", []))

        prob = float(nlp.get("nlp_probability", 0.0))
        if prob > nlp_score:
            nlp_score = prob
            intent = nlp.get("intent", "Normal")
            if "detailed_probs" in nlp:
                best_probs = {
                    "normal": float(nlp["detailed_probs"].get("normal", 0.0)),
                    "suspicious": float(nlp["detailed_probs"].get("suspicious", 0.0)),
                    "scam": float(nlp["detailed_probs"].get("scam", 0.0)),
                }

    full_transcript = " ".join(transcripts) if transcripts else "No speech detected."
    return {
        "transcript": full_transcript,
        "nlp_score": nlp_score,
        "intent": intent,
        "best_probs": best_probs,
        "detected_phrases": detected_phrases,
    }

# ═══════════════════════════════════════════════════════════════
#  UPLOAD & LIVE MIC MODE
# ═══════════════════════════════════════════════════════════════
if mode in ["Upload Audio", "Live Microphone"]:
    # Cyber mic hero image
    if _ICON_MIC:
        st.markdown(
            f"<div class='cyber-mic-hero'>"
            f"<img src='{_ICON_MIC}' alt='Voice Analysis'/>"
            f"<div class='mic-label'>Voice Threat Scanner</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if mode == "Upload Audio":
        uploaded = st.file_uploader(
            "Upload call audio for analysis",
            type=["wav", "flac", "mp3", "ogg", "m4a"],
            help="Supported formats: WAV, FLAC, MP3, OGG, M4A",
        )
        input_desc = "Step 1 · Upload a phone-call recording to begin the scan."
    else:
        uploaded = st.audio_input("Record voice sample for real-time analysis")
        input_desc = "Step 1 · Record your voice or a live call to begin the scan."

    col_main, col_right = st.columns([2.4, 1.6])

    if uploaded is None:
        with col_main:
            st.markdown(
                f"<div class='glass-card'><h3>Awaiting Audio Input</h3>"
                f"<p class='sub-label'>{input_desc}</p>"
                f"<p>The system will run deepfake detection, transcribe the speech, and analyze it with BERT for scam patterns.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_right:
            st.markdown(
                "<div class='glass-card'><h3>Threat Summary</h3>"
                "<p>No audio analyzed yet.</p>"
                "<p class='sub-label'>Risk score, alerts, and explanations will appear here.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        # Left: input + rich analysis cards; Right: threat overview
        with col_main:
            st.markdown(
                "<div class='glass-card'><h3>Audio Input</h3>"
                "<p class='sub-label'>Step 1 · Uploaded Call Recording</p></div>",
                unsafe_allow_html=True,
            )
            st.audio(uploaded, format="audio/wav")

            fpath = _save_uploaded_file(uploaded)

            with st.spinner("Running deepfake & NLP analysis..."):
                detector = RealtimeDetector(model=model)
                result = detector.analyze_file(fpath)

            assessment = result["assessment"]
            alert = result["alert"]
            level: RiskLevel = assessment["level"]
            frame_results = result.get("frame_results", [])
            nlp_info = _aggregate_nlp_results(frame_results) if frame_results else {
                "transcript": "No speech detected.",
                "nlp_score": 0.0,
                "intent": "Normal",
                "best_probs": {"normal": 1.0, "suspicious": 0.0, "scam": 0.0},
                "detected_phrases": set(),
            }

            deepfake_prob = float(assessment.get("voice_score", assessment.get("score", 0.0)))
            nlp_prob = float(assessment.get("nlp_score", nlp_info["nlp_score"] * 100.0))

            # ── Top analysis cards ──
            st.markdown(
                "<div class='glass-card'><h3>Analysis Snapshot</h3>"
                "<p class='sub-label'>Deepfake · NLP Scam Analysis · Combined Risk</p>",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                verdict = "Fake" if deepfake_prob >= (threshold * 100.0) else "Real"
                st.markdown("**Deepfake Detection**")
                st.metric("Voice Verdict", verdict, f"{deepfake_prob:.1f}% prob")
                st.caption("Probability that the **voice itself** is AI-generated.")

            with c2:
                st.markdown("**NLP Scam Analysis (BERT)**")
                st.metric("Conversation Class", nlp_info["intent"], f"{nlp_info['nlp_score'] * 100:.1f}% risk")
                st.caption("BERT classifier scoring the **conversation content**.")

            with c3:
                st.markdown("**Combined Risk Score**")
                st.metric("Risk Level", level.name.title(), f"{assessment['score']:.1f}% overall")
                st.caption("Takes the worst of voice + language risk for safety.")

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Advanced panels: Voice Analysis, Conversation Intelligence, System Explanation ──
            vcol1, vcol2 = st.columns(2)

            with vcol1:
                st.markdown("<div class='glass-card'><h3>Voice Analysis Overview</h3>", unsafe_allow_html=True)

                if frame_results:
                    probs = [fr["probability"] * 100.0 for fr in frame_results]
                    df = pd.DataFrame({
                        "Frame": list(range(len(probs))),
                        "Deepfake Probability (%)": probs,
                    })
                    fig = px.area(
                        df,
                        x="Frame",
                        y="Deepfake Probability (%)",
                        range_y=[0, 100],
                        template="plotly_dark",
                    )
                    fig.update_traces(line_color="#38bdf8")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No frame-level probabilities available.")

                st.markdown(
                    "Deepfake probability is computed from spectral features of each audio frame. "
                    "Sharper peaks and unstable prosody often indicate synthesis artifacts.",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with vcol2:
                st.markdown("<div class='glass-card'><h3>Conversation Intelligence</h3>", unsafe_allow_html=True)
                st.markdown(f"**Transcript:**\n> {nlp_info['transcript']}")

                if nlp_info["detected_phrases"]:
                    st.warning(
                        "**Suspicious Phrases Detected:** "
                        + ", ".join(sorted(nlp_info["detected_phrases"])),
                    )

                st.markdown(
                    "The BERT model flags conversations as **Suspicious** or **Scam** when it sees "
                    "patterns like urgent payment requests, OTP collection, remote-access demands, "
                    "or impersonation of banks and authorities.",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # System explanation panel
            st.markdown(
                """<div class='glass-card'>
                <h3>How This System Works</h3>
                <p class='sub-label'>Pipeline · From audio to risk score</p>
                <ol>
                    <li><strong>Audio Input</strong> – You upload or stream a call recording.</li>
                    <li><strong>Deepfake Detection</strong> – A CNN analyzes the voice signal for synthesis artifacts.</li>
                    <li><strong>Speech-to-Text (Whisper)</strong> – The audio is transcribed into text.</li>
                    <li><strong>NLP Analysis (BERT)</strong> – The transcript is classified as Normal / Suspicious / Scam.</li>
                    <li><strong>Risk Scoring</strong> – Voice and text risks are fused into a single threat level.</li>
                </ol>
                </div>""",
                unsafe_allow_html=True,
            )

            # Technical feature breakdown (optional for advanced users)
            if frame_results and frame_results[-1].get("features"):
                with st.expander("🔬 Technical Audio Features (Advanced)"):
                    features = frame_results[-1]["features"]
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        st.markdown("**Spectral Features**")
                        if "mfcc_mean" in features:
                            st.write(f"MFCC Mean: `{features['mfcc_mean']:.4f}`")
                        if "spectral_centroid_mean" in features:
                            st.write(
                                f"Spectral Centroid: `{features['spectral_centroid_mean']:.1f}` Hz",
                            )
                        if "spectral_centroid_std" in features:
                            st.write(
                                f"Spectral Centroid Std: `{features['spectral_centroid_std']:.1f}` Hz",
                            )

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

            # Per-frame table
            if frame_results and len(frame_results) > 1:
                with st.expander(f"📋 Per-Frame Details ({result['num_frames']} frames)"):
                    rows = []
                    for i, fr in enumerate(frame_results):
                        rows.append(
                            {
                                "Frame": i,
                                "Probability": f"{fr['probability']:.4f}",
                                "Verdict": "FAKE" if fr["probability"] >= threshold else "REAL",
                            },
                        )
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Right-hand threat assessment column
        with col_right:
            st.markdown("<div class='glass-card'><h3>Threat Assessment</h3>", unsafe_allow_html=True)

            st.markdown(
                f"<p class='sub-label'>Overall Risk · {level.name.title()}</p>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div class='risk-badge risk-{level.name}'>"
                f"<span>{level.name.title()} Risk</span>"
                f"<span>{assessment['score']:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.progress(min(1.0, assessment["score"] / 100.0))

            st.markdown("---")
            st.markdown("**Signal Breakdown**")
            st.metric("Deepfake Score", f"{deepfake_prob:.1f}%")
            st.metric("NLP Scam Score", f"{nlp_prob:.1f}%")

            st.markdown("---")
            if level == RiskLevel.HIGH:
                st.error(f"⚠️ {alert['title']}: {alert['message']}")
                st.caption("High risk: treat this call as a likely scam attempt.")
            elif level == RiskLevel.MEDIUM:
                st.warning(f"🔶 {alert['title']}: {alert['message']}")
                st.caption("Medium risk: proceed with caution and verify identity.")
            else:
                st.success(f"✅ {alert['title']}: {alert['message']}")
                st.caption("Low risk: no strong deepfake or scam indicators detected.")

            if alert.get("explanation"):
                with st.expander("Why was this score assigned?", expanded=True):
                    for exp in alert["explanation"]:
                        st.markdown(f"- {exp}")
                    if alert.get("action"):
                        st.info(f"**Recommended action:** {alert['action']}")

# ═══════════════════════════════════════════════════════════════
#  DEMO MODE
# ═══════════════════════════════════════════════════════════════
elif mode == "Analyze Demo":
    st.markdown("### Demo Mode")
    st.markdown("Run analysis on pre-generated sample audio to see the system in action.")

    # Find demo files
    demo_files = []
    for root, _dirs, files in os.walk(RAW_DATA_DIR):
        for f in sorted(files):
            if f.endswith(".wav"):
                demo_files.append(os.path.join(root, f))

    if not demo_files:
        st.warning(
            "No demo data found. Generate it by running:\n\n"
            "```bash\npython scripts/prepare_dataset.py --generate-demo\n```"
        )
    else:
        real_files = [df for df in demo_files if "/real/" in df][:5]
        fake_files = [df for df in demo_files if "/fake/" in df][:5]

        col_r, col_f = st.columns(2)

        with col_r:
            st.markdown("#### Real Voice Samples")
            for fpath in real_files:
                fname = os.path.basename(fpath)
                if st.button(f"Analyze {fname}", key=f"real_{fname}"):
                    _run_demo_analysis(fpath, model, threshold)

        with col_f:
            st.markdown("#### Fake Voice Samples")
            for fpath in fake_files:
                fname = os.path.basename(fpath)
                if st.button(f"Analyze {fname}", key=f"fake_{fname}"):
                    _run_demo_analysis(fpath, model, threshold)

# ═══════════════════════════════════════════════════════════════
#  HISTORY AND METRICS (NEW)
# ═══════════════════════════════════════════════════════════════
elif mode == "Call History & Reports":
    st.markdown("### Historical Logs & System Metrics")
    import json
    from src.utils.config import RESULTS_DIR, LOG_PATH
    
    tab1, tab2 = st.tabs(["Call History", "Evaluation Metrics"])
    
    with tab1:
        st.markdown("#### Recent Call Analysis Logs")
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
        st.markdown("#### System Performance Visualizations")
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
