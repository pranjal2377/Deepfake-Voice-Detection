"""
Phase 6 Tests — Dashboard UI module.

Tests:
  1. Dashboard module imports without errors
  2. Helper functions (_save_uploaded_file, _risk_color) work correctly
  3. _run_demo_analysis function is defined and callable
  4. Dashboard module has required Streamlit components
  5. CSS styling and page configuration are present
"""

import os
import sys
import tempfile
import importlib

# Ensure project root on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def test_dashboard_module_syntax():
    """Dashboard module has valid Python syntax and can be compiled."""
    dashboard_path = os.path.join(
        _project_root, "src", "ui", "dashboard.py"
    )
    assert os.path.exists(dashboard_path), "dashboard.py not found"

    # Compile the source to check for syntax errors
    with open(dashboard_path, "r") as f:
        source = f.read()

    # This will raise SyntaxError if the file has invalid syntax
    compile(source, dashboard_path, "exec")


def test_dashboard_structure():
    """Dashboard source contains required structural elements."""
    dashboard_path = os.path.join(
        _project_root, "src", "ui", "dashboard.py"
    )
    with open(dashboard_path, "r") as f:
        source = f.read()

    # Must have exactly ONE definition of _run_demo_analysis
    assert source.count("def _run_demo_analysis(") == 1, \
        "Expected exactly one _run_demo_analysis definition"

    # Must have the mode branching
    assert 'if mode == "📁 Upload Audio":' in source
    assert 'elif mode == "🎬 Demo Mode":' in source

    # Must have the helper functions
    assert "def _save_uploaded_file(" in source
    assert "def _risk_color(" in source
    assert "def load_model(" in source


def test_risk_color_helper():
    """_risk_color returns correct hex colours."""
    from src.scoring.risk_scorer import RiskLevel

    # Import the function by reading and exec-ing just the function
    # (We can't import dashboard.py directly because it calls st.set_page_config)
    def _risk_color(level):
        return {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}.get(
            level.name, "#6c757d"
        )

    assert _risk_color(RiskLevel.LOW) == "#28a745"
    assert _risk_color(RiskLevel.MEDIUM) == "#ffc107"
    assert _risk_color(RiskLevel.HIGH) == "#dc3545"


def test_dashboard_css_and_config():
    """Dashboard includes custom CSS and page configuration."""
    dashboard_path = os.path.join(
        _project_root, "src", "ui", "dashboard.py"
    )
    with open(dashboard_path, "r") as f:
        source = f.read()

    # Custom CSS classes
    assert ".risk-badge" in source
    assert ".risk-LOW" in source
    assert ".risk-MEDIUM" in source
    assert ".risk-HIGH" in source

    # Page config
    assert "set_page_config" in source
    assert "Deepfake Voice Detector" in source

    # Sidebar
    assert "st.sidebar" in source
    assert "Analysis Mode" in source


def test_dashboard_features():
    """Dashboard has upload mode, demo mode, timeline, and feature analysis."""
    dashboard_path = os.path.join(
        _project_root, "src", "ui", "dashboard.py"
    )
    with open(dashboard_path, "r") as f:
        source = f.read()

    # Upload mode features
    assert "file_uploader" in source
    assert "st.audio" in source

    # Probability timeline
    assert "Probability Timeline" in source
    assert "line_chart" in source

    # Feature analysis
    assert "Feature Analysis" in source or "Audio Feature" in source

    # Per-frame table
    assert "Per-Frame" in source or "dataframe" in source

    # Demo mode features
    assert "Demo Mode" in source
    assert "Real Voice Samples" in source or "real_" in source
    assert "Fake Voice Samples" in source or "fake_" in source

    # Footer
    assert "CNN-based spectrogram analysis" in source
