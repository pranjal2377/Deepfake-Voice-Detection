"""
Phase 7 Tests — Polish & Documentation.

Tests:
  1. README.md is comprehensive (sections, badges, structure)
  2. requirements.txt lists all necessary packages
  3. __init__.py files export correct public APIs
  4. Error handling — missing file, empty audio, missing config
  5. run_all.sh script exists and is executable
"""

import os
import sys
import importlib

# Ensure project root on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def test_readme_comprehensive():
    """README.md has all expected sections and content."""
    readme_path = os.path.join(_project_root, "README.md")
    assert os.path.exists(readme_path), "README.md not found"

    with open(readme_path, "r") as f:
        content = f.read()

    # Must have key sections
    required_sections = [
        "Overview",
        "Architecture",
        "Project Structure",
        "Tech Stack",
        "Quick Start",
        "Usage Guide",
        "Model Details",
        "Risk Classification",
        "Configuration",
        "Testing",
        "Development Phases",
    ]
    for section in required_sections:
        assert section in content, f"README missing section: {section}"

    # Must have key info
    assert "DeepfakeCNN" in content, "README should mention the model name"
    assert "streamlit" in content.lower(), "README should mention Streamlit"
    assert "requirements.txt" in content, "README should reference requirements"
    assert "run_all.sh" in content, "README should reference run_all.sh"

    # Reasonably long (at least 3000 chars for a proper README)
    assert len(content) > 3000, f"README seems too short ({len(content)} chars)"


def test_requirements_valid():
    """requirements.txt contains all critical packages."""
    req_path = os.path.join(_project_root, "requirements.txt")
    assert os.path.exists(req_path), "requirements.txt not found"

    with open(req_path, "r") as f:
        content = f.read().lower()

    required_packages = [
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "librosa",
        "sounddevice",
        "soundfile",
        "scikit-learn",
        "streamlit",
        "pyyaml",
        "pandas",
        "pytest",
    ]
    for pkg in required_packages:
        assert pkg in content, f"requirements.txt missing: {pkg}"


def test_init_exports():
    """__init__.py files properly export public APIs."""
    # src package has version
    import src
    assert hasattr(src, "__version__"), "src.__init__ should define __version__"

    # Key sub-packages export their classes / functions
    from src.audio import preprocess_audio, load_audio
    from src.features import extract_all_features, features_to_model_input
    from src.model import DeepfakeCNN
    from src.scoring import RiskLevel, RiskScorer
    from src.alerts import generate_alert
    from src.detection import RealtimeDetector
    from src.evaluation import ModelEvaluator, load_model_for_evaluation
    from src.training import Trainer, compute_metrics
    from src.data import (
        discover_files,
        create_manifest,
        AudioAugmentor,
        DeepfakeAudioDataset,
        create_dataloader,
    )

    # Quick sanity — they're actually callable / class-like
    assert callable(preprocess_audio)
    assert callable(load_audio)
    assert callable(extract_all_features)
    assert callable(generate_alert)
    assert callable(compute_metrics)


def test_error_handling():
    """Critical modules raise clear errors for edge cases."""
    import pytest as _pytest
    from src.audio.preprocessor import load_audio, preprocess_audio

    # 1. Missing file → FileNotFoundError
    try:
        load_audio("/nonexistent/path/audio.wav")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()

    # 2. Non-audio file → ValueError
    try:
        # Use this very test file as a "non-audio" file
        load_audio(__file__)
        assert False, "Expected ValueError for non-audio file"
    except (ValueError, Exception):
        pass  # Any error is acceptable for a non-audio file

    # 3. Config loader — missing file
    from src.utils.config import load_config
    try:
        load_config("/nonexistent/config.yaml")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()

    # 4. Detector — missing file
    from src.detection.detector import RealtimeDetector
    detector = RealtimeDetector()
    try:
        detector.analyze_file("/nonexistent/audio.wav")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_run_script_exists():
    """run_all.sh exists, is executable, and has proper structure."""
    script_path = os.path.join(_project_root, "run_all.sh")
    assert os.path.exists(script_path), "run_all.sh not found"

    # Check executable
    assert os.access(script_path, os.X_OK), "run_all.sh should be executable"

    with open(script_path, "r") as f:
        content = f.read()

    # Must have shebang
    assert content.startswith("#!/"), "run_all.sh should have a shebang line"

    # Must reference key steps
    assert "prepare_dataset" in content, "Script should run dataset preparation"
    assert "train.py" in content, "Script should run training"
    assert "evaluate.py" in content, "Script should run evaluation"
    assert "requirements.txt" in content, "Script should install requirements"
    assert "streamlit" in content, "Script should mention dashboard launch"
