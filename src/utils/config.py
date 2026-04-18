"""
Configuration loader utility.
Loads config.yaml and provides easy access to all settings.
"""

import os
import logging
import yaml
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Raises:
        FileNotFoundError: If the config file is missing.
        yaml.YAMLError: If the config file has invalid YAML.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: configs/config.yaml relative to project root."
        )
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Invalid YAML in config file '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config).__name__}")

    logger.debug("Loaded configuration from %s", path)
    return config


# Load once at import time
_config = load_config()


# ─── Audio Settings ───────────────────────────────────────
SAMPLE_RATE: int = _config["audio"]["sample_rate"]
FRAME_DURATION: float = _config["audio"]["frame_duration"]
HOP_DURATION: float = _config["audio"]["hop_duration"]
CHANNELS: int = _config["audio"]["channels"]

# ─── Feature Settings ────────────────────────────────────
N_MFCC: int = _config["features"]["n_mfcc"]
N_MELS: int = _config["features"]["n_mels"]
N_FFT: int = _config["features"]["n_fft"]
HOP_LENGTH: int = _config["features"]["hop_length"]
FMAX: int = _config["features"]["fmax"]

# ─── Model Settings ──────────────────────────────────────
MODEL_ARCH: str = _config["model"]["architecture"]
INPUT_LENGTH: float = _config["model"]["input_length"]
BATCH_SIZE: int = _config["model"]["batch_size"]
LEARNING_RATE: float = _config["model"]["learning_rate"]
EPOCHS: int = _config["model"]["epochs"]
EARLY_STOPPING_PATIENCE: int = _config["model"]["early_stopping_patience"]
CONFIDENCE_THRESHOLD: float = _config["model"]["confidence_threshold"]

# ─── Risk Settings (Replaces Scoring) ───────────────────
LOW_THRESHOLD: float = _config["risk"]["medium_risk_threshold"] * 100
MEDIUM_THRESHOLD: float = _config["risk"]["high_risk_threshold"] * 100
AGGREGATION_WINDOW: int = _config["risk"]["aggregation_window"]
SMOOTHING_FACTOR: float = _config["risk"]["smoothing_factor"]

# ─── Paths ────────────────────────────────────────────────
RAW_DATA_DIR: str = os.path.join(PROJECT_ROOT, _config["paths"]["raw_data"])
PROCESSED_DATA_DIR: str = os.path.join(PROJECT_ROOT, _config["paths"]["processed_data"])
TEST_DATA_DIR: str = os.path.join(PROJECT_ROOT, _config["paths"]["test_data"])
MODELS_DIR: str = os.path.join(PROJECT_ROOT, _config["paths"]["models"])
DEEPFAKE_MODEL_PATH: str = os.path.join(PROJECT_ROOT, _config["paths"]["deepfake_model"])
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, _config["paths"]["results"])
DOCS_DIR: str = os.path.join(PROJECT_ROOT, _config["paths"]["docs"])

# ─── Dataset Settings ─────────────────────────────────────
TRAIN_RATIO: float = _config["dataset"]["train_ratio"]
VAL_RATIO: float = _config["dataset"]["val_ratio"]
TEST_RATIO: float = _config["dataset"]["test_ratio"]
DATASET_SEED: int = _config["dataset"]["seed"]
AUGMENT_TRAIN: bool = _config["dataset"]["augment_train"]
AUGMENT_PROB: float = _config["dataset"]["augment_prob"]
TRAIN_DATA_PATH: str = os.path.join(PROJECT_ROOT, _config["dataset"]["train_path"])
TEST_DATA_PATH: str = os.path.join(PROJECT_ROOT, _config["dataset"]["test_path"])
SCAM_DATASET_PATH: str = os.path.join(PROJECT_ROOT, _config["dataset"]["scam_dataset"])

# ─── Logging Settings ─────────────────────────────────────
ENABLE_LOGS: bool = _config["logging"]["enable_logs"]
LOG_PATH: str = os.path.join(PROJECT_ROOT, _config["logging"]["log_path"])



def get_config() -> Dict[str, Any]:
    """Return the full config dictionary."""
    return _config

# ─── NLP Settings ──────────────────────────────────────
NLP_BERT_MODEL_NAME: str = _config.get("nlp", {}).get("bert_model_name", "distilbert-base-uncased")
NLP_CONFIDENCE_THRESHOLD: float = _config.get("nlp", {}).get("confidence_threshold", 0.5)
NLP_DEVICE: str = _config.get("nlp", {}).get("device", "cpu")

