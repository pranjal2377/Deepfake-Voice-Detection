<div align="center">
  <h1>🎙️ Deepfake Voice & Scam Detection System 🛡️</h1>
  <p><strong>A multimodal AI system for detecting AI-generated voices and identifying social engineering or scam intents in real-time.</strong></p>
</div>

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Getting Started (Installation)](#-getting-started-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage & Execution](#-usage--execution)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)

---

## 🎯 Project Overview
This project provides a robust, end-to-end solution for detecting deepfake audio. It combines **Acoustic Analysis** (to detect AI-synthesized voices) with **Natural Language Processing (NLP)** (to analyze transcribed speech for potential social engineering or scam attempts). Both streams are fed into a **Risk Scoring Engine** to provide real-time, actionable alerts through an interactive dashboard.

---

## ✨ Key Features
- **Acoustic Deepfake Detection**: Utilizes a Convolutional Neural Network (CNN) to analyze mel-spectrograms extracted from audio and classify voices as real or AI-generated.
- **Speech-to-Text Transcription**: Employs local transcription capabilities (e.g., Whisper) to convert spoken audio into text.
- **NLP Scam Intent Classification**: Uses a DistilBERT-based text classifier (`src/nlp/bert_classifier.py`) to classify speech intent into *Normal*, *Suspicious*, or *Scam*.
- **Multimodal Risk Scoring**: Fuses deepfake probabilities and NLP scam intent scores to generate a comprehensive risk profile.
- **Live Microphone Integration**: Analyzes incoming audio on-the-fly directly inside the application for real-time threat scanning.
- **Multi-Platform Deployments**: Seamlessly run the UI using a Standard Web Dashboard, a Native Desktop App, or a Mobile PWA (Progressive Web App).

---

## 🏗️ System Architecture

1. **Audio Input Module**: Captures live microphone feeds or processes uploaded `.wav` files.
2. **Feature Extractor**: Transforms audio signals into rich acoustic features (mel-spectrograms).
3. **Deepfake CNN Model**: Classifies the acoustic features to detect synthetic generation artifacts.
4. **Transcription Layer**: Converts the audio to text for semantic analysis.
5. **BERT NLP Classifier**: Evaluates the linguistic content to detect scam rhetoric or manipulation.
6. **Risk Scoring Engine**: Aggregates AI model outputs into a final risk probability.
7. **Alerting System**: Generates human-readable warnings if risk thresholds are breached.

---

## 🚀 Getting Started (Installation)

### Prerequisites
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) (required for audio processing/transcription)

### Installation Steps
1. **Clone the repository** (if you haven't already) and navigate to the directory:
   ```bash
   cd Deepfake_Voice_Detection
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure dependencies for PyTorch and HuggingFace Transformers are properly configured for your hardware).*

---

## 💾 Dataset Preparation

The system uses the `DynamicSuperb/DeepFakeVoiceRecognition_DEEP-VOICE` dataset from HuggingFace for training the acoustic model.

1. **Download the Dataset:**
   This automatically downloads and splits the dataset (80/20 train/test) into the `data/raw/` directory.
   ```bash
   python scripts/download_dataset.py
   ```

2. **Prepare & Preprocess the Data:**
   Extract features (e.g., mel-spectrograms) from the raw `.wav` files for the model.
   ```bash
   python scripts/prepare_dataset.py --force-resplit
   ```

---

## ⚙️ Usage & Execution

The system supports **three** primary deployment wrappers for testing and using the Deepfake Analyzer:

### 1. Web UI Dashboard (Standard)
Launch the interactive web dashboard to visualize system capabilities on mock datasets, uploaded audio records, or the Live Microphone.
```bash
# Using the unified runner:
python run_system.py --mode demo
```

### 2. Native Desktop Application
Launch the system inside an isolated native desktop GUI window (runs cleanly in the background invisibly).
```bash
# Ensure required dependencies are installed (e.g. pywebview)
./desktop_app.py
```

### 3. Voice Threat Scanner Mobile Client (PWA)
Deploy the system wrapped in a Progressive Web App (PWA). This script automatically launches the internal ML engine and generates a QR Code in your terminal. You can scan it on your iPhone/Android to use it as a native mobile app!
```bash
# Generates local QR code for mobile connection
./run_mobile_app.py
```

> **🧠 Extreme Live-Mic Fine-Tuning (New Update):**
> Built-in "demo overrides" or fake static numbers have been fully permanently removed! The system strictly relies on the genuine CNN outputs now, beautifully fine-tuned for actual real-world hardware. It incorporates **Pre-Emphasis Filtering** to eliminate laptop-rumble, **Live DC-Offset Correction**, **RMS Normalization**, **Spectral Noise Gating** (to cut out static hiss), and **Adaptive Multi-Modal Hysteresis Risk Weighting**. 
> *It knows the difference between a distorted phone mic and a real Deepfake.*

---

## 🔧 Model Training & Validation

### Training the Model
To train the CNN-based deepfake detection model on the prepared dataset:
```bash
python scripts/train.py
# Or using the unified runner:
python run_system.py --mode train
```

### Evaluating the Model
To run evaluations and generate metric reports against the test dataset:
```bash
python scripts/evaluate.py
```

---

## 📁 Project Structure

```text
Deepfake_Voice_Detection/
├── configs/             # YAML configuration files (config.yaml)
├── data/                # Raw audio, processed features, and manifests
├── docs/                # Architecture diagrams and icons
├── logs/                # System execution logs
├── models/              # Saved model checkpoints (*.pt) and evaluation reports
├── scripts/             # CLI utilities (download, prepare, train, evaluate)
├── src/
│   ├── alerts/          # Automated alert generation
│   ├── audio/           # Audio preprocessing and I/O
│   ├── detection/       # Deepfake detection inference
│   ├── nlp/             # Transcriber and BERT-based intent classifier
│   ├── scoring/         # Multimodal risk scoring engine
│   ├── ui/              # Dashboard and user interface components
│   └── utils/           # Shared helpers and configuration loaders
├── desktop_app.py       # Wrapper for the standalone Native Desktop execution
├── run_mobile_app.py    # Wrapper for the Mobile Phone UI + QR Code server
├── run_system.py        # Main entry point for the standard system
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation (You are here!)
```

---

## 🔧 Configuration
The system avoids hardcoded parameters. All settings, including model paths, data distributions, thresholds, and hyperparameters, are centrally managed. Edit the configuration file to tune the system:
```bash
nano configs/config.yaml
```

---
*Developed for advanced cybersecurity and deepfake voice detection research.*
