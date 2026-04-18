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
- **Interactive UI Dashboard**: A sleek, real-time dashboard for visualization, evaluation, and monitoring of incoming audio streams.

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

### 1. Training the Model
To train the CNN-based deepfake detection model on the prepared dataset:
```bash
python scripts/train.py
# Or using the unified runner:
python run_system.py --mode train
```

### 2. Evaluating the Model
To run evaluations and generate metrics/reports against the test dataset:
```bash
python scripts/evaluate.py
```

### 3. Running the UI Dashboard (Demo Mode)
Launch the interactive dashboard to visualize system capabilities on mock or uploaded audio records.
```bash
python run_system.py --mode demo
# Alternatively, you can run:
./run_all.sh
```

> **Note for Demonstrations:**
> If you need to force predefined confidence scores for presentation purposes based on filenames (e.g., forcing a high probability for files named `fake_...`), you can apply the demo override patch:
> ```bash
> python patch.py
> ```

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
├── run_system.py        # Main entry point for the system
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
