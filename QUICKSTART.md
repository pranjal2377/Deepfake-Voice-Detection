# Quick Start

This repository is intended to be clone-and-run friendly.

## Requirements

- Python 3.10-3.12 recommended
- Git
- FFmpeg on PATH for audio/transcription workflows
- A microphone for live microphone mode
- Internet access on first use so Hugging Face models can be downloaded

The repository contains the project source, configuration, tracked model/test artifacts, sample audio, tests, and documentation. Large external training datasets are not bundled; use the dataset download script when training from scratch.

## Windows

Double-click **SETUP_AND_RUN.bat** and choose an action.

Or run manually:

```bat
py -m venv .venv
.venv\\Scripts\\python.exe -m pip install --upgrade pip
.venv\\Scripts\\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m streamlit run src\\ui\\dashboard.py
```

## Linux / macOS

Run:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

## Useful commands

Dashboard:
```bash
python -m streamlit run src/ui/dashboard.py
```

Demo:
```bash
python scripts/realtime.py --mode demo
```

Live microphone:
```bash
python scripts/realtime.py --mode mic
```

Train:
```bash
python scripts/train.py
```

Evaluate:
```bash
python scripts/evaluate.py
```

Tests:
```bash
python -m pytest tests/ -v
```

## Dataset

For training from scratch:

```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py --force-resplit
```

The full external training dataset is intentionally not committed to GitHub.

## Model and configuration

System settings and model paths are controlled through `configs/config.yaml`. The application can use the tracked model artifact when present. If it is missing or you want to retrain, run `python scripts/train.py`.

## Troubleshooting

- Install FFmpeg and ensure it is on PATH if audio/transcription fails.
- Check OS microphone permissions for live mode.
- Hugging Face models require internet access on first use.
- If PyTorch installation fails, verify the Python version and use the CPU installation command above.
