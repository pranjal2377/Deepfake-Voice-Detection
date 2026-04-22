"""
Real-Time Detection Engine — Full implementation.

Supports:
  1. File-based analysis (analyze_file)
  2. Live microphone streaming (start_realtime / stop_realtime)
  3. Simulation mode for testing without a microphone (start_simulation)

The real-time pipeline captures audio in a circular buffer, processes
fixed-length frames, runs the CNN model, and feeds results into the
risk scorer. Each frame produces a callback with the latest assessment.
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, Optional, Callable, List
from collections import deque

from src.audio.preprocessor import (
    preprocess_audio, split_into_frames, normalize_audio, trim_silence,
)
from src.features.extractor import extract_all_features, features_to_model_input
from src.scoring.risk_scorer import RiskScorer
from src.alerts.alert_system import generate_alert
from src.utils.config import SAMPLE_RATE, FRAME_DURATION, HOP_DURATION
from src.nlp.transcriber import AudioTranscriber
from src.nlp.bert_classifier import BertClassifier

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """
    Orchestrates the full detection pipeline:
    Audio → Preprocess → Features → Model → Score → Alert.

    Supports file analysis, live microphone capture, and simulation mode.
    """

    def __init__(self, model=None, sr: int = SAMPLE_RATE):
        self.model = model
        self.sr = sr
        self.scorer = RiskScorer()
        self.transcriber = AudioTranscriber()
        self.bert_classifier = BertClassifier()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

        # Audio buffer for real-time processing
        self._frame_samples = int(sr * FRAME_DURATION)
        self._hop_samples = int(sr * HOP_DURATION)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        # History of frame results
        self.frame_history: List[Dict] = []

    # ─── File-based analysis ────────────────────────────────

    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze a single audio file end-to-end.

        Args:
            file_path: Path to an audio file

        Returns:
            Dict with assessment, alert, and features

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If the audio file is invalid or unreadable.
        """
        import os
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # 1. Preprocess
        try:
            audio, sr = preprocess_audio(file_path)
            # MEGA-SPEEDUP: Run Transcription & BERT ONCE on the full file!
            # Instead of choking the CPU by transcribing every 2-second overlapping chunk 
            full_transcript = self.transcriber.transcribe(audio, sr)
            full_nlp_result = self.bert_classifier.analyze_transcript(full_transcript)
        except Exception as exc:
            raise ValueError(f"Failed to preprocess '{file_path}': {exc}") from exc

        # 2. Split into frames
        frames = split_into_frames(audio, sr, FRAME_DURATION)
        if len(frames) == 0:
            frames = [audio]  # use entire clip as one frame

        results = []
        self.scorer.reset()

        for frame in frames:
            # 3. Extract features
            features = extract_all_features(frame, sr)

            # 4. Model inference
            # (Note: Removed hardcoded demo overrides. System strictly relies on CNN weights now)
            probability = self._predict(frame, sr)
            
            # 4.5. NLP inference (MOCKED to the pre-computed full file result for 15x speeds!)
            nlp_result = full_nlp_result
            
            # 5. Score
            self.scorer.add_prediction(probability, nlp_result["nlp_probability"])

            results.append({
                "features": features,
                "probability": probability,
                "nlp_result": nlp_result
            })

        # 6. Final assessment & alert
        assessment = self.scorer.get_assessment()
        alert = generate_alert(
            assessment,
            features=results[-1]["features"] if results else None,
        )

        return {
            "assessment": assessment,
            "alert": alert,
            "frame_results": results,
            "num_frames": len(frames),
        }

    # ─── Real-time microphone capture ──────────────────────

    def start_realtime(self, callback: Optional[Callable] = None):
        """
        Start live microphone capture and continuous analysis.

        Args:
            callback: Function called with each frame result dict:
                      {assessment, alert, features, probability, frame_index}
        """
        if self._is_running:
            logger.warning("Real-time detection already running")
            return

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("sounddevice required: pip install sounddevice")

        if callback:
            self._callbacks.append(callback)

        self.scorer.reset()
        self.frame_history.clear()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._is_running = True

        # Audio callback — runs in sounddevice's thread
        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio status: {status}")
            audio_chunk = indata[:, 0].copy()  # mono
            with self._lock:
                self._buffer = np.concatenate([self._buffer, audio_chunk])

        # Processing thread — runs analysis on buffered frames
        def _processing_loop():
            frame_index = 0
            while self._is_running:
                frame = self._extract_frame()
                if frame is not None:
                    result = self._process_frame(frame, frame_index)
                    frame_index += 1
                    self._notify_callbacks(result)
                else:
                    time.sleep(0.05)  # wait for more audio

        # Start audio stream
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=_audio_callback,
            blocksize=int(self.sr * 0.1),  # 100ms blocks
        )
        self._stream.start()

        # Start processing thread
        self._thread = threading.Thread(target=_processing_loop, daemon=True)
        self._thread.start()

        logger.info("Real-time detection started (microphone)")

    def stop_realtime(self):
        """Stop live capture and processing."""
        self._is_running = False

        if hasattr(self, "_stream"):
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._callbacks.clear()
        logger.info("Real-time detection stopped")

    # ─── Simulation mode ───────────────────────────────────

    def start_simulation(
        self,
        audio: np.ndarray,
        callback: Optional[Callable] = None,
        speed: float = 1.0,
    ):
        """
        Simulate real-time analysis on a pre-loaded audio array.
        Useful for testing, demos, and environments without microphones.

        Args:
            audio: Audio waveform (numpy array at self.sr)
            callback: Function called with each frame result
            speed: Playback speed multiplier (1.0 = real-time, >1 = faster)
        """
        if self._is_running:
            logger.warning("Detection already running")
            return

        if callback:
            self._callbacks.append(callback)

        self.scorer.reset()
        self.frame_history.clear()
        self._is_running = True

        def _simulation_loop():
            # Normalize and prepare audio
            processed = normalize_audio(audio)
            frames = split_into_frames(processed, self.sr, FRAME_DURATION)

            if len(frames) == 0:
                frames = [processed]

            delay = HOP_DURATION / speed

            for i, frame in enumerate(frames):
                if not self._is_running:
                    break

                result = self._process_frame(frame, i)
                self._notify_callbacks(result)

                # Simulate real-time delay
                time.sleep(delay)

            self._is_running = False
            logger.info("Simulation complete")

        self._thread = threading.Thread(target=_simulation_loop, daemon=True)
        self._thread.start()

        logger.info(f"Simulation started ({len(audio)/self.sr:.1f}s audio, speed={speed}x)")

    # ─── Internal methods ──────────────────────────────────

    def _extract_frame(self) -> Optional[np.ndarray]:
        """Extract a frame from the audio buffer if enough data is available."""
        with self._lock:
            if len(self._buffer) < self._frame_samples:
                return None

            frame = self._buffer[:self._frame_samples].copy()
            # Advance by hop length (sliding window)
            self._buffer = self._buffer[self._hop_samples:]

        return frame

    def _process_frame(self, frame: np.ndarray, frame_index: int) -> Dict:
        """Process a single audio frame through the full pipeline."""
        # Extract features
        features = extract_all_features(frame, self.sr)

        # Model inference
        probability = self._predict(frame, self.sr)

        # LIVE MIC OPTIMIZATION: Only run heavy NLP inference every 4th frame (~2 secs loop delay)
        # Re-using previous result guarantees real-time UI without huge lag spikes!
        if not hasattr(self, "_cached_nlp"):
            self._cached_nlp = {"nlp_probability": 0.0, "risk_level": "Low"}
            
        if frame_index % 4 == 0:
            transcript = self.transcriber.transcribe(frame, self.sr)
            if transcript.strip():
                self._cached_nlp = self.bert_classifier.analyze_transcript(transcript)
        
        nlp_result = self._cached_nlp

        # Update scorer
        self.scorer.add_prediction(probability, nlp_result["nlp_probability"])
        assessment = self.scorer.get_assessment()

        # Generate alert
        alert = generate_alert(assessment, features=features)

        result = {
            "frame_index": frame_index,
            "probability": probability,
            "nlp_result": nlp_result,
            "assessment": assessment,
            "alert": alert,
            "features": features,
            "timestamp": time.time(),
        }

        self.frame_history.append(result)
        return result

    def _predict(self, frame: np.ndarray, sr: int = 16000) -> float:
        """
        Run model inference on a single frame.
        Returns 0.5 (uncertain) if no model is loaded.
        """
        if self.model is None:
            return 0.5

        import torch
        mel = features_to_model_input(frame, sr)
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(tensor).squeeze().item()
            
        # --- FINE-TUNING CALIBRATION FOR LIVE MICROPHONE ---
        # CNNs trained on pristine datasets over-activate (predict Fake)
        # when processing noisy/distorted WebM or live mics. 
        # Here we quadratically penalize marginal predictions and apply a scalar.
        prob = float(prob)
        prob = (prob ** 1.3) * 0.85
        prob = max(0.0, min(1.0, prob))

        return prob

    def _notify_callbacks(self, result: Dict):
        """Send result to all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def add_callback(self, callback: Callable):
        """Register an additional callback."""
        self._callbacks.append(callback)

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_summary(self) -> Dict:
        """Get a summary of the current session."""
        if not self.frame_history:
            return {"frames_analyzed": 0}

        probs = [r["probability"] for r in self.frame_history]
        return {
            "frames_analyzed": len(self.frame_history),
            "avg_probability": float(np.mean(probs)),
            "max_probability": float(np.max(probs)),
            "min_probability": float(np.min(probs)),
            "latest_assessment": self.frame_history[-1]["assessment"],
            "latest_alert": self.frame_history[-1]["alert"],
        }
