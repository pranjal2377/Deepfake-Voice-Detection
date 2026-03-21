import os
import time
from datetime import datetime
import numpy as np

# Load internal modules
from src.detection.detector import RealtimeDetector
from src.utils.logger import global_logger
from src.utils.config import load_config

import librosa

class DemoRunner:
    """
    Runs an end-to-end evaluation/demo pipeline on a single audio file.
    Audio Upload -> Deepfake Detection -> Speech-to-Text -> NLP Scam Detection -> Risk Score -> Alert
    """
    
    def __init__(self):
        self.detector = RealtimeDetector()
        
    def process_file(self, audio_path: str):
        print(f"--- Starting Demo for: {os.path.basename(audio_path)} ---")
        
        # 1. Audio Upload / Load
        try:
            # Simple simulate real-time loop by processing chunks, or process entire file at once
            print("[1/5] Loading Audio...")
            audio_data, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return
            
        print("[2/5] Deepfake Voice Detection...")
        # To simulate pipeline on full file, we do detection
        # The detector handles everything if we push frames, but for a 1-shot demo we can use the transcriber and model directly or feed all data through detector.
        self.detector.reset()
        
        # Feed all audio to detector block by block
        # the detector expects 2 seconds of audio max in frame? Currently the process chunk handles an arbitrary size but let's test.
        # Detector takes 16000 hz float32
        chunk_size = sr * 2 # 2 seconds chunks
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < sr * 0.5:
                continue # skip tiny ends
            
            # 3 & 4. STT and NLP inside process_audio_chunk 
            result = self.detector.process_audio_chunk(chunk)
            
        final_assessment = self.detector.risk_scorer.get_assessment()
        transcript = self.detector.get_current_transcript()
        
        print("[3/5] Speech-to-Text Transcription...")
        print(f"Transcript: {transcript}")
        
        print("[4/5] Risk Scoring...")
        print(f"Audio Deepfake Probability: {final_assessment.get('voice_score', 0)}%")
        print(f"NLP Scam Probability: {final_assessment.get('nlp_score', 0)}%")
        
        combined_score = final_assessment.get('score', 0)
        
        print(f"[5/5] Generating Alert...")
        level = final_assessment.get('level', "Unknown").value if hasattr(final_assessment.get('level'), 'value') else "Unknown"
        print(f"Final Combined Risk Score: {combined_score}% (Level: {level})")
        
        # Evaluate Scam Intent
        nlp_result = self.detector.scam_detector.analyze_transcript(transcript)
        
        # Log to JSON
        timestamp = datetime.now().isoformat()
        global_logger.log_analysis(
            timestamp=timestamp,
            transcript=transcript,
            voice_probability=final_assessment.get('voice_score', 0) / 100.0,
            scam_probability=final_assessment.get('nlp_score', 0) / 100.0,
            combined_risk_score=combined_score / 100.0,
            detected_phrases=nlp_result.get('detected_phrases', []),
            intent=nlp_result.get('intent', 'None')
        )
        
        print("Demo complete. Log written.")
        return final_assessment

if __name__ == "__main__":
    demo = DemoRunner()
    # A dummy run with a test wav could be initiated here, but leaving to user as they would provide sample
    print("Demo module ready. Import and call `DemoRunner().process_file('path.wav')`")
