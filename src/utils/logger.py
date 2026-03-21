import json
import logging
import os
from datetime import datetime
from src.utils.config import ENABLE_LOGS, LOG_PATH

class CallLogger:
    """Logs call analysis details to a JSON file."""
    
    def __init__(self):
        self.enable_logs = ENABLE_LOGS
        self.log_path = LOG_PATH
        
        if self.enable_logs:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            
            # Initialize empty file if not exists
            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w') as f:
                    json.dump([], f)
    
    def log_analysis(self, timestamp: str, transcript: str, voice_probability: float, scam_probability: float, combined_risk_score: float, detected_phrases: list, intent: str):
        if not self.enable_logs:
            return
            
        log_entry = {
            "timestamp": timestamp,
            "transcript": transcript,
            "voice_probability": voice_probability,
            "scam_probability": scam_probability,
            "combined_risk_score": combined_risk_score,
            "detected_phrases": detected_phrases,
            "intent": intent
        }
        
        try:
            with open(self.log_path, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
                    
            logs.append(log_entry)
            
            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to write log entry: {e}")

global_logger = CallLogger()
