"""
BERT-based Text Classification for Scam Detection.
Analyzes voice transcripts and predicts intent using DistilBERT.
"""

import os
import torch
import logging
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from src.utils.config import NLP_BERT_MODEL_NAME, NLP_CONFIDENCE_THRESHOLD, NLP_DEVICE

logger = logging.getLogger(__name__)

# Map indices to human-readable labels
LABEL_MAP = {
    0: "Normal",
    1: "Suspicious",
    2: "Scam"
}

class BertClassifier:
    """
    HuggingFace BERT-based classifier for detecting scam intents in transcripts.
    Implements lazy loading to save memory.
    """
    
    def __init__(self, model_name: str = NLP_BERT_MODEL_NAME, device: str = NLP_DEVICE, threshold: float = NLP_CONFIDENCE_THRESHOLD):
        self.model_name = model_name
        self._device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.threshold = threshold
        self.device = torch.device(self._device)
        
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def _load_model(self):
        """Lazy load the model when first needed to save memory."""
        if not self._is_loaded:
            logger.info(f"Loading BERT model ({self.model_name}) on {self.device}...")
            # Using num_labels=3 for Normal, Suspicious, Scam
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=3
            )
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True

    def analyze_transcript(self, text: str) -> Dict:
        """
        Analyze a transcript segment using BERT.
        
        Returns:
            Dict containing: nlp_probability, detected_phrases, intent, detailed_probs
        """
        if not text or not text.strip():
             return {
                "transcript": text,
                "nlp_probability": 0.0,
                "detected_phrases": [],
                "intent": "Normal",
                "detailed_probs": {"normal": 1.0, "suspicious": 0.0, "scam": 0.0}
            }
            
        self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        if probs.ndim == 0:
            probs = probs.reshape(-1)
            
        prob_normal = float(probs[0])
        prob_suspicious = float(probs[1]) if len(probs) > 1 else 0.0
        prob_scam = float(probs[2]) if len(probs) > 2 else 0.0
        
        # Calculate risk score, heavier weight on scam
        nlp_score = prob_scam + (0.5 * prob_suspicious)
        nlp_score = min(1.0, max(0.0, float(nlp_score)))
        
        # Intent classification
        pred_idx = int(probs.argmax())
        
        if pred_idx == 2:
            intent = "Scam"
        elif pred_idx == 1:
            intent = "Suspicious"
        else:
            intent = "Normal"
            
        return {
            "transcript": text,
            "nlp_probability": nlp_score,
            "detailed_probs": {
                "normal": prob_normal,
                "suspicious": prob_suspicious,
                "scam": prob_scam
            },
            "detected_phrases": [], # Kept for backward compatibility
            "intent": intent
        }
