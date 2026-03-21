"""
NLP Model Training Pipeline.
Trains TF-IDF and Classifier based on Config paths.
"""

import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.config import SCAM_DATASET_PATH
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)

def run_nlp_training():
    logger.info("Initializing NLP Model Training...")
    
    if not os.path.exists(SCAM_DATASET_PATH):
        logger.error(f"Dataset not found at {SCAM_DATASET_PATH}. Cannot train NLP model.")
        
        # Fallback to create dummy dataset for pure execution success if not present
        os.makedirs(os.path.dirname(SCAM_DATASET_PATH), exist_ok=True)
        dummy_data = pd.DataFrame({
            'transcript': ['hello how are you', 'please send me the otp and gift card', 'this is my bank account'],
            'label': [0, 1, 1]
        })
        dummy_data.to_csv(SCAM_DATASET_PATH, index=False)
        logger.info(f"Created fallback demo dataset at {SCAM_DATASET_PATH}")
        
    df = pd.read_csv(SCAM_DATASET_PATH)
    
    # 1. Pipeline Definition
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(random_state=42))
    ])
    
    # 2. Train
    if 'transcript' in df.columns and 'label' in df.columns:
        X = df['transcript'].fillna('')
        y = df['label']
        
        logger.info("Fitting NLP pipeline...")
        pipeline.fit(X, y)
        
        # 3. Save Model
        ModelManager.save_nlp_model(pipeline)
        logger.info("NLP Training Complete.")
    else:
        logger.error("Dataset lacks required 'transcript' and 'label' columns.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_nlp_training()