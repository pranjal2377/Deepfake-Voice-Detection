import os
import torch
import logging
from src.utils.config import DEEPFAKE_MODEL_PATH, MODELS_DIR

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages saving and loading of models."""
    
    @staticmethod
    def ensure_dir():
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    @staticmethod
    def save_deepfake_model(model, optimizer=None, epoch=0):
        ModelManager.ensure_dir()
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        }
        torch.save(state, DEEPFAKE_MODEL_PATH)
        logger.info(f"Deepfake model saved to {DEEPFAKE_MODEL_PATH}")

    @staticmethod
    def load_deepfake_model(model):
        if not os.path.exists(DEEPFAKE_MODEL_PATH):
            logger.warning("Deepfake model not found.")
            return False
        checkpoint = torch.load(DEEPFAKE_MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Deepfake model loaded from {DEEPFAKE_MODEL_PATH}")
        return True

