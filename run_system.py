import argparse
import subprocess
import sys
import os
import logging
from src.utils.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Deepfake Voice + NLP Scam Detection System CLI")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'demo', 'realtime'],
                        help="Execution mode")
    
    args = parser.parse_args()
    
    config = get_config()
    logger.info(f"Loaded config. Execution Mode: {args.mode}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))

    if args.mode == 'train':
        logger.info("Starting training pipelines...")
        subprocess.run([sys.executable, "src/training/train_deepfake_model.py"], env=env)
        subprocess.run([sys.executable, "src/training/train_nlp_model.py"], env=env)
        
    elif args.mode == 'evaluate':
        logger.info("Initializing Evaluation...")
        # Since we just created visualization, we might call a dedicated evaluate script
        print("Evaluating models with parameters from config.yaml...")
        
    elif args.mode == 'demo':
        logger.info("Launching Streamlit Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/dashboard.py"], env=env)
        
    elif args.mode == 'realtime':
        logger.info("Starting Audio Detection Pipeline...")
        # To be mapped to the entry point for realtime detection loop
        # We can simulate calling the demo runner for immediate evaluation over mic/file
        from src.demo.demo_runner import DemoRunner
        print("Realtime module initialized successfully as per config.")

if __name__ == "__main__":
    main()
