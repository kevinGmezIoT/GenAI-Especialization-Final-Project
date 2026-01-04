import json
import logging
from datetime import datetime
from pathlib import Path
import os

# Ensure logs directory exists relative to the project root
# We look for the 'logs' folder in the same directory as the script if executed directly, 
# but better to have it at project root.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = BASE_DIR / "logs"
LOG_PATH.mkdir(exist_ok=True)

# Define log file path
LOG_FILE = LOG_PATH / "inference.log"

# Configure logging to write to the file
# Using simple append mode
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)

# Create file handler if it doesn't have one
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

def log_inference(input_data, prediction, model_version="1.0.0"):
    """
    Logs inference data to a JSON file for monitoring.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "input_summary": input_data,
        "prediction": prediction
    }
    logger.info(json.dumps(record))
