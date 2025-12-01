"""Logging configuration for TCS Stock Forecast application."""
import logging
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
logger = logging.getLogger("tcs_forecast")
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(LOGS_DIR / "tcs-stock.log")
fh.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers
logger.addHandler(fh)
logger.addHandler(ch)

# Prevent duplicate logging
logger.propagate = False

if __name__ == "__main__":
    logger.info("Logging configured successfully")
