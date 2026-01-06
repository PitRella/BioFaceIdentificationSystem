"""Logging configuration for the system."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from config import LOGS_DIR


def setup_logger(name: str = "BioFaceIdentification", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers if any
    logger.handlers.clear()
    
    # Log format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = LOGS_DIR / f"{name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
