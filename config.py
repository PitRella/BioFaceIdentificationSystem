"""Biometric identification system configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"
LOGS_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Database (PostgreSQL)
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = int(os.getenv("DATABASE_PORT", "5432"))
DATABASE_NAME = os.getenv("DATABASE_NAME", "face_recognition_db")
DATABASE_USER = os.getenv("DATABASE_USER", "user")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")

# SQLAlchemy URL (async)
DATABASE_URL = (
    f"postgresql+asyncpg://{DATABASE_USER}:{DATABASE_PASSWORD}"
    f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)

# Recognition thresholds
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.6"))
FACE_ANGLE_THRESHOLD = int(os.getenv("FACE_ANGLE_THRESHOLD", "30"))  # degrees
SHARPNESS_THRESHOLD = float(os.getenv("SHARPNESS_THRESHOLD", "100"))

# Camera parameters
CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))

# Processing parameters
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))  # Process every Nth frame
MAX_FACES_PER_FRAME = int(os.getenv("MAX_FACES_PER_FRAME", "10"))

# Performance settings
CACHE_DESCRIPTORS = os.getenv("CACHE_DESCRIPTORS", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))

# User registration
MIN_PHOTOS_FOR_REGISTRATION = 5
MAX_PHOTOS_FOR_REGISTRATION = 10
