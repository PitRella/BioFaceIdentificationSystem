"""Video capture module for camera."""
import cv2
import numpy as np
from typing import Optional
from config import CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from utils.logger import setup_logger

logger = setup_logger()


class VideoCapture:
    """Video capture class for camera operations."""
    
    def __init__(
        self,
        camera_id: int = CAMERA_ID,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS
    ):
        """Initialize video capture.
        
        Args:
            camera_id: Camera device ID (default: 0)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        logger.info(f"VideoCapture initialized: camera_id={camera_id}, resolution={width}x{height}, fps={fps}")
    
    def start(self) -> bool:
        """Start video capture.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera started: resolution={actual_width}x{actual_height}, fps={actual_fps}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Read frame from camera.
        
        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        if not self.is_opened():
            logger.warning("Camera not opened, cannot read frame")
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
            return frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
    def stop(self) -> None:
        """Stop video capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera stopped and released")
    
    def is_opened(self) -> bool:
        """Check if camera is opened.
        
        Returns:
            True if camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
