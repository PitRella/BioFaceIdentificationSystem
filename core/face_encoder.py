"""Face encoding module for creating biometric descriptors."""
import os
import face_recognition
import cv2
import numpy as np
from typing import List, Optional
from utils.logger import setup_logger

logger = setup_logger()

# Try to set face_recognition models path if available
try:
    import face_recognition_models
    import pathlib
    models_path = pathlib.Path(face_recognition_models.__file__).parent / "models"
    if models_path.exists():
        os.environ.setdefault("FACE_RECOGNITION_MODELS_PATH", str(models_path))
        logger.debug(f"Set FACE_RECOGNITION_MODELS_PATH to {models_path}")
except ImportError:
    logger.warning("face_recognition_models not found. face_recognition may not work properly.")


class FaceEncoder:
    """Face encoder for creating 128-dimensional face descriptors."""
    
    def __init__(self):
        """Initialize face encoder."""
        logger.info("FaceEncoder initialized (using face_recognition library)")
    
    def encode_face(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Encode face to 128-dimensional descriptor.
        
        Args:
            face_image: Face image (BGR or RGB format)
            landmarks: Optional facial landmarks (not used by face_recognition)
        
        Returns:
            128-dimensional descriptor array or None if failed
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided")
            return None
        
        try:
            # face_recognition expects RGB format
            if len(face_image.shape) == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Get face encodings (128-dimensional)
            encodings = face_recognition.face_encodings(rgb_image)
            
            if len(encodings) == 0:
                logger.warning("No face encoding found in image")
                return None
            
            # Return first encoding (normalized)
            descriptor = encodings[0]
            return descriptor
            
        except Exception as e:
            logger.error(f"Error encoding face: {e}")
            return None
    
    def encode_faces_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Encode multiple faces in batch.
        
        Args:
            face_images: List of face images
        
        Returns:
            List of descriptors (may be shorter than input if some failed)
        """
        descriptors = []
        for face_image in face_images:
            descriptor = self.encode_face(face_image)
            if descriptor is not None:
                descriptors.append(descriptor)
        return descriptors
    
    def encode_from_location(
        self,
        frame: np.ndarray,
        face_location
    ) -> Optional[np.ndarray]:
        """Encode face from frame using face location.
        
        Args:
            frame: Full frame (BGR format)
            face_location: FaceLocation object
        
        Returns:
            128-dimensional descriptor or None if failed
        """
        try:
            # Extract face region
            top, right, bottom, left = face_location.to_tuple()
            
            # Add padding
            padding = 20
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(frame.shape[0], bottom + padding)
            right = min(frame.shape[1], right + padding)
            
            face_image = frame[top:bottom, left:right]
            
            if face_image.size == 0:
                logger.warning("Empty face region extracted")
                return None
            
            return self.encode_face(face_image)
            
        except Exception as e:
            logger.error(f"Error encoding face from location: {e}")
            return None
