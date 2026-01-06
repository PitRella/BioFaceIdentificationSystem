"""Face detection module using dlib."""
import dlib
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger()


class FaceLocation:
    """Face location data structure."""
    def __init__(self, top: int, right: int, bottom: int, left: int):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple (top, right, bottom, left)."""
        return (self.top, self.right, self.bottom, self.left)
    
    def width(self) -> int:
        """Get face width."""
        return self.right - self.left
    
    def height(self) -> int:
        """Get face height."""
        return self.bottom - self.top
    
    def center(self) -> Tuple[int, int]:
        """Get face center coordinates."""
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)


class FaceDetector:
    """Face detector using dlib HOG detector."""
    
    def __init__(self):
        """Initialize face detector."""
        try:
            # Initialize HOG face detector
            self.detector = dlib.get_frontal_face_detector()
            
            # Initialize facial landmarks predictor
            # Try to find shape predictor file
            predictor_path = self._find_predictor_file()
            if predictor_path:
                self.predictor = dlib.shape_predictor(str(predictor_path))
                logger.info(f"Facial landmarks predictor loaded from {predictor_path}")
            else:
                self.predictor = None
                logger.warning("Facial landmarks predictor not found. Some features may be unavailable.")
            
            logger.info("FaceDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FaceDetector: {e}")
            raise
    
    def _find_predictor_file(self) -> Optional[Path]:
        """Find shape predictor file (68 facial landmarks).
        
        Returns:
            Path to predictor file or None if not found
        """
        # Common locations for shape predictor
        possible_paths = [
            Path(__file__).parent.parent / "models" / "shape_predictor_68_face_landmarks.dat",
            Path(__file__).parent.parent / "data" / "models" / "shape_predictor_68_face_landmarks.dat",
            Path.home() / ".dlib" / "shape_predictor_68_face_landmarks.dat",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        logger.warning("shape_predictor_68_face_landmarks.dat not found. Please download it from:")
        logger.warning("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return None
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceLocation]:
        """Detect faces in frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of FaceLocation objects
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Convert BGR to RGB for dlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector(rgb_frame, 1)  # 1 = upsampling factor
            
            face_locations = []
            for detection in detections:
                # dlib rectangle: (left, top, right, bottom)
                top = detection.top()
                right = detection.right()
                bottom = detection.bottom()
                left = detection.left()
                
                face_location = FaceLocation(top, right, bottom, left)
                face_locations.append(face_location)
            
            return face_locations
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def get_landmarks(self, frame: np.ndarray, face_location: FaceLocation) -> Optional[np.ndarray]:
        """Get 68 facial landmarks for a face.
        
        Args:
            frame: Input frame (BGR format)
            face_location: Face location in frame
        
        Returns:
            Numpy array of shape (68, 2) with landmark coordinates or None if failed
        """
        if self.predictor is None:
            logger.warning("Landmarks predictor not available")
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create dlib rectangle
            rect = dlib.rectangle(
                face_location.left,
                face_location.top,
                face_location.right,
                face_location.bottom
            )
            
            # Get landmarks
            shape = self.predictor(rgb_frame, rect)
            
            # Convert to numpy array
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error getting landmarks: {e}")
            return None
    
    def get_face_angle(self, landmarks: np.ndarray) -> float:
        """Calculate face rotation angle based on landmarks.
        
        Args:
            landmarks: Array of shape (68, 2) with facial landmarks
        
        Returns:
            Rotation angle in degrees (positive = right, negative = left)
        """
        if landmarks is None or len(landmarks) < 68:
            return 0.0
        
        try:
            # Use nose tip and eye corners to estimate angle
            # Landmark indices: 30 = nose tip, 36 = left eye corner, 45 = right eye corner
            nose_tip = landmarks[30]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            
            # Calculate angle from horizontal
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logger.error(f"Error calculating face angle: {e}")
            return 0.0
    
    def is_valid_angle(self, angle: float, threshold: float = 30.0) -> bool:
        """Check if face angle is within acceptable range.
        
        Args:
            angle: Face rotation angle in degrees
            threshold: Maximum allowed angle in degrees (default: 30)
        
        Returns:
            True if angle is within threshold, False otherwise
        """
        return abs(angle) <= threshold
