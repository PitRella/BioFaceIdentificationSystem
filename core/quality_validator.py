"""Image quality validation module."""
import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
from config import FACE_ANGLE_THRESHOLD, SHARPNESS_THRESHOLD
from utils.logger import setup_logger

if TYPE_CHECKING:
    from core.face_detector import FaceLocation

logger = setup_logger()


@dataclass
class QualityReport:
    """Quality validation report."""
    is_valid: bool
    sharpness_score: float
    lighting_score: float
    angle: float
    angle_valid: bool
    face_size_valid: bool
    issues: List[str]
    
    def __init__(self):
        self.is_valid = False
        self.sharpness_score = 0.0
        self.lighting_score = 0.0
        self.angle = 0.0
        self.angle_valid = False
        self.face_size_valid = False
        self.issues = []


class QualityValidator:
    """Validator for face image quality."""
    
    def __init__(
        self,
        sharpness_threshold: float = SHARPNESS_THRESHOLD,
        angle_threshold: float = FACE_ANGLE_THRESHOLD,
        min_face_size: int = 100
    ):
        """Initialize quality validator.
        
        Args:
            sharpness_threshold: Minimum sharpness score (Laplacian variance)
            angle_threshold: Maximum allowed face angle in degrees
            min_face_size: Minimum face size in pixels
        """
        self.sharpness_threshold = sharpness_threshold
        self.angle_threshold = angle_threshold
        self.min_face_size = min_face_size
        logger.info(f"QualityValidator initialized: sharpness_threshold={sharpness_threshold}, "
                   f"angle_threshold={angle_threshold}, min_face_size={min_face_size}")
    
    def validate_sharpness(self, image: np.ndarray) -> tuple[bool, float]:
        """Validate image sharpness using Laplacian variance.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Tuple of (is_valid, sharpness_score)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            is_valid = variance >= self.sharpness_threshold
            return is_valid, variance
            
        except Exception as e:
            logger.error(f"Error validating sharpness: {e}")
            return False, 0.0
    
    def validate_lighting(self, image: np.ndarray) -> tuple[bool, float]:
        """Validate image lighting (brightness and contrast).
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Tuple of (is_valid, lighting_score)
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Calculate mean brightness
            mean_brightness = np.mean(l_channel)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(l_channel)
            
            # Combined score (normalized)
            # Good lighting: brightness 50-200, contrast > 20
            brightness_score = 1.0 - abs(mean_brightness - 125) / 125  # Optimal around 125
            contrast_score = min(contrast / 50.0, 1.0)  # Normalize to 0-1
            
            lighting_score = (brightness_score + contrast_score) / 2.0
            
            # Valid if brightness is reasonable (30-220) and contrast is good (>15)
            is_valid = 30 <= mean_brightness <= 220 and contrast >= 15
            
            return is_valid, lighting_score
            
        except Exception as e:
            logger.error(f"Error validating lighting: {e}")
            return False, 0.0
    
    def validate_angle(self, angle: float, threshold: Optional[float] = None) -> bool:
        """Validate face rotation angle.
        
        Args:
            angle: Face rotation angle in degrees
            threshold: Maximum allowed angle (uses default if None)
        
        Returns:
            True if angle is within threshold, False otherwise
        """
        if threshold is None:
            threshold = self.angle_threshold
        return abs(angle) <= threshold
    
    def validate_face_size(
        self,
        face_location: 'FaceLocation',
        min_size: Optional[int] = None
    ) -> bool:
        """Validate face size.
        
        Args:
            face_location: FaceLocation object
            min_size: Minimum face size (uses default if None)
        
        Returns:
            True if face size is valid, False otherwise
        """
        if min_size is None:
            min_size = self.min_face_size
        
        width = face_location.width()
        height = face_location.height()
        min_dimension = min(width, height)
        
        return min_dimension >= min_size
    
    def validate_all(
        self,
        image: np.ndarray,
        face_location: 'FaceLocation',
        landmarks: Optional[np.ndarray] = None,
        angle: Optional[float] = None
    ) -> QualityReport:
        """Perform all quality validations.
        
        Args:
            image: Face image (BGR format)
            face_location: FaceLocation object
            landmarks: Optional facial landmarks for angle calculation
            angle: Optional pre-calculated face angle
        
        Returns:
            QualityReport with validation results
        """
        report = QualityReport()
        
        # Validate sharpness
        sharpness_valid, sharpness_score = self.validate_sharpness(image)
        report.sharpness_score = sharpness_score
        if not sharpness_valid:
            report.issues.append(f"Low sharpness (score: {sharpness_score:.2f})")
        
        # Validate lighting
        lighting_valid, lighting_score = self.validate_lighting(image)
        report.lighting_score = lighting_score
        if not lighting_valid:
            report.issues.append(f"Poor lighting (score: {lighting_score:.2f})")
        
        # Validate angle
        if angle is not None:
            report.angle = angle
            report.angle_valid = self.validate_angle(angle)
            if not report.angle_valid:
                report.issues.append(f"Face angle too large ({angle:.1f}°)")
        elif landmarks is not None:
            # Calculate angle from landmarks if not provided
            from core.face_detector import FaceDetector
            detector = FaceDetector()
            report.angle = detector.get_face_angle(landmarks)
            report.angle_valid = self.validate_angle(report.angle)
            if not report.angle_valid:
                report.issues.append(f"Face angle too large ({report.angle:.1f}°)")
        else:
            report.angle_valid = True  # Assume valid if can't check
            logger.warning("No angle information provided, assuming valid")
        
        # Validate face size
        report.face_size_valid = self.validate_face_size(face_location)
        if not report.face_size_valid:
            report.issues.append(f"Face too small (min: {self.min_face_size}px)")
        
        # Overall validation
        report.is_valid = (
            sharpness_valid and
            lighting_valid and
            report.angle_valid and
            report.face_size_valid
        )
        
        return report
