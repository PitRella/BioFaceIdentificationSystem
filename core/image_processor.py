"""Image preprocessing module."""
import cv2
import numpy as np
from typing import Tuple, Optional
from utils.logger import setup_logger

logger = setup_logger()


class ImageProcessor:
    """Image preprocessing utilities."""
    
    @staticmethod
    def resize_frame(
        frame: np.ndarray,
        max_width: int = 1280,
        max_height: int = 720,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """Resize frame to specified dimensions.
        
        Args:
            frame: Input frame
            max_width: Maximum width
            max_height: Maximum height
            maintain_aspect: Whether to maintain aspect ratio
        
        Returns:
            Resized frame
        """
        if frame is None or frame.size == 0:
            return frame
        
        height, width = frame.shape[:2]
        
        if width <= max_width and height <= max_height:
            return frame
        
        if maintain_aspect:
            # Calculate scaling factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width = max_width
            new_height = max_height
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def extract_face_region(
        frame: np.ndarray,
        top: int,
        right: int,
        bottom: int,
        left: int,
        padding: int = 20
    ) -> Optional[np.ndarray]:
        """Extract face region from frame with padding.
        
        Args:
            frame: Input frame
            top: Top coordinate
            right: Right coordinate
            bottom: Bottom coordinate
            left: Left coordinate
            padding: Padding in pixels
        
        Returns:
            Extracted face region or None if failed
        """
        try:
            height, width = frame.shape[:2]
            
            # Add padding and clamp to frame boundaries
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(height, bottom + padding)
            right = min(width, right + padding)
            
            face_region = frame[top:bottom, left:right]
            
            if face_region.size == 0:
                logger.warning("Empty face region extracted")
                return None
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            return None
    
    @staticmethod
    def align_face(
        frame: np.ndarray,
        landmarks: np.ndarray,
        desired_size: Tuple[int, int] = (150, 150)
    ) -> Optional[np.ndarray]:
        """Align face based on landmarks (eyes alignment).
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks (68 points)
            desired_size: Output size (width, height)
        
        Returns:
            Aligned face image or None if failed
        """
        if landmarks is None or len(landmarks) < 68:
            logger.warning("Insufficient landmarks for alignment")
            return None
        
        try:
            # Get eye landmarks
            # Left eye: 36-41, Right eye: 42-47
            left_eye_center = np.mean(landmarks[36:42], axis=0)
            right_eye_center = np.mean(landmarks[42:48], axis=0)
            
            # Calculate angle
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate center point
            eye_center = ((left_eye_center + right_eye_center) / 2).astype(int)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(
                tuple(eye_center),
                angle,
                1.0
            )
            
            # Rotate image
            aligned = cv2.warpAffine(
                frame,
                rotation_matrix,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_CUBIC
            )
            
            # Crop face region (approximate)
            face_width = int(np.linalg.norm(right_eye_center - left_eye_center) * 2.5)
            face_height = int(face_width * 1.2)
            
            x = int(eye_center[0] - face_width / 2)
            y = int(eye_center[1] - face_height / 2)
            
            x = max(0, x)
            y = max(0, y)
            
            face_crop = aligned[y:y+face_height, x:x+face_width]
            
            if face_crop.size == 0:
                return None
            
            # Resize to desired size
            resized = cv2.resize(face_crop, desired_size, interpolation=cv2.INTER_CUBIC)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error aligning face: {e}")
            return None
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image quality (contrast and brightness).
        
        Args:
            image: Input image
        
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to 0-255 range.
        
        Args:
            image: Input image
        
        Returns:
            Normalized image
        """
        if image.dtype != np.uint8:
            # Normalize to 0-255
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)
        return image
