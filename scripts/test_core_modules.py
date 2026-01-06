"""Test script for core modules."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from core.video_capture import VideoCapture
from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.quality_validator import QualityValidator
from core.face_recognizer import FaceRecognizer
from utils.logger import setup_logger

logger = setup_logger()


def test_video_capture():
    """Test video capture module."""
    logger.info("Testing VideoCapture...")
    try:
        with VideoCapture() as cap:
            if not cap.is_opened():
                logger.error("Failed to open camera")
                return False
            
            # Try to read a frame
            frame = cap.read()
            if frame is not None:
                logger.info(f"Successfully captured frame: {frame.shape}")
                return True
            else:
                logger.error("Failed to read frame")
                return False
    except Exception as e:
        logger.error(f"Error testing VideoCapture: {e}")
        return False


def test_face_detector():
    """Test face detector module."""
    logger.info("Testing FaceDetector...")
    try:
        detector = FaceDetector()
        
        # Try to capture a frame
        with VideoCapture() as cap:
            if not cap.is_opened():
                logger.warning("Camera not available, skipping face detection test")
                return True  # Not a failure, just no camera
            
            frame = cap.read()
            if frame is None:
                logger.warning("Could not read frame, skipping face detection test")
                return True
            
            # Detect faces
            faces = detector.detect_faces(frame)
            logger.info(f"Detected {len(faces)} faces")
            
            if len(faces) > 0:
                # Try to get landmarks
                landmarks = detector.get_landmarks(frame, faces[0])
                if landmarks is not None:
                    logger.info(f"Successfully got landmarks: {landmarks.shape}")
                    angle = detector.get_face_angle(landmarks)
                    logger.info(f"Face angle: {angle:.2f}°")
            
            return True
            
    except Exception as e:
        logger.error(f"Error testing FaceDetector: {e}")
        return False


def test_face_encoder():
    """Test face encoder module."""
    logger.info("Testing FaceEncoder...")
    try:
        encoder = FaceEncoder()
        
        # Create a dummy face image
        dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Try to encode
        descriptor = encoder.encode_face(dummy_face)
        if descriptor is not None:
            logger.info(f"Successfully encoded face: descriptor shape={descriptor.shape}")
            return True
        else:
            logger.warning("Could not encode dummy face (expected for random image)")
            return True  # Not a failure, just no real face
            
    except Exception as e:
        logger.error(f"Error testing FaceEncoder: {e}")
        return False


def test_quality_validator():
    """Test quality validator module."""
    logger.info("Testing QualityValidator...")
    try:
        validator = QualityValidator()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Test sharpness
        is_sharp, score = validator.validate_sharpness(test_image)
        logger.info(f"Sharpness validation: valid={is_sharp}, score={score:.2f}")
        
        # Test lighting
        is_light_ok, light_score = validator.validate_lighting(test_image)
        logger.info(f"Lighting validation: valid={is_light_ok}, score={light_score:.2f}")
        
        # Test angle
        is_angle_ok = validator.validate_angle(15.0)
        logger.info(f"Angle validation (15°): valid={is_angle_ok}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing QualityValidator: {e}")
        return False


def test_face_recognizer():
    """Test face recognizer module."""
    logger.info("Testing FaceRecognizer...")
    try:
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create test descriptors
        desc1 = np.random.rand(128).astype(np.float32)
        desc2 = desc1 + 0.1  # Similar descriptor
        desc3 = np.random.rand(128).astype(np.float32)  # Different descriptor
        
        # Test distance calculation
        distance = recognizer.calculate_distance(desc1, desc2)
        logger.info(f"Distance between similar descriptors: {distance:.4f}")
        
        distance2 = recognizer.calculate_distance(desc1, desc3)
        logger.info(f"Distance between different descriptors: {distance2:.4f}")
        
        # Test verification
        is_match, dist = recognizer.verify(desc1, desc2)
        logger.info(f"Verification result: match={is_match}, distance={dist:.4f}")
        
        # Test identification
        database = [
            (1, desc1),
            (2, desc3),
        ]
        result = recognizer.identify(desc2, database)
        if result:
            user_id, distance = result
            logger.info(f"Identification result: user_id={user_id}, distance={distance:.4f}")
        else:
            logger.info("Identification: no match found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing FaceRecognizer: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("Testing core modules")
    logger.info("=" * 50)
    
    results = {
        "VideoCapture": test_video_capture(),
        "FaceDetector": test_face_detector(),
        "FaceEncoder": test_face_encoder(),
        "QualityValidator": test_quality_validator(),
        "FaceRecognizer": test_face_recognizer(),
    }
    
    logger.info("=" * 50)
    logger.info("Test Results:")
    for module, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {module}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 50)
    
    if all_passed:
        logger.info("All tests passed!")
    else:
        logger.warning("Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
