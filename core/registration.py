"""User registration module."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from core.video_capture import VideoCapture
from core.face_detector import FaceDetector, FaceLocation
from core.face_encoder import FaceEncoder
from core.quality_validator import QualityValidator, QualityReport
from core.image_processor import ImageProcessor
from database.connection import AsyncSessionLocal
from database.repositories import UserRepository, BiometricTemplateRepository
from config import (
    MIN_PHOTOS_FOR_REGISTRATION,
    MAX_PHOTOS_FOR_REGISTRATION,
    PHOTOS_DIR,
    FACE_ANGLE_THRESHOLD
)
from utils.logger import setup_logger

logger = setup_logger()


class RegistrationResult:
    """Registration result data structure."""
    def __init__(self):
        self.success: bool = False
        self.user_id: Optional[int] = None
        self.photos_captured: int = 0
        self.templates_created: int = 0
        self.photo_path: Optional[str] = None
        self.issues: List[str] = []
        self.message: str = ""


class UserRegistration:
    """User registration handler."""
    
    def __init__(self):
        """Initialize registration handler."""
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.validator = QualityValidator()
        self.processor = ImageProcessor()
        logger.info("UserRegistration initialized")
    
    async def register_user(
        self,
        name: str,
        surname: str,
        access_level: int = 0,
        callback_progress: Optional[callable] = None
    ) -> RegistrationResult:
        """Register a new user by capturing photos and creating templates.
        
        Args:
            name: User's first name
            surname: User's last name
            access_level: Access level (default: 0)
            callback_progress: Optional callback function(photo_num, total, message)
        
        Returns:
            RegistrationResult with registration status
        """
        result = RegistrationResult()
        
        try:
            logger.info(f"Starting registration for {name} {surname}")
            
            # Capture photos
            photos_data = await self._capture_photos(callback_progress)
            
            if len(photos_data) < MIN_PHOTOS_FOR_REGISTRATION:
                result.message = f"Failed to capture enough photos. Got {len(photos_data)}, need {MIN_PHOTOS_FOR_REGISTRATION}"
                result.issues.append(result.message)
                logger.error(result.message)
                return result
            
            result.photos_captured = len(photos_data)
            logger.info(f"Captured {result.photos_captured} photos")
            
            # Create user in database
            async with AsyncSessionLocal() as session:
                try:
                    # Save reference photo
                    reference_photo_path = await self._save_reference_photo(
                        photos_data[0]['frame'],
                        name,
                        surname
                    )
                    
                    # Create user
                    user = await UserRepository.create(
                        session,
                        name=name,
                        surname=surname,
                        photo_path=str(reference_photo_path),
                        access_level=access_level
                    )
                    await session.flush()
                    result.user_id = user.id
                    result.photo_path = str(reference_photo_path)
                    logger.info(f"Created user with ID: {result.user_id}")
                    
                    # Create templates from all photos
                    templates_created = 0
                    for i, photo_data in enumerate(photos_data):
                        descriptor = photo_data.get('descriptor')
                        if descriptor is not None:
                            await BiometricTemplateRepository.create(
                                session,
                                user_id=user.id,
                                descriptor=descriptor.tolist()
                            )
                            templates_created += 1
                            logger.debug(f"Created template {i+1}/{len(photos_data)}")
                    
                    await session.commit()
                    result.templates_created = templates_created
                    result.success = True
                    result.message = f"User registered successfully with {templates_created} templates"
                    logger.info(result.message)
                    
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Error during database operations: {e}")
                    result.issues.append(f"Database error: {str(e)}")
                    raise
            
        except Exception as e:
            logger.error(f"Registration failed: {e}", exc_info=True)
            result.issues.append(f"Registration error: {str(e)}")
            result.message = f"Registration failed: {str(e)}"
        
        return result
    
    async def _capture_photos(
        self,
        callback_progress: Optional[callable] = None
    ) -> List[dict]:
        """Capture photos with quality validation.
        
        Args:
            callback_progress: Optional callback function(photo_num, total, message)
        
        Returns:
            List of dictionaries with 'frame', 'descriptor', 'quality_report'
        """
        photos_data = []
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        if callback_progress:
            callback_progress(0, MAX_PHOTOS_FOR_REGISTRATION, "Starting photo capture...")
        
        try:
            with VideoCapture() as cap:
                if not cap.is_opened():
                    raise RuntimeError("Failed to open camera")
                
                logger.info("Camera opened, starting photo capture")
                
                while len(photos_data) < MAX_PHOTOS_FOR_REGISTRATION:
                    frame = cap.read()
                    if frame is None:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Too many consecutive frame read failures")
                            break
                        continue
                    
                    consecutive_failures = 0
                    frame_count += 1
                    
                    # Process every 3rd frame for performance
                    if frame_count % 3 != 0:
                        continue
                    
                    # Detect faces
                    faces = self.detector.detect_faces(frame)
                    
                    if len(faces) == 0:
                        if callback_progress:
                            callback_progress(
                                len(photos_data),
                                MAX_PHOTOS_FOR_REGISTRATION,
                                "No face detected. Please position yourself in front of the camera."
                            )
                        continue
                    
                    if len(faces) > 1:
                        if callback_progress:
                            callback_progress(
                                len(photos_data),
                                MAX_PHOTOS_FOR_REGISTRATION,
                                "Multiple faces detected. Please ensure only one person is in frame."
                            )
                        continue
                    
                    face_location = faces[0]
                    
                    # Get landmarks
                    landmarks = self.detector.get_landmarks(frame, face_location)
                    
                    # Calculate angle
                    angle = 0.0
                    if landmarks is not None:
                        angle = self.detector.get_face_angle(landmarks)
                    
                    # Extract face region
                    top, right, bottom, left = face_location.to_tuple()
                    face_image = self.processor.extract_face_region(
                        frame, top, right, bottom, left
                    )
                    
                    if face_image is None:
                        continue
                    
                    # Validate quality
                    quality_report = self.validator.validate_all(
                        face_image,
                        face_location,
                        landmarks,
                        angle
                    )
                    
                    if not quality_report.is_valid:
                        issues_str = ", ".join(quality_report.issues)
                        if callback_progress:
                            callback_progress(
                                len(photos_data),
                                MAX_PHOTOS_FOR_REGISTRATION,
                                f"Quality issues: {issues_str}. Please adjust position."
                            )
                        logger.debug(f"Photo rejected: {issues_str}")
                        continue
                    
                    # Encode face
                    descriptor = self.encoder.encode_face(face_image)
                    
                    if descriptor is None:
                        if callback_progress:
                            callback_progress(
                                len(photos_data),
                                MAX_PHOTOS_FOR_REGISTRATION,
                                "Failed to encode face. Please try again."
                            )
                        continue
                    
                    # Photo is valid, save it
                    photos_data.append({
                        'frame': frame.copy(),
                        'face_image': face_image,
                        'descriptor': descriptor,
                        'quality_report': quality_report,
                        'angle': angle
                    })
                    
                    logger.info(f"Captured photo {len(photos_data)}/{MAX_PHOTOS_FOR_REGISTRATION} "
                              f"(angle: {angle:.1f}°, sharpness: {quality_report.sharpness_score:.2f})")
                    
                    if callback_progress:
                        callback_progress(
                            len(photos_data),
                            MAX_PHOTOS_FOR_REGISTRATION,
                            f"Photo {len(photos_data)}/{MAX_PHOTOS_FOR_REGISTRATION} captured. "
                            f"Please move slightly for next photo."
                        )
                    
                    # If we have minimum required, check if we should continue
                    if len(photos_data) >= MIN_PHOTOS_FOR_REGISTRATION:
                        # Continue capturing up to MAX for better quality
                        pass
                
                logger.info(f"Photo capture completed: {len(photos_data)} photos")
                return photos_data
                
        except Exception as e:
            logger.error(f"Error during photo capture: {e}", exc_info=True)
            raise
    
    async def _save_reference_photo(
        self,
        frame: np.ndarray,
        name: str,
        surname: str
    ) -> Path:
        """Save reference photo to disk.
        
        Args:
            frame: Frame to save
            name: User's first name
            surname: User's last name
        
        Returns:
            Path to saved photo
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{surname}_{timestamp}.jpg"
        photo_path = PHOTOS_DIR / filename
        
        # Save image
        cv2.imwrite(str(photo_path), frame)
        logger.info(f"Saved reference photo: {photo_path}")
        
        return photo_path
