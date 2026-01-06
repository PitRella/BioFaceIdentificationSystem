"""Face identification module."""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from core.video_capture import VideoCapture
from core.face_detector import FaceDetector, FaceLocation
from core.face_encoder import FaceEncoder
from core.face_recognizer import FaceRecognizer
from core.image_processor import ImageProcessor
from database.connection import AsyncSessionLocal
from database.repositories import BiometricTemplateRepository, AccessLogRepository, UserRepository
from config import FRAME_SKIP, MAX_FACES_PER_FRAME
from utils.logger import setup_logger

logger = setup_logger()


class IdentificationResult:
    """Identification result data structure."""
    def __init__(self):
        self.user_id: Optional[int] = None
        self.user_name: Optional[str] = None
        self.user_surname: Optional[str] = None
        self.confidence: float = 0.0
        self.distance: float = 0.0
        self.success: bool = False
        self.timestamp: datetime = datetime.utcnow()


class FaceIdentification:
    """Face identification handler."""
    
    def __init__(self):
        """Initialize identification handler."""
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.recognizer = FaceRecognizer()
        self.processor = ImageProcessor()
        self._descriptors_cache: Optional[List[Tuple[int, np.ndarray]]] = None
        self._cache_timestamp: Optional[datetime] = None
        logger.info("FaceIdentification initialized")
    
    async def load_descriptors_cache(self, force_reload: bool = False) -> int:
        """Load all descriptors from database into cache.
        
        Args:
            force_reload: Force reload even if cache exists
        
        Returns:
            Number of descriptors loaded
        """
        # Check if cache is still valid (reload every 5 minutes)
        if not force_reload and self._descriptors_cache is not None:
            if self._cache_timestamp:
                cache_age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
                if cache_age < 300:  # 5 minutes
                    logger.debug("Using existing descriptors cache")
                    return len(self._descriptors_cache)
        
        try:
            async with AsyncSessionLocal() as session:
                templates = await BiometricTemplateRepository.get_all(session)
                
                descriptors = []
                for template in templates:
                    try:
                        descriptor = np.array(template.descriptor, dtype=np.float32)
                        descriptors.append((template.user_id, descriptor))
                    except Exception as e:
                        logger.warning(f"Failed to load descriptor for template {template.id}: {e}")
                
                self._descriptors_cache = descriptors
                self._cache_timestamp = datetime.utcnow()
                
                logger.info(f"Loaded {len(descriptors)} descriptors into cache")
                return len(descriptors)
                
        except Exception as e:
            logger.error(f"Error loading descriptors cache: {e}")
            return 0
    
    async def identify_frame(
        self,
        frame: np.ndarray,
        access_type: str = "identification"
    ) -> List[IdentificationResult]:
        """Identify faces in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            access_type: Type of access ('verification' or 'identification')
        
        Returns:
            List of IdentificationResult objects (one per detected face)
        """
        results = []
        
        try:
            # Ensure cache is loaded
            if self._descriptors_cache is None or len(self._descriptors_cache) == 0:
                await self.load_descriptors_cache()
            
            if len(self._descriptors_cache) == 0:
                logger.warning("No descriptors in database for identification")
                return results
            
            # Detect faces
            faces = self.detector.detect_faces(frame)
            
            if len(faces) == 0:
                return results
            
            # Limit number of faces to process
            faces = faces[:MAX_FACES_PER_FRAME]
            
            # Process each face
            for face_location in faces:
                result = await self._identify_face(frame, face_location, access_type)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error identifying frame: {e}", exc_info=True)
            return results
    
    async def _identify_face(
        self,
        frame: np.ndarray,
        face_location: FaceLocation,
        access_type: str
    ) -> Optional[IdentificationResult]:
        """Identify a single face.
        
        Args:
            frame: Input frame
            face_location: Face location in frame
            access_type: Type of access
        
        Returns:
            IdentificationResult or None if failed
        """
        result = IdentificationResult()
        
        try:
            # Extract face region
            top, right, bottom, left = face_location.to_tuple()
            face_image = self.processor.extract_face_region(
                frame, top, right, bottom, left
            )
            
            if face_image is None:
                return None
            
            # Encode face
            descriptor = self.encoder.encode_face(face_image)
            
            if descriptor is None:
                logger.debug("Failed to encode face")
                return None
            
            # Identify against database
            match = self.recognizer.identify(descriptor, self._descriptors_cache)
            
            if match:
                user_id, distance = match
                
                # Get user information
                async with AsyncSessionLocal() as session:
                    user = await UserRepository.get_by_id(session, user_id)
                    
                    if user:
                        result.user_id = user_id
                        result.user_name = user.name
                        result.user_surname = user.surname
                        result.distance = distance
                        result.confidence = 1.0 - (distance / self.recognizer.threshold)  # Normalize to 0-1
                        result.confidence = max(0.0, min(1.0, result.confidence))  # Clamp
                        result.success = True
                        
                        # Log access
                        await AccessLogRepository.create(
                            session,
                            user_id=user_id,
                            access_type=access_type,
                            result="success",
                            confidence=result.confidence
                        )
                        await session.commit()
                        
                        logger.info(f"Identified: {user.name} {user.surname} (ID: {user_id}, "
                                  f"distance: {distance:.4f}, confidence: {result.confidence:.2%})")
                    else:
                        logger.warning(f"User {user_id} not found in database")
                        # Log failed access
                        async with AsyncSessionLocal() as session:
                            await AccessLogRepository.create(
                                session,
                                user_id=None,
                                access_type=access_type,
                                result="failure",
                                confidence=None
                            )
                            await session.commit()
            else:
                # No match found, log failure
                async with AsyncSessionLocal() as session:
                    await AccessLogRepository.create(
                        session,
                        user_id=None,
                        access_type=access_type,
                        result="failure",
                        confidence=None
                    )
                    await session.commit()
                
                logger.debug("No match found for face")
            
            return result if result.success else None
            
        except Exception as e:
            logger.error(f"Error identifying face: {e}", exc_info=True)
            return None
    
    async def verify_user(
        self,
        frame: np.ndarray,
        user_id: int
    ) -> Optional[IdentificationResult]:
        """Verify if face belongs to specific user (1:1 verification).
        
        Args:
            frame: Input frame
            user_id: User ID to verify against
        
        Returns:
            IdentificationResult or None if failed
        """
        try:
            # Detect face
            faces = self.detector.detect_faces(frame)
            
            if len(faces) == 0:
                logger.debug("No face detected for verification")
                return None
            
            face_location = faces[0]
            
            # Extract and encode face
            top, right, bottom, left = face_location.to_tuple()
            face_image = self.processor.extract_face_region(
                frame, top, right, bottom, left
            )
            
            if face_image is None:
                return None
            
            descriptor = self.encoder.encode_face(face_image)
            
            if descriptor is None:
                return None
            
            # Get user's templates from database
            async with AsyncSessionLocal() as session:
                templates = await BiometricTemplateRepository.get_by_user_id(session, user_id)
                
                if not templates:
                    logger.warning(f"No templates found for user {user_id}")
                    return None
                
                # Compare with all user's templates
                best_match = None
                best_distance = float('inf')
                best_template_descriptor = None
                
                for template in templates:
                    template_descriptor = np.array(template.descriptor, dtype=np.float32)
                    distance = self.recognizer.calculate_distance(descriptor, template_descriptor)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = template
                        best_template_descriptor = template_descriptor
                
                if best_template_descriptor is None:
                    return None
                
                # Check if match
                is_match, distance = self.recognizer.verify(descriptor, best_template_descriptor)
                
                result = IdentificationResult()
                
                if is_match:
                    user = await UserRepository.get_by_id(session, user_id)
                    if user:
                        result.user_id = user_id
                        result.user_name = user.name
                        result.user_surname = user.surname
                        result.distance = distance
                        result.confidence = 1.0 - (distance / self.recognizer.threshold)
                        result.confidence = max(0.0, min(1.0, result.confidence))
                        result.success = True
                        
                        # Log access
                        await AccessLogRepository.create(
                            session,
                            user_id=user_id,
                            access_type="verification",
                            result="success",
                            confidence=result.confidence
                        )
                        await session.commit()
                        
                        logger.info(f"Verification successful: {user.name} {user.surname}")
                    else:
                        result.success = False
                else:
                    result.success = False
                    # Log failed verification
                    await AccessLogRepository.create(
                        session,
                        user_id=user_id,
                        access_type="verification",
                        result="failure",
                        confidence=None
                    )
                    await session.commit()
                    logger.info(f"Verification failed for user {user_id}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            return None
