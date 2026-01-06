"""Face recognition module for comparing descriptors."""
import numpy as np
from typing import List, Tuple, Optional
from config import FACE_RECOGNITION_THRESHOLD
from utils.logger import setup_logger

logger = setup_logger()


class FaceRecognizer:
    """Face recognizer for comparing face descriptors."""
    
    def __init__(self, threshold: float = FACE_RECOGNITION_THRESHOLD):
        """Initialize face recognizer.
        
        Args:
            threshold: Distance threshold for face matching (default: 0.6)
                      Lower values = stricter matching
        """
        self.threshold = threshold
        logger.info(f"FaceRecognizer initialized with threshold={threshold}")
    
    def calculate_distance(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> float:
        """Calculate Euclidean distance between two descriptors.
        
        Args:
            desc1: First face descriptor (128-dimensional)
            desc2: Second face descriptor (128-dimensional)
        
        Returns:
            Euclidean distance (lower = more similar)
        """
        try:
            if desc1.shape != desc2.shape:
                logger.error(f"Descriptor shape mismatch: {desc1.shape} vs {desc2.shape}")
                return float('inf')
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(desc1 - desc2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def verify(
        self,
        descriptor1: np.ndarray,
        descriptor2: np.ndarray
    ) -> Tuple[bool, float]:
        """Verify if two descriptors belong to the same person (1:1 comparison).
        
        Args:
            descriptor1: First face descriptor
            descriptor2: Second face descriptor
        
        Returns:
            Tuple of (is_match, distance)
        """
        distance = self.calculate_distance(descriptor1, descriptor2)
        is_match = distance <= self.threshold
        
        logger.debug(f"Verification: distance={distance:.4f}, threshold={self.threshold}, match={is_match}")
        return is_match, distance
    
    def identify(
        self,
        descriptor: np.ndarray,
        database: List[Tuple[int, np.ndarray]]
    ) -> Optional[Tuple[int, float]]:
        """Identify person from descriptor by comparing with database (1:N comparison).
        
        Args:
            descriptor: Face descriptor to identify
            database: List of tuples (user_id, descriptor) from database
        
        Returns:
            Tuple of (user_id, distance) if match found, None otherwise
        """
        if not database:
            logger.warning("Empty database provided for identification")
            return None
        
        try:
            best_match = None
            best_distance = float('inf')
            
            for user_id, db_descriptor in database:
                distance = self.calculate_distance(descriptor, db_descriptor)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = user_id
            
            # Check if best match is within threshold
            if best_distance <= self.threshold:
                logger.debug(f"Identification: user_id={best_match}, distance={best_distance:.4f}")
                return (best_match, best_distance)
            else:
                logger.debug(f"Identification: no match found (best distance={best_distance:.4f} > threshold={self.threshold})")
                return None
                
        except Exception as e:
            logger.error(f"Error during identification: {e}")
            return None
    
    def identify_multiple(
        self,
        descriptor: np.ndarray,
        database: List[Tuple[int, np.ndarray]],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Identify top K matches from database.
        
        Args:
            descriptor: Face descriptor to identify
            database: List of tuples (user_id, descriptor) from database
            top_k: Number of top matches to return
        
        Returns:
            List of tuples (user_id, distance) sorted by distance (best first)
        """
        if not database:
            return []
        
        try:
            matches = []
            
            for user_id, db_descriptor in database:
                distance = self.calculate_distance(descriptor, db_descriptor)
                matches.append((user_id, distance))
            
            # Sort by distance and return top K
            matches.sort(key=lambda x: x[1])
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error during multiple identification: {e}")
            return []
