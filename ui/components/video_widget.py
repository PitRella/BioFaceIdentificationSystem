"""Video widget for displaying camera feed with face detection overlay."""
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional, List, Tuple
from utils.logger import setup_logger

logger = setup_logger()


class VideoWidget(QLabel):
    """Widget for displaying video stream with face detection overlay."""
    
    def __init__(self, parent=None):
        """Initialize video widget."""
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setText("No video feed")
        self._current_frame: Optional[np.ndarray] = None
        self._face_results: List[dict] = []
        
    def update_frame(self, frame: np.ndarray) -> None:
        """Update displayed frame.
        
        Args:
            frame: Frame as numpy array (BGR format)
        """
        if frame is None or frame.size == 0:
            return
        
        self._current_frame = frame.copy()
        self._draw_frame()
    
    def update_face_results(self, results: List[dict]) -> None:
        """Update face detection/recognition results.
        
        Args:
            results: List of dicts with face information:
                - 'location': (top, right, bottom, left)
                - 'name': Optional user name
                - 'surname': Optional user surname
                - 'confidence': Optional confidence score
                - 'success': bool
        """
        self._face_results = results
        if self._current_frame is not None:
            self._draw_frame()
    
    def _draw_frame(self) -> None:
        """Draw frame with face detection overlay."""
        if self._current_frame is None:
            return
        
        frame = self._current_frame.copy()
        
        # Draw face bounding boxes and labels
        for result in self._face_results:
            location = result.get('location')
            if location is None:
                continue
            
            top, right, bottom, left = location
            
            # Determine color based on recognition result
            if result.get('success', False):
                color = (0, 255, 0)  # Green for recognized
                name = result.get('name', '')
                surname = result.get('surname', '')
                confidence = result.get('confidence', 0.0)
                
                if name and surname:
                    label = f"{name} {surname} ({confidence:.0%})"
                else:
                    label = f"Recognized ({confidence:.0%})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(top - 10, label_size[1])
            cv2.rectangle(
                frame,
                (left, label_y - label_size[1] - 5),
                (left + label_size[0] + 10, label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (left + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Convert BGR to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # Scale to fit widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def clear(self) -> None:
        """Clear video display."""
        self._current_frame = None
        self._face_results = []
        self.setText("No video feed")
        self.setStyleSheet("background-color: black;")
