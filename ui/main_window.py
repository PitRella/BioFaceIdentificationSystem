"""Main window for face identification system."""
import sys
import os
import asyncio
from typing import Optional

# Fix Qt plugin path issue with OpenCV
os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '')

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QFont

from core.video_capture import VideoCapture
from core.identification import FaceIdentification, IdentificationResult
from ui.components.video_widget import VideoWidget
from ui.admin_window import AdminWindow
from config import FRAME_SKIP
from utils.logger import setup_logger

logger = setup_logger()


class VideoProcessor(QObject):
    """Video processor worker object."""
    
    frame_ready = pyqtSignal(object)  # Emits frame (np.ndarray)
    faces_identified = pyqtSignal(list)  # Emits list of IdentificationResult
    error_occurred = pyqtSignal(str)  # Emits error message
    finished = pyqtSignal()  # Emits when processing is done
    
    def __init__(self):
        """Initialize video processor."""
        super().__init__()
        self.identification: Optional[FaceIdentification] = None
        self.cap: Optional[VideoCapture] = None
        self.running = False
        self.frame_count = 0
    
    def set_identification(self, identification: FaceIdentification) -> None:
        """Set face identification instance.
        
        Args:
            identification: FaceIdentification instance
        """
        self.identification = identification
    
    def start_processing(self) -> None:
        """Start video processing."""
        self.running = True
        self.frame_count = 0
        try:
            self.cap = VideoCapture()
            if not self.cap.start():
                self.error_occurred.emit("Failed to open camera")
                self.running = False
                return
            logger.info("Video capture started")
            self._process_loop()
        except Exception as e:
            self.error_occurred.emit(f"Error starting camera: {str(e)}")
            logger.error(f"Error starting camera: {e}")
            self.running = False
    
    def stop_processing(self) -> None:
        """Stop video processing."""
        self.running = False
        if self.cap:
            self.cap.stop()
            self.cap = None
        logger.info("Video capture stopped")
        self.finished.emit()
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        if not self.cap:
            return
        
        if not self.cap.is_opened():
            self.error_occurred.emit("Camera not opened")
            return
        
        while self.running:
            frame = self.cap.read()
            if frame is None:
                continue
            
            self.frame_count += 1
            
            # Emit frame for display
            self.frame_ready.emit(frame)
            
            # Process every Nth frame for identification
            if self.frame_count % FRAME_SKIP == 0 and self.identification:
                try:
                    # Run async identification
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(
                        self.identification.identify_frame(frame)
                    )
                    loop.close()
                    
                    # Convert results to dict format for display
                    face_results = []
                    for face_location, result in results:
                        top, right, bottom, left = face_location.to_tuple()
                        face_results.append({
                            'location': (top, right, bottom, left),
                            'name': result.user_name,
                            'surname': result.user_surname,
                            'confidence': result.confidence,
                            'success': result.success
                        })
                    
                    self.faces_identified.emit(face_results)
                    
                except Exception as e:
                    logger.error(f"Error during identification: {e}")
                    self.error_occurred.emit(f"Identification error: {str(e)}")
        
        self.finished.emit()


class VideoProcessingThread(QThread):
    """Thread for processing video frames."""
    
    def __init__(self, processor: VideoProcessor):
        """Initialize video processing thread.
        
        Args:
            processor: VideoProcessor instance
        """
        super().__init__()
        self.processor = processor
        # CRITICAL: Move processor to this thread BEFORE starting thread
        # This must be done in the main thread before start()
        self.processor.moveToThread(self)
    
    def run(self) -> None:
        """Run video processing."""
        self.processor.start_processing()
    
    def stop(self) -> None:
        """Stop video processing."""
        if self.processor:
            self.processor.stop_processing()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.identification: Optional[FaceIdentification] = None
        self.video_thread: Optional[VideoProcessingThread] = None
        self.admin_window: Optional[AdminWindow] = None
        self._init_ui()
        self._init_identification()
    
    def _init_ui(self) -> None:
        """Initialize user interface."""
        self.setWindowTitle("Face Identification System")
        self.setMinimumSize(1280, 720)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("Face Identification System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Video widget
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(1280, 720)
        main_layout.addWidget(self.video_widget, stretch=1)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.start_button = QPushButton("Start Identification")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_identification)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_identification)
        button_layout.addWidget(self.stop_button)
        
        self.admin_button = QPushButton("Admin Panel")
        self.admin_button.setMinimumHeight(40)
        self.admin_button.clicked.connect(self.open_admin_panel)
        button_layout.addWidget(self.admin_button)
        
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _init_identification(self) -> None:
        """Initialize face identification system."""
        try:
            self.identification = FaceIdentification()
            logger.info("Face identification system initialized")
        except Exception as e:
            logger.error(f"Error initializing identification: {e}")
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize face identification system:\n{str(e)}"
            )
    
    def start_identification(self) -> None:
        """Start face identification."""
        if self.identification is None:
            QMessageBox.warning(
                self,
                "Not Ready",
                "Face identification system not initialized"
            )
            return
        
        if self.video_thread and self.video_thread.isRunning():
            return
        
        # Load descriptors cache
        self.status_bar.showMessage("Loading descriptors...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            count = loop.run_until_complete(
                self.identification.load_descriptors_cache()
            )
            loop.close()
            
            if count == 0:
                QMessageBox.warning(
                    self,
                    "No Users",
                    "No users registered in database.\n"
                    "Please register users first using:\n"
                    "python scripts/register_user.py"
                )
                self.status_bar.showMessage("No users in database")
                return
            
            self.status_bar.showMessage(f"Loaded {count} descriptors. Starting identification...")
            
        except Exception as e:
            logger.error(f"Error loading descriptors: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load descriptors:\n{str(e)}"
            )
            return
        
        # Create video processor
        processor = VideoProcessor()
        processor.set_identification(self.identification)
        processor.frame_ready.connect(self.video_widget.update_frame)
        processor.faces_identified.connect(self._on_faces_identified)
        processor.error_occurred.connect(self._on_error)
        
        # Create and start thread
        self.video_thread = VideoProcessingThread(processor)
        self.video_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_bar.showMessage("Identification running...")
        logger.info("Identification started")
    
    def stop_identification(self) -> None:
        """Stop face identification."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()
            self.video_thread = None
        
        self.video_widget.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Stopped")
        logger.info("Identification stopped")
    
    def _on_faces_identified(self, results: list[dict]) -> None:
        """Handle face identification results.
        
        Args:
            results: List of identification results
        """
        self.video_widget.update_face_results(results)
        
        # Update status
        if results:
            recognized = sum(1 for r in results if r.get('success', False))
            if recognized > 0:
                self.status_bar.showMessage(
                    f"Identified {recognized} face(s)"
                )
            else:
                self.status_bar.showMessage("Unknown face(s) detected")
    
    def _on_error(self, error_message: str) -> None:
        """Handle error from video thread.
        
        Args:
            error_message: Error message
        """
        logger.error(f"Video thread error: {error_message}")
        QMessageBox.warning(self, "Error", error_message)
        self.stop_identification()
    
    def open_admin_panel(self) -> None:
        """Open administrative panel."""
        if self.admin_window is None:
            self.admin_window = AdminWindow(self)
        
        self.admin_window.show()
        self.admin_window.raise_()
        self.admin_window.activateWindow()
        logger.info("Admin panel opened")
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self.stop_identification()
        if self.admin_window:
            self.admin_window.close()
        event.accept()
