"""Entry point for biometric identification system application."""
import sys
import os

# CRITICAL: Block OpenCV's Qt plugins completely
# OpenCV includes Qt plugins that conflict with PyQt5
# We need to prevent Qt from finding them

# Method 1: Remove cv2/qt/plugins from any plugin path
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    paths = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'].split(':')
    paths = [p for p in paths if 'cv2/qt/plugins' not in p]
    if paths:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ':'.join(paths)
    else:
        del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

# Method 2: Set explicit platform to avoid plugin discovery issues
if 'QT_QPA_PLATFORM' not in os.environ:
    # Try to use system Qt plugins, not OpenCV's
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Method 3: Disable OpenCV's GUI completely
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = ''

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
from utils.logger import setup_logger
from ui.main_window import MainWindow

logger = setup_logger()


def main():
    """Main application function."""
    logger.info("Starting biometric identification system...")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Face Identification System")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info("Application started")
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)

