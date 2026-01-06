"""Administrative window."""
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt
from ui.admin_panel import AdminPanel
from utils.logger import setup_logger

logger = setup_logger()


class AdminWindow(QMainWindow):
    """Administrative window for system management."""
    
    def __init__(self, parent=None):
        """Initialize admin window."""
        super().__init__(parent)
        self.setWindowTitle("Administrative Panel - Face Identification System")
        self.setMinimumSize(1000, 700)
        
        # Set admin panel as central widget
        self.admin_panel = AdminPanel()
        self.setCentralWidget(self.admin_panel)
        
        logger.info("Admin window initialized")
