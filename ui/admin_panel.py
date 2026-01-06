"""Administrative panel for user management and system statistics."""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QLineEdit, QMessageBox, QTabWidget, QHeaderView,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

from database.connection import AsyncSessionLocal
from database.repositories import (
    UserRepository, BiometricTemplateRepository, AccessLogRepository
)
from database.models import User, AccessLog
from config import FACE_RECOGNITION_THRESHOLD
from utils.logger import setup_logger

logger = setup_logger()


class UserManagementWidget(QWidget):
    """Widget for user management."""
    
    user_updated = pyqtSignal()  # Emitted when user list is updated
    
    def __init__(self, parent=None):
        """Initialize user management widget."""
        super().__init__(parent)
        self._init_ui()
        self._load_users()
    
    def _init_ui(self) -> None:
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._load_users)
        button_layout.addWidget(self.refresh_button)
        
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._delete_selected_user)
        button_layout.addWidget(self.delete_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Users table
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(5)
        self.users_table.setHorizontalHeaderLabels([
            "ID", "Name", "Surname", "Registration Date", "Templates"
        ])
        self.users_table.horizontalHeader().setStretchLastSection(True)
        self.users_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.users_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.users_table)
    
    def _load_users(self) -> None:
        """Load users from database."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            users = loop.run_until_complete(self._fetch_users())
            loop.close()
            
            self.users_table.setRowCount(len(users))
            
            for row, user in enumerate(users):
                self.users_table.setItem(row, 0, QTableWidgetItem(str(user.id)))
                self.users_table.setItem(row, 1, QTableWidgetItem(user.name))
                self.users_table.setItem(row, 2, QTableWidgetItem(user.surname))
                self.users_table.setItem(
                    row, 3,
                    QTableWidgetItem(user.registration_date.strftime("%Y-%m-%d %H:%M:%S"))
                )
                template_count = len(user.templates) if user.templates else 0
                self.users_table.setItem(row, 4, QTableWidgetItem(str(template_count)))
            
            self.users_table.resizeColumnsToContents()
            logger.info(f"Loaded {len(users)} users")
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load users:\n{str(e)}")
    
    async def _fetch_users(self) -> List[User]:
        """Fetch users from database."""
        async with AsyncSessionLocal() as session:
            return await UserRepository.get_all(session)
    
    def _delete_selected_user(self) -> None:
        """Delete selected user."""
        selected_rows = self.users_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a user to delete")
            return
        
        row = selected_rows[0].row()
        user_id = int(self.users_table.item(row, 0).text())
        user_name = self.users_table.item(row, 1).text()
        user_surname = self.users_table.item(row, 2).text()
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete user:\n{user_name} {user_surname}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(self._delete_user(user_id))
                loop.close()
                
                if success:
                    QMessageBox.information(self, "Success", "User deleted successfully")
                    self._load_users()
                    self.user_updated.emit()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete user")
                    
            except Exception as e:
                logger.error(f"Error deleting user: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete user:\n{str(e)}")
    
    async def _delete_user(self, user_id: int) -> bool:
        """Delete user from database."""
        async with AsyncSessionLocal() as session:
            try:
                success = await UserRepository.delete(session, user_id)
                await session.commit()
                return success
            except Exception as e:
                await session.rollback()
                logger.error(f"Error deleting user: {e}")
                return False


class AccessLogWidget(QWidget):
    """Widget for viewing access logs."""
    
    def __init__(self, parent=None):
        """Initialize access log widget."""
        super().__init__(parent)
        self._init_ui()
        self._load_logs()
        
        # Auto-refresh every 5 seconds
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._load_logs)
        self.refresh_timer.start(5000)
    
    def _init_ui(self) -> None:
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._load_logs)
        button_layout.addWidget(self.refresh_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Logs table
        self.logs_table = QTableWidget()
        self.logs_table.setColumnCount(6)
        self.logs_table.setHorizontalHeaderLabels([
            "Timestamp", "User", "Type", "Result", "Confidence", "ID"
        ])
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        self.logs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.logs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.logs_table)
    
    def _load_logs(self, limit: int = 100) -> None:
        """Load access logs from database."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logs = loop.run_until_complete(self._fetch_logs(limit))
            loop.close()
            
            self.logs_table.setRowCount(len(logs))
            
            for row, log_entry in enumerate(logs):
                self.logs_table.setItem(
                    row, 0,
                    QTableWidgetItem(log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                )
                
                if log_entry.user:
                    user_name = f"{log_entry.user.name} {log_entry.user.surname}"
                else:
                    user_name = "Unknown"
                self.logs_table.setItem(row, 1, QTableWidgetItem(user_name))
                
                self.logs_table.setItem(row, 2, QTableWidgetItem(log_entry.access_type))
                self.logs_table.setItem(row, 3, QTableWidgetItem(log_entry.result))
                
                confidence = f"{log_entry.confidence:.2%}" if log_entry.confidence else "N/A"
                self.logs_table.setItem(row, 4, QTableWidgetItem(confidence))
                
                self.logs_table.setItem(row, 5, QTableWidgetItem(str(log_entry.id)))
            
            self.logs_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
    
    async def _fetch_logs(self, limit: int) -> List[AccessLog]:
        """Fetch access logs from database."""
        async with AsyncSessionLocal() as session:
            return await AccessLogRepository.get_recent(session, limit=limit)


class StatisticsWidget(QWidget):
    """Widget for system statistics."""
    
    def __init__(self, parent=None):
        """Initialize statistics widget."""
        super().__init__(parent)
        self._init_ui()
        self._load_statistics()
        
        # Auto-refresh every 10 seconds
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._load_statistics)
        self.refresh_timer.start(10000)
    
    def _init_ui(self) -> None:
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        
        # Statistics group
        stats_group = QGroupBox("System Statistics")
        stats_layout = QFormLayout()
        
        self.total_users_label = QLabel("0")
        stats_layout.addRow("Total Users:", self.total_users_label)
        
        self.total_templates_label = QLabel("0")
        stats_layout.addRow("Total Templates:", self.total_templates_label)
        
        self.total_logs_label = QLabel("0")
        stats_layout.addRow("Total Log Entries:", self.total_logs_label)
        
        self.successful_logins_label = QLabel("0")
        stats_layout.addRow("Successful Identifications:", self.successful_logins_label)
        
        self.failed_logins_label = QLabel("0")
        stats_layout.addRow("Failed Identifications:", self.failed_logins_label)
        
        self.last_24h_label = QLabel("0")
        stats_layout.addRow("Last 24 Hours:", self.last_24h_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._load_statistics)
        layout.addWidget(refresh_button)
        
        layout.addStretch()
    
    def _load_statistics(self) -> None:
        """Load system statistics."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stats = loop.run_until_complete(self._fetch_statistics())
            loop.close()
            
            self.total_users_label.setText(str(stats['total_users']))
            self.total_templates_label.setText(str(stats['total_templates']))
            self.total_logs_label.setText(str(stats['total_logs']))
            self.successful_logins_label.setText(str(stats['successful']))
            self.failed_logins_label.setText(str(stats['failed']))
            self.last_24h_label.setText(str(stats['last_24h']))
            
        except Exception as e:
            logger.error(f"Error loading statistics: {e}")
    
    async def _fetch_statistics(self) -> dict:
        """Fetch system statistics."""
        async with AsyncSessionLocal() as session:
            users = await UserRepository.get_all(session)
            templates = await BiometricTemplateRepository.get_all(session)
            logs = await AccessLogRepository.get_recent(session, limit=10000)
            
            # Calculate statistics
            total_users = len(users)
            total_templates = len(templates)
            total_logs = len(logs)
            
            successful = sum(1 for log in logs if log.result == "success")
            failed = total_logs - successful
            
            # Last 24 hours
            last_24h = datetime.utcnow() - timedelta(hours=24)
            last_24h_count = sum(1 for log in logs if log.timestamp >= last_24h)
            
            return {
                'total_users': total_users,
                'total_templates': total_templates,
                'total_logs': total_logs,
                'successful': successful,
                'failed': failed,
                'last_24h': last_24h_count
            }


class SettingsWidget(QWidget):
    """Widget for system settings."""
    
    settings_changed = pyqtSignal()  # Emitted when settings are changed
    
    def __init__(self, parent=None):
        """Initialize settings widget."""
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        
        settings_group = QGroupBox("Recognition Settings")
        settings_layout = QFormLayout()
        
        # Recognition threshold
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.1, 1.0)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setValue(FACE_RECOGNITION_THRESHOLD)
        self.threshold_spinbox.setDecimals(2)
        settings_layout.addRow("Recognition Threshold:", self.threshold_spinbox)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Note
        note_label = QLabel(
            "Note: Settings changes require application restart to take effect."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: gray;")
        layout.addWidget(note_label)
        
        layout.addStretch()
    
    def get_threshold(self) -> float:
        """Get recognition threshold."""
        return self.threshold_spinbox.value()


class AdminPanel(QWidget):
    """Administrative panel main widget."""
    
    def __init__(self, parent=None):
        """Initialize admin panel."""
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Administrative Panel")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # User Management tab
        self.user_management = UserManagementWidget()
        self.user_management.user_updated.connect(self._on_user_updated)
        self.tabs.addTab(self.user_management, "User Management")
        
        # Access Log tab
        self.access_log = AccessLogWidget()
        self.tabs.addTab(self.access_log, "Access Log")
        
        # Statistics tab
        self.statistics = StatisticsWidget()
        self.tabs.addTab(self.statistics, "Statistics")
        
        # Settings tab
        self.settings = SettingsWidget()
        self.tabs.addTab(self.settings, "Settings")
        
        layout.addWidget(self.tabs)
    
    def _on_user_updated(self) -> None:
        """Handle user update event."""
        # Refresh statistics when user is updated
        self.statistics._load_statistics()
