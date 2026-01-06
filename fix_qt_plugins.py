"""Script to fix Qt plugin conflicts between OpenCV and PyQt5."""
import os
import shutil
from pathlib import Path

def fix_qt_plugins():
    """Rename OpenCV's Qt plugins directory to prevent conflicts."""
    try:
        import cv2
        cv2_path = Path(cv2.__file__).parent
        qt_plugins_path = cv2_path / "qt" / "plugins"
        qt_plugins_backup = cv2_path / "qt" / "plugins.disabled"
        
        if qt_plugins_path.exists() and not qt_plugins_backup.exists():
            print(f"Renaming {qt_plugins_path} to {qt_plugins_backup}")
            qt_plugins_path.rename(qt_plugins_backup)
            print("OpenCV Qt plugins disabled successfully!")
            return True
        elif qt_plugins_backup.exists():
            print("OpenCV Qt plugins already disabled")
            return True
        else:
            print("OpenCV Qt plugins directory not found")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    fix_qt_plugins()
