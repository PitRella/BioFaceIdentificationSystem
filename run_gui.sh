#!/bin/bash
# Script to run GUI application with proper Qt settings

# Completely remove OpenCV's Qt plugins from path
# Find cv2 qt plugins directory and exclude it
CV2_QT_PLUGINS=$(python -c "import cv2; import os; print(os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins'))" 2>/dev/null)

if [ -n "$CV2_QT_PLUGINS" ] && [ -d "$CV2_QT_PLUGINS" ]; then
    # Remove cv2 plugins from path if present
    if [ -n "$QT_QPA_PLATFORM_PLUGIN_PATH" ]; then
        export QT_QPA_PLATFORM_PLUGIN_PATH=$(echo "$QT_QPA_PLATFORM_PLUGIN_PATH" | tr ':' '\n' | grep -v "$CV2_QT_PLUGINS" | tr '\n' ':' | sed 's/:$//')
    fi
fi

# Set platform explicitly
export QT_QPA_PLATFORM=xcb

# Run the application
python main.py "$@"
