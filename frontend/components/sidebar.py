"""Sidebar components for the MeshBuilder UI"""
import os
import logging
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon

# Set up logger
logger = logging.getLogger(__name__)

class DraggableSidebar(QFrame):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.main_window = main_window
        self.initUI()

    def initUI(self):
        self.setFixedWidth(68)
        self.setStyleSheet("""
            QFrame#sidebar {
                background-color: #F5F5F5;
                border-radius: 10px;
                margin: 6px;
            }
            QPushButton {
                border: none;
                padding: 6px;
                margin: 4px;
                border-radius: 5px;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Define the assets directory
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
        
        # Fix: Use lambda for choose_export_format to prevent attribute error
        buttons = [
            (os.path.join(assets_dir, "plus.svg"), "New Project", self.main_window.open_upload_option_dialog),
            (os.path.join(assets_dir, "file-multiple.svg"), "Projects", lambda: logger.info("Projects clicked")),
            (os.path.join(assets_dir, "image-multiple-outline.svg"), "Media", self.main_window.toggle_image_list_sidebar),
            (os.path.join(assets_dir, "tune-variant.svg"), "Advanced", self.main_window.toggle_settings_sidebar),
            (os.path.join(assets_dir, "cube-scan.svg"), "Start Processing", self.main_window.start_processing),
            (os.path.join(assets_dir, "content-save-outline.svg"), "Save/Load", lambda: logger.info("Save/Load clicked")),
            (os.path.join(assets_dir, "export.svg"), "Export", lambda: self.handle_export_click()),
            (os.path.join(assets_dir, "logmain.png"), "Show Logs", self.main_window.show_logs),  # Updated icon path
            (os.path.join(assets_dir, "mesh.png"), "Refine Mesh", self.main_window.refine_mesh),  # Added refine mesh button
            (os.path.join(assets_dir, "restart.svg"), "Cancel", self.main_window.cancel_processing),
        ]

        for icon_path, tooltip, action in buttons:
            btn = QPushButton()
            icon = QIcon(icon_path)
            if icon.isNull():
                logger.warning(f"Icon not found: {icon_path}")
                # Use text instead if icon is missing
                btn.setText(tooltip[:1])  # First character as button text
            else:
                btn.setIcon(icon)
                btn.setIconSize(QSize(20, 20))
            btn.setToolTip(tooltip)
            btn.clicked.connect(action)
            layout.addWidget(btn)

        layout.addStretch()
        self.setLayout(layout)
        
    def handle_export_click(self):
        """Handle export button click by showing the right sidebar"""
        try:
            # Try to call the export method if it exists
            if hasattr(self.main_window, 'choose_export_format'):
                self.main_window.choose_export_format()
            else:
                # Fallback behavior if the method doesn't exist
                logger.info("Export clicked")
                # Show the right sidebar directly if it exists
                if hasattr(self.main_window, 'right_sidebar_dock') and self.main_window.right_sidebar_dock:
                    if not self.main_window.right_sidebar_dock.isVisible():
                        self.main_window.right_sidebar_dock.show()
        except Exception as e:
            logger.error(f"Error handling export click: {str(e)}")