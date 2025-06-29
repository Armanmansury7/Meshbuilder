"""Label components for the MeshBuilder UI"""
import os
import logging
from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

# Set up logger
logger = logging.getLogger(__name__)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class BrandingLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setText("MeshBuilder")
        self.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 28px;
                font-weight: bold;
            }
        """)

class VersionLabel(QLabel):
    def __init__(self, version="v0.1.0", parent=None):
        super().__init__(parent)
        self.initUI(version)

    def initUI(self, version):
        self.setText(version)
        self.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 8px;
            }
        """)

class PoweredByLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        text_label = QLabel("Powered by")
        text_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 10px;
            }
        """)
        layout.addWidget(text_label)

        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "assets", "ie_logo.png")
        
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(120, 30, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            logo_label.setText("Logo Missing")
            logo_label.setStyleSheet("color: #FF0000; font-size: 10px;")
            logger.warning(f"Logo not found at {logo_path}")
            
        layout.addWidget(logo_label)

        layout.addStretch()
        self.setLayout(layout)