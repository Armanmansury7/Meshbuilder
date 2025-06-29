"""Dialog components for the MeshBuilder UI"""
import logging
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal

# Set up logger
logger = logging.getLogger(__name__)

class UploadOptionDialog(QDialog):
    start_new_project_signal = pyqtSignal()
    upload_existing_project_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload Options")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        label = QLabel("Choose an option:")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        new_project_btn = QPushButton("Start New Project")
        new_project_btn.setFixedHeight(40)
        new_project_btn.clicked.connect(self.on_new_project)
        layout.addWidget(new_project_btn)

        upload_existing_btn = QPushButton("Upload to Existing Project")
        upload_existing_btn.setFixedHeight(40)
        upload_existing_btn.clicked.connect(self.on_upload_existing)
        layout.addWidget(upload_existing_btn)

        self.setLayout(layout)

    def on_new_project(self):
        self.start_new_project_signal.emit()
        self.accept()

    def on_upload_existing(self):
        self.upload_existing_project_signal.emit()
        self.accept()