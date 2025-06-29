"""List view components for the MeshBuilder UI"""
import os
import logging
from PyQt6.QtWidgets import QWidget, QListWidget, QListWidgetItem, QHBoxLayout, QPushButton, QLabel, QDialog, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap

from frontend.components.labels import ClickableLabel

# Set up logger
logger = logging.getLogger(__name__)

class ThumbnailListItem(QWidget):
    deleteRequested = pyqtSignal()  # Signal to request deletion of this item

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self.filename_label = ClickableLabel()
        self.filename_label.setText(os.path.basename(self.image_path))
        self.filename_label.setStyleSheet("color: #00aaff; text-decoration: underline; font-size: 11px;")
        self.filename_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.filename_label.clicked.connect(self.show_full_image)
        layout.addWidget(self.filename_label)

        self.delete_button = QPushButton()
        trash_icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                "assets", "trashcan.svg")
        trash_icon = QIcon(trash_icon_path)
        self.delete_button.setIcon(trash_icon)
        self.delete_button.setFixedSize(20, 20)
        self.delete_button.setStyleSheet("background-color: transparent; border: none;")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        layout.addWidget(self.delete_button)

        self.setLayout(layout)

    def on_delete_clicked(self):
        self.deleteRequested.emit()

    def show_full_image(self):
        """Opens a modal dialog showing the full image."""
        dialog = QDialog(self)
        dialog.setWindowTitle(os.path.basename(self.image_path))
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        vbox = QVBoxLayout(dialog)
        label = QLabel()
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            # Scale large images to fit dialog better
            screen_size = dialog.screen().size()
            max_width = screen_size.width() * 0.8
            max_height = screen_size.height() * 0.8
            
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(
                    int(max_width), 
                    int(max_height),
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
            
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            label.setText(f"Failed to load image: {self.image_path}")
        
        vbox.addWidget(label)
        dialog.exec()

class ImageListView(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        self.setStyleSheet("""
            QListWidget {
                background-color: #2E2E2E;
                border: none;
                padding: 0px;
            }
            QListWidget::item {
                margin: 1px;
                padding: 3px;
                border-bottom: 1px solid #3c3c3c;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
                border-radius: 3px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_thumbnail_item(self, image_path):
        """Creates and adds a new thumbnail item to the list."""
        item = QListWidgetItem(self)
        widget = ThumbnailListItem(image_path)
        widget.deleteRequested.connect(lambda: self.remove_item(item))
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)
        logger.info(f"Added file to list: {image_path}")

    def remove_item(self, item):
        row = self.row(item)
        widget = self.itemWidget(item)
        if widget and hasattr(widget, 'image_path'):
            logger.info(f"Removing file from list: {widget.image_path}")
        self.takeItem(row)