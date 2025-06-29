"""
MeshBuilder Frontend Module
Enhanced UI for the 3D reconstruction application
Updated to work with the enhanced MeshBuilderConnector
"""
import os
import sys
import logging
import datetime
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QProgressBar, QTabWidget, QListWidget, QListWidgetItem, QSplitter,
    QGroupBox, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QScrollArea,
    QSizePolicy, QFrame, QToolBar, QStatusBar, QMenu, QMenuBar, QDialog,
    QDockWidget, QInputDialog, QTextEdit, QSlider, QToolButton, QButtonGroup,
    QRadioButton
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread, QTimer, QPoint, QEvent
from PyQt6.QtGui import QIcon, QPixmap, QAction, QFont, QColor, QPainter, QImage, QCursor

# Import the ENHANCED connector for backend communication
try:
    from interface.connector import MeshBuilderConnector
    from interface.project import ProjectManager, Project
    CONNECTOR_AVAILABLE = True
    print("✓ Enhanced connector imported successfully")
except ImportError as e:
    print(f"✗ Enhanced connector import error: {e}")
    CONNECTOR_AVAILABLE = False
    # Keep the existing dummy classes as fallback
    class MeshBuilderConnector:
        def __init__(self, parent): 
            self.parent = parent
            self.current_project = None
        def create_project(self, name, quality): return True
        def get_project_info(self): return {"name": "test", "quality_level": "high", "media_count": {"total": 0}}
        def add_media_files(self, files): return {"total": len(files), "images": len(files), "videos": 0}
        def start_processing(self, output_path, settings): return True
        def cancel_processing(self): pass
        def get_processing_status(self): return {"is_processing": False}
        def load_project(self, name): return True
        def update_settings(self, settings): pass
        def get_system_recommendations(self): return {}
        def validate_license(self): return True
        def cleanup(self): pass
        # Dummy signals
        processingStarted = None
        processingFinished = None
        processingError = None
        progressUpdated = None
        stageChanged = None
        previewReady = None
        resourceUpdate = None
        memoryOptimization = None
        licenseValidationRequired = None
        systemSpecsDetected = None
    class ProjectManager:
        def list_projects(self): return []
        def save_project(self, project): return True
        def get_projects_with_details(self): return []
        def duplicate_project(self, old_name, new_name): return None
    class Project:
        def __init__(self): self.name = "test"

# Import the ModelViewer for 3D model display
try:
    from frontend.components.model_viewer import ModelViewer
except ImportError:
    # Create a placeholder ModelViewer
    class ModelViewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setMinimumSize(400, 300)
            self.current_model = None
            layout = QVBoxLayout(self)
            label = QLabel("3D Model Viewer\n(PyQtGraph/trimesh required)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: white; font-size: 14px;")
            layout.addWidget(label)
        def load_model(self, path): return False
        def auto_orbit(self, enable): pass
        def zoom_in(self): pass
        def zoom_out(self): pass
        def take_screenshot(self, path): return False

# Set up logging
logger = logging.getLogger("frontend.window")

class LogWindow(QFrame):
    """Log window to display processing messages"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #2E2E2E;
                border-radius: 10px;
                margin: 6px;
                padding: 10px;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 12px;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: #AAAAAA;
                border: 1px solid #555555;
                border-radius: 3px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        title = QLabel("Processing Log")
        layout.addWidget(title)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: #FFFFFF;
                padding: 4px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        layout.addWidget(clear_btn)
        
        self.setLayout(layout)
    
    def append_log(self, message):
        """Add a message to the log"""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_log(self):
        """Clear the log"""
        self.log_text.clear()


class ContactDialog(QDialog):
    """Dialog for contact information"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Contact Us")
        self.setMinimumWidth(300)
        self.setStyleSheet("""
            QDialog {
                background-color: #2E2E2E;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
            }
            QPushButton {
                background-color: #444444;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        message = QLabel("Contact us at: nick@immersive-engineering.com")
        layout.addWidget(message)
        
        response = QLabel("We will revert you within 48 hours.")
        layout.addWidget(response)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)
        
        self.setLayout(layout)


class SettingsDialog(QDialog):
    """Modal settings dialog with tabs"""
    settingsUpdated = pyqtSignal(dict)
    
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Project Settings")
        self.setFixedSize(400, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #2E2E2E;
                border-radius: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #333333;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-bottom-color: #555555;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color: #444444;
                border-bottom-color: #444444;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 12px;
            }
            QGroupBox {
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 5px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #444444;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QCheckBox, QRadioButton {
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #444444;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.basic_tab = QWidget()
        self.advanced_tab = QWidget()
        
        self.tabs.addTab(self.basic_tab, "Basic")
        self.tabs.addTab(self.advanced_tab, "Advanced")
        
        # Set up Basic tab
        self.setup_basic_tab()
        
        # Set up Advanced tab
        self.setup_advanced_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(button_layout)
    
    def setup_basic_tab(self):
        """Set up the Basic tab with settings"""
        layout = QVBoxLayout(self.basic_tab)
        layout.setSpacing(15)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        
        # Quality settings
        quality_group = QGroupBox("Image Quality Settings")
        quality_layout = QVBoxLayout()
        
        quality_layout.addWidget(QLabel("Image Quality Threshold:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["fast", "balanced", "high", "very_high", "photorealistic"])
        self.quality_combo.setCurrentText(self.current_settings.get("Processing", {}).get("quality_level", "high"))
        quality_layout.addWidget(self.quality_combo)
        
        quality_layout.addWidget(QLabel("Feature Matching:"))
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["sequential", "exhaustive"])
        current_matcher = self.current_settings.get("Processing", {}).get("feature_matcher", "exhaustive")
        self.feature_combo.setCurrentText(current_matcher)
        quality_layout.addWidget(self.feature_combo)
        
        quality_group.setLayout(quality_layout)
        scroll_layout.addWidget(quality_group)
        
        # Point Cloud settings
        point_cloud_group = QGroupBox("Point Cloud Settings")
        point_cloud_layout = QVBoxLayout()
        
        point_cloud_layout.addWidget(QLabel("Point Density:"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["medium", "high", "ultra"])
        self.density_combo.setCurrentText(self.current_settings.get("Processing", {}).get("point_density", "high"))
        point_cloud_layout.addWidget(self.density_combo)
        
        point_cloud_group.setLayout(point_cloud_layout)
        scroll_layout.addWidget(point_cloud_group)
        
        # Mesh settings
        mesh_group = QGroupBox("Mesh Settings")
        mesh_layout = QVBoxLayout()
        
        mesh_layout.addWidget(QLabel("Mesh Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["medium", "high", "ultra"])
        self.resolution_combo.setCurrentText(self.current_settings.get("Processing", {}).get("mesh_resolution", "high"))
        mesh_layout.addWidget(self.resolution_combo)
        
        mesh_layout.addWidget(QLabel("Texture Resolution:"))
        self.texture_combo = QComboBox()
        self.texture_combo.addItems(["2048", "4096", "8192"])
        self.texture_combo.setCurrentText(self.current_settings.get("Processing", {}).get("texture_resolution", "4096"))
        mesh_layout.addWidget(self.texture_combo)
        
        mesh_layout.addWidget(QLabel("Target Face Count:"))
        self.faces_spinbox = QSpinBox()
        self.faces_spinbox.setMinimum(5000)
        self.faces_spinbox.setMaximum(1000000)
        self.faces_spinbox.setSingleStep(5000)
        target_faces = self.current_settings.get("Processing", {}).get("target_faces", "100000")
        self.faces_spinbox.setValue(int(target_faces))
        mesh_layout.addWidget(self.faces_spinbox)
        
        mesh_group.setLayout(mesh_layout)
        scroll_layout.addWidget(mesh_group)
        
        # Color Map Type
        color_group = QGroupBox("Color Map Type")
        color_layout = QVBoxLayout()
        
        self.flat_grey_radio = QRadioButton("Flat Grey")
        self.medium_quality_radio = QRadioButton("Medium Quality")
        self.high_quality_radio = QRadioButton("High Quality")
        self.photo_real_radio = QRadioButton("Photo Real (Coming Soon)")
        self.photo_real_radio.setEnabled(False)
        self.photo_real_radio.setToolTip("Coming Soon")
        
        flat_grey_model = self.current_settings.get("Processing", {}).get("flat_grey_model", "false")
        if flat_grey_model.lower() == "true":
            self.flat_grey_radio.setChecked(True)
        else:
            self.high_quality_radio.setChecked(True)
        
        color_layout.addWidget(self.flat_grey_radio)
        color_layout.addWidget(self.medium_quality_radio)
        color_layout.addWidget(self.high_quality_radio)
        color_layout.addWidget(self.photo_real_radio)
        
        color_group.setLayout(color_layout)
        scroll_layout.addWidget(color_group)
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
    
    def setup_advanced_tab(self):
        """Set up the Advanced tab with settings"""
        layout = QVBoxLayout(self.advanced_tab)
        layout.setSpacing(15)
        
        # Hardware settings
        hardware_group = QGroupBox("Hardware Settings")
        hardware_layout = QVBoxLayout()
        
        self.use_gpu_checkbox = QCheckBox("Use GPU Acceleration")
        use_gpu = self.current_settings.get("Hardware", {}).get("use_gpu", "true")
        self.use_gpu_checkbox.setChecked(use_gpu.lower() == "true")
        hardware_layout.addWidget(self.use_gpu_checkbox)
        
        gs_layout = QHBoxLayout()
        self.use_gs_checkbox = QCheckBox("Use Gaussian Splatting")
        self.use_gs_checkbox.setChecked(False)
        self.use_gs_checkbox.setEnabled(False)
        gs_layout.addWidget(self.use_gs_checkbox)
        
        gs_label = QLabel("Coming Soon")
        gs_label.setStyleSheet("color: #FF9900; font-style: italic;")
        gs_layout.addWidget(gs_label)
        gs_layout.addStretch()
        
        hardware_layout.addLayout(gs_layout)
        
        hardware_group.setLayout(hardware_layout)
        layout.addWidget(hardware_group)
        
        # Export settings
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout()
        
        export_layout.addWidget(QLabel("Default Export Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["obj", "ply", "fbx (Coming Soon)", "gltf (Coming Soon)"])
        export_format = self.current_settings.get("Output", {}).get("format", "obj")
        self.format_combo.setCurrentText(export_format)
        export_layout.addWidget(self.format_combo)
        
        self.compress_checkbox = QCheckBox("Compress Output Files")
        compress_output = self.current_settings.get("Output", {}).get("compress_output", "false")
        self.compress_checkbox.setChecked(compress_output.lower() == "true")
        export_layout.addWidget(self.compress_checkbox)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Cache settings
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QVBoxLayout()
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        cache_layout.addWidget(self.clear_cache_btn)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)
        
        layout.addStretch()
    
    def clear_cache(self):
        """Clear application cache"""
        reply = QMessageBox.question(
            self, 
            'Clear Cache',
            'Are you sure you want to clear all cache files?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                cache_dir = os.path.join(tempfile.gettempdir(), "meshbuilder")
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    os.makedirs(cache_dir, exist_ok=True)
                    QMessageBox.information(self, "Cache Cleared", "Cache files cleared successfully.")
                else:
                    QMessageBox.information(self, "No Cache", "No cache directory found.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error clearing cache: {str(e)}")
    
    def save_settings(self):
        """Save settings and emit signal"""
        settings = {
            "Processing": {
                "quality_level": self.quality_combo.currentText(),
                "quality_threshold": 0.5,
                "color_enhancement_mode": "balanced",
                "feature_matcher": self.feature_combo.currentText(),
                "mesh_resolution": self.resolution_combo.currentText(),
                "texture_resolution": self.texture_combo.currentText(),
                "target_faces": str(self.faces_spinbox.value()),
                "flat_grey_model": str(self.flat_grey_radio.isChecked()).lower(),
                "use_deep_enhancement": False
            },
            "Hardware": {
                "use_gpu": str(self.use_gpu_checkbox.isChecked()).lower(),
                "use_gaussian_splatting": "false"
            },
            "Output": {
                "format": self.format_combo.currentText().split()[0].lower(),  # Remove "(Coming Soon)"
                "compress_output": str(self.compress_checkbox.isChecked()).lower()
            }
        }
        
        self.settingsUpdated.emit(settings)
        self.accept()


class SideBarButton(QPushButton):
    """Custom button for the sidebar"""
    def __init__(self, text, icon_name=None, parent=None):
        super().__init__(text, parent)
        self.setToolTip(text)
        
        if icon_name:
            self.setIcon(QIcon.fromTheme(icon_name))
        
        self.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                border-radius: 5px;
                padding: 8px;
                text-align: center;
                margin: 2px 5px;
            }
            QPushButton:hover {
                background-color: #4D6666;
            }
            QPushButton:pressed {
                background-color: #4D6666;
            }
        """)
        
        if icon_name or self.icon():
            self.setIconSize(QSize(20, 20))
        
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setFlat(False)


class CollapsibleSection(QWidget):
    """Collapsible section for sidebar"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QWidget { background-color: transparent; }")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.header_button = QPushButton(title)
        self.header_button.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px;
                text-align: center;
                margin: 2px 5px;
            }
            QPushButton:hover {
                background-color: #4D6666;
            }
            QPushButton:pressed {
                background-color: #4D6666;
            }
        """)
        self.header_button.setIcon(QIcon.fromTheme("go-down"))
        self.header_button.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.header_button.clicked.connect(self.toggle_content)
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(5, 0, 5, 0)
        self.content_layout.setSpacing(2)
        
        self.layout.addWidget(self.header_button)
        self.layout.addWidget(self.content)
        
        self.is_expanded = True
    
    def add_button(self, button):
        """Add a button to the section"""
        self.content_layout.addWidget(button)
    
    def toggle_content(self):
        """Toggle the visibility of the content section"""
        self.is_expanded = not self.is_expanded
        self.content.setVisible(self.is_expanded)
        
        if self.is_expanded:
            self.header_button.setIcon(QIcon.fromTheme("go-down"))
        else:
            self.header_button.setIcon(QIcon.fromTheme("go-next"))


class EnhancedLeftSidebar(QFrame):
    """Enhanced left sidebar with project menu"""
    
    # Define signals
    newProjectClicked = pyqtSignal()
    openProjectClicked = pyqtSignal()
    saveProjectClicked = pyqtSignal()
    cropModelClicked = pyqtSignal()
    takeScreenshotClicked = pyqtSignal()
    settingsClicked = pyqtSignal()
    selectPortionClicked = pyqtSignal()
    deleteSelectionClicked = pyqtSignal()
    showLogsClicked = pyqtSignal()
    refineMeshClicked = pyqtSignal()
    uploadImagesClicked = pyqtSignal()
    viewUploadedImagesClicked = pyqtSignal()
    browseOutputClicked = pyqtSignal()
    generateModelClicked = pyqtSignal()
    cancelProcessingClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.initUI()
    
    def initUI(self):
        self.setMinimumWidth(90)
        self.setMaximumWidth(90)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        
        self.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-radius: 10px;
                margin: 6px;
            }
        """)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #333333;
                width: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(5)
        
        # Logo for collapsed state
        self.logo_label = QLabel()
        logo_path = os.path.join("assets", "icon 1c_1.png")
        if os.path.exists(logo_path):
            self.logo_pixmap = QPixmap(logo_path)
            self.logo_pixmap = self.logo_pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, 
                                                 Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(self.logo_pixmap)
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.logo_label.setText("MB")
            self.logo_label.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.logo_label.setVisible(True)
        layout.addWidget(self.logo_label)
        
        # Header
        self.header_label = QLabel("Project Menu")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setStyleSheet("""
            color: #000000;
            font-size: 12px;
            font-weight: bold;
            padding: 5px;
            background-color: #00ffff;
            border-radius: 5px;
        """)
        self.header_label.setVisible(False)
        layout.addWidget(self.header_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #444444;")
        layout.addWidget(separator)
        
        # Buttons
        self.new_project_btn = SideBarButton("", "")
        self.new_project_btn.clicked.connect(self.newProjectClicked.emit)
        plus_icon_path = os.path.join("assets", "Plus.png")
        if os.path.exists(plus_icon_path):
            self.new_project_btn.setIcon(QIcon(plus_icon_path))
        layout.addWidget(self.new_project_btn)
        
        self.upload_btn = SideBarButton("", "")
        self.upload_btn.clicked.connect(self.uploadImagesClicked.emit)
        upload_icon_path = os.path.join("assets", "AddImage.png")
        if os.path.exists(upload_icon_path):
            self.upload_btn.setIcon(QIcon(upload_icon_path))
        layout.addWidget(self.upload_btn)
        
        self.view_images_btn = SideBarButton("", "")
        self.view_images_btn.clicked.connect(self.viewUploadedImagesClicked.emit)
        view_images_icon_path = os.path.join("assets", "ViewImage.png")
        if os.path.exists(view_images_icon_path):
            self.view_images_btn.setIcon(QIcon(view_images_icon_path))
        layout.addWidget(self.view_images_btn)
        
        self.start_btn = SideBarButton("", "")
        self.start_btn.clicked.connect(self.generateModelClicked.emit)
        start_icon_path = os.path.join("assets", "Play.png")
        if os.path.exists(start_icon_path):
            self.start_btn.setIcon(QIcon(start_icon_path))
        layout.addWidget(self.start_btn)
        
        self.stop_btn = SideBarButton("", "")
        self.stop_btn.clicked.connect(self.cancelProcessingClicked.emit)
        stop_icon_path = os.path.join("assets", "Stop.png")
        if os.path.exists(stop_icon_path):
            self.stop_btn.setIcon(QIcon(stop_icon_path))
        layout.addWidget(self.stop_btn)
        
        # Mesh Tools section
        self.mesh_tools_section = CollapsibleSection("")
        
        mesh_icon_path = os.path.join("assets", "mesh.png")
        if os.path.exists(mesh_icon_path):
            self.mesh_tools_section.header_button.setIcon(QIcon(mesh_icon_path))
        
        self.crop_model_btn = SideBarButton("", "edit-cut")
        self.crop_model_btn.clicked.connect(self.cropModelClicked.emit)
        self.mesh_tools_section.add_button(self.crop_model_btn)
        
        self.select_portion_btn = SideBarButton("", "edit-select-all")
        self.select_portion_btn.clicked.connect(self.selectPortionClicked.emit)
        self.mesh_tools_section.add_button(self.select_portion_btn)
        
        self.delete_selection_btn = SideBarButton("", "edit-delete")
        self.delete_selection_btn.clicked.connect(self.deleteSelectionClicked.emit)
        self.mesh_tools_section.add_button(self.delete_selection_btn)
        
        self.refine_mesh_btn = SideBarButton("", "edit-find-replace")
        refine_icon_path = os.path.join("assets", "mesh.png")
        if os.path.exists(refine_icon_path):
            self.refine_mesh_btn.setIcon(QIcon(refine_icon_path))
        self.refine_mesh_btn.clicked.connect(self.refineMeshClicked.emit)
        self.mesh_tools_section.add_button(self.refine_mesh_btn)
        
        self.screenshot_btn = SideBarButton("", "camera-photo")
        self.screenshot_btn.clicked.connect(self.takeScreenshotClicked.emit)
        self.mesh_tools_section.add_button(self.screenshot_btn)
        
        layout.addWidget(self.mesh_tools_section)
        
        self.browse_path_btn = SideBarButton("", "")
        self.browse_path_btn.clicked.connect(self.browseOutputClicked.emit)
        browse_icon_path = os.path.join("assets", "SaveAs.png")
        if os.path.exists(browse_icon_path):
            self.browse_path_btn.setIcon(QIcon(browse_icon_path))
        layout.addWidget(self.browse_path_btn)
        
        layout.addStretch()
        
        # Collapse/expand button
        collapse_layout = QHBoxLayout()
        collapse_layout.setContentsMargins(0, 0, 7, 0)
        collapse_layout.addStretch()
        
        self.collapse_btn = QPushButton()
        self.collapse_btn.setIcon(QIcon.fromTheme("go-next"))
        self.collapse_btn.setToolTip("Expand Sidebar")
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                border-radius: 15px;
                padding: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #4D6666;
            }
        """)
        self.collapse_btn.setFixedSize(30, 30)
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        
        collapse_layout.addWidget(self.collapse_btn)
        layout.addLayout(collapse_layout)
        
        self.scroll_area.setWidget(content_widget)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.scroll_area)
        
        self.is_collapsed = True
        self.original_width = 200
    
    def toggle_collapse(self):
        """Toggle sidebar collapse state"""
        if self.is_collapsed:
            # Expand
            self.setMinimumWidth(220)
            self.setMaximumWidth(300)
            self.new_project_btn.setText("New Project")
            self.upload_btn.setText("Upload/Image")
            self.view_images_btn.setText("View Uploaded Images")
            self.start_btn.setText("Start")
            self.stop_btn.setText("Stop")
            self.browse_path_btn.setText("Browse Path")
            self.crop_model_btn.setText("Crop Model")
            self.select_portion_btn.setText("Select Portion")
            self.delete_selection_btn.setText("Delete Selection")
            self.refine_mesh_btn.setText("Refine Mesh")
            self.screenshot_btn.setText("Take Screenshot")
            self.mesh_tools_section.header_button.setText("Mesh Tools")
            self.collapse_btn.setIcon(QIcon.fromTheme("go-previous"))
            self.collapse_btn.setToolTip("Collapse Sidebar")
            
            self.header_label.setVisible(True)
            self.logo_label.setVisible(False)
        else:
            # Collapse
            self.original_width = self.width()
            self.setMinimumWidth(80)
            self.setMaximumWidth(80)
            self.new_project_btn.setText("")
            self.upload_btn.setText("")
            self.view_images_btn.setText("")
            self.start_btn.setText("")
            self.stop_btn.setText("")
            self.browse_path_btn.setText("")
            self.crop_model_btn.setText("")
            self.select_portion_btn.setText("")
            self.delete_selection_btn.setText("")
            self.refine_mesh_btn.setText("")
            self.screenshot_btn.setText("")
            self.mesh_tools_section.header_button.setText("")
            self.collapse_btn.setIcon(QIcon.fromTheme("go-next"))
            self.collapse_btn.setToolTip("Expand Sidebar")
            
            self.header_label.setVisible(False)
            self.logo_label.setVisible(True)
        
        self.is_collapsed = not self.is_collapsed


# Try to import the image list view
try:
    from frontend.components.list_view import ImageListView
except ImportError:
    # Create a placeholder ImageListView
    class ImageListView(QListWidget):
        def __init__(self):
            super().__init__()
            self.setStyleSheet("background-color: #1E1E1E; color: white;")
        def add_thumbnail_item(self, path): 
            self.addItem(os.path.basename(path))
        def clear(self): 
            super().clear()

# Try to import label components
try:
    from frontend.components.labels import BrandingLabel, VersionLabel, PoweredByLabel
except ImportError:
    # Create placeholder labels
    class BrandingLabel(QLabel):
        def __init__(self):
            super().__init__("MeshBuilder")
            self.setStyleSheet("color: #00ffff; font-size: 24px; font-weight: bold;")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    class VersionLabel(QLabel):
        def __init__(self, version):
            super().__init__(f"Version {version}")
            self.setStyleSheet("color: white; font-size: 12px;")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    class PoweredByLabel(QLabel):
        def __init__(self):
            super().__init__("Powered by Immersive Engineering")
            self.setStyleSheet("color: #888888; font-size: 10px;")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class MainWindowCompatMixin:
    """Mixin class to add compatibility methods"""
    
    def open_upload_option_dialog(self):
        """Open dialog for new project options"""
        try:
            from frontend.components.dialogs import UploadOptionDialog
            dialog = UploadOptionDialog(self)
            dialog.start_new_project_signal.connect(self.start_new_project)
            dialog.upload_existing_project_signal.connect(self.upload_images)
            dialog.exec()
        except ImportError:
            # Fallback behavior
            self.import_images()
    
    def start_new_project(self):
        """Start a new project (called from upload dialog)"""
        self.new_project()
    
    def upload_images(self):
        """Upload images to the project (called from upload dialog)"""
        self.import_images()
    
    def toggle_image_list_sidebar(self):
        """Toggle image list sidebar visibility"""
        if hasattr(self, 'image_list_dock') and self.image_list_dock.isVisible():
            self.image_list_dock.hide()
        else:
            self.image_list_dock.setFloating(True)
            
            main_width = self.width()
            main_height = self.height()
            dock_width = self.image_list_dock.width()
            dock_height = self.image_list_dock.height()
            
            x_position = (main_width - dock_width) // 2
            y_position = (main_height - dock_height) // 2
            
            self.image_list_dock.move(x_position, y_position)
            self.image_list_dock.show()
    
    def toggle_settings_sidebar(self):
        """Toggle settings sidebar visibility"""
        if hasattr(self, 'settings_dock') and self.settings_dock.isVisible():
            self.settings_dock.hide()
        else:
            self.settings_dock.show()


class MeshbuilderMainWindow(MainWindowCompatMixin, QMainWindow):
    """Main window for the Meshbuilder application"""
    
    def __init__(self):
        super().__init__()
        
        self.logger = logging.getLogger("MeshBuilder.UI")
        
        # Initialize state variables
        self.current_project = None
        self.is_processing = False
        self.output_path = ""
        
        # Initialize enhanced backend interface
        if CONNECTOR_AVAILABLE:
            self.interface = MeshBuilderConnector(self)
            self.logger.info("Enhanced connector initialized")
        else:
            # Fallback to dummy connector
            self.interface = MeshBuilderConnector(self)
            self.logger.warning("Using fallback connector")
        
        # Initialize project manager
        self.project_manager = ProjectManager()
        
        # Set up the UI
        self.setup_ui()
        
        # Connect signals AFTER interface is created
        self.connect_signals()
        
        # Show initial message
        self.add_log_message("Enhanced MeshBuilder initialized. Ready to process images.")
    
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("MeshBuilder")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet("background-color: #1E1E1E;")
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Add project name label in top right corner
        self.project_name_label = QLabel("File Name- No Project")
        self.project_name_label.setStyleSheet("color: #00ffff; font-size: 14px; font-weight: bold;")
        self.project_name_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.main_layout.addWidget(self.project_name_label)
        
        # Header section
        header_layout = QVBoxLayout()
        
        brand = BrandingLabel()
        version = VersionLabel("v0.1.0")
        powered_by = PoweredByLabel()
        header_layout.addWidget(brand)
        header_layout.addWidget(version)
        header_layout.addWidget(powered_by)
        header_layout.addStretch()
        
        # Create middle section with 3D viewer
        middle_layout = QHBoxLayout()
        
        # Create a container for the model viewer
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        
        # Add model viewer
        self.model_viewer = ModelViewer()
        viewer_layout.addWidget(self.model_viewer)
        
        middle_layout.addWidget(viewer_container)
        
        # Add layouts to main layout
        self.main_layout.addLayout(header_layout)
        self.main_layout.addLayout(middle_layout)
        
        # Add processing controls at the BOTTOM
        bottom_controls = QVBoxLayout()
        
        # Create a horizontal layout for buttons and progress bar
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        controls_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(QIcon.fromTheme("process-stop"))
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.cancel_button.setEnabled(False)
        controls_layout.addWidget(self.cancel_button)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                text-align: center;
                background-color: #2E2E2E;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.progress_bar.setValue(0)
        controls_layout.addWidget(self.progress_bar)
        
        bottom_controls.addLayout(controls_layout)
        
        # Add status label BELOW the progress bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: white; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_controls.addWidget(self.status_label)
        
        self.main_layout.addLayout(bottom_controls)
        
        # Create status bar with progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_progress_bar = QProgressBar()
        self.status_progress_bar.setTextVisible(True)
        self.status_progress_bar.setRange(0, 100)
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.status_progress_bar)
        
        self.status_text = QLabel("Ready")
        self.status_bar.addWidget(self.status_text)
        
        # Set up dock widgets
        self.setup_dock_widgets()
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)
        
        # File menu
        file_menu = self.menu_bar.addMenu("&File")
        
        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        self.recent_projects_menu = QMenu("Recent Projects", self)
        self.update_recent_projects_menu()
        file_menu.addMenu(self.recent_projects_menu)
        
        file_menu.addSeparator()
        
        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        export_format_menu = QMenu("Export Format", self)
        
        obj_format_action = QAction("OBJ", self)
        obj_format_action.triggered.connect(lambda: self.set_export_format("obj"))
        export_format_menu.addAction(obj_format_action)
        
        ply_format_action = QAction("PLY", self)
        ply_format_action.triggered.connect(lambda: self.set_export_format("ply"))
        export_format_menu.addAction(ply_format_action)
        
        fbx_format_action = QAction("FBX (Coming Soon)", self)
        fbx_format_action.setEnabled(False)
        export_format_menu.addAction(fbx_format_action)
        
        gltf_format_action = QAction("GLTF (Coming Soon)", self)
        gltf_format_action.setEnabled(False)
        export_format_menu.addAction(gltf_format_action)
        
        file_menu.addMenu(export_format_menu)
        
        file_menu.addSeparator()
        
        import_images_action = QAction("Import &Images", self)
        import_images_action.triggered.connect(self.import_images)
        file_menu.addAction(import_images_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = self.menu_bar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo_action)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        crop_model_action = QAction("&Crop Model", self)
        crop_model_action.triggered.connect(self.crop_model)
        edit_menu.addAction(crop_model_action)
        
        delete_selection_action = QAction("&Delete Selection", self)
        delete_selection_action.setShortcut("Delete")
        delete_selection_action.triggered.connect(self.delete_selection)
        edit_menu.addAction(delete_selection_action)
        
        refine_mesh_action = QAction("&Refine Mesh", self)
        refine_mesh_action.triggered.connect(self.refine_mesh)
        edit_menu.addAction(refine_mesh_action)
        
        # View menu
        view_menu = self.menu_bar.addMenu("&View")
        
        view_360_action = QAction("&360° View", self)
        view_360_action.triggered.connect(self.view_360)
        view_menu.addAction(view_360_action)
        
        preview_action = QAction("&Preview", self)
        preview_action.triggered.connect(self.show_preview)
        view_menu.addAction(preview_action)
        
        view_menu.addSeparator()
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Settings menu
        settings_menu = self.menu_bar.addMenu("&Settings")
        
        show_logs_action = QAction("Show &Logs", self)
        show_logs_action.triggered.connect(self.show_logs)
        settings_menu.addAction(show_logs_action)
        
        image_list_action = QAction("Show &Image List", self)
        image_list_action.triggered.connect(self.toggle_image_list_sidebar)
        settings_menu.addAction(image_list_action)
        
        start_processing_action = QAction("&Start Processing", self)
        start_processing_action.triggered.connect(self.start_processing)
        settings_menu.addAction(start_processing_action)
        
        cancel_processing_action = QAction("&Cancel Processing", self)
        cancel_processing_action.triggered.connect(self.cancel_processing)
        settings_menu.addAction(cancel_processing_action)
        
        settings_menu.addSeparator()
        
        project_settings_action = QAction("&Project Settings", self)
        project_settings_action.triggered.connect(self.show_settings_dialog)
        settings_menu.addAction(project_settings_action)
        
        screenshot_action = QAction("Take &Screenshot", self)
        screenshot_action.triggered.connect(self.take_screenshot)
        settings_menu.addAction(screenshot_action)
        
        # Help menu
        help_menu = self.menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        contact_action = QAction("&Contact Us", self)
        contact_action.triggered.connect(self.show_contact)
        help_menu.addAction(contact_action)
        
        tutorial_action = QAction("&Tutorial", self)
        tutorial_action.setEnabled(False)
        tutorial_action.setToolTip("Coming Soon")
        help_menu.addAction(tutorial_action)
        
        help_menu.addSeparator()
        
        check_updates_action = QAction("&Check for Updates", self)
        check_updates_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(check_updates_action)
        
        user_guide_action = QAction("&User Guide", self)
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)
        
        report_bug_action = QAction("&Report Bug", self)
        report_bug_action.triggered.connect(self.report_bug)
        help_menu.addAction(report_bug_action)
    
    def setup_dock_widgets(self):
        """Set up the dock widgets"""
        # Left sidebar dock
        self.left_sidebar_dock = QDockWidget("", self)
        self.left_sidebar_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.left_sidebar_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.left_sidebar_dock.setTitleBarWidget(QWidget())
        
        self.left_sidebar = EnhancedLeftSidebar(self)
        self.left_sidebar_dock.setWidget(self.left_sidebar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_sidebar_dock)
        
        # Connect left sidebar signals
        self.left_sidebar.newProjectClicked.connect(self.new_project)
        self.left_sidebar.uploadImagesClicked.connect(self.import_images)
        self.left_sidebar.viewUploadedImagesClicked.connect(self.toggle_image_list_sidebar)
        self.left_sidebar.generateModelClicked.connect(self.start_processing)
        self.left_sidebar.cancelProcessingClicked.connect(self.cancel_processing)
        self.left_sidebar.cropModelClicked.connect(self.crop_model)
        self.left_sidebar.takeScreenshotClicked.connect(self.take_screenshot)
        self.left_sidebar.selectPortionClicked.connect(self.select_portion)
        self.left_sidebar.deleteSelectionClicked.connect(self.delete_selection)
        self.left_sidebar.refineMeshClicked.connect(self.refine_mesh)
        self.left_sidebar.browseOutputClicked.connect(self.browse_output_path)
        
        # Image list dock
        self.image_list_dock = QDockWidget("Imported Images", self)
        self.image_list_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | 
                                           Qt.DockWidgetArea.LeftDockWidgetArea)
        self.image_list_view = ImageListView()
        self.image_list_dock.setWidget(self.image_list_view)
        
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.image_list_dock)
        self.image_list_dock.hide()
        
        self.image_list_dock.setFixedWidth(300)
        self.image_list_dock.setFixedHeight(600)
        
        # Log window
        self.log_window = LogWindow()
    
    def connect_signals(self):
        """Connect signals from the enhanced backend interface"""
        if not CONNECTOR_AVAILABLE:
            self.add_log_message("Warning: Enhanced connector not available, using fallback mode")
            return
        
        try:
            # Basic processing signals - these should always be available
            if hasattr(self.interface, 'processingStarted'):
                self.interface.processingStarted.connect(self.on_processing_started)
            if hasattr(self.interface, 'processingFinished'):
                self.interface.processingFinished.connect(self.on_processing_finished)
            if hasattr(self.interface, 'processingError'):
                self.interface.processingError.connect(self.on_processing_error)
            if hasattr(self.interface, 'progressUpdated'):
                self.interface.progressUpdated.connect(self.on_progress_updated)
            if hasattr(self.interface, 'stageChanged'):
                self.interface.stageChanged.connect(self.on_stage_changed)
            
            # Enhanced signals - these are optional
            if hasattr(self.interface, 'previewReady'):
                self.interface.previewReady.connect(self.on_preview_ready)
            if hasattr(self.interface, 'resourceUpdate'):
                self.interface.resourceUpdate.connect(self.on_resource_update)
            if hasattr(self.interface, 'memoryOptimization'):
                self.interface.memoryOptimization.connect(self.on_memory_optimization)
            if hasattr(self.interface, 'licenseValidationRequired'):
                self.interface.licenseValidationRequired.connect(self.on_license_validation_required)
            if hasattr(self.interface, 'systemSpecsDetected'):
                self.interface.systemSpecsDetected.connect(self.on_system_specs_detected)
                
            self.add_log_message("Enhanced connector signals connected successfully")
            
        except AttributeError as e:
            self.add_log_message(f"Warning: Some enhanced interface signals not available: {e}")
        except Exception as e:
            self.add_log_message(f"Error connecting signals: {e}")
    
    # Enhanced signal handlers
    def on_system_specs_detected(self, specs):
        """Handle system specifications detection"""
        try:
            memory_gb = specs.get('total_memory_gb', 0)
            quality_level = specs.get('quality_level', 'medium')
            gpu_available = specs.get('gpu_available', False)
            
            gpu_text = f", GPU: {specs.get('gpu_name', 'Available')}" if gpu_available else ", GPU: Not available"
            
            self.add_log_message(f"System detected: {memory_gb:.1f} GB RAM{gpu_text}")
            self.add_log_message(f"Recommended quality: {quality_level}")
            
            # Update status bar with system info
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"System: {memory_gb:.1f}GB RAM, Quality: {quality_level}", 5000)
                
        except Exception as e:
            self.add_log_message(f"Error handling system specs: {e}")
    
    def on_resource_update(self, resource_data):
        """Handle resource usage updates"""
        try:
            memory_usage = resource_data.get('memory_usage_percent', 0)
            cpu_usage = resource_data.get('cpu_usage_percent', 0)
            
            # Only log if usage is high
            if memory_usage > 80:
                self.add_log_message(f"High memory usage: {memory_usage:.1f}%")
            if cpu_usage > 90:
                self.add_log_message(f"High CPU usage: {cpu_usage:.1f}%")
                
        except Exception as e:
            self.add_log_message(f"Error handling resource update: {e}")
    
    def on_memory_optimization(self, message):
        """Handler for memory optimization signal"""
        try:
            self.add_log_message(f"Memory optimization: {message}")
            
            # Show warning dialog for critical memory issues
            if "warning" in message.lower() or "high" in message.lower():
                QMessageBox.warning(self, "Memory Warning", message)
                
        except Exception as e:
            self.add_log_message(f"Error handling memory optimization: {e}")
    
    def on_processing_started(self):
        """Handle processing started"""
        self.is_processing = True
        self.update_ui()
        self.add_log_message("Enhanced processing started...")
    
    def on_processing_finished(self, success, result_path):
        """Handle processing finished"""
        self.is_processing = False
        self.update_ui()
        
        if success:
            self.add_log_message(f"Enhanced processing completed successfully: {result_path}")
            self.show_info("Processing Complete", f"Model saved to: {result_path}")
            
            # Try to load the model in the viewer
            try:
                if hasattr(self, 'model_viewer') and self.model_viewer.load_model(result_path):
                    self.add_log_message("3D model loaded in viewer")
                    self.model_viewer.setFocus()
            except Exception as e:
                self.add_log_message(f"Could not load model in viewer: {e}")
        else:
            self.add_log_message(f"Enhanced processing failed: {result_path}")
    
    def on_processing_error(self, title, message):
        """Handle processing error"""
        self.is_processing = False
        self.update_ui()
        self.add_log_message(f"Error: {message}")
        self.show_error(message, title)
    
    def on_progress_updated(self, message, percentage):
        """Handle progress update"""
        self.progress_bar.setValue(percentage)
        self.status_progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.status_text.setText(message)
        self.add_log_message(f"Progress: {message} ({percentage}%)")
    
    def on_stage_changed(self, stage_id, description):
        """Handle stage change"""
        self.add_log_message(f"Stage: {description}")
    
    def on_preview_ready(self, preview_type, path):
        """Handle preview ready"""
        self.add_log_message(f"Preview available: {preview_type} - {path}")
    
    def on_license_validation_required(self, message):
        """Handle license validation required"""
        QMessageBox.information(self, "License", message)
    
    def update_ui(self):
        """Update UI state based on processing status"""
        self.start_button.setEnabled(not self.is_processing and self.current_project is not None)
        self.cancel_button.setEnabled(self.is_processing)
        
        if not self.is_processing:
            self.progress_bar.setValue(0)
            self.status_progress_bar.setValue(0)
            self.status_label.setText("Ready")
            self.status_text.setText("Ready")
    
    def add_log_message(self, message):
        """Add message to log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        if hasattr(self, 'log_window'):
            self.log_window.append_log(formatted_message)
        
        print(formatted_message)  # Also print to console
    
    def show_error(self, message, title="Error"):
        """Show error message"""
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title, message):
        """Show information message"""
        QMessageBox.information(self, title, message)
    
    def set_export_format(self, format_str):
        """Set the export format"""
        if self.current_project and hasattr(self.current_project, 'output_format'):
            format_lower = format_str.lower()
            
            supported_formats = ["obj", "ply"]
            if format_lower not in supported_formats:
                QMessageBox.warning(
                    self, 
                    "Unsupported Format", 
                    f"{format_str.upper()} export is not yet supported. Using OBJ format instead."
                )
                format_lower = "obj"
            
            self.current_project.output_format = format_lower
            
            if self.current_project.output_path:
                base_path = os.path.splitext(self.current_project.output_path)[0]
                new_path = f"{base_path}.{format_lower}"
                self.current_project.set_output_path(new_path, format_lower)
                self.output_path = new_path
                
            self.add_log_message(f"Set export format to {format_lower.upper()}")
    
    def refine_mesh(self):
        """Refine mesh functionality"""
        if not self.current_project or not hasattr(self.current_project, 'model_path') or not self.current_project.model_path:
            QMessageBox.information(self, "No Model", "No 3D model available to refine")
            return
            
        QMessageBox.information(
            self, 
            "Feature Coming Soon", 
            "The Refine Mesh feature is currently under development and will be available in the next version.\n\n"
            "This feature will allow you to enhance mesh quality by:\n"
            "- Improving polygon distribution\n"
            "- Smoothing rough areas\n"
            "- Increasing detail in specific regions"
        )
        self.add_log_message("Refine mesh feature will be available in the next version")
    
    def new_project(self):
        """Create a new project with enhanced connector"""
        name, ok = QInputDialog.getText(self, "New Project", "Project Name:")
        
        if ok and name:
            try:
                # Validate project name
                if not name.strip():
                    QMessageBox.warning(self, "Invalid Name", "Project name cannot be empty")
                    return
                    
                if self.interface.create_project(name.strip(), "high"):
                    project_info = self.interface.get_project_info()
                    if project_info:
                        self.current_project = self.interface.current_project
                        self.project_name_label.setText(f"File Name- {name}")
                        
                        # Clear image list
                        if hasattr(self, 'image_list_view'):
                            self.image_list_view.clear()
                        
                        self.progress_bar.setValue(0)
                        self.update_ui()
                        
                        self.add_log_message(f"Created new project: {name}")
                        self.status_label.setText(f"Created new project: {name}")
                        
                        # Set default output path
                        default_output = os.path.join(os.path.expanduser("~"), "Documents", f"{name}.obj")
                        self.output_path = default_output
                        
                        self.update_recent_projects_menu()
                        
                        # Get system recommendations
                        try:
                            recommendations = self.interface.get_system_recommendations()
                            if recommendations:
                                rec_quality = recommendations.get("recommended_quality", "high")
                                if rec_quality != "high":
                                    self.add_log_message(f"System recommendation: {rec_quality} quality level for optimal performance")
                        except Exception as e:
                            self.add_log_message(f"Could not get system recommendations: {e}")
                            
                    else:
                        QMessageBox.warning(self, "Error", "Failed to get project information")
                else:
                    QMessageBox.warning(self, "Error", "Failed to create project")
                    
            except Exception as e:
                self.add_log_message(f"Error creating project: {e}")
                QMessageBox.warning(self, "Error", f"Failed to create project: {str(e)}")
    
    def open_project(self):
        """Open an existing project"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Open Project")
        dialog.resize(400, 300)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2E2E2E;
            }
            QLabel {
                color: #FFFFFF;
            }
            QListWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #555555;
            }
            QPushButton {
                background-color: #444444;
                color: #FFFFFF;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        projects_list = QListWidget()
        layout.addWidget(QLabel("Select a project:"))
        layout.addWidget(projects_list)
        
        projects = self.project_manager.list_projects()
        for project in projects:
            projects_list.addItem(project)
        
        button_layout = QHBoxLayout()
        open_btn = QPushButton("Open")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(open_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        open_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = projects_list.currentItem()
            if selected:
                project_name = selected.text()
                self.load_project(project_name)
    
    def load_project(self, project_name):
        """Load a project by name with enhanced connector"""
        try:
            if self.interface.load_project(project_name):
                project_info = self.interface.get_project_info()
                if project_info:
                    self.current_project = self.interface.current_project
                    self.project_name_label.setText(f"File Name- {project_info['name']}")
                    
                    if hasattr(self, 'image_list_view'):
                        self.image_list_view.clear()
                        if self.current_project and hasattr(self.current_project, 'image_paths'):
                            for img_path in self.current_project.image_paths:
                                self.image_list_view.add_thumbnail_item(img_path)
                            
                            if len(self.current_project.image_paths) > 0:
                                self.toggle_image_list_sidebar()
                    
                    if project_info.get('output_path'):
                        self.output_path = project_info['output_path']
                    
                    self.add_log_message(f"Opened project: {project_info['name']}")
                    self.status_label.setText(f"Opened project: {project_info['name']}")
                    
                    # Log system specs if available
                    system_specs = project_info.get('system_specs')
                    if system_specs:
                        memory_gb = system_specs.get('total_memory_gb', 0)
                        self.add_log_message(f"System: {memory_gb:.1f} GB RAM available")
                    
                    self.update_recent_projects_menu()
                    
                    # Load model if available
                    if (self.current_project and hasattr(self.current_project, 'model_path') 
                        and self.current_project.model_path and os.path.exists(self.current_project.model_path)):
                        try:
                            self.add_log_message(f"Loading 3D model: {os.path.basename(self.current_project.model_path)}")
                            if hasattr(self, 'model_viewer'):
                                if self.model_viewer.load_model(self.current_project.model_path):
                                    self.add_log_message("3D model loaded successfully")
                                    self.model_viewer.setFocus()
                                else:
                                    self.add_log_message("Failed to load 3D model. Check that required libraries are installed.")
                        except Exception as e:
                            self.add_log_message(f"Error loading 3D model: {str(e)}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to load project: {project_name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load project: {str(e)}")
    
    def save_project(self):
        """Save the current project"""
        if self.current_project:
            try:
                if self.project_manager.save_project(self.current_project):
                    self.add_log_message(f"Saved project: {self.current_project.name}")
                    self.status_label.setText(f"Saved project: {self.current_project.name}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save project")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save project: {str(e)}")
        else:
            QMessageBox.information(self, "No Project", "No project is currently open")
    
    def save_project_as(self):
        """Save the current project with a new name"""
        if not self.current_project:
            QMessageBox.information(self, "No Project", "No project is currently open")
            return
        
        name, ok = QInputDialog.getText(
            self, "Save Project As", "New Project Name:", 
            text=f"{self.current_project.name} - Copy"
        )
        
        if ok and name:
            try:
                duplicate = self.project_manager.duplicate_project(self.current_project.name, name)
                
                if duplicate:
                    self.current_project = duplicate
                    self.project_name_label.setText(f"File Name- {name}")
                    
                    self.add_log_message(f"Saved project as: {name}")
                    self.status_label.setText(f"Saved project as: {name}")
                    
                    self.update_recent_projects_menu()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to save project as: {name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save project as: {str(e)}")
    
    def update_recent_projects_menu(self):
        """Update the Recent Projects menu"""
        if hasattr(self, 'recent_projects_menu'):
            self.recent_projects_menu.clear()
            
            try:
                projects = self.project_manager.list_projects()
                if not projects:
                    no_projects_action = QAction("No Recent Projects", self)
                    no_projects_action.setEnabled(False)
                    self.recent_projects_menu.addAction(no_projects_action)
                else:
                    project_details = self.project_manager.get_projects_with_details()
                    project_details = sorted(
                        project_details, 
                        key=lambda p: p.get("modified_date", ""), 
                        reverse=True
                    )
                    
                    for i, project in enumerate(project_details[:5]):
                        name = project.get("name", "Unknown")
                        action = QAction(name, self)
                        action.triggered.connect(lambda checked, n=name: self.load_project(n))
                        self.recent_projects_menu.addAction(action)
            except Exception as e:
                self.add_log_message(f"Error updating recent projects menu: {e}")
    
    def import_images(self):
        """Import images into the project with enhanced connector"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            files = file_dialog.selectedFiles()
            
            if not self.current_project:
                self.new_project()
            
            if self.current_project:
                try:
                    result = self.interface.add_media_files(files)
                    
                    for file_path in files:
                        if hasattr(self, 'image_list_view'):
                            self.image_list_view.add_thumbnail_item(file_path)
                    
                    if result["total"] > 0:
                        self.toggle_image_list_sidebar()
                    
                    self.add_log_message(f"Added {result['total']} files: {result['images']} images, {result['videos']} videos")
                    self.status_label.setText(f"Added {result['total']} media files")
                    
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to add media files: {str(e)}")
    
    def browse_output_path(self):
        """Browse for output path"""
        format_ext = "obj"
        
        if self.current_project and hasattr(self.current_project, 'output_format'):
            format_ext = self.current_project.output_format
        
        project_name = self.current_project.name if self.current_project and hasattr(self.current_project, 'name') else "untitled"
        
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilter(f"3D Models (*.{format_ext})")
        file_dialog.selectFile(f"{project_name}.{format_ext}")
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            output_path = file_dialog.selectedFiles()[0]
            
            if not output_path.lower().endswith(f".{format_ext}"):
                output_path += f".{format_ext}"
            
            if self.current_project and hasattr(self.current_project, 'set_output_path'):
                self.current_project.set_output_path(output_path, format_ext)
            
            self.output_path = output_path
            self.add_log_message(f"Set output path: {output_path}")
    
    def start_processing(self):
        """Start processing with enhanced connector"""
        if not self.current_project:
            QMessageBox.information(self, "No Project", "Please create or open a project first")
            return
        
        try:
            # Validate project has media files
            project_info = self.interface.get_project_info()
            if not project_info or project_info['media_count']['total'] == 0:
                QMessageBox.warning(self, "No Input Files", "Please add images or videos first")
                return
                
            # Validate output path
            if not self.output_path:
                self.browse_output_path()
                if not self.output_path:
                    return
            
            # Get settings from current project or defaults
            settings = self.get_settings_from_ui()
            
            # Log processing start
            self.add_log_message(f"Starting processing with {project_info['media_count']['total']} media files")
            self.add_log_message(f"Quality level: {project_info.get('quality_level', 'unknown')}")
            self.add_log_message(f"Output path: {self.output_path}")
            
            # Start processing
            success = self.interface.start_processing(self.output_path, settings)
            
            if not success:
                QMessageBox.warning(self, "Processing Error", "Failed to start enhanced processing")
            else:
                self.add_log_message("Started enhanced processing pipeline")
                
        except Exception as e:
            self.add_log_message(f"Error starting processing: {e}")
            QMessageBox.warning(self, "Processing Error", f"Failed to start processing: {str(e)}")
    
    def get_settings_from_ui(self):
        """Get settings from UI elements for enhanced connector"""
        settings = {}
        
        # Get quality level from current project
        quality_level = "high"
        if self.current_project and hasattr(self.current_project, 'quality_level'):
            quality_level = self.current_project.quality_level
        
        # Validate quality level
        valid_qualities = ["fast", "balanced", "high", "very_high", "photorealistic"]
        if quality_level not in valid_qualities:
            quality_level = "high"
        
        # Get export format
        export_format = "obj"
        if self.current_project and hasattr(self.current_project, 'output_format'):
            export_format = self.current_project.output_format
            if export_format not in ["obj", "ply"]:
                export_format = "obj"
        
        # Build settings dictionary
        settings["Processing"] = {
            "quality_level": quality_level,
            "quality_threshold": 0.5,
            "color_enhancement_mode": "balanced",
            "feature_matcher": "exhaustive",
            "mesh_resolution": "high",
            "texture_resolution": "4096",
            "target_faces": "100000",
            "flat_grey_model": "false",
            "use_deep_enhancement": False
        }
        
        settings["Hardware"] = {
            "use_gpu": "true",
            "use_gaussian_splatting": "false"
        }
        
        settings["Output"] = {
            "format": export_format,
            "compress_output": "false"
        }
        
        return settings
    
    def cancel_processing(self):
        """Cancel ongoing processing with enhanced connector"""
        try:
            status = self.interface.get_processing_status()
            if status["is_processing"]:
                reply = QMessageBox.question(
                    self, 
                    "Cancel Processing", 
                    "Are you sure you want to cancel the current enhanced processing job?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.interface.cancel_processing()
                    self.add_log_message("Enhanced processing cancelled by user")
        except Exception as e:
            self.add_log_message(f"Error cancelling enhanced processing: {e}")
    
    def show_settings_dialog(self):
        """Show enhanced settings dialog"""
        current_settings = {}
        
        if self.current_project and hasattr(self.current_project, 'settings'):
            current_settings = self.current_project.settings
        
        dialog = SettingsDialog(self, current_settings)
        dialog.settingsUpdated.connect(self.update_settings)
        dialog.exec()
    
    def update_settings(self, settings):
        """Update settings with enhanced connector"""
        try:
            self.interface.update_settings(settings)
            
            if self.current_project and hasattr(self.current_project, 'update_settings'):
                self.current_project.update_settings(settings)
            
            self.add_log_message("Updated enhanced project settings")
        except Exception as e:
            self.add_log_message(f"Error updating enhanced settings: {e}")
    
    def show_logs(self):
        """Show the log window as a dialog"""
        log_dialog = QDialog(self)
        log_dialog.setWindowTitle("Enhanced Processing Logs")
        log_dialog.resize(700, 500)
        
        layout = QVBoxLayout(log_dialog)
        layout.addWidget(self.log_window)
        
        log_dialog.exec()
    
    def crop_model(self):
        """Crop the model functionality"""
        if not self.current_project or not hasattr(self.current_project, 'model_path') or not self.current_project.model_path:
            QMessageBox.information(self, "No Model", "No 3D model available to crop")
            return
            
        QMessageBox.information(
            self, 
            "Feature Coming Soon", 
            "The Crop Model feature is currently under development and will be available in the next version.\n\n"
            "This feature will allow you to:\n"
            "- Define a custom bounding box\n"
            "- Trim unwanted portions of the model\n"
            "- Focus on specific areas of interest"
        )
        self.add_log_message("Crop model feature will be available in the next version")
    
    def select_portion(self):
        """Select portion of model functionality"""
        if not self.current_project or not hasattr(self.current_project, 'model_path') or not self.current_project.model_path:
            QMessageBox.information(self, "No Model", "No 3D model available to select from")
            return
            
        QMessageBox.information(
            self, 
            "Feature Coming Soon", 
            "The Select Portion feature is currently under development and will be available in the next version.\n\n"
            "This feature will allow you to:\n"
            "- Select specific parts of your model\n"
            "- Create selections for further editing\n"
            "- Isolate distinct components"
        )
        self.add_log_message("Select portion feature will be available in the next version")
    
    def delete_selection(self):
        """Delete selected portion of model functionality"""
        if not self.current_project or not hasattr(self.current_project, 'model_path') or not self.current_project.model_path:
            QMessageBox.information(self, "No Model", "No 3D model available")
            return
            
        QMessageBox.information(
            self, 
            "Feature Coming Soon", 
            "The Delete Selection feature is currently under development and will be available in the next version.\n\n"
            "This feature will allow you to:\n"
            "- Remove selected portions of the model\n"
            "- Clean up unwanted artifacts\n"
            "- Refine your 3D model"
        )
        self.add_log_message("Delete selection feature will be available in the next version")
    
    def undo_action(self):
        """Undo the last action"""
        self.add_log_message("Undo feature will be implemented in a future update")
    
    def redo_action(self):
        """Redo the previously undone action"""
        self.add_log_message("Redo feature will be implemented in a future update")
    
    def view_360(self):
        """Show 360-degree view of the model"""
        if not self.current_project or not hasattr(self.current_project, 'model_path') or not self.current_project.model_path:
            QMessageBox.information(self, "No Model", "No 3D model available to view")
            return
            
        model_path = self.current_project.model_path
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found", f"3D model file not found: {model_path}")
            return
        
        if not hasattr(self.model_viewer, 'current_model') or self.model_viewer.current_model != model_path:
            self.add_log_message(f"Loading 3D model: {os.path.basename(model_path)}")
            if not self.model_viewer.load_model(model_path):
                QMessageBox.warning(self, "Loading Error", "Failed to load 3D model. Check that PyQtGraph and trimesh are installed.")
                return
        
        if hasattr(self.model_viewer, '_orbit_running') and self.model_viewer._orbit_running:
            self.model_viewer.auto_orbit(False)
            self.add_log_message("Stopped 360° rotation")
        else:
            self.model_viewer.auto_orbit(True)
            self.add_log_message("Started 360° rotation")
            
        self.model_viewer.setFocus()
    
    def show_preview(self):
        """Show preview of the model"""
        if not self.current_project or not hasattr(self.current_project, 'previews') or not self.current_project.previews:
            QMessageBox.information(self, "No Preview", "No previews available")
            return
        
        preview_types = list(self.current_project.previews.keys())
        if not preview_types:
            return
            
        priority_order = ["textured", "mesh", "dense", "sparse"]
        selected_preview = None
        
        for preview_type in priority_order:
            if preview_type in self.current_project.previews:
                selected_preview = self.current_project.previews[preview_type]
                break
        
        if not selected_preview or not os.path.exists(selected_preview):
            QMessageBox.information(self, "Preview Not Found", "Preview file not found")
            return
            
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Model Preview")
        preview_dialog.resize(800, 600)
        
        layout = QVBoxLayout(preview_dialog)
        
        pixmap = QPixmap(selected_preview)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                preview_dialog.width() - 40, 
                preview_dialog.height() - 40,
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            preview_label = QLabel()
            preview_label.setPixmap(pixmap)
            preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(preview_label)
            
            preview_dialog.exec()
    
    def zoom_in(self):
        """Zoom in on the model view"""
        if not hasattr(self, 'model_viewer'):
            self.add_log_message("Model viewer not initialized")
            return
            
        if not self.model_viewer.current_model:
            QMessageBox.information(self, "No Model", "No 3D model is currently loaded")
            return
            
        self.model_viewer.zoom_in()
        self.add_log_message("Zoomed in on model")
        self.model_viewer.setFocus()
    
    def zoom_out(self):
        """Zoom out on the model view"""
        if not hasattr(self, 'model_viewer'):
            self.add_log_message("Model viewer not initialized")
            return
            
        if not self.model_viewer.current_model:
            QMessageBox.information(self, "No Model", "No 3D model is currently loaded")
            return
            
        self.model_viewer.zoom_out()
        self.add_log_message("Zoomed out from model")
        self.model_viewer.setFocus()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About MeshBuilder",
            "MeshBuilder Enhanced\n\n"
            "A photorealistic 3D reconstruction application\n"
            "Powered by 3D Gaussian Splatting technology\n\n"
            "Version: 0.1.0\n"
            "Build Date: April 2025\n\n"
            "© 2025 Immersive Engineering"
        )
    
    def show_contact(self):
        """Show contact dialog"""
        dialog = ContactDialog(self)
        dialog.exec()
    
    def check_for_updates(self):
        """Check for application updates"""
        QMessageBox.information(
            self,
            "Check for Updates",
            "You are running the latest version (0.1.0)"
        )
    
    def show_user_guide(self):
        """Show user guide"""
        QMessageBox.information(
            self,
            "User Guide",
            "The user guide will be available in a future update."
        )
    
    def report_bug(self):
        """Report a bug"""
        QMessageBox.information(
            self,
            "Report Bug",
            "To report a bug, please contact us at: nick@immersive-engineering.com\n\n"
            "Please include details about the issue and steps to reproduce it."
        )
    
    def take_screenshot(self):
        """Take a screenshot of the 3D model view"""
        if not hasattr(self, 'model_viewer') or not self.model_viewer.current_model:
            # Fall back to taking a screenshot of the entire application
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(self.winId())
            
            image = screenshot.toImage()
            
            painter = QPainter(image)
            
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            
            watermark_text = "Generated by MeshBuilder"
            text_width = painter.fontMetrics().horizontalAdvance(watermark_text)
            painter.drawText(
                20,
                image.height() - 20,
                watermark_text
            )
            
            logo_path = os.path.join("assets", "ie_logo.png")
            if os.path.exists(logo_path):
                logo = QPixmap(logo_path)
                scaled_logo = logo.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                painter.drawPixmap(
                    image.width() - scaled_logo.width() - 20, 
                    image.height() - scaled_logo.height() - 20, 
                    scaled_logo
                )
            
            painter.end()
        else:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            if not self.model_viewer.take_screenshot(temp_path):
                QMessageBox.warning(self, "Screenshot Failed", "Failed to capture 3D model screenshot")
                return
            
            # Load the image
            screenshot = QPixmap(temp_path)
            image = screenshot.toImage()
            
            # Create a painter to add watermark
            painter = QPainter(image)
            
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            
            watermark_text = "Generated by MeshBuilder"
            painter.drawText(
                20,
                image.height() - 20,
                watermark_text
            )
            
            # Add logo if exists
            logo_path = os.path.join("assets", "ie_logo.png")
            if os.path.exists(logo_path):
                logo = QPixmap(logo_path)
                scaled_logo = logo.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                painter.drawPixmap(
                    image.width() - scaled_logo.width() - 20, 
                    image.height() - scaled_logo.height() - 20, 
                    scaled_logo
                )
            
            painter.end()
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Save screenshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"MeshBuilder_Screenshot_{timestamp}.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            default_filename,
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            if image.save(file_path):
                self.add_log_message(f"Screenshot saved: {file_path}")
                QMessageBox.information(self, "Screenshot Saved", f"Screenshot saved to: {file_path}")
            else:
                QMessageBox.warning(self, "Save Failed", "Failed to save screenshot")
    
    def closeEvent(self, event):
        """Handle application close with enhanced cleanup"""
        if self.is_processing:
            reply = QMessageBox.question(
                self, "Processing Active", 
                "Enhanced processing is still active. Do you want to cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.interface.cancel_processing()
                    self.add_log_message("Processing cancelled for application exit")
                except Exception as e:
                    self.add_log_message(f"Error cancelling processing: {e}")
                
                # Enhanced cleanup
                try:
                    if hasattr(self.interface, 'cleanup'):
                        self.interface.cleanup()
                        self.add_log_message("Enhanced connector cleanup completed")
                except Exception as e:
                    self.add_log_message(f"Error during cleanup: {e}")
                
                event.accept()
            else:
                event.ignore()
        else:
            # Enhanced cleanup on normal close
            try:
                if hasattr(self.interface, 'cleanup'):
                    self.interface.cleanup()
            except Exception as e:
                self.add_log_message(f"Error during cleanup: {e}")
            event.accept()


# Compatibility alias for main.py import
MainWindow = MeshbuilderMainWindow

# Main entry point for testing
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("MeshBuilder Enhanced")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Immersive Engineering")
    
    # Create and show main window
    try:
        window = MeshbuilderMainWindow()
        window.show()
        
        # Check connector availability and show status
        if CONNECTOR_AVAILABLE:
            window.add_log_message("Enhanced MeshBuilder Connector loaded successfully")
            try:
                # Test enhanced features
                recommendations = window.interface.get_system_recommendations()
                if recommendations:
                    quality_level = recommendations.get("quality_level", "medium")
                    window.add_log_message(f"System recommended quality: {quality_level}")
            except:
                pass
        else:
            window.add_log_message("Warning: Enhanced connector not available, using fallback mode")
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error starting enhanced application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)