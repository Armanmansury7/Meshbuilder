"""
Enhanced Progress Dialog for MeshBuilder
Displays detailed progress for photorealistic 3D reconstruction
"""
from PyQt6.QtWidgets import (
    QDialog, QProgressBar, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QScrollArea, QWidget, QGridLayout, QFrame,
    QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize, QTime
from PyQt6.QtGui import QIcon, QPixmap

class EnhancedProgressDialog(QDialog):
    """Progress dialog with detailed steps for photorealistic 3D processing"""
    
    cancelled = pyqtSignal()  # Signal when user cancels processing
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up window
        self.setWindowTitle("Processing 3D Model")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        
        # Create components
        self.create_header()
        self.create_progress_area()
        self.create_preview_area()
        self.create_buttons()
        
        # Define processing steps
        self.steps = {
            "preprocessing": {
                "name": "Image Preprocessing",
                "progress": 0,
                "status": "pending",
                "substeps": ["Loading images", "Quality checking", "Image enhancement"]
            },
            "sparse": {
                "name": "Structure from Motion",
                "progress": 0,
                "status": "pending",
                "substeps": ["Feature extraction", "Feature matching", "Camera calibration", "Sparse reconstruction"]
            },
            "dense": {
                "name": "Dense Reconstruction",
                "progress": 0,
                "status": "pending",
                "substeps": ["Image undistortion", "Depth map generation", "Point cloud fusion"]
            },
            "mesh": {
                "name": "Mesh Generation",
                "progress": 0,
                "status": "pending", 
                "substeps": ["Point cloud processing", "Surface reconstruction", "Mesh cleaning"]
            },
            "texture": {
                "name": "Texture Generation",
                "progress": 0,
                "status": "pending",
                "substeps": ["UV mapping", "Texture projection", "Initial texture baking"]
            },
            "enhancement": {
                "name": "Photorealistic Enhancement",
                "progress": 0,
                "status": "pending",
                "substeps": ["Color enhancement", "HDR tone mapping", "Texture optimization"]
            },
            "optimization": {
                "name": "Model Optimization",
                "progress": 0,
                "status": "pending",
                "substeps": ["Mesh simplification", "Texture packing", "Final cleanup"]
            },
            "export": {
                "name": "Export",
                "progress": 0,
                "status": "pending",
                "substeps": ["Format conversion", "Metadata generation", "File saving"]
            }
        }
        
        # Initialize UI elements for steps
        self.step_widgets = {}
        self.create_steps_ui()
        
        # Set initial status
        self.set_status("Ready to process", 0)
    
    def create_header(self):
        """Create the header section with main progress bar"""
        header_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Preparing...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_layout.addWidget(self.status_label)
        
        # Overall progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #aaa;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 1px;
            }
            """
        )
        header_layout.addWidget(self.progress_bar)
        
        # Time estimate
        self.time_label = QLabel("Estimated time remaining: calculating...")
        header_layout.addWidget(self.time_label)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        header_layout.addWidget(separator)
        
        # Add to main layout
        self.main_layout.addLayout(header_layout)
    
    def create_progress_area(self):
        """Create the area with detailed progress steps"""
        # Create scroll area for steps
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create container widget for steps
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(8)
        
        # Add container to scroll area
        scroll_area.setWidget(self.steps_container)
        
        # Add scroll area to main layout
        self.main_layout.addWidget(scroll_area, 1)  # Stretch factor 1
    
    def create_preview_area(self):
        """Create the area for previews"""
        # Create preview frame
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setStyleSheet("background-color: #f0f0f0;")
        preview_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preview_frame.setMinimumHeight(150)
        preview_frame.setMaximumHeight(150)
        
        preview_layout = QVBoxLayout(preview_frame)
        
        # Preview label
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        # Preview image holders in a horizontal layout
        preview_images_layout = QHBoxLayout()
        
        # Create placeholders for previews
        self.preview_labels = {}
        preview_types = ["sparse", "dense", "mesh", "textured"]
        
        for preview_type in preview_types:
            # Container for each preview with label
            preview_container = QVBoxLayout()
            
            # Image label
            image_label = QLabel()
            image_label.setFixedSize(QSize(100, 100))
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("background-color: #ddd; border: 1px solid #ccc;")
            image_label.setText(f"No {preview_type}\npreview yet")
            self.preview_labels[preview_type] = image_label
            
            # Type label
            type_label = QLabel(preview_type.capitalize())
            type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Add to container
            preview_container.addWidget(image_label)
            preview_container.addWidget(type_label)
            
            # Add to layout
            preview_images_layout.addLayout(preview_container)
        
        preview_layout.addLayout(preview_images_layout)
        
        # Add to main layout
        self.main_layout.addWidget(preview_frame)
    
    def create_buttons(self):
        """Create the bottom buttons"""
        button_layout = QHBoxLayout()
        
        # Add buttons
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        
        # Add stretch to push buttons to the right
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        
        # Add to main layout
        self.main_layout.addLayout(button_layout)
    
    def create_steps_ui(self):
        """Create UI elements for each processing step"""
        for step_id, step_data in self.steps.items():
            # Create step container
            step_frame = QFrame()
            step_frame.setFrameShape(QFrame.Shape.StyledPanel)
            step_frame.setStyleSheet(
                """
                QFrame {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background-color: #f8f8f8;
                }
                """
            )
            
            step_layout = QVBoxLayout(step_frame)
            step_layout.setContentsMargins(10, 10, 10, 10)
            
            # Header layout with name and status
            header_layout = QHBoxLayout()
            
            # Step name
            name_label = QLabel(step_data["name"])
            name_label.setStyleSheet("font-weight: bold;")
            header_layout.addWidget(name_label)
            
            # Status
            status_label = QLabel("Pending")
            status_label.setStyleSheet("color: #888;")
            status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            header_layout.addWidget(status_label)
            
            step_layout.addLayout(header_layout)
            
            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setStyleSheet(
                """
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 2px;
                    text-align: center;
                    height: 5px;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
                """
            )
            step_layout.addWidget(progress_bar)
            
            # Substeps
            substeps_layout = QVBoxLayout()
            substeps_layout.setContentsMargins(10, 5, 10, 0)
            substeps_layout.setSpacing(2)
            
            substep_labels = []
            for substep in step_data["substeps"]:
                label = QLabel(f"• {substep}")
                label.setStyleSheet("color: #888; font-size: 11px;")
                substeps_layout.addWidget(label)
                substep_labels.append(label)
            
            step_layout.addLayout(substeps_layout)
            
            # Save widgets for updating
            self.step_widgets[step_id] = {
                "frame": step_frame,
                "status": status_label,
                "progress": progress_bar,
                "substeps": substep_labels
            }
            
            # Add to main steps layout
            self.steps_layout.addWidget(step_frame)
    
    def update_progress(self, step_id, progress, current_substep=0):
        """
        Update progress for a specific step
        
        Args:
            step_id: Step identifier
            progress: Progress value (0-100)
            current_substep: Current substep index
        """
        if step_id not in self.step_widgets:
            return
        
        # Update step status
        widgets = self.step_widgets[step_id]
        
        if progress == 0:
            widgets["status"].setText("Pending")
            widgets["status"].setStyleSheet("color: #888;")
            widgets["frame"].setStyleSheet(
                """
                QFrame {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background-color: #f8f8f8;
                }
                """
            )
        elif progress < 100:
            widgets["status"].setText("In Progress")
            widgets["status"].setStyleSheet("color: #0066cc;")
            widgets["frame"].setStyleSheet(
                """
                QFrame {
                    border: 1px solid #0066cc;
                    border-radius: 4px;
                    background-color: #e6f2ff;
                }
                """
            )
        else:
            widgets["status"].setText("Completed")
            widgets["status"].setStyleSheet("color: #4CAF50;")
            widgets["frame"].setStyleSheet(
                """
                QFrame {
                    border: 1px solid #4CAF50;
                    border-radius: 4px;
                    background-color: #f0f9f0;
                }
                """
            )
        
        # Update progress bar
        widgets["progress"].setValue(progress)
        
        # Update substeps
        for i, label in enumerate(widgets["substeps"]):
            if i < current_substep:
                # Completed substep
                label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            elif i == current_substep:
                # Current substep
                label.setStyleSheet("color: #0066cc; font-size: 11px; font-weight: bold;")
            else:
                # Pending substep
                label.setStyleSheet("color: #888; font-size: 11px;")
        
        # Store progress in steps data
        self.steps[step_id]["progress"] = progress
        
        # Update overall progress
        self.update_overall_progress()
    
    def update_overall_progress(self):
        """Update overall progress based on individual steps"""
        # Define weights for each step
        weights = {
            "preprocessing": 5,
            "sparse": 15,
            "dense": 25,
            "mesh": 15,
            "texture": 15,
            "enhancement": 10,
            "optimization": 10,
            "export": 5
        }
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        weighted_progress = sum(
            self.steps[step_id]["progress"] * weight / 100
            for step_id, weight in weights.items()
        )
        
        overall_progress = int(weighted_progress * 100 / total_weight)
        
        # Update progress bar
        self.progress_bar.setValue(overall_progress)
        
        # Update time estimate (simple estimate)
        if overall_progress > 0:
            # This is a very simple estimate - in a real app, you'd track actual time
            # spent and calculate a more accurate estimate
            if not hasattr(self, 'start_time'):
                setattr(self, 'start_time', QTime.currentTime())
            
            elapsed = self.start_time.secsTo(QTime.currentTime())
            if elapsed > 0:
                total_estimated = elapsed * 100 / overall_progress
                remaining = total_estimated - elapsed
                
                # Format nicely
                if remaining < 60:
                    time_str = f"about {int(remaining)} seconds"
                elif remaining < 3600:
                    time_str = f"about {int(remaining / 60)} minutes"
                else:
                    time_str = f"about {int(remaining / 3600)} hours and {int((remaining % 3600) / 60)} minutes"
                
                self.time_label.setText(f"Estimated time remaining: {time_str}")
    
    def set_status(self, message, progress=None):
        """Set overall status message and optionally update progress"""
        self.status_label.setText(message)
        
        if progress is not None:
            self.progress_bar.setValue(progress)
    
    def update_preview(self, preview_type, image_path):
        """
        Update preview image
        
        Args:
            preview_type: Type of preview (sparse, dense, mesh, textured)
            image_path: Path to preview image
        """
        if preview_type not in self.preview_labels:
            return
        
        # Load image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale the pixmap to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                self.preview_labels[preview_type].size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_labels[preview_type].setPixmap(pixmap)
        else:
            # Failed to load image
            self.preview_labels[preview_type].setText(f"Preview\nload failed")
    
    def on_cancel(self):
        """Handle cancel button click"""
        # Confirm cancellation
        result = QMessageBox.question(
            self,
            "Cancel Processing",
            "Are you sure you want to cancel the current processing job?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            # Emit cancelled signal
            self.cancelled.emit()
            
            # Update UI
            self.set_status("Processing cancelled", self.progress_bar.value())
            self.cancel_button.setEnabled(False)
    
    def process_completed(self, success, message):
        """
        Called when processing is completed
        
        Args:
            success: Whether processing was successful
            message: Status message
        """
        if success:
            self.set_status(f"Processing completed: {message}", 100)
            
            # Change cancel button to close
            self.cancel_button.setText("Close")
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.accept)
        else:
            self.set_status(f"Processing failed: {message}", self.progress_bar.value())
            
            # Change cancel button to close
            self.cancel_button.setText("Close")
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.reject)