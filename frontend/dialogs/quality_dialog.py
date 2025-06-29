# frontend/dialogs/quality_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QRadioButton, QGroupBox, QButtonGroup, QFrame, QSizePolicy,
    QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap

class QualitySelectionDialog(QDialog):
    """Dialog for selecting quality level for 3D reconstruction"""
    
    qualitySelected = pyqtSignal(str)  # Signal emitted when a quality is selected
    
    def __init__(self, parent=None, current_quality="high"):
        super().__init__(parent)
        
        self.setWindowTitle("Select Quality Level")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        # Store quality descriptions
        self.quality_descriptions = {
            "medium": {
                "name": "Medium Quality",
                "description": "Decent quality with faster processing times",
                "use_case": "Previews, drafts, or when processing time is limited",
                "estimated_time": "Faster (30-60% less time than High Quality)",
                "memory_usage": "Lower (suitable for machines with 8GB RAM)",
                "icon": ":/icons/quality_medium.png"
            },
            "high": {
                "name": "High Quality",
                "description": "Well-balanced quality and processing time",
                "use_case": "General purpose 3D reconstruction for most applications",
                "estimated_time": "Standard baseline",
                "memory_usage": "Medium (recommended 16GB RAM)",
                "icon": ":/icons/quality_high.png"
            },
            "photorealistic": {
                "name": "Photorealistic",
                "description": "Maximum detail and visual realism",
                "use_case": "Professional visualization, rendering, and high-end applications",
                "estimated_time": "Slower (2-3x longer than High Quality)",
                "memory_usage": "High (recommended 32GB RAM or more)",
                "icon": ":/icons/quality_photorealistic.png"
            }
        }
        
        self.current_quality = current_quality
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Select Quality Level")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Choose the quality level for your 3D reconstruction. "
                          "Higher quality levels produce better results but require more processing time and memory.")
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        # Radio button group for quality selection
        quality_group = QGroupBox("Quality Options")
        quality_layout = QVBoxLayout()
        quality_group.setLayout(quality_layout)
        
        # Button group for radio buttons
        self.button_group = QButtonGroup(self)
        
        # Create radio buttons for each quality level
        self.quality_radios = {}
        self.quality_info_labels = {}
        
        for quality_level, desc in self.quality_descriptions.items():
            # Create horizontal layout for this quality option
            option_layout = QHBoxLayout()
            
            # Radio button
            radio = QRadioButton(desc["name"])
            radio.setChecked(quality_level == self.current_quality)
            self.button_group.addButton(radio)
            self.quality_radios[quality_level] = radio
            
            # Try to add icon if available (fallback gracefully)
            try:
                if desc.get("icon"):
                    icon = QIcon(desc["icon"])
                    radio.setIcon(icon)
            except:
                pass  # Icon not found, continue without it
            
            option_layout.addWidget(radio)
            
            # Quality info label
            info_label = QLabel(desc["description"])
            info_label.setWordWrap(True)
            self.quality_info_labels[quality_level] = info_label
            option_layout.addWidget(info_label, 1)  # Add stretch factor
            
            quality_layout.addLayout(option_layout)
            
            # Add details below radio button
            details_layout = QVBoxLayout()
            details_layout.setContentsMargins(20, 0, 0, 10)  # Indent details
            
            use_case_label = QLabel(f"<b>Use case:</b> {desc['use_case']}")
            use_case_label.setWordWrap(True)
            details_layout.addWidget(use_case_label)
            
            time_label = QLabel(f"<b>Processing time:</b> {desc['estimated_time']}")
            time_label.setWordWrap(True)
            details_layout.addWidget(time_label)
            
            memory_label = QLabel(f"<b>Memory usage:</b> {desc['memory_usage']}")
            memory_label.setWordWrap(True)
            details_layout.addWidget(memory_label)
            
            quality_layout.addLayout(details_layout)
            
            # Add separator except for last item
            if quality_level != "photorealistic":
                sep_line = QFrame()
                sep_line.setFrameShape(QFrame.Shape.HLine)
                sep_line.setFrameShadow(QFrame.Shadow.Sunken)
                quality_layout.addWidget(sep_line)
        
        main_layout.addWidget(quality_group)
        
        # Add spacer
        main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        buttons_layout.addWidget(ok_button)
        
        main_layout.addLayout(buttons_layout)
    
    def get_selected_quality(self):
        """Get the currently selected quality level"""
        for quality, radio in self.quality_radios.items():
            if radio.isChecked():
                return quality
        return self.current_quality  # Fallback
    
    def accept(self):
        """Override accept to emit qualitySelected signal"""
        selected_quality = self.get_selected_quality()
        self.qualitySelected.emit(selected_quality)
        super().accept()


if __name__ == "__main__":
    # Test the dialog
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = QualitySelectionDialog(current_quality="high")
    
    def quality_selected(quality):
        print(f"Selected quality: {quality}")
    
    dialog.qualitySelected.connect(quality_selected)
    dialog.exec()