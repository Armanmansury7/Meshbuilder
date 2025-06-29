"""
Settings Dialog for MeshBuilder
Allows configuration of processing parameters and photorealistic options
"""
from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QFormLayout, QLabel, QComboBox, QSlider, QSpinBox, 
    QCheckBox, QPushButton, QGroupBox, QSizePolicy, QLineEdit,
    QFileDialog, QSpacerItem
)
from PyQt6.QtCore import Qt, QSettings

class SettingsDialog(QDialog):
    """Settings dialog for MeshBuilder with photorealistic options"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("MeshBuilder Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget(self)
        
        # Create tabs
        general_tab = QWidget()
        processing_tab = QWidget()
        photorealistic_tab = QWidget()  # New photorealistic tab
        output_tab = QWidget()
        
        # Set up tab layouts
        self.setup_general_tab(general_tab)
        self.setup_processing_tab(processing_tab)
        self.setup_photorealistic_tab(photorealistic_tab)  # Setup new tab
        self.setup_output_tab(output_tab)
        
        # Add tabs to widget
        tab_widget.addTab(general_tab, "General")
        tab_widget.addTab(processing_tab, "Processing")
        tab_widget.addTab(photorealistic_tab, "Photorealistic")  # Add new tab
        tab_widget.addTab(output_tab, "Output")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        # Save button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        # Reset to defaults button
        defaults_button = QPushButton("Reset to Defaults")
        defaults_button.clicked.connect(self.reset_defaults)
        
        # Add buttons to layout
        button_layout.addWidget(defaults_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
        
        # Load current settings
        self.load_settings()
        
    def setup_general_tab(self, tab):
        """Set up the general settings tab"""
        layout = QFormLayout(tab)
        
        # COLMAP path
        self.colmap_path_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_colmap_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.colmap_path_edit)
        path_layout.addWidget(browse_button)
        
        layout.addRow("COLMAP Path:", path_layout)
        
        # Temporary directory
        self.temp_dir_edit = QLineEdit()
        temp_browse_button = QPushButton("Browse...")
        temp_browse_button.clicked.connect(self.browse_temp_dir)
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temp_dir_edit)
        temp_layout.addWidget(temp_browse_button)
        
        layout.addRow("Temporary Directory:", temp_layout)
        
        # Output directory
        self.output_dir_edit = QLineEdit()
        output_browse_button = QPushButton("Browse...")
        output_browse_button.clicked.connect(self.browse_output_dir)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_browse_button)
        
        layout.addRow("Output Directory:", output_layout)
        
        # GPU Acceleration
        self.use_gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        layout.addRow("", self.use_gpu_checkbox)
        
        # Add spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
    def setup_processing_tab(self, tab):
        """Set up the processing settings tab"""
        layout = QVBoxLayout(tab)
        form_layout = QFormLayout()
        
        # Image processing settings
        img_group = QGroupBox("Image Processing")
        img_layout = QFormLayout(img_group)
        
        # Max image dimension
        self.max_image_dimension = QSpinBox()
        self.max_image_dimension.setRange(512, 8192)
        self.max_image_dimension.setSingleStep(512)
        self.max_image_dimension.setSuffix(" px")
        img_layout.addRow("Max Image Dimension:", self.max_image_dimension)
        
        # Image quality threshold
        self.quality_threshold = QSlider(Qt.Orientation.Horizontal)
        self.quality_threshold.setRange(0, 100)
        self.quality_threshold.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_threshold.setTickInterval(10)
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.quality_threshold)
        
        self.quality_value = QLabel("0.5")
        quality_layout.addWidget(self.quality_value)
        
        # Connect slider to label update
        self.quality_threshold.valueChanged.connect(
            lambda v: self.quality_value.setText(str(v/100))
        )
        
        img_layout.addRow("Quality Threshold:", quality_layout)
        
        # Add the image group to the main layout
        layout.addWidget(img_group)
        
        # Reconstruction settings
        recon_group = QGroupBox("Reconstruction")
        recon_layout = QFormLayout(recon_group)
        
        # Feature matcher
        self.feature_matcher = QComboBox()
        self.feature_matcher.addItems(["exhaustive", "sequential"])
        recon_layout.addRow("Feature Matcher:", self.feature_matcher)
        
        # Point density
        self.point_density = QComboBox()
        self.point_density.addItems(["low", "medium", "high"])
        recon_layout.addRow("Point Cloud Density:", self.point_density)
        
        # Mesh resolution
        self.mesh_resolution = QComboBox()
        self.mesh_resolution.addItems(["low", "medium", "high"])
        recon_layout.addRow("Mesh Resolution:", self.mesh_resolution)
        
        # Meshing algorithm
        self.meshing_algorithm = QComboBox()
        self.meshing_algorithm.addItems(["poisson", "bpa"])
        self.meshing_algorithm.setToolTip("Poisson: Smoother surfaces. BPA: Better for sharp features.")
        recon_layout.addRow("Meshing Algorithm:", self.meshing_algorithm)
        
        # Smoothing
        self.smoothing = QSlider(Qt.Orientation.Horizontal)
        self.smoothing.setRange(0, 100)
        self.smoothing.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smoothing.setTickInterval(10)
        
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(self.smoothing)
        
        self.smoothing_value = QLabel("0.5")
        smoothing_layout.addWidget(self.smoothing_value)
        
        # Connect slider to label update
        self.smoothing.valueChanged.connect(
            lambda v: self.smoothing_value.setText(str(v/100))
        )
        
        recon_layout.addRow("Smoothing:", smoothing_layout)
        
        # Add the reconstruction group to the main layout
        layout.addWidget(recon_group)
        
        # Add spacer
        layout.addStretch()
        
    def setup_photorealistic_tab(self, tab):
        """Set up the photorealistic settings tab"""
        layout = QVBoxLayout(tab)
        
        # Texture Settings Group
        texture_group = QGroupBox("Texture Settings")
        texture_layout = QFormLayout(texture_group)
        
        # Texture Resolution
        self.texture_resolution = QComboBox()
        self.texture_resolution.addItems(["1024", "2048", "4096", "8192"])
        self.texture_resolution.setToolTip("Higher resolution gives better quality but requires more memory")
        texture_layout.addRow("Texture Resolution:", self.texture_resolution)
        
        # Color Enhancement Mode
        self.color_enhance_mode = QComboBox()
        self.color_enhance_mode.addItems(["none", "balanced", "vibrant", "realistic"])
        self.color_enhance_mode.setToolTip(
            "none: No color enhancement\n"
            "balanced: Good all-around enhancement\n"
            "vibrant: More saturated, higher contrast\n"
            "realistic: Natural look with tone mapping"
        )
        texture_layout.addRow("Color Enhancement:", self.color_enhance_mode)
        
        # Add texture group to layout
        layout.addWidget(texture_group)
        
        # Advanced Settings Group
        advanced_group = QGroupBox("Advanced Enhancement")
        advanced_layout = QFormLayout(advanced_group)
        
        # HDR Tone Mapping
        self.enable_hdr = QCheckBox("Enable HDR Tone Mapping")
        self.enable_hdr.setToolTip("Simulates high dynamic range for more realistic lighting")
        advanced_layout.addRow("", self.enable_hdr)
        
        # Saturation Boost
        self.saturation_boost = QSlider(Qt.Orientation.Horizontal)
        self.saturation_boost.setRange(80, 150)  # 0.8 to 1.5
        self.saturation_boost.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.saturation_boost.setTickInterval(10)
        
        saturation_layout = QHBoxLayout()
        saturation_layout.addWidget(self.saturation_boost)
        
        self.saturation_value = QLabel("1.0")
        saturation_layout.addWidget(self.saturation_value)
        
        # Connect slider to label update
        self.saturation_boost.valueChanged.connect(
            lambda v: self.saturation_value.setText(f"{v/100:.2f}")
        )
        
        advanced_layout.addRow("Saturation Boost:", saturation_layout)
        
        # Contrast Boost
        self.contrast_boost = QSlider(Qt.Orientation.Horizontal)
        self.contrast_boost.setRange(80, 150)  # 0.8 to 1.5
        self.contrast_boost.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.contrast_boost.setTickInterval(10)
        
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(self.contrast_boost)
        
        self.contrast_value = QLabel("1.0")
        contrast_layout.addWidget(self.contrast_value)
        
        # Connect slider to label update
        self.contrast_boost.valueChanged.connect(
            lambda v: self.contrast_value.setText(f"{v/100:.2f}")
        )
        
        advanced_layout.addRow("Contrast Boost:", contrast_layout)
        
        # Denoise
        self.apply_denoise = QCheckBox("Apply Denoising")
        self.apply_denoise.setToolTip("Reduces noise in textures")
        advanced_layout.addRow("", self.apply_denoise)
        
        # UV Mapping Method
        self.uv_mapping_method = QComboBox()
        self.uv_mapping_method.addItems(["simple", "advanced"])
        self.uv_mapping_method.setToolTip(
            "simple: Faster but may cause stretching\n"
            "advanced: Better quality but slower"
        )
        advanced_layout.addRow("UV Mapping:", self.uv_mapping_method)
        
        # Add advanced group to layout
        layout.addWidget(advanced_group)
        
        # Quality Preset Buttons
        preset_group = QGroupBox("Quality Presets")
        preset_layout = QHBoxLayout(preset_group)
        
        # Standard preset
        standard_button = QPushButton("Standard")
        standard_button.clicked.connect(self.apply_standard_preset)
        preset_layout.addWidget(standard_button)
        
        # Photorealistic preset
        photorealistic_button = QPushButton("Photorealistic")
        photorealistic_button.clicked.connect(self.apply_photorealistic_preset)
        preset_layout.addWidget(photorealistic_button)
        
        # Performance preset
        performance_button = QPushButton("Performance")
        performance_button.clicked.connect(self.apply_performance_preset)
        preset_layout.addWidget(performance_button)
        
        # Add preset group to layout
        layout.addWidget(preset_group)
        
        # Add spacer
        layout.addStretch()
    
    def setup_output_tab(self, tab):
        """Set up the output settings tab"""
        layout = QFormLayout(tab)
        
        # Default export format
        self.default_format = QComboBox()
        self.default_format.addItems(["obj", "ply", "fbx", "gltf"])
        layout.addRow("Default Format:", self.default_format)
        
        # Target face count for optimization
        self.target_faces = QSpinBox()
        self.target_faces.setRange(1000, 1000000)
        self.target_faces.setSingleStep(10000)
        self.target_faces.setSuffix(" faces")
        layout.addRow("Target Mesh Size:", self.target_faces)
        
        # Compress output files
        self.compress_output = QCheckBox("Compress Output Files")
        layout.addRow("", self.compress_output)
        
        # Clean up temporary files
        self.remove_temp = QCheckBox("Remove Temporary Files After Processing")
        layout.addRow("", self.remove_temp)
        
        # Keep processing logs
        self.keep_logs = QCheckBox("Keep Processing Logs")
        layout.addRow("", self.keep_logs)
        
        # Add spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
    
    def load_settings(self):
        """Load current settings from config"""
        # General tab
        self.colmap_path_edit.setText(self.config.get("Paths", "colmap_bin", fallback="colmap"))
        self.temp_dir_edit.setText(self.config.get("Paths", "temp_dir", fallback=""))
        self.output_dir_edit.setText(self.config.get("Paths", "output_dir", fallback=""))
        self.use_gpu_checkbox.setChecked(self.config.getboolean("Processing", "use_gpu", fallback=True))
        
        # Processing tab
        self.max_image_dimension.setValue(self.config.getint("Processing", "max_image_dimension", fallback=2048))
        self.quality_threshold.setValue(int(self.config.getfloat("Processing", "quality_threshold", fallback=0.5) * 100))
        self.quality_value.setText(str(self.config.getfloat("Processing", "quality_threshold", fallback=0.5)))
        
        index = self.feature_matcher.findText(self.config.get("Processing", "feature_matcher", fallback="exhaustive"))
        self.feature_matcher.setCurrentIndex(max(0, index))
        
        index = self.point_density.findText(self.config.get("Processing", "point_density", fallback="medium"))
        self.point_density.setCurrentIndex(max(0, index))
        
        index = self.mesh_resolution.findText(self.config.get("Processing", "mesh_resolution", fallback="medium"))
        self.mesh_resolution.setCurrentIndex(max(0, index))
        
        index = self.meshing_algorithm.findText(self.config.get("Processing", "meshing_algorithm", fallback="poisson"))
        self.meshing_algorithm.setCurrentIndex(max(0, index))
        
        self.smoothing.setValue(int(self.config.getfloat("Processing", "smoothing", fallback=0.5) * 100))
        self.smoothing_value.setText(str(self.config.getfloat("Processing", "smoothing", fallback=0.5)))
        
        # Photorealistic tab
        index = self.texture_resolution.findText(str(self.config.getint("Processing", "texture_resolution", fallback=4096)))
        self.texture_resolution.setCurrentIndex(max(0, index))
        
        index = self.color_enhance_mode.findText(self.config.get("Processing", "color_enhance_mode", fallback="balanced"))
        self.color_enhance_mode.setCurrentIndex(max(0, index))
        
        self.enable_hdr.setChecked(self.config.getboolean("Photorealistic", "enable_hdr_tone_mapping", fallback=True))
        
        sat_value = int(self.config.getfloat("Photorealistic", "saturation_boost", fallback=1.15) * 100)
        self.saturation_boost.setValue(sat_value)
        self.saturation_value.setText(f"{sat_value/100:.2f}")
        
        contrast_value = int(self.config.getfloat("Photorealistic", "contrast_boost", fallback=1.2) * 100)
        self.contrast_boost.setValue(contrast_value)
        self.contrast_value.setText(f"{contrast_value/100:.2f}")
        
        self.apply_denoise.setChecked(self.config.getboolean("Photorealistic", "apply_denoise", fallback=True))
        
        index = self.uv_mapping_method.findText(self.config.get("Photorealistic", "uv_mapping_method", fallback="advanced"))
        self.uv_mapping_method.setCurrentIndex(max(0, index))
        
        # Output tab
        index = self.default_format.findText(self.config.get("Output", "default_format", fallback="obj"))
        self.default_format.setCurrentIndex(max(0, index))
        
        self.target_faces.setValue(self.config.getint("Processing", "target_faces", fallback=100000))
        self.compress_output.setChecked(self.config.getboolean("Output", "compress_output", fallback=False))
        self.remove_temp.setChecked(self.config.getboolean("Cleanup", "remove_temp_files", fallback=True))
        self.keep_logs.setChecked(self.config.getboolean("Cleanup", "keep_logs", fallback=True))
    
    def save_settings(self):
        """Save settings to config"""
        # General tab
        self.config.set("Paths", "colmap_bin", self.colmap_path_edit.text())
        self.config.set("Paths", "temp_dir", self.temp_dir_edit.text())
        self.config.set("Paths", "output_dir", self.output_dir_edit.text())
        self.config.set("Processing", "use_gpu", str(self.use_gpu_checkbox.isChecked()))
        
        # Processing tab
        self.config.set("Processing", "max_image_dimension", str(self.max_image_dimension.value()))
        self.config.set("Processing", "quality_threshold", str(float(self.quality_value.text())))
        self.config.set("Processing", "feature_matcher", self.feature_matcher.currentText())
        self.config.set("Processing", "point_density", self.point_density.currentText())
        self.config.set("Processing", "mesh_resolution", self.mesh_resolution.currentText())
        self.config.set("Processing", "meshing_algorithm", self.meshing_algorithm.currentText())
        self.config.set("Processing", "smoothing", str(float(self.smoothing_value.text())))
        
        # Photorealistic tab
        self.config.set("Processing", "texture_resolution", self.texture_resolution.currentText())
        self.config.set("Processing", "color_enhance_mode", self.color_enhance_mode.currentText())
        self.config.set("Photorealistic", "enable_hdr_tone_mapping", str(self.enable_hdr.isChecked()))
        self.config.set("Photorealistic", "saturation_boost", str(float(self.saturation_value.text())))
        self.config.set("Photorealistic", "contrast_boost", str(float(self.contrast_value.text())))
        self.config.set("Photorealistic", "apply_denoise", str(self.apply_denoise.isChecked()))
        self.config.set("Photorealistic", "uv_mapping_method", self.uv_mapping_method.currentText())
        
        # Output tab
        self.config.set("Output", "default_format", self.default_format.currentText())
        self.config.set("Processing", "target_faces", str(self.target_faces.value()))
        self.config.set("Output", "compress_output", str(self.compress_output.isChecked()))
        self.config.set("Cleanup", "remove_temp_files", str(self.remove_temp.isChecked()))
        self.config.set("Cleanup", "keep_logs", str(self.keep_logs.isChecked()))
        
        # Save to file
        self.config_manager.save_config(self.config)
        
        # Accept dialog
        self.accept()
    
    def reset_defaults(self):
        """Reset settings to defaults"""
        # Create a new default config
        self.config = self.config_manager.create_default_config()
        
        # Load the default values into UI
        self.load_settings()
    
    def browse_colmap_path(self):
        """Browse for COLMAP executable"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select COLMAP Executable",
            "",
            "Executables (*.exe);;All Files (*)"
        )
        if path:
            self.colmap_path_edit.setText(path)
    
    def browse_temp_dir(self):
        """Browse for temporary directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Temporary Directory",
            ""
        )
        if directory:
            self.temp_dir_edit.setText(directory)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        if directory:
            self.output_dir_edit.setText(directory)
    
    def apply_standard_preset(self):
        """Apply standard quality preset"""
        # Processing
        self.max_image_dimension.setValue(2048)
        self.feature_matcher.setCurrentText("exhaustive")
        self.point_density.setCurrentText("medium")
        self.mesh_resolution.setCurrentText("medium")
        self.meshing_algorithm.setCurrentText("poisson")
        self.smoothing.setValue(50)  # 0.5
        
        # Photorealistic
        self.texture_resolution.setCurrentText("2048")
        self.color_enhance_mode.setCurrentText("balanced")
        self.enable_hdr.setChecked(True)
        self.saturation_boost.setValue(115)  # 1.15
        self.contrast_boost.setValue(120)    # 1.2
        self.apply_denoise.setChecked(True)
        self.uv_mapping_method.setCurrentText("advanced")
        
        # Output
        self.target_faces.setValue(100000)
    
    def apply_photorealistic_preset(self):
        """Apply photorealistic quality preset"""
        # Processing
        self.max_image_dimension.setValue(3200)
        self.feature_matcher.setCurrentText("exhaustive")
        self.point_density.setCurrentText("high")
        self.mesh_resolution.setCurrentText("high")
        self.meshing_algorithm.setCurrentText("poisson")
        self.smoothing.setValue(30)  # 0.3 - less smoothing for detail
        
        # Photorealistic
        self.texture_resolution.setCurrentText("4096")
        self.color_enhance_mode.setCurrentText("realistic")
        self.enable_hdr.setChecked(True)
        self.saturation_boost.setValue(110)  # 1.1 - more natural
        self.contrast_boost.setValue(115)    # 1.15
        self.apply_denoise.setChecked(True)
        self.uv_mapping_method.setCurrentText("advanced")
        
        # Output
        self.target_faces.setValue(150000)
    
    def apply_performance_preset(self):
        """Apply performance-oriented preset"""
        # Processing
        self.max_image_dimension.setValue(1600)
        self.feature_matcher.setCurrentText("sequential")
        self.point_density.setCurrentText("low")
        self.mesh_resolution.setCurrentText("low")
        self.meshing_algorithm.setCurrentText("poisson")
        self.smoothing.setValue(60)  # 0.6
        
        # Photorealistic
        self.texture_resolution.setCurrentText("1024")
        self.color_enhance_mode.setCurrentText("balanced")
        self.enable_hdr.setChecked(False)
        self.saturation_boost.setValue(110)  # 1.1
        self.contrast_boost.setValue(110)    # 1.1
        self.apply_denoise.setChecked(False)
        self.uv_mapping_method.setCurrentText("simple")
        
        # Output
        self.target_faces.setValue(50000)