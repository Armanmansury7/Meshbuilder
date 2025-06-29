# frontend/components/model_viewer.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QCursor, QIcon
import os
import numpy as np
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

class ModelViewer(QWidget):
    """3D model viewer for displaying OBJ, PLY and other 3D model formats"""
    viewChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Model data
        self.current_model = None
        self.model_mesh = None
        self._orbit_running = False
        self._orbit_timer = None
        self.last_pos = None
        
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("background-color: #2A2A2A; border-radius: 8px;")
        
        # Set minimum size to 800x800 and allow expanding
        self.setMinimumSize(800, 800)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # To receive keyboard events
        
        # Create placeholder for when no model is loaded
        self.placeholder = QLabel(" Meshbuilder:Model Viewer ")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 16px;")
        layout.addWidget(self.placeholder)
        
        if PYQTGRAPH_AVAILABLE:
            # Create 3D view widget
            self.view = gl.GLViewWidget()
            # Changed default camera position to isometric view for better initial visualization
            self.view.setCameraPosition(distance=40, elevation=35, azimuth=45)
            self.view.setBackgroundColor('#2A2A2A')
            
            # Improve viewing experience
            self.view.opts['fov'] = 60
            self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # Add axes for orientation
            self.axes = gl.GLAxisItem()
            self.axes.setSize(x=10, y=10, z=10)
            self.view.addItem(self.axes)
            
            # Add grid for reference
            self.grid = gl.GLGridItem()
            self.grid.setSize(x=20, y=20, z=0)
            self.grid.setSpacing(x=1, y=1, z=1)
            self.view.addItem(self.grid)
            
            # Add 3D view to layout
            layout.addWidget(self.view)
            
            # Add view control buttons
            control_layout = QHBoxLayout()
            control_layout.setContentsMargins(10, 5, 10, 5)
            control_layout.setSpacing(10)
            
            # Create navigation buttons for orientation
            self.top_btn = QPushButton("Top")
            self.top_btn.setToolTip("Top view (aerial)")
            self.top_btn.clicked.connect(lambda: self.rotate_to_preset('top'))
            self.top_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
            """)
            
            self.front_btn = QPushButton("Front")
            self.front_btn.setToolTip("Front view")
            self.front_btn.clicked.connect(lambda: self.rotate_to_preset('front'))
            self.front_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
            """)
            
            self.side_btn = QPushButton("Side")
            self.side_btn.setToolTip("Side view")
            self.side_btn.clicked.connect(lambda: self.rotate_to_preset('right'))
            self.side_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
            """)
            
            self.iso_btn = QPushButton("Isometric")
            self.iso_btn.setToolTip("Isometric view")
            self.iso_btn.clicked.connect(lambda: self.rotate_to_preset('isometric'))
            self.iso_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
            """)
            
            self.landscape_btn = QPushButton("Flat")
            self.landscape_btn.setToolTip("Align model flat on XY plane")
            self.landscape_btn.clicked.connect(self.align_model_to_xy_plane)
            self.landscape_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QPushButton:pressed {
                    background-color: #3498db;
                }
            """)
            
            # Add reset view button
            self.reset_view_btn = QPushButton("Reset View")
            self.reset_view_btn.setToolTip("Reset camera view")
            self.reset_view_btn.clicked.connect(self.reset_view)
            self.reset_view_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 3px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
            """)
            
            # Add buttons to layout
            control_layout.addWidget(self.top_btn)
            control_layout.addWidget(self.front_btn)
            control_layout.addWidget(self.side_btn)
            control_layout.addWidget(self.iso_btn)
            control_layout.addWidget(self.landscape_btn)
            control_layout.addWidget(self.reset_view_btn)
            control_layout.addStretch()
            
            layout.addLayout(control_layout)
            
            # Initially hide the view - will show when model is loaded
            self.view.hide()
        else:
            # If pyqtgraph is not available, show info message
            self.view = None
            self.placeholder.setText("3D viewer requires PyQtGraph and PyOpenGL.\n"
                                    "Please install with: pip install pyqtgraph pyopengl trimesh")
    
    def load_model(self, model_path):
        """Load and display a 3D model"""
        if not os.path.exists(model_path):
            self.placeholder.setText(f"Error: Model file not found:\n{model_path}")
            if hasattr(self, 'view') and self.view:
                self.view.hide()
            self.placeholder.show()
            return False
            
        # Check if required libraries are available
        if not PYQTGRAPH_AVAILABLE or not TRIMESH_AVAILABLE:
            self.placeholder.setText("3D viewer requires PyQtGraph, PyOpenGL and trimesh.\n"
                                    "Please install with: pip install pyqtgraph pyopengl trimesh")
            return False
            
        try:
            # Clear current model if any
            if self.model_mesh is not None and self.view:
                self.view.removeItem(self.model_mesh)
                self.model_mesh = None
            
            # Load model using trimesh (supports OBJ, PLY, STL, etc.)
            self.original_mesh = trimesh.load(model_path)
            
            # Create a copy of the mesh to manipulate
            mesh = self.original_mesh.copy()
            
            # Extract mesh data
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            
            # Normalize size and center model
            scale = 20.0 / max(mesh.extents)
            vertices *= scale
            center = np.mean(vertices, axis=0)
            vertices -= center
            
            # Determine principal axes using PCA (Principal Component Analysis)
            # This helps identify the main orientation of the model
            if len(vertices) > 3:  # Need at least 3 points for PCA
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=3)
                    pca.fit(vertices)
                    
                    # Store principal components for later alignment
                    self.principal_components = pca.components_
                except ImportError:
                    # If sklearn is not available, we'll use a simple method
                    self.principal_components = None
            else:
                self.principal_components = None
            
            # Create colors based on vertex normals if available
            if hasattr(mesh, 'visual') and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors.copy()
                colors = colors[:, :3] / 255.0  # Normalize to 0-1 range
                
                # Create mesh item with vertex colors
                self.model_mesh = gl.GLMeshItem(
                    vertexes=vertices, 
                    faces=faces, 
                    vertexColors=colors,
                    smooth=True, 
                    shader='shaded'
                )
            else:
                # Create mesh with default coloring
                self.model_mesh = gl.GLMeshItem(
                    vertexes=vertices, 
                    faces=faces,
                    color=(0.7, 0.7, 0.8, 1.0),
                    smooth=True, 
                    shader='shaded'
                )
            
            # Add to view
            self.view.addItem(self.model_mesh)
            
            # Hide placeholder, show model
            self.placeholder.hide()
            self.view.show()
            
            # Store path for reference
            self.current_model = model_path
            self.vertices = vertices
            self.faces = faces
            
            # Align model to XY plane automatically for better initial view
            self.align_model_to_xy_plane()
            
            # Reset view to see the model properly (isometric view by default)
            self.rotate_to_preset('isometric')
            
            # Set focus to this widget to enable keyboard navigation
            self.setFocus()
            
            return True
            
        except Exception as e:
            self.placeholder.setText(f"Error loading model: {str(e)}")
            if hasattr(self, 'view') and self.view:
                self.view.hide()
            self.placeholder.show()
            print(f"Model loading error: {str(e)}")
            return False
    
    def align_model_to_xy_plane(self):
        """Align the model to lie flat on the XY plane and center it on the origin"""
        if not self.view or not self.model_mesh or not hasattr(self, 'vertices'):
            return
            
        try:
            # Get a fresh copy of vertices to work with
            vertices = self.vertices.copy()
            
            # Step 1: First make the model flat on XY plane
            # Get the extents in each dimension
            min_vals = np.min(vertices, axis=0)
            max_vals = np.max(vertices, axis=0)
            extents = max_vals - min_vals
            
            # Identify which dimension is likely the "height" (typically the smallest)
            height_dim = np.argmin(extents)
            
            # Create transformed vertices to align with Z-axis up
            transformed = vertices.copy()
            
            # Swap axes to make the model flat on XY plane
            if height_dim == 0:  # X is height, so X should be mapped to Z
                # X -> Z, Y -> Y, Z -> X
                transformed[:, 0] = vertices[:, 2]
                transformed[:, 2] = vertices[:, 0]
            elif height_dim == 1:  # Y is height, so Y should be mapped to Z
                # X -> X, Y -> Z, Z -> Y
                transformed[:, 1] = vertices[:, 2]
                transformed[:, 2] = vertices[:, 1]
            # If Z is already the height dimension, no need to swap
            
            # Step 2: Ensure the model is oriented correctly (Z-up)
            z_min = np.min(transformed[:, 2])
            z_max = np.max(transformed[:, 2])
            
            # If most of the model is in negative Z, flip it
            if abs(z_min) > z_max:
                transformed[:, 2] = -transformed[:, 2]
            
            # Step 3: CENTER THE MODEL ON THE ORIGIN (0,0,0)
            # Calculate the center of the model in XY plane
            x_min = np.min(transformed[:, 0])
            x_max = np.max(transformed[:, 0])
            y_min = np.min(transformed[:, 1])
            y_max = np.max(transformed[:, 1])
            
            # Calculate the center offset
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Translate to center the model on the origin
            transformed[:, 0] -= x_center
            transformed[:, 1] -= y_center
            
            # Ensure Z is aligned with the ground plane (min Z should be at 0)
            z_min = np.min(transformed[:, 2])
            transformed[:, 2] -= z_min
            
            # Update the mesh with transformed vertices
            self.model_mesh.setMeshData(vertexes=transformed, faces=self.faces)
            print(f"Model aligned to XY plane and centered on origin")
            
            # Step 4: Reset the camera to view the centered model
            self.view.setCameraPosition(distance=40, elevation=35, azimuth=45)  # Isometric view first
            
            # Reset all view parameters to ensure proper centering
            self.view.opts['center'] = pg.Vector(0, 0, 0)  # Center view on origin
            
            # Update the view
            self.view.update()
            
            # Give user feedback
            print("Model is now centered on the grid origin")
            
        except Exception as e:
            print(f"Model alignment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # Finally, set the view to look at the centered model
        self.rotate_to_preset('isometric')  # Start with isometric view
        self.view.pan(0, 0, 0)  # Ensure panning is reset
            
    def reset_view(self):
        """Reset camera view to default isometric position"""
        if self.view and self.model_mesh:
            # Reset camera to isometric view
            self.view.setCameraPosition(distance=40, elevation=35, azimuth=45)
            self.viewChanged.emit()
            
    def zoom_in(self):
        """Zoom in to the model"""
        if self.view and self.model_mesh:
            # Get current distance
            distance = self.view.opts['distance']
            
            # Zoom in by reducing distance
            new_distance = max(5, distance * 0.8)
            self.view.setCameraPosition(distance=new_distance)
            self.viewChanged.emit()
            
    def zoom_out(self):
        """Zoom out from the model"""
        if self.view and self.model_mesh:
            # Get current distance
            distance = self.view.opts['distance']
            
            # Zoom out by increasing distance
            new_distance = min(100, distance * 1.2)
            self.view.setCameraPosition(distance=new_distance)
            self.viewChanged.emit()
            
    def rotate_to_preset(self, preset):
        """Rotate to a preset view"""
        if not self.view or not self.model_mesh:
            return
            
        # Define presets
        presets = {
            'front': (0, 0),          # (azimuth, elevation)
            'back': (180, 0),
            'left': (-90, 0),
            'right': (90, 0),
            'top': (0, 90),           # True top-down view
            'bottom': (0, -90),
            'isometric': (45, 35),    # Standard isometric view
            'landscape': (0, 90)      # Same as top, for backwards compatibility
        }
        
        if preset in presets:
            azimuth, elevation = presets[preset]
            
            # Set camera position
            self.view.setCameraPosition(azimuth=azimuth, elevation=elevation)
            
            # Adjust distance to ensure the model is visible
            self.view.setCameraPosition(distance=40)
            
            # Special handling for top/landscape view to ensure good alignment
            if preset in ['top', 'landscape']:
                # Center the view on the model
                self.view.pan(0, 0, 0)
                
            self.viewChanged.emit()
            print(f"View changed to {preset}")
            
    def auto_orbit(self, enable=True):
        """Enable or disable automatic 360-degree rotation"""
        if not self.view or not self.model_mesh:
            return
            
        # Setup orbit timer if needed
        if not self._orbit_timer:
            self._orbit_timer = QTimer()
            self._orbit_timer.timeout.connect(self._orbit_update)
            
        if enable:
            # Start with a more interesting view for orbiting
            self.rotate_to_preset('isometric')
            self._orbit_timer.start(50)  # Update every 50ms
            self._orbit_running = True
            print("Auto-orbit enabled")
        else:
            self._orbit_timer.stop()
            self._orbit_running = False
            print("Auto-orbit disabled")
            
    def _orbit_update(self):
        """Update orbit position"""
        if self.view and self.model_mesh:
            azimuth = self.view.opts['azimuth']
            self.view.setCameraPosition(azimuth=(azimuth + 1) % 360)
            self.viewChanged.emit()
            
    def take_screenshot(self, output_path):
        """Take a screenshot of the current view"""
        if not self.view or not self.model_mesh:
            return False
            
        try:
            # Capture the 3D view
            screenshot = self.view.grabFrameBuffer()
            
            # Save to file
            screenshot.save(output_path)
            print(f"Screenshot saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Screenshot error: {str(e)}")
            return False
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.view and self.model_mesh:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            super().wheelEvent(event)
                
    def mousePressEvent(self, event):
        """Store mouse position for dragging"""
        self.last_pos = event.position()
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation"""
        if self.view and self.model_mesh and self.last_pos:
            dx = event.position().x() - self.last_pos.x()
            dy = event.position().y() - self.last_pos.y()
            
            # Improve mouse sensitivity for smoother rotation
            sensitivity = 0.5
            
            if event.buttons() & Qt.MouseButton.LeftButton:
                # Left button: rotate around model
                azimuth = self.view.opts['azimuth']
                elevation = self.view.opts['elevation']
                
                # Adjust azimuth (horizontal rotation)
                azimuth -= dx * sensitivity
                
                # Adjust elevation (vertical rotation), with limits
                elevation += dy * sensitivity
                elevation = min(90, max(-90, elevation))
                
                self.view.setCameraPosition(azimuth=azimuth, elevation=elevation)
                self.viewChanged.emit()
                
            elif event.buttons() & Qt.MouseButton.RightButton:
                # Right button: pan view
                self.view.pan(dx * sensitivity, dy * sensitivity, 0)
                self.viewChanged.emit()
                
            self.last_pos = event.position()
        
        super().mouseMoveEvent(event)
    
    def keyPressEvent(self, event):
        """Handle key press events for model rotation and arrow keys"""
        if not self.view or not self.model_mesh:
            return super().keyPressEvent(event)
            
        # Current camera parameters
        azimuth = self.view.opts['azimuth']
        elevation = self.view.opts['elevation']
        
        # Rotation step
        step = 5.0
        
        # Handle W, S, A, D keys and arrow keys
        if event.key() == Qt.Key.Key_W or event.key() == Qt.Key.Key_Up:
            # W or Up - rotate up (increase elevation)
            elevation = min(90, elevation + step)
            self.view.setCameraPosition(elevation=elevation)
            self.viewChanged.emit()
        elif event.key() == Qt.Key.Key_S or event.key() == Qt.Key.Key_Down:
            # S or Down - rotate down (decrease elevation)
            elevation = max(-90, elevation - step)
            self.view.setCameraPosition(elevation=elevation)
            self.viewChanged.emit()
        elif event.key() == Qt.Key.Key_A or event.key() == Qt.Key.Key_Left:
            # A or Left - rotate left (decrease azimuth)
            azimuth = (azimuth - step) % 360
            self.view.setCameraPosition(azimuth=azimuth)
            self.viewChanged.emit()
        elif event.key() == Qt.Key.Key_D or event.key() == Qt.Key.Key_Right:
            # D or Right - rotate right (increase azimuth)
            azimuth = (azimuth + step) % 360
            self.view.setCameraPosition(azimuth=azimuth)
            self.viewChanged.emit()
        # Additional keys for convenience
        elif event.key() == Qt.Key.Key_T:
            # T - top view
            self.rotate_to_preset('top')
        elif event.key() == Qt.Key.Key_F:
            # F - front view
            self.rotate_to_preset('front')
        elif event.key() == Qt.Key.Key_R:
            # R - right side view
            self.rotate_to_preset('right')
        elif event.key() == Qt.Key.Key_I:
            # I - isometric view
            self.rotate_to_preset('isometric')
        elif event.key() == Qt.Key.Key_X:
            # X - align model to XY plane 
            self.align_model_to_xy_plane()
        elif event.key() == Qt.Key.Key_Space:
            # Space - toggle auto-orbit
            self.auto_orbit(not self._orbit_running)
        elif event.key() == Qt.Key.Key_Home:
            # Home - reset view
            self.reset_view()
        else:
            super().keyPressEvent(event)
        
    def resizeEvent(self, event):
        """Handle resizing of the widget"""
        super().resizeEvent(event)
        # Ensure the view is centered when resized
        if self.view and self.model_mesh:
            self.view.update()