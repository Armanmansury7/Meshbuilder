"""
MeshBuilder Enhanced Connector
Integrates all backend functionality and provides clean interface to frontend
"""
import os
import sys
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QThread

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import backend components
from backend.processors.image_processor import ImageProcessor
from backend.processors.dataset_builder import DatasetBuilder
from backend.processors.splat_trainer import GaussianSplattingTrainer
from backend.processors.point_cloud_converter import PointCloudToMeshConverter
from backend.processors.quality_manager import QualityManager
from interface.project import ProjectManager, Project

logger = logging.getLogger("MeshBuilder.Connector")


class ProcessingThread(QThread):
    """Thread for running the processing pipeline"""
    
    # Signals
    stageChanged = pyqtSignal(str, str)  # stage_id, description
    progressUpdated = pyqtSignal(str, int)  # message, percentage
    processingError = pyqtSignal(str, str)  # title, message
    processingFinished = pyqtSignal(bool, str)  # success, result_path
    previewReady = pyqtSignal(str, str)  # preview_type, path
    
    def __init__(self, project, output_path, settings):
        super().__init__()
        self.project = project
        self.output_path = output_path
        self.settings = settings
        self.should_cancel = False
        
    def run(self):
        """Run the complete processing pipeline"""
        try:
            # Stage 1: Image Processing
            self.stageChanged.emit("image_processing", "Processing images...")
            processed_paths = self._process_images()
            
            if self.should_cancel:
                self.processingFinished.emit(False, "Processing cancelled")
                return
                
            # Stage 2: Dataset Building
            self.stageChanged.emit("dataset_building", "Building 3DGS dataset...")
            dataset_path = self._build_dataset(processed_paths)
            
            if self.should_cancel:
                self.processingFinished.emit(False, "Processing cancelled")
                return
                
            # Stage 3: 3D Gaussian Splatting Training
            self.stageChanged.emit("gaussian_splatting", "Training 3D model...")
            training_result = self._train_gaussian_splatting()
            
            if self.should_cancel:
                self.processingFinished.emit(False, "Processing cancelled")
                return
                
            # Stage 4: Point Cloud to Mesh Conversion
            self.stageChanged.emit("mesh_conversion", "Converting to mesh...")
            mesh_path = self._convert_to_mesh()
            
            if mesh_path:
                self.processingFinished.emit(True, mesh_path)
            else:
                self.processingFinished.emit(False, "Mesh conversion failed")
                
        except Exception as e:
            logger.error(f"Processing pipeline error: {e}")
            self.processingError.emit("Processing Error", str(e))
            self.processingFinished.emit(False, str(e))
    
    def _process_images(self):
        """Process images stage"""
        processor = ImageProcessor()
        
        output_dir = Path(self.project.project_dir) / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # Get quality settings
        quality_threshold = float(self.settings.get("Processing", {}).get("quality_threshold", 0.5))
        enhancement_mode = self.settings.get("Processing", {}).get("color_enhancement_mode", "balanced")
        use_gpu = self.settings.get("Hardware", {}).get("use_gpu", "true").lower() == "true"
        
        def progress_callback(message, progress):
            self.progressUpdated.emit(message, int(progress * 0.2))  # 0-20%
            
        processed_paths = processor.preprocess_images(
            self.project.image_paths,
            output_dir,
            quality_threshold=quality_threshold,
            enhancement_mode=enhancement_mode,
            use_gpu=use_gpu,
            callback=progress_callback
        )
        
        self.project.processed_images_dir = str(output_dir)
        return processed_paths
    
    def _build_dataset(self, processed_images):
        """Build COLMAP dataset for 3DGS"""
        builder = DatasetBuilder()
        
        def progress_callback(message, progress):
            self.progressUpdated.emit(message, 20 + int(progress * 0.2))  # 20-40%
            
        dataset_path = builder.build_dataset(
            self.project.name,
            self.project.processed_images_dir,
            output_dir=str(self.project.project_dir.parent),
            callback=progress_callback
        )
        
        if dataset_path:
            self.project.dataset_path = str(dataset_path)
            
        return dataset_path
    
    def _train_gaussian_splatting(self):
        """Train 3D Gaussian Splatting model"""
        trainer = GaussianSplattingTrainer(
            base_output_dir=str(self.project.project_dir.parent)
        )
        
        # Get quality-based parameters
        quality_level = self.settings.get("Processing", {}).get("quality_level", "high")
        
        # Map quality to iterations and resolution
        quality_map = {
            "fast": (5000, 1000),
            "balanced": (10000, 1200),
            "high": (15000, 1200),
            "very_high": (20000, 1400),
            "photorealistic": (30000, 1600)
        }
        
        iterations, resolution = quality_map.get(quality_level, (15000, 1200))
        
        def progress_callback(message, progress):
            self.progressUpdated.emit(message, 40 + int(progress * 0.4))  # 40-80%
            
        result = trainer.train_model(
            self.project.name,
            resolution=resolution,
            iterations=iterations,
            callback=progress_callback
        )
        
        if result["success"] and result["output_path"]:
            self.project.model_path = result["output_path"]
            
        return result
    
    def _convert_to_mesh(self):
        """Convert point cloud to mesh"""
        converter = PointCloudToMeshConverter(
            base_output_dir=str(self.project.project_dir.parent)
        )
        
        # Get target triangles from settings
        target_triangles = int(self.settings.get("Processing", {}).get("target_faces", 100000))
        
        def progress_callback(message, progress):
            self.progressUpdated.emit(message, 80 + int(progress * 0.2))  # 80-100%
            
        success = converter.convert_project(
            self.project.name,
            method='poisson',
            target_triangles=target_triangles,
            callback=progress_callback
        )
        
        if success:
            # Get the output path
            mesh_dir = Path(self.project.project_dir.parent) / self.project.name / "mesh_output"
            mesh_path = mesh_dir / "model.obj"
            
            if mesh_path.exists():
                # Copy to user's specified output path
                import shutil
                output_format = self.settings.get("Output", {}).get("format", "obj")
                final_path = Path(self.output_path)
                
                if not final_path.suffix:
                    final_path = final_path.with_suffix(f".{output_format}")
                    
                shutil.copy2(mesh_path, final_path)
                return str(final_path)
                
        return None
    
    def cancel(self):
        """Cancel processing"""
        self.should_cancel = True


class MeshBuilderConnector(QObject):
    """Enhanced connector for MeshBuilder backend integration"""
    
    # Signals
    processingStarted = pyqtSignal()
    processingFinished = pyqtSignal(bool, str)  # success, result_path
    processingError = pyqtSignal(str, str)  # title, message
    progressUpdated = pyqtSignal(str, int)  # message, percentage
    stageChanged = pyqtSignal(str, str)  # stage_id, description
    previewReady = pyqtSignal(str, str)  # preview_type, path
    resourceUpdate = pyqtSignal(dict)  # resource usage data
    memoryOptimization = pyqtSignal(str)  # optimization message
    licenseValidationRequired = pyqtSignal(str)  # message
    systemSpecsDetected = pyqtSignal(dict)  # system specifications
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Initialize managers
        self.project_manager = ProjectManager()
        self.quality_manager = QualityManager()
        
        # Current state
        self.current_project = None
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize system specs
        self._detect_system_specs()
        
        logger.info("MeshBuilderConnector initialized")
    
    def _detect_system_specs(self):
        """Detect and emit system specifications"""
        try:
            import psutil
            specs = {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'gpu_available': False,
                'gpu_name': 'Unknown'
            }
            
            # Check for GPU
            try:
                import torch
                if torch.cuda.is_available():
                    specs['gpu_available'] = True
                    specs['gpu_name'] = torch.cuda.get_device_name(0)
                    specs['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
            
            # Get recommended quality
            recommendations = self.quality_manager.recommend_quality_level()
            specs['quality_level'] = recommendations['recommended_quality'].value
            
            self.systemSpecsDetected.emit(specs)
            
        except Exception as e:
            logger.error(f"Error detecting system specs: {e}")
    
    def create_project(self, name: str, quality: str = "high") -> bool:
        """Create a new project"""
        try:
            self.current_project = self.project_manager.create_project(name, quality)
            return True
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False
    
    def load_project(self, name: str) -> bool:
        """Load an existing project"""
        try:
            self.current_project = self.project_manager.load_project(name)
            return self.current_project is not None
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return False
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get current project information"""
        if not self.current_project:
            return {}
            
        return {
            "name": self.current_project.name,
            "quality_level": self.current_project.quality_level,
            "media_count": self.current_project.get_media_count(),
            "output_path": self.current_project.output_path,
            "processing_status": self.current_project.processing_status,
            "system_specs": {
                "total_memory_gb": self.quality_manager.system_specs.ram_gb,
                "gpu_available": self.quality_manager.system_specs.gpu_available
            }
        }
    
    def add_media_files(self, files: List[str]) -> Dict[str, int]:
        """Add media files to current project"""
        if not self.current_project:
            raise ValueError("No project loaded")
            
        total_added = 0
        images_added = 0
        videos_added = 0
        
        # Add images
        image_files = [f for f in files if Path(f).suffix.lower() in 
                      ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']]
        if image_files:
            images_added = self.current_project.add_images(image_files)
            total_added += images_added
        
        # Add videos (if any)
        video_files = [f for f in files if Path(f).suffix.lower() in 
                      ['.mp4', '.avi', '.mov', '.mkv']]
        if video_files:
            videos_added = self.current_project.add_videos(video_files)
            total_added += videos_added
        
        # Save project
        self.project_manager.save_project(self.current_project)
        
        return {
            "total": total_added,
            "images": images_added,
            "videos": videos_added
        }
    
    def start_processing(self, output_path: str, settings: Dict[str, Any]) -> bool:
        """Start the processing pipeline"""
        if not self.current_project:
            self.processingError.emit("No Project", "No project loaded")
            return False
            
        if self.is_processing:
            self.processingError.emit("Already Processing", "Processing already in progress")
            return False
        
        # Validate media count
        media_count = self.current_project.get_media_count()
        if media_count["total"] < 20:
            self.processingError.emit(
                "Insufficient Images", 
                f"Need at least 20 images, found {media_count['total']}"
            )
            return False
            
        # Update project settings
        self.current_project.update_settings(settings)
        self.current_project.set_output_path(output_path)
        
        # Create processing thread
        self.processing_thread = ProcessingThread(
            self.current_project,
            output_path,
            settings
        )
        
        # Connect signals
        self.processing_thread.stageChanged.connect(self.stageChanged.emit)
        self.processing_thread.progressUpdated.connect(self.progressUpdated.emit)
        self.processing_thread.processingError.connect(self.processingError.emit)
        self.processing_thread.processingFinished.connect(self._on_processing_finished)
        self.processing_thread.previewReady.connect(self.previewReady.emit)
        
        # Start processing
        self.is_processing = True
        self.processingStarted.emit()
        self.processing_thread.start()
        
        return True
    
    def cancel_processing(self):
        """Cancel ongoing processing"""
        if self.processing_thread and self.is_processing:
            self.processing_thread.cancel()
            self.processing_thread.wait()
            self.is_processing = False
    
    def _on_processing_finished(self, success: bool, result_path: str):
        """Handle processing completion"""
        self.is_processing = False
        
        if success and self.current_project:
            # Update project with results
            self.current_project.update_processing_status("completed", 100)
            self.project_manager.save_project(self.current_project)
        
        self.processingFinished.emit(success, result_path)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            "is_processing": self.is_processing,
            "current_project": self.current_project.name if self.current_project else None
        }
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update project settings"""
        if self.current_project:
            self.current_project.update_settings(settings)
            self.project_manager.save_project(self.current_project)
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system recommendations"""
        recommendations = self.quality_manager.recommend_quality_level()
        return {
            "recommended_quality": recommendations["recommended_quality"].value,
            "reasoning": recommendations["reasoning"],
            "alternatives": [alt["quality"].value for alt in recommendations.get("alternatives", [])]
        }
    
    def validate_license(self) -> bool:
        """Validate license (placeholder)"""
        # License validation is handled in main.py
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_processing:
            self.cancel_processing()
        logger.info("Connector cleanup completed")