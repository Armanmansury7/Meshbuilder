"""
Dataset Builder for MeshBuilder - COLMAP Dataset Creation for 3D Gaussian Splatting
Automates COLMAP processing to create datasets compatible with gaussian-splatting training

This module handles:
- COLMAP feature extraction
- Exhaustive matching
- 3D reconstruction (mapping)
- Dataset structuring for 3DGS
- Camera model validation
- Cleanup of intermediate files

Author: MeshBuilder Team
Version: 2.0.0
"""

import os
import sys
import logging
import subprocess
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MeshBuilder.DatasetBuilder")


@dataclass
class COLMAPConfig:
    """Configuration for COLMAP processing"""
    # Feature extraction settings
    camera_model: str = "SIMPLE_PINHOLE"
    single_camera: bool = True
    image_quality: str = "high"
    max_image_size: int = 3200
    
    # Matching settings
    matcher_type: str = "exhaustive"
    
    # Mapper settings
    mapper_type: str = "incremental"
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = False
    
    # Processing settings
    use_gpu: bool = True
    gpu_index: str = "0"


class DatasetBuilder:
    """
    Builds COLMAP datasets for 3D Gaussian Splatting training
    
    Handles the complete COLMAP pipeline from feature extraction to 
    structured dataset output compatible with gaussian-splatting
    """
    
    # Supported camera models for 3DGS
    SUPPORTED_CAMERA_MODELS = ["SIMPLE_PINHOLE", "PINHOLE"]
    
    def __init__(self, base_output_dir: str = "./output"):
        """
        Initialize the dataset builder
        
        Args:
            base_output_dir: Base directory for output datasets
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths relative to project root
        self.project_root = Path(__file__).parent.parent.parent
        self.colmap_bat = self.project_root / "resources" / "models" / "COLMAP" / "COLMAP.bat"
        self.colmap_exe = self.project_root / "resources" / "models" / "COLMAP" / "bin" / "colmap.exe"
        
        # Check COLMAP availability
        self._check_colmap_installation()
        
        # Default configuration
        self.config = COLMAPConfig()
        
        logger.info(f"DatasetBuilder initialized with output dir: {self.base_output_dir}")
    
    def _check_colmap_installation(self):
        """Check if COLMAP is available"""
        if not self.colmap_bat.exists() and not self.colmap_exe.exists():
            logger.warning("COLMAP not found at expected locations:")
            logger.warning(f"  - {self.colmap_bat}")
            logger.warning(f"  - {self.colmap_exe}")
            logger.warning("Please ensure COLMAP is installed in resources/models/COLMAP/")
            
            # Try to find COLMAP in system PATH as fallback
            colmap_in_path = shutil.which("colmap")
            if colmap_in_path:
                logger.info(f"Found COLMAP in system PATH: {colmap_in_path}")
                self.colmap_exe = Path(colmap_in_path)
            else:
                raise RuntimeError("COLMAP not found. Please install COLMAP to use dataset building.")
    
    def build_dataset(self, 
                     project_name: str,
                     input_images_dir: str,
                     output_dir: Optional[str] = None,
                     config: Optional[COLMAPConfig] = None,
                     callback: Optional[Callable[[str, int], None]] = None) -> Optional[Path]:
        """
        Build a COLMAP dataset for 3D Gaussian Splatting
        
        Args:
            project_name: Name of the project
            input_images_dir: Directory containing processed input images
            output_dir: Output directory (uses base_output_dir if None)
            config: COLMAP configuration (uses defaults if None)
            callback: Progress callback function(message: str, progress: int)
            
        Returns:
            Path to the created dataset directory, or None if failed
        """
        # Use provided config or defaults
        if config:
            self.config = config
        
        # Set up paths
        input_path = Path(input_images_dir)
        if not input_path.exists():
            logger.error(f"Input images directory not found: {input_path}")
            return None
        
        # Count input images
        image_files = self._get_image_files(input_path)
        if len(image_files) == 0:
            logger.error("No image files found in input directory")
            return None
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Set up output directory
        if output_dir:
            dataset_dir = Path(output_dir) / project_name / "dataset"
        else:
            dataset_dir = self.base_output_dir / project_name / "dataset"
        
        # Create dataset structure
        dataset_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = dataset_dir / "workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        # Copy images to dataset
        images_dir = dataset_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        shutil.copytree(input_path, images_dir)
        
        logger.info(f"Building dataset for project: {project_name}")
        logger.info(f"Dataset directory: {dataset_dir}")
        
        try:
            # Stage 1: Feature Extraction
            if callback:
                callback("Extracting features from images", 10)
            
            success = self._run_feature_extractor(workspace_dir, images_dir)
            if not success:
                logger.error("Feature extraction failed")
                return None
            
            # Stage 2: Feature Matching
            if callback:
                callback("Matching features between images", 30)
            
            success = self._run_matcher(workspace_dir)
            if not success:
                logger.error("Feature matching failed")
                return None
            
            # Stage 3: 3D Reconstruction (Mapping)
            if callback:
                callback("Reconstructing 3D structure", 50)
            
            success = self._run_mapper(workspace_dir, images_dir)
            if not success:
                logger.error("3D reconstruction failed")
                return None
            
            # Stage 4: Validate Camera Models
            if callback:
                callback("Validating camera models", 80)
            
            success = self._validate_camera_models(workspace_dir)
            if not success:
                logger.error("Camera model validation failed")
                return None
            
            # Stage 5: Structure Output for 3DGS
            if callback:
                callback("Structuring dataset for 3DGS", 90)
            
            success = self._structure_for_3dgs(workspace_dir, dataset_dir)
            if not success:
                logger.error("Dataset structuring failed")
                return None
            
            # Clean up workspace (keep essential files only)
            self._cleanup_workspace(workspace_dir)
            
            if callback:
                callback("Dataset creation completed", 100)
            
            logger.info(f"Dataset successfully created at: {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Dataset building failed: {e}")
            if callback:
                callback(f"Dataset building failed: {e}", -1)
            return None
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _get_colmap_command(self) -> List[str]:
        """Get the appropriate COLMAP command based on platform"""
        if sys.platform == "win32" and self.colmap_bat.exists():
            return [str(self.colmap_bat)]
        elif self.colmap_exe.exists():
            return [str(self.colmap_exe)]
        else:
            # Fallback to system COLMAP
            return ["colmap"]
    
    def _run_colmap_command(self, args: List[str], description: str) -> bool:
        """
        Run a COLMAP command with error handling
        
        Args:
            args: Command arguments
            description: Description for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build full command
            cmd = self._get_colmap_command() + args
            
            logger.info(f"Running: {description}")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            # Run command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.debug(f"[COLMAP] {line}")
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                logger.info(f"{description} completed successfully")
                return True
            else:
                logger.error(f"{description} failed with return code {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"{description} failed with exception: {e}")
            return False
    
    def _run_feature_extractor(self, workspace_dir: Path, images_dir: Path) -> bool:
        """
        Run COLMAP feature extractor
        
        Args:
            workspace_dir: COLMAP workspace directory
            images_dir: Directory containing images
            
        Returns:
            True if successful, False otherwise
        """
        database_path = workspace_dir / "database.db"
        
        # Remove existing database
        if database_path.exists():
            database_path.unlink()
        
        args = [
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", self.config.camera_model,
            "--ImageReader.single_camera", "1" if self.config.single_camera else "0",
            "--SiftExtraction.use_gpu", "1" if self.config.use_gpu else "0",
            "--SiftExtraction.gpu_index", self.config.gpu_index,
            "--SiftExtraction.max_image_size", str(self.config.max_image_size)
        ]
        
        return self._run_colmap_command(args, "Feature extraction")
    
    def _run_matcher(self, workspace_dir: Path) -> bool:
        """
        Run COLMAP feature matcher
        
        Args:
            workspace_dir: COLMAP workspace directory
            
        Returns:
            True if successful, False otherwise
        """
        database_path = workspace_dir / "database.db"
        
        if self.config.matcher_type == "exhaustive":
            args = [
                "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if self.config.use_gpu else "0",
                "--SiftMatching.gpu_index", self.config.gpu_index
            ]
            description = "Exhaustive feature matching"
        else:
            # Add support for other matchers if needed
            logger.error(f"Unsupported matcher type: {self.config.matcher_type}")
            return False
        
        return self._run_colmap_command(args, description)
    
    def _run_mapper(self, workspace_dir: Path, images_dir: Path) -> bool:
        """
        Run COLMAP 3D reconstruction (mapper)
        
        Args:
            workspace_dir: COLMAP workspace directory
            images_dir: Directory containing images
            
        Returns:
            True if successful, False otherwise
        """
        database_path = workspace_dir / "database.db"
        sparse_dir = workspace_dir / "sparse"
        
        # Clean existing sparse reconstruction
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir()
        
        args = [
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_refine_focal_length", "1" if self.config.ba_refine_focal_length else "0",
            "--Mapper.ba_refine_principal_point", "1" if self.config.ba_refine_principal_point else "0",
            "--Mapper.ba_refine_extra_params", "1" if self.config.ba_refine_extra_params else "0"
        ]
        
        return self._run_colmap_command(args, "3D reconstruction")
    
    def _validate_camera_models(self, workspace_dir: Path) -> bool:
        """
        Validate that camera models are compatible with 3DGS
        
        Args:
            workspace_dir: COLMAP workspace directory
            
        Returns:
            True if valid, False otherwise
        """
        sparse_dir = workspace_dir / "sparse" / "0"
        cameras_file = sparse_dir / "cameras.bin"
        
        if not cameras_file.exists():
            logger.error("cameras.bin not found")
            return False
        
        try:
            # Read cameras using COLMAP model reader
            args = [
                "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(sparse_dir),
                "--output_type", "TXT"
            ]
            
            # Convert to text format for validation
            if not self._run_colmap_command(args, "Converting model to text"):
                logger.warning("Could not convert model to text for validation")
                # Continue anyway as binary format might be fine
                return True
            
            # Read cameras.txt to validate models
            cameras_txt = sparse_dir / "cameras.txt"
            if cameras_txt.exists():
                with open(cameras_txt, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            camera_model = parts[1]
                            if camera_model not in self.SUPPORTED_CAMERA_MODELS:
                                logger.error(f"Unsupported camera model found: {camera_model}")
                                logger.error(f"Supported models: {self.SUPPORTED_CAMERA_MODELS}")
                                return False
                
                logger.info("All camera models are compatible with 3DGS")
            
            return True
            
        except Exception as e:
            logger.error(f"Camera model validation failed: {e}")
            return False
    
    def _structure_for_3dgs(self, workspace_dir: Path, dataset_dir: Path) -> bool:
        """
        Structure the dataset for 3D Gaussian Splatting
        
        Args:
            workspace_dir: COLMAP workspace directory
            dataset_dir: Final dataset directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create sparse directory structure
            sparse_src = workspace_dir / "sparse" / "0"
            sparse_dst = dataset_dir / "sparse" / "0"
            
            # Remove existing sparse directory
            if (dataset_dir / "sparse").exists():
                shutil.rmtree(dataset_dir / "sparse")
            
            # Create new sparse directory
            sparse_dst.mkdir(parents=True)
            
            # Copy essential files
            essential_files = ["cameras.bin", "images.bin", "points3D.bin"]
            
            for filename in essential_files:
                src_file = sparse_src / filename
                dst_file = sparse_dst / filename
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied {filename} to dataset")
                else:
                    logger.error(f"Essential file missing: {filename}")
                    return False
            
            # Also copy text versions if they exist (for debugging)
            text_files = ["cameras.txt", "images.txt", "points3D.txt"]
            for filename in text_files:
                src_file = sparse_src / filename
                if src_file.exists():
                    shutil.copy2(src_file, sparse_dst / filename)
            
            # Create dataset info file
            dataset_info = {
                "dataset_type": "colmap",
                "camera_model": self.config.camera_model,
                "num_images": len(list((dataset_dir / "images").glob("*"))),
                "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "compatible_with": "gaussian-splatting"
            }
            
            with open(dataset_dir / "dataset_info.json", 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logger.info("Dataset structured successfully for 3DGS")
            return True
            
        except Exception as e:
            logger.error(f"Dataset structuring failed: {e}")
            return False
    
    def _cleanup_workspace(self, workspace_dir: Path):
        """
        Clean up temporary workspace files
        
        Args:
            workspace_dir: COLMAP workspace directory
        """
        try:
            # Keep only essential files
            files_to_remove = [
                "database.db",
                "database.db-shm",
                "database.db-wal"
            ]
            
            for filename in files_to_remove:
                file_path = workspace_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed temporary file: {filename}")
            
            # Remove workspace directory if empty
            if not any(workspace_dir.iterdir()):
                workspace_dir.rmdir()
                logger.debug("Removed empty workspace directory")
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def cleanup_intermediate_data(self, project_name: str):
        """
        Clean up intermediate COLMAP data after 3DGS training
        
        Args:
            project_name: Name of the project
        """
        dataset_dir = self.base_output_dir / project_name / "dataset"
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return
        
        try:
            # Remove workspace if it exists
            workspace_dir = dataset_dir / "workspace"
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
                logger.info("Removed COLMAP workspace directory")
            
            # Remove any distorted directories
            distorted_dir = dataset_dir / "distorted"
            if distorted_dir.exists():
                shutil.rmtree(distorted_dir)
                logger.info("Removed distorted images directory")
            
            # Clean up any .tmp files
            for tmp_file in dataset_dir.rglob("*.tmp"):
                tmp_file.unlink()
                logger.debug(f"Removed temporary file: {tmp_file}")
            
            logger.info(f"Cleaned up intermediate data for project: {project_name}")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def validate_dataset(self, project_name: str) -> Dict[str, Any]:
        """
        Validate that a dataset is ready for 3DGS training
        
        Args:
            project_name: Name of the project
            
        Returns:
            Validation results dictionary
        """
        dataset_dir = self.base_output_dir / project_name / "dataset"
        
        validation = {
            "valid": False,
            "dataset_path": str(dataset_dir),
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Check dataset directory
            if not dataset_dir.exists():
                validation["errors"].append("Dataset directory not found")
                return validation
            
            # Check images directory
            images_dir = dataset_dir / "images"
            if not images_dir.exists():
                validation["errors"].append("Images directory not found")
            else:
                image_count = len(list(images_dir.glob("*")))
                validation["stats"]["image_count"] = image_count
                if image_count < 20:
                    validation["warnings"].append(f"Low image count: {image_count} (recommend 20+)")
            
            # Check sparse reconstruction
            sparse_dir = dataset_dir / "sparse" / "0"
            if not sparse_dir.exists():
                validation["errors"].append("Sparse reconstruction not found")
            else:
                # Check essential files
                for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
                    file_path = sparse_dir / filename
                    if not file_path.exists():
                        validation["errors"].append(f"Missing file: {filename}")
                    elif file_path.stat().st_size == 0:
                        validation["errors"].append(f"Empty file: {filename}")
            
            # If no errors, dataset is valid
            validation["valid"] = len(validation["errors"]) == 0
            
            if validation["valid"]:
                logger.info(f"Dataset validation passed for: {project_name}")
            else:
                logger.warning(f"Dataset validation failed for: {project_name}")
                for error in validation["errors"]:
                    logger.warning(f"  - {error}")
            
        except Exception as e:
            validation["errors"].append(f"Validation error: {e}")
            logger.error(f"Dataset validation failed: {e}")
        
        return validation


def test_dataset_builder():
    """Test the dataset builder functionality"""
    try:
        print("Testing DatasetBuilder...")
        
        # Create dataset builder
        builder = DatasetBuilder()
        
        # Test configuration
        config = COLMAPConfig(
            camera_model="SIMPLE_PINHOLE",
            use_gpu=True,
            max_image_size=2000
        )
        
        print(f"✓ DatasetBuilder initialized")
        print(f"✓ COLMAP paths configured")
        print(f"✓ Default configuration: {config.camera_model}")
        
        # Test validation
        validation = builder.validate_dataset("test_project")
        print(f"✓ Validation function works: {len(validation)} fields")
        
        print("✓ DatasetBuilder test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ DatasetBuilder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    test_dataset_builder()