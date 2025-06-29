"""
File Handler Module for Meshbuilder
Handles file operations, validation, and workspace management
"""
import os
import shutil
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

logger = logging.getLogger("MeshBuilder.FileHandler")

class FileHandler:
    """Handles file operations, validation, and workspace management"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config
        self.temp_dir = Path(self.config.get("Paths", "temp_dir", fallback="./temp"))
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def validate_inputs(self, input_paths: List[str], min_images: int = None) -> bool:
        """
        Validate input files
        
        Args:
            input_paths: List of paths to input files
            min_images: Minimum number of required images (overrides config)
            
        Returns:
            bool: Validity of inputs
        """
        if not input_paths:
            logger.error("No input files provided")
            return False
            
        # Count image files
        image_count = 0
        video_count = 0
        
        # Accepted file extensions
        image_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        
        for path in input_paths:
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                return False
                
            ext = os.path.splitext(path)[1].lower()
            
            if ext in image_exts:
                image_count += 1
            elif ext in video_exts:
                video_count += 1
            else:
                logger.error(f"Unsupported file format: {path}")
                return False
                
        # Check if enough images are provided
        if min_images is None:
            # Use the configuration setting with fallback to 20
            min_images = int(self.config.get("Processing", "min_images", fallback=20))
            
            # For test mode, allow a smaller number of images
            if os.environ.get("MESHBUILDER_TEST_MODE", "0") == "1":
                min_images = min(min_images, 3)  # Allow as few as 3 images for testing
        
        # Only enforce minimum if all inputs are images (videos are processed differently)
        if video_count == 0 and image_count < min_images:
            logger.error(f"Insufficient number of images. At least {min_images} required, but only {image_count} provided.")
            return False
            
        logger.info(f"Validated {image_count} images and {video_count} videos")
        return True
    
    def create_temp_workspace(self) -> Path:
        """
        Create a temporary workspace
        
        Returns:
            Path to temporary workspace
        """
        # Create a unique workspace identifier
        workspace_id = uuid.uuid4().hex[:8]
        workspace_path = self.temp_dir / f"workspace_{workspace_id}"
        
        # Create workspace and subdirectories
        os.makedirs(workspace_path, exist_ok=True)
        os.makedirs(workspace_path / "images", exist_ok=True)
        os.makedirs(workspace_path / "processed", exist_ok=True)
        os.makedirs(workspace_path / "output", exist_ok=True)
        
        logger.info(f"Created temporary workspace at {workspace_path}")
        return workspace_path
    
    def copy_to_workspace(self, file_path: str, target_dir: Path) -> str:
        """
        Copy a file to the workspace
        
        Args:
            file_path: Path to file
            target_dir: Target directory
            
        Returns:
            Path to file in workspace
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # Get file name and create destination path
        file_name = os.path.basename(file_path)
        dest_path = target_dir / file_name
        
        # Copy file
        shutil.copy2(file_path, dest_path)
        logger.debug(f"Copied {file_path} to {dest_path}")
        
        return str(dest_path)
    
    def cleanup_workspace(self, workspace_path: Path) -> bool:
        """
        Clean up workspace
        
        Args:
            workspace_path: Path to workspace
            
        Returns:
            Success or failure
        """
        try:
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
                logger.info(f"Cleaned up workspace at {workspace_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to clean up workspace: {str(e)}")
            return False
    
    def get_file_type(self, file_path: str) -> str:
        """
        Determine file type
        
        Args:
            file_path: Path to file
            
        Returns:
            File type ('image', 'video', or 'unknown')
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        image_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        
        if ext in image_exts:
            return 'image'
        elif ext in video_exts:
            return 'video'
        else:
            return 'unknown'