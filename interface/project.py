"""
MeshBuilder Project Management System
Handles project creation, loading, saving, and management

This module provides project-based organization for 3D reconstruction workflows
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("MeshBuilder.Project")


class Project:
    """Represents a MeshBuilder project"""
    
    def __init__(self, name: str, quality_level: str = "high", project_dir: Optional[Path] = None):
        """
        Initialize a project
        
        Args:
            name: Project name
            quality_level: Quality level for processing
            project_dir: Project directory (auto-generated if None)
        """
        self.name = name
        self.quality_level = quality_level
        self.created_date = datetime.now().isoformat()
        self.modified_date = self.created_date
        
        # Paths
        if project_dir is None:
            base_dir = Path.home() / "Documents" / "MeshBuilder" / "projects"
            self.project_dir = base_dir / name
        else:
            self.project_dir = Path(project_dir)
        
        # File lists
        self.image_paths = []
        self.video_paths = []
        
        # Processing results
        self.processed_images_dir = None
        self.dataset_path = None
        self.model_path = None
        self.output_path = None
        self.output_format = "obj"
        
        # Processing status
        self.processing_status = "created"  # created, processing, completed, failed
        self.processing_progress = 0
        
        # Preview paths
        self.previews = {}
        
        # Settings
        self.settings = self._get_default_settings()
        
        logger.info(f"Project initialized: {self.name}")
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default project settings"""
        return {
            "Processing": {
                "quality_level": self.quality_level,
                "feature_matcher": "exhaustive",
                "point_density": "high",
                "mesh_resolution": "high",
                "texture_resolution": "4096",
                "target_faces": "100000",
                "flat_grey_model": "false"
            },
            "Hardware": {
                "use_gpu": "true",
                "use_gaussian_splatting": "false"
            },
            "Output": {
                "format": self.output_format,
                "compress_output": "false"
            }
        }
    
    def create_directories(self):
        """Create project directory structure"""
        try:
            # Main project directory
            self.project_dir.mkdir(parents=True, exist_ok=True)
            
            # Subdirectories
            (self.project_dir / "input" / "images").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "input" / "videos").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "processed").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "dataset").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "output").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "previews").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created project directories for: {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to create project directories: {e}")
            raise
    
    def add_images(self, image_paths: List[str]) -> int:
        """
        Add images to the project
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Number of images successfully added
        """
        added_count = 0
        target_dir = self.project_dir / "input" / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_paths:
            try:
                img_path = Path(img_path)
                if img_path.exists() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    # Copy image to project directory
                    target_path = target_dir / img_path.name
                    if not target_path.exists():
                        shutil.copy2(img_path, target_path)
                        self.image_paths.append(str(target_path))
                        added_count += 1
                    else:
                        logger.warning(f"Image already exists in project: {img_path.name}")
                        # Add to list if not already there
                        if str(target_path) not in self.image_paths:
                            self.image_paths.append(str(target_path))
                            added_count += 1
                else:
                    logger.warning(f"Invalid image file: {img_path}")
            except Exception as e:
                logger.error(f"Failed to add image {img_path}: {e}")
        
        if added_count > 0:
            self.modified_date = datetime.now().isoformat()
        
        logger.info(f"Added {added_count} images to project {self.name}")
        return added_count
    
    def add_videos(self, video_paths: List[str]) -> int:
        """
        Add videos to the project
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            Number of videos successfully added
        """
        added_count = 0
        target_dir = self.project_dir / "input" / "videos"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for video_path in video_paths:
            try:
                video_path = Path(video_path)
                if video_path.exists() and video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Copy video to project directory
                    target_path = target_dir / video_path.name
                    if not target_path.exists():
                        shutil.copy2(video_path, target_path)
                        self.video_paths.append(str(target_path))
                        added_count += 1
                    else:
                        logger.warning(f"Video already exists in project: {video_path.name}")
                        # Add to list if not already there
                        if str(target_path) not in self.video_paths:
                            self.video_paths.append(str(target_path))
                            added_count += 1
                else:
                    logger.warning(f"Invalid video file: {video_path}")
            except Exception as e:
                logger.error(f"Failed to add video {video_path}: {e}")
        
        if added_count > 0:
            self.modified_date = datetime.now().isoformat()
        
        logger.info(f"Added {added_count} videos to project {self.name}")
        return added_count
    
    def get_media_count(self) -> Dict[str, int]:
        """Get count of media files in project"""
        return {
            "images": len(self.image_paths),
            "videos": len(self.video_paths),
            "total": len(self.image_paths) + len(self.video_paths)
        }
    
    def get_all_media_paths(self) -> List[str]:
        """Get all media file paths"""
        return self.image_paths + self.video_paths
    
    def set_output_path(self, output_path: str, output_format: str = "obj"):
        """Set output path and format"""
        self.output_path = str(Path(output_path).resolve())
        self.output_format = output_format.lower()
        self.modified_date = datetime.now().isoformat()
        logger.info(f"Set output path: {self.output_path} ({self.output_format})")
    
    def add_preview(self, preview_type: str, preview_path: str):
        """Add a preview image"""
        self.previews[preview_type] = str(Path(preview_path).resolve())
        self.modified_date = datetime.now().isoformat()
        logger.info(f"Added {preview_type} preview: {preview_path}")
    
    def update_processing_status(self, status: str, progress: int = 0):
        """Update processing status"""
        self.processing_status = status
        self.processing_progress = progress
        self.modified_date = datetime.now().isoformat()
        logger.info(f"Project {self.name} status: {status} ({progress}%)")
    
    def update_processing_results(self, success: bool, results: Dict[str, Any]):
        """Update processing results"""
        if success:
            self.processing_status = "completed"
            if "model_path" in results:
                self.model_path = results["model_path"]
            if "dataset_path" in results:
                self.dataset_path = results["dataset_path"]
            if "previews" in results:
                self.previews.update(results["previews"])
        else:
            self.processing_status = "failed"
        
        self.modified_date = datetime.now().isoformat()
        logger.info(f"Updated processing results for {self.name}: success={success}")
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update project settings"""
        self.settings.update(settings)
        self.modified_date = datetime.now().isoformat()
        logger.info(f"Updated settings for project {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary for serialization"""
        return {
            "name": self.name,
            "quality_level": self.quality_level,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "project_dir": str(self.project_dir),
            "image_paths": self.image_paths,
            "video_paths": self.video_paths,
            "processed_images_dir": self.processed_images_dir,
            "dataset_path": self.dataset_path,
            "model_path": self.model_path,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "processing_status": self.processing_status,
            "processing_progress": self.processing_progress,
            "previews": self.previews,
            "settings": self.settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create project from dictionary"""
        project = cls(
            name=data["name"],
            quality_level=data.get("quality_level", "high"),
            project_dir=Path(data["project_dir"])
        )
        
        # Restore all attributes
        project.created_date = data.get("created_date", project.created_date)
        project.modified_date = data.get("modified_date", project.modified_date)
        project.image_paths = data.get("image_paths", [])
        project.video_paths = data.get("video_paths", [])
        project.processed_images_dir = data.get("processed_images_dir")
        project.dataset_path = data.get("dataset_path")
        project.model_path = data.get("model_path")
        project.output_path = data.get("output_path")
        project.output_format = data.get("output_format", "obj")
        project.processing_status = data.get("processing_status", "created")
        project.processing_progress = data.get("processing_progress", 0)
        project.previews = data.get("previews", {})
        project.settings = data.get("settings", project._get_default_settings())
        
        return project


class ProjectManager:
    """Manages MeshBuilder projects"""
    
    def __init__(self, projects_dir: Optional[Path] = None):
        """
        Initialize project manager
        
        Args:
            projects_dir: Directory to store projects (defaults to user Documents)
        """
        if projects_dir is None:
            self.projects_dir = Path.home() / "Documents" / "MeshBuilder" / "projects"
        else:
            self.projects_dir = Path(projects_dir)
        
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for quick project lookup
        self.index_file = self.projects_dir / "projects_index.json"
        
        logger.info(f"ProjectManager initialized: {self.projects_dir}")
    
    def create_project(self, name: str, quality_level: str = "high") -> Project:
        """
        Create a new project
        
        Args:
            name: Project name
            quality_level: Quality level for processing
            
        Returns:
            Created project instance
        """
        # Sanitize project name
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        project_dir = self.projects_dir / safe_name
        
        # Check if project already exists
        if project_dir.exists():
            counter = 1
            while (self.projects_dir / f"{safe_name}_{counter}").exists():
                counter += 1
            safe_name = f"{safe_name}_{counter}"
            project_dir = self.projects_dir / safe_name
        
        # Create project
        project = Project(safe_name, quality_level, project_dir)
        project.create_directories()
        
        # Save project
        self.save_project(project)
        
        # Update index
        self._update_index()
        
        logger.info(f"Created project: {safe_name}")
        return project
    
    def save_project(self, project: Project) -> bool:
        """
        Save project to disk
        
        Args:
            project: Project to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project_file = project.project_dir / "project.json"
            
            with open(project_file, 'w') as f:
                json.dump(project.to_dict(), f, indent=2)
            
            logger.info(f"Saved project: {project.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save project {project.name}: {e}")
            return False
    
    def load_project(self, name: str) -> Optional[Project]:
        """
        Load project by name
        
        Args:
            name: Project name
            
        Returns:
            Loaded project or None if not found
        """
        try:
            project_dir = self.projects_dir / name
            project_file = project_dir / "project.json"
            
            if not project_file.exists():
                logger.warning(f"Project file not found: {project_file}")
                return None
            
            with open(project_file, 'r') as f:
                data = json.load(f)
            
            project = Project.from_dict(data)
            logger.info(f"Loaded project: {name}")
            return project
            
        except Exception as e:
            logger.error(f"Failed to load project {name}: {e}")
            return None
    
    def list_projects(self) -> List[str]:
        """
        List all project names
        
        Returns:
            List of project names
        """
        projects = []
        
        try:
            for item in self.projects_dir.iterdir():
                if item.is_dir():
                    project_file = item / "project.json"
                    if project_file.exists():
                        projects.append(item.name)
            
            projects.sort()
            
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
        
        return projects
    
    def get_projects_with_details(self) -> List[Dict[str, Any]]:
        """
        Get projects with detailed information
        
        Returns:
            List of project detail dictionaries
        """
        projects = []
        
        for project_name in self.list_projects():
            try:
                project_dir = self.projects_dir / project_name
                project_file = project_dir / "project.json"
                
                with open(project_file, 'r') as f:
                    data = json.load(f)
                
                # Extract key details
                project_info = {
                    "name": data.get("name", project_name),
                    "created_date": data.get("created_date", ""),
                    "modified_date": data.get("modified_date", ""),
                    "quality_level": data.get("quality_level", "unknown"),
                    "processing_status": data.get("processing_status", "unknown"),
                    "image_count": len(data.get("image_paths", [])),
                    "video_count": len(data.get("video_paths", [])),
                    "has_output": bool(data.get("model_path"))
                }
                
                projects.append(project_info)
                
            except Exception as e:
                logger.warning(f"Could not get details for project {project_name}: {e}")
        
        # Sort by modified date (newest first)
        projects.sort(key=lambda p: p.get("modified_date", ""), reverse=True)
        
        return projects
    
    def duplicate_project(self, source_name: str, new_name: str) -> Optional[Project]:
        """
        Duplicate an existing project
        
        Args:
            source_name: Name of source project
            new_name: Name for the new project
            
        Returns:
            Duplicated project or None if failed
        """
        try:
            source_project = self.load_project(source_name)
            if not source_project:
                logger.error(f"Source project not found: {source_name}")
                return None
            
            # Create new project with same settings
            new_project = self.create_project(new_name, source_project.quality_level)
            
            # Copy settings
            new_project.settings = source_project.settings.copy()
            new_project.output_format = source_project.output_format
            
            # Copy input files
            if source_project.image_paths:
                new_project.add_images(source_project.image_paths)
            
            if source_project.video_paths:
                new_project.add_videos(source_project.video_paths)
            
            # Save the duplicated project
            self.save_project(new_project)
            
            logger.info(f"Duplicated project {source_name} -> {new_name}")
            return new_project
            
        except Exception as e:
            logger.error(f"Failed to duplicate project {source_name}: {e}")
            return None
    
    def delete_project(self, name: str) -> bool:
        """
        Delete a project
        
        Args:
            name: Project name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project_dir = self.projects_dir / name
            
            if project_dir.exists():
                shutil.rmtree(project_dir)
                logger.info(f"Deleted project: {name}")
                
                # Update index
                self._update_index()
                
                return True
            else:
                logger.warning(f"Project directory not found: {project_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete project {name}: {e}")
            return False
    
    def _update_index(self):
        """Update the projects index file"""
        try:
            projects = self.get_projects_with_details()
            
            index_data = {
                "last_updated": datetime.now().isoformat(),
                "projects_count": len(projects),
                "projects": projects
            }
            
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update projects index: {e}")


def test_project_manager():
    """Test the project management system"""
    try:
        # Create project manager
        manager = ProjectManager()
        
        # Test project creation
        project = manager.create_project("test_project", "high")
        
        # Test project listing
        projects = manager.list_projects()
        
        # Test project details
        details = manager.get_projects_with_details()
        
        logger.info("ProjectManager test successful:")
        logger.info(f"  - Created project: {project.name}")
        logger.info(f"  - Found {len(projects)} projects")
        logger.info(f"  - Project details: {len(details)} entries")
        
        return True
        
    except Exception as e:
        logger.error(f"ProjectManager test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the project manager
    logging.basicConfig(level=logging.INFO)
    success = test_project_manager()
    if success:
        print("SUCCESS: ProjectManager is ready for use")
    else:
        print("FAILED: ProjectManager has issues")