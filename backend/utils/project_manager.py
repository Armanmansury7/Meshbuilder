"""
Project Manager Module for MeshBuilder
Handles project creation, loading, and management
"""
import os
import json
import logging
import datetime
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("MeshBuilder.ProjectManager")

class ProjectManager:
    """Manages MeshBuilder projects, including saving and loading"""
    
    def __init__(self, base_dir=None):
        """
        Initialize project manager
        
        Args:
            base_dir: Base directory for projects
        """
        self.base_dir = base_dir or os.path.join(os.path.expanduser("~"), "MeshBuilder", "Projects")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Load project registry
        self.registry_path = os.path.join(self.base_dir, "project_registry.json")
        self.project_registry = self._load_registry()
    
    def create_project(self, name: str, description: str = "") -> str:
        """
        Create a new project
        
        Args:
            name: Project name
            description: Optional project description
            
        Returns:
            Project ID
        """
        # Generate project ID (timestamp-based for uniqueness)
        project_id = f"proj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create project directory
        project_dir = os.path.join(self.base_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(project_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "output"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "temp"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "previews"), exist_ok=True)
        
        # Create project metadata
        metadata = {
            "project_id": project_id,
            "name": name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "last_modified": datetime.datetime.now().isoformat(),
            "status": "created",
            "image_count": 0,
            "model_output": None
        }
        
        # Save metadata
        with open(os.path.join(project_dir, "project.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update registry
        self.project_registry[project_id] = {
            "name": name,
            "description": description,
            "path": project_dir,
            "created_at": metadata["created_at"],
            "last_modified": metadata["last_modified"],
            "status": metadata["status"]
        }
        self._save_registry()
        
        logger.info(f"Created new project: {name} (ID: {project_id})")
        return project_id
    
    def get_project_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all projects
        
        Returns:
            List of project dictionaries
        """
        projects = []
        
        for project_id, info in self.project_registry.items():
            projects.append({
                "project_id": project_id,
                "name": info.get("name", "Unnamed Project"),
                "description": info.get("description", ""),
                "created_at": info.get("created_at", ""),
                "last_modified": info.get("last_modified", ""),
                "status": info.get("status", "unknown"),
                "path": info.get("path", "")
            })
            
        # Sort by last modified (newest first)
        projects.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
        
        return projects
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete project information
        
        Args:
            project_id: Project ID
            
        Returns:
            Project information dictionary or None if not found
        """
        if project_id not in self.project_registry:
            logger.warning(f"Project not found: {project_id}")
            return None
            
        project_info = self.project_registry[project_id].copy()
        
        # Add detailed information
        try:
            # Load project metadata
            metadata = self.get_project_metadata(project_id)
            if metadata:
                project_info.update(metadata)
                
            # Get image files
            project_dir = project_info["path"]
            input_dir = os.path.join(project_dir, "input")
            
            image_files = []
            if os.path.exists(input_dir):
                for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                    image_files.extend([
                        os.path.join(input_dir, f) for f in os.listdir(input_dir)
                        if f.lower().endswith(ext)
                    ])
            
            project_info["image_files"] = image_files
            project_info["image_count"] = len(image_files)
            
            # Get model files
            output_dir = os.path.join(project_dir, "output")
            
            model_files = []
            if os.path.exists(output_dir):
                for ext in ['.obj', '.fbx', '.gltf', '.glb', '.ply']:
                    model_files.extend([
                        os.path.join(output_dir, f) for f in os.listdir(output_dir)
                        if f.lower().endswith(ext)
                    ])
            
            project_info["model_files"] = model_files
            
            # Get preview files
            preview_dir = os.path.join(project_dir, "previews")
            
            preview_files = []
            if os.path.exists(preview_dir):
                for ext in ['.png', '.jpg', '.jpeg']:
                    preview_files.extend([
                        os.path.join(preview_dir, f) for f in os.listdir(preview_dir)
                        if f.lower().endswith(ext)
                    ])
            
            project_info["preview_files"] = preview_files
            
            return project_info
            
        except Exception as e:
            logger.error(f"Error getting project details: {str(e)}")
            return project_info  # Return basic info if detailed info fails
    
    def get_project_path(self, project_id: str) -> Optional[str]:
        """
        Get project directory path
        
        Args:
            project_id: Project ID
            
        Returns:
            Project directory path or None if not found
        """
        if project_id in self.project_registry:
            return self.project_registry[project_id].get("path")
        return None
    
    def get_project_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project metadata
        
        Args:
            project_id: Project ID
            
        Returns:
            Project metadata dictionary or None if not found
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            return None
            
        try:
            metadata_path = os.path.join(project_path, "project.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading project metadata: {str(e)}")
            return None
    
    def update_project_metadata(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update project metadata
        
        Args:
            project_id: Project ID
            updates: Dictionary with metadata updates
            
        Returns:
            Success or failure
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            logger.error(f"Project not found: {project_id}")
            return False
            
        try:
            # Update metadata file
            metadata_path = os.path.join(project_path, "project.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"project_id": project_id}
                
            # Update last_modified
            metadata["last_modified"] = datetime.datetime.now().isoformat()
            
            # Apply updates
            metadata.update(updates)
                
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Update registry
            if project_id in self.project_registry:
                # Update basic registry info
                for key in ["name", "description", "status"]:
                    if key in updates:
                        self.project_registry[project_id][key] = updates[key]
                
                self.project_registry[project_id]["last_modified"] = metadata["last_modified"]
                self._save_registry()
                
            logger.info(f"Updated project metadata: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating project metadata: {str(e)}")
            return False
    
    def add_images_to_project(self, project_id: str, image_paths: List[str], 
                            copy_files: bool = True) -> int:
        """
        Add images to a project
        
        Args:
            project_id: Project ID
            image_paths: List of image file paths
            copy_files: Whether to copy files to project directory (True) or move them (False)
            
        Returns:
            Number of images successfully added
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            logger.error(f"Project not found: {project_id}")
            return 0
            
        input_dir = os.path.join(project_path, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        added_count = 0
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
                
            # Get destination path
            dest_path = os.path.join(input_dir, os.path.basename(img_path))
            
            try:
                # Copy or move the file
                if copy_files:
                    shutil.copy2(img_path, dest_path)
                else:
                    shutil.move(img_path, dest_path)
                    
                added_count += 1
                    
            except Exception as e:
                logger.error(f"Error adding image {img_path}: {str(e)}")
        
        # Update metadata with new image count
        if added_count > 0:
            metadata = self.get_project_metadata(project_id) or {}
            current_count = metadata.get("image_count", 0)
            
            self.update_project_metadata(project_id, {
                "image_count": current_count + added_count,
                "status": "images_added"
            })
            
        logger.info(f"Added {added_count} images to project {project_id}")
        return added_count
    
    def save_model_to_project(self, project_id: str, model_path: str, 
                            copy_file: bool = True) -> bool:
        """
        Save a model to a project
        
        Args:
            project_id: Project ID
            model_path: Path to model file
            copy_file: Whether to copy file (True) or move it (False)
            
        Returns:
            Success or failure
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            logger.error(f"Project not found: {project_id}")
            return False
            
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        try:
            # Create output directory
            output_dir = os.path.join(project_path, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get destination path
            dest_path = os.path.join(output_dir, os.path.basename(model_path))
            
            # Copy or move the file
            if copy_file:
                shutil.copy2(model_path, dest_path)
            else:
                shutil.move(model_path, dest_path)
                
            # Update metadata
            model_rel_path = os.path.relpath(dest_path, project_path)
            self.update_project_metadata(project_id, {
                "model_output": model_rel_path,
                "status": "model_generated",
                "model_generated_at": datetime.datetime.now().isoformat()
            })
            
            logger.info(f"Saved model to project {project_id}: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to project: {str(e)}")
            return False
    
    def save_project_settings(self, project_id: str, settings: Dict[str, Any]) -> bool:
        """
        Save processing settings to project
        
        Args:
            project_id: Project ID
            settings: Processing settings dictionary
            
        Returns:
            Success or failure
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            logger.error(f"Project not found: {project_id}")
            return False
            
        try:
            # Create settings directory
            settings_dir = os.path.join(project_path, "settings")
            os.makedirs(settings_dir, exist_ok=True)
            
            # Save settings to file
            settings_path = os.path.join(settings_dir, "processing_settings.json")
            
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
                
            # Update metadata
            settings_timestamp = datetime.datetime.now().isoformat()
            self.update_project_metadata(project_id, {
                "settings_saved_at": settings_timestamp
            })
            
            logger.info(f"Saved project settings for {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving project settings: {str(e)}")
            return False
    
    def load_project_settings(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Load processing settings from project
        
        Args:
            project_id: Project ID
            
        Returns:
            Settings dictionary or None if not found
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            logger.error(f"Project not found: {project_id}")
            return None
            
        try:
            # Get settings file path
            settings_path = os.path.join(project_path, "settings", "processing_settings.json")
            
            if not os.path.exists(settings_path):
                logger.warning(f"No settings file found for project {project_id}")
                return None
                
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                
            logger.info(f"Loaded project settings for {project_id}")
            return settings
            
        except Exception as e:
            logger.error(f"Error loading project settings: {str(e)}")
            return None
    
    def save_previews_to_project(self, project_id: str, preview_paths: List[str], 
                               copy_files: bool = True) -> int:
        """
        Save preview images to project
        
        Args:
            project_id: Project ID
            preview_paths: List of preview image paths
            copy_files: Whether to copy files (True) or move them (False)
            
        Returns:
            Number of previews successfully saved
        """
        project_path = self.get_project_path(project_id)
        
        if not project_path:
            logger.error(f"Project not found: {project_id}")
            return 0
            
        preview_dir = os.path.join(project_path, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        
        saved_count = 0
        saved_paths = []
        
        for prev_path in preview_paths:
            if not os.path.exists(prev_path):
                logger.warning(f"Preview file not found: {prev_path}")
                continue
                
            # Get destination path
            dest_path = os.path.join(preview_dir, os.path.basename(prev_path))
            
            try:
                # Copy or move the file
                if copy_files:
                    shutil.copy2(prev_path, dest_path)
                else:
                    shutil.move(prev_path, dest_path)
                    
                saved_paths.append(os.path.relpath(dest_path, project_path))
                saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving preview {prev_path}: {str(e)}")
        
        # Update metadata with preview information
        if saved_count > 0:
            self.update_project_metadata(project_id, {
                "preview_count": saved_count,
                "preview_paths": saved_paths,
                "previews_generated_at": datetime.datetime.now().isoformat()
            })
            
        logger.info(f"Saved {saved_count} previews to project {project_id}")
        return saved_count
    
    def delete_project(self, project_id: str, delete_files: bool = False) -> bool:
        """
        Delete project from registry and optionally delete files
        
        Args:
            project_id: Project ID
            delete_files: Whether to delete project files
            
        Returns:
            Success or failure
        """
        if project_id not in self.project_registry:
            logger.error(f"Project not found: {project_id}")
            return False
            
        project_path = self.project_registry[project_id].get("path")
        
        # Remove from registry
        del self.project_registry[project_id]
        self._save_registry()
        
        # Delete files if requested
        if delete_files and project_path and os.path.exists(project_path):
            try:
                shutil.rmtree(project_path)
                logger.info(f"Deleted project files: {project_path}")
            except Exception as e:
                logger.error(f"Error deleting project files: {str(e)}")
                return False
                
        logger.info(f"Deleted project: {project_id}")
        return True
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load project registry from file"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading project registry: {str(e)}")
                
        return {}
    
    def _save_registry(self):
        """Save project registry to file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.project_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving project registry: {str(e)}")