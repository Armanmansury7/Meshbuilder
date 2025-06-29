"""
Settings Manager Module for MeshBuilder
Handles processing settings and allows reprocessing with updated settings
"""
import os
import json
import logging
import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("MeshBuilder.SettingsManager")

class SettingsManager:
    """Manages processing settings and allows reprocessing with updated settings"""
    
    def __init__(self, config_path=None):
        """
        Initialize settings manager
        
        Args:
            config_path: Path to the global configuration file
        """
        self.config_path = config_path
        self.project_settings = {}
        self.global_defaults = self._load_global_defaults()
    
    def save_project_settings(self, project_id: str, settings: Dict[str, Any], project_dir: str) -> bool:
        """
        Save project settings to file
        
        Args:
            project_id: Project identifier
            settings: Settings dictionary
            project_dir: Project directory
            
        Returns:
            Success or failure
        """
        settings_dir = os.path.join(project_dir, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        
        settings_path = os.path.join(settings_dir, "processing_settings.json")
        
        try:
            # Add timestamp and version information
            settings_with_meta = settings.copy()
            settings_with_meta["_timestamp"] = datetime.datetime.now().isoformat()
            settings_with_meta["_version"] = "1.0"
            settings_with_meta["_project_id"] = project_id
            
            with open(settings_path, 'w') as f:
                json.dump(settings_with_meta, f, indent=2)
                
            logger.info(f"Saved settings for project {project_id} to {settings_path}")
            self.project_settings[project_id] = settings_with_meta
            return True
            
        except Exception as e:
            logger.error(f"Error saving project settings: {str(e)}")
            return False
    
    def load_project_settings(self, project_id: str, project_dir: str) -> Optional[Dict[str, Any]]:
        """
        Load project settings from file
        
        Args:
            project_id: Project identifier
            project_dir: Project directory
            
        Returns:
            Project settings dictionary or None if not found
        """
        settings_path = os.path.join(project_dir, "settings", "processing_settings.json")
        
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    
                logger.info(f"Loaded settings for project {project_id} from {settings_path}")
                self.project_settings[project_id] = settings
                return settings
            else:
                logger.warning(f"No settings file found for project {project_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading project settings: {str(e)}")
            return None
    
    def update_project_settings(self, project_id: str, updates: Dict[str, Any], project_dir: str) -> Dict[str, Any]:
        """
        Update project settings
        
        Args:
            project_id: Project identifier
            updates: Dictionary with setting updates
            project_dir: Project directory
            
        Returns:
            Updated settings dictionary
        """
        # Load current settings if not already loaded
        if project_id not in self.project_settings:
            self.load_project_settings(project_id, project_dir)
            
        current_settings = self.project_settings.get(project_id, {})
        
        # Update settings
        for key, value in updates.items():
            # Skip metadata keys
            if not key.startswith('_'):
                current_settings[key] = value
                
        # Update timestamp
        current_settings["_timestamp"] = datetime.datetime.now().isoformat()
        
        # Save updated settings
        self.save_project_settings(project_id, current_settings, project_dir)
        
        return current_settings
    
    def get_setting_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined setting presets
        
        Returns:
            Dictionary of preset settings
        """
        presets = {
            "draft": {
                "feature_matcher": "sequential",
                "point_density": "low",
                "mesh_resolution": "low",
                "texture_resolution": 1024,
                "target_faces": 50000,
                "smoothing": 0.7,
                "trim_mesh": False,
                "enhance_color": False,
                "add_watermark": True
            },
            "standard": {
                "feature_matcher": "exhaustive",
                "point_density": "medium",
                "mesh_resolution": "medium",
                "texture_resolution": 2048,
                "target_faces": 100000,
                "smoothing": 0.5,
                "trim_mesh": False,
                "enhance_color": True,
                "color_enhancement_mode": "balanced",
                "add_watermark": True
            },
            "high_quality": {
                "feature_matcher": "exhaustive",
                "point_density": "high",
                "mesh_resolution": "high",
                "texture_resolution": 4096,
                "target_faces": 200000,
                "smoothing": 0.3,
                "trim_mesh": False,
                "enhance_color": True,
                "color_enhancement_mode": "realistic",
                "add_watermark": True
            },
            "maximum": {
                "feature_matcher": "exhaustive",
                "point_density": "high",
                "mesh_resolution": "high",
                "texture_resolution": 8192,
                "target_faces": 500000,
                "smoothing": 0.2,
                "trim_mesh": False,
                "enhance_color": True,
                "color_enhancement_mode": "realistic",
                "add_watermark": True
            }
        }
        
        return presets
    
    def get_preset_settings(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get settings for a specific preset
        
        Args:
            preset_name: Name of the preset ("draft", "standard", "high_quality", "maximum")
            
        Returns:
            Settings dictionary or None if preset not found
        """
        presets = self.get_setting_presets()
        
        if preset_name in presets:
            # Make a deep copy to avoid modifying the original
            settings = presets[preset_name].copy()
            
            # Add metadata
            settings["_preset"] = preset_name
            settings["_timestamp"] = datetime.datetime.now().isoformat()
            
            return settings
        
        logger.warning(f"Preset not found: {preset_name}")
        return None
    
    def validate_settings(self, settings: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate settings against allowed values and constraints
        
        Args:
            settings: Settings dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        valid = True
        errors = []
        
        # Known settings with valid values
        valid_values = {
            "feature_matcher": ["exhaustive", "sequential", "vocab_tree"],
            "point_density": ["low", "medium", "high"],
            "mesh_resolution": ["low", "medium", "high"],
            "color_enhancement_mode": ["balanced", "vibrant", "realistic"]
        }
        
        # Range constraints
        range_constraints = {
            "texture_resolution": (512, 8192),
            "target_faces": (5000, 1000000),
            "smoothing": (0.0, 1.0)
        }
        
        # Boolean settings
        boolean_settings = ["trim_mesh", "enhance_color", "add_watermark", "use_gpu"]
        
        # Check each setting
        for key, value in settings.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
                
            # Check enum settings
            if key in valid_values:
                if value not in valid_values[key]:
                    valid = False
                    errors.append(f"Invalid value for {key}: {value}. Valid values are {valid_values[key]}")
            
            # Check range constraints
            elif key in range_constraints:
                min_val, max_val = range_constraints[key]
                if not (min_val <= value <= max_val):
                    valid = False
                    errors.append(f"Invalid value for {key}: {value}. Must be between {min_val} and {max_val}")
            
            # Check boolean settings
            elif key in boolean_settings:
                if not isinstance(value, bool):
                    valid = False
                    errors.append(f"Invalid value for {key}: {value}. Must be a boolean")
        
        return valid, errors
    
    def generate_diff(self, old_settings: Dict[str, Any], new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a diff between two settings dictionaries
        
        Args:
            old_settings: Old settings dictionary
            new_settings: New settings dictionary
            
        Returns:
            Dictionary with changes (key: [old_value, new_value])
        """
        diff = {}
        
        # Find changed and added keys
        for key, new_value in new_settings.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
                
            if key in old_settings:
                old_value = old_settings[key]
                if old_value != new_value:
                    diff[key] = [old_value, new_value]
            else:
                diff[key] = [None, new_value]
        
        # Find removed keys
        for key in old_settings:
            # Skip metadata keys
            if key.startswith('_'):
                continue
                
            if key not in new_settings:
                diff[key] = [old_settings[key], None]
        
        return diff
    
    def merge_with_defaults(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge settings with defaults for missing values
        
        Args:
            settings: Settings dictionary
            
        Returns:
            Merged settings dictionary
        """
        # Get default settings (standard preset)
        defaults = self.get_preset_settings("standard") or {}
        
        # Remove metadata keys from defaults
        defaults = {k: v for k, v in defaults.items() if not k.startswith('_')}
        
        # Start with defaults, then override with provided settings
        merged = defaults.copy()
        
        # Override with provided settings
        for key, value in settings.items():
            merged[key] = value
        
        return merged
    
    def _load_global_defaults(self) -> Dict[str, Any]:
        """Load global default settings from config file"""
        if not self.config_path or not os.path.exists(self.config_path):
            return {}
            
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(self.config_path)
            
            # Convert config to dictionary
            defaults = {}
            
            # Extract settings from Processing section
            if "Processing" in config:
                processing = config["Processing"]
                
                # Convert string values to appropriate types
                if "feature_matcher" in processing:
                    defaults["feature_matcher"] = processing["feature_matcher"]
                
                if "point_density" in processing:
                    defaults["point_density"] = processing["point_density"]
                
                if "mesh_resolution" in processing:
                    defaults["mesh_resolution"] = processing["mesh_resolution"]
                
                if "texture_resolution" in processing:
                    defaults["texture_resolution"] = int(processing["texture_resolution"])
                
                if "target_faces" in processing:
                    defaults["target_faces"] = int(processing["target_faces"])
                
                if "smoothing" in processing:
                    defaults["smoothing"] = float(processing["smoothing"])
                
                if "use_gpu" in processing:
                    defaults["use_gpu"] = processing.getboolean("use_gpu")
            
            return defaults
                
        except Exception as e:
            logger.error(f"Error loading global defaults: {str(e)}")
            return {}