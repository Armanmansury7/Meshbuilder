"""
Configuration Manager Module for Meshbuilder
Handles configuration loading and saving - FIXED VERSION
"""
import os
import configparser
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up basic logging (don't depend on other modules yet)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Handles configuration operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration path"""
        # Default to user's home directory
        if config_path is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".meshbuilder")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.ini")
        
        self.config_path = config_path
        logger.info(f"Config manager initialized with path: {self.config_path}")
        
    def create_default_config(self) -> configparser.ConfigParser:
        """
        Create default configuration
        
        Returns:
            ConfigParser with default configuration
        """
        config = configparser.ConfigParser()
        
        # Paths section
        config["Paths"] = {
            "base_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "temp_dir": os.path.join(os.path.expanduser("~"), ".meshbuilder", "temp"),
            "output_dir": os.path.join(os.path.expanduser("~"), "MeshBuilder_Output"),
            "colmap_bin": "colmap"  # Assume COLMAP is in PATH
        }
        
        # Processing section
        config["Processing"] = {
            "use_gpu": "true",
            "min_images": "20",
            "max_images": "200",
            "max_image_dimension": "4096",
            "enhance_images": "true",
            "quality_threshold": "0.5",
            "blur_threshold": "100.0",
            "frame_extraction_rate": "1",
            "feature_matcher": "exhaustive",
            "point_density": "high",
            "mesh_resolution": "high",
            "smoothing": "0.5",
            "texture_resolution": "4096",
            "target_faces": "1000000",
            "color_enhance_mode": "realistic",
            "meshing_algorithm": "poisson",
            "pms_quality": "normal",
            "denoise_strength": "0.5",
            "max_workers": "4",
            "max_num_features": "16000",
            "use_deep_enhancement": "false"
        }
        
        # Output section
        config["Output"] = {
            "default_format": "obj",
            "export_formats": "obj,fbx,gltf",
            "compress_output": "false"
        }
        
        # Cleanup section
        config["Cleanup"] = {
            "remove_temp_files": "true",
            "keep_logs": "true"
        }
        
        # Deep Enhancement section
        config["DeepEnhancement"] = {
            "use_realesrgan": "true",
            "use_gfpgan": "true",
            "realesrgan_model_path": "models/RealESRGAN_x4plus.pth",
            "gfpgan_model_path": "models/GFPGANv1.4.pth",
            "tile_size": "400",
            "half_precision": "true",
            "gfpgan_upscale": "2",
            "mode": "both"
        }
        
        return config
    
    def load_config(self) -> configparser.ConfigParser:
        """
        Load configuration from file, creating default if not exists
        
        Returns:
            Configuration as ConfigParser
        """
        config = self.create_default_config()
        
        if os.path.exists(self.config_path):
            try:
                config.read(self.config_path)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                logger.info("Using default configuration")
        else:
            # Save default configuration
            self.save_config(config)
            logger.info(f"Created default configuration at {self.config_path}")
            
        return config
    
    def save_config(self, config: configparser.ConfigParser) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            
        Returns:
            Success or failure
        """
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as config_file:
                config.write(config_file)
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def update_config(self, section: str, key: str, value: str) -> bool:
        """
        Update a specific configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
            
        Returns:
            Success or failure
        """
        config = self.load_config()
        
        if section not in config:
            config[section] = {}
            
        config[section][key] = value
        
        return self.save_config(config)
    
    def get_config_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration as a nested dictionary
        
        Returns:
            Configuration as nested dict
        """
        config = self.load_config()
        result = {}
        
        for section_name in config.sections():
            result[section_name] = {}
            for key, value in config[section_name].items():
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    result[section_name][key] = value.lower() == 'true'
                elif value.isdigit():
                    result[section_name][key] = int(value)
                elif self._is_float(value):
                    result[section_name][key] = float(value)
                else:
                    result[section_name][key] = value
        
        return result
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def get_photorealistic_settings(self) -> Dict[str, Any]:
        """
        Get all photorealistic rendering settings
        
        Returns:
            Dictionary with photorealistic settings
        """
        config = self.load_config()
        settings = {}
        
        # Get general processing settings that affect photorealistic quality
        settings.update({
            "texture_resolution": config.getint("Processing", "texture_resolution", fallback=4096),
            "color_enhance_mode": config.get("Processing", "color_enhance_mode", fallback="balanced"),
            "mesh_resolution": config.get("Processing", "mesh_resolution", fallback="medium"),
            "smoothing": config.getfloat("Processing", "smoothing", fallback=0.5),
            "use_deep_enhancement": config.getboolean("Processing", "use_deep_enhancement", fallback=False)
        })
        
        return settings
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration
        
        Returns:
            True if configuration is valid
        """
        try:
            config = self.load_config()
            
            # Check required sections
            required_sections = ["Paths", "Processing", "Output"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate some critical settings
            processing = config["Processing"]
            
            # Check numeric values
            numeric_settings = {
                "max_image_dimension": (512, 8192),
                "quality_threshold": (0.0, 1.0),
                "max_workers": (1, 32)
            }
            
            for setting, (min_val, max_val) in numeric_settings.items():
                try:
                    value = float(processing.get(setting, 0))
                    if not (min_val <= value <= max_val):
                        logger.error(f"Setting {setting} ({value}) out of range [{min_val}, {max_val}]")
                        return False
                except ValueError:
                    logger.error(f"Setting {setting} is not a valid number")
                    return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False


# Test function to verify this module works
def test_config_manager():
    """Test the config manager"""
    try:
        print("Testing ConfigManager...")
        
        # Create config manager
        cm = ConfigManager()
        
        # Load config
        config = cm.load_config()
        print(f"✓ Config loaded with {len(config.sections())} sections")
        
        # Validate config
        is_valid = cm.validate_config()
        print(f"✓ Config validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Get as dict
        config_dict = cm.get_config_dict()
        print(f"✓ Config dict has {len(config_dict)} sections")
        
        print("ConfigManager test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False


if __name__ == "__main__":
    test_config_manager()