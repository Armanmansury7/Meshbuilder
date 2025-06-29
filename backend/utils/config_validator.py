import os
import sys
import json
import jsonschema
from jsonschema import validate
import configparser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.advanced_logger import AdvancedLogger

class ConfigValidator:
    """
    Validates configuration settings to ensure they meet required schema and constraints.
    Provides error checking and default values for missing settings.
    """
    def __init__(self):
        self.logger = AdvancedLogger().get_logger(self.__class__.__name__)
        self.schemas = {}
        self.default_configs = {}
        self.load_schemas()
    
    def load_schemas(self):
        """Load validation schemas for different configuration types."""
        # General application settings schema
        self.schemas['app_settings'] = {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "version": {"type": "string"},
                "temp_folder": {"type": "string"},
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "max_threads": {"type": "integer", "minimum": 1, "maximum": 32},
                "gpu_acceleration": {"type": "boolean"},
                "output_formats": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["app_name", "version", "temp_folder", "log_level"]
        }
        
        # Reconstruction settings schema
        self.schemas['reconstruction'] = {
            "type": "object",
            "properties": {
                "quality": {"type": "string", "enum": ["low", "medium", "high", "ultra"]},
                "use_gpu": {"type": "boolean"},
                "advanced_mode": {"type": "boolean"},
                "use_gaussian_splatting": {"type": "boolean"},
                "colmap_path": {"type": "string"},
                "match_features": {"type": "string", "enum": ["exhaustive", "sequential", "vocab_tree"]},
                "image_downscale": {"type": "integer", "minimum": 1, "maximum": 8},
                "max_image_size": {"type": "integer", "minimum": 512, "maximum": 8192},
                "max_num_features": {"type": "integer", "minimum": 1000, "maximum": 100000},
                "texture_size": {"type": "integer", "minimum": 512, "maximum": 8192},
                "min_num_matches": {"type": "integer", "minimum": 3, "maximum": 100}
            },
            "required": ["quality", "use_gpu", "image_downscale", "max_image_size"]
        }
        
        # Preset configurations for different quality levels
        self.default_configs['reconstruction'] = {
            "low": {
                "quality": "low",
                "use_gpu": True,
                "advanced_mode": False,
                "use_gaussian_splatting": False,
                "match_features": "sequential",
                "image_downscale": 4,
                "max_image_size": 1024,
                "max_num_features": 8000,
                "texture_size": 2048,
                "min_num_matches": 15
            },
            "medium": {
                "quality": "medium",
                "use_gpu": True,
                "advanced_mode": False,
                "use_gaussian_splatting": False,
                "match_features": "sequential",
                "image_downscale": 2,
                "max_image_size": 2048,
                "max_num_features": 16000,
                "texture_size": 4096,
                "min_num_matches": 10
            },
            "high": {
                "quality": "high",
                "use_gpu": True,
                "advanced_mode": True,
                "use_gaussian_splatting": False,
                "match_features": "exhaustive",
                "image_downscale": 1,
                "max_image_size": 4096,
                "max_num_features": 32000,
                "texture_size": 4096,
                "min_num_matches": 6
            },
            "ultra": {
                "quality": "ultra",
                "use_gpu": True,
                "advanced_mode": True,
                "use_gaussian_splatting": True,
                "match_features": "exhaustive",
                "image_downscale": 1,
                "max_image_size": 8192,
                "max_num_features": 64000,
                "texture_size": 8192,
                "min_num_matches": 4
            }
        }
    
    def validate_config(self, config_data, schema_name):
        """
        Validate configuration data against a schema.
        
        Args:
            config_data: Configuration data to validate (dict)
            schema_name: Name of the schema to validate against
            
        Returns:
            Tuple of (is_valid, errors)
        """
        if schema_name not in self.schemas:
            self.logger.error(f"Schema '{schema_name}' not found")
            return False, ["Schema not found"]
        
        schema = self.schemas[schema_name]
        
        try:
            validate(instance=config_data, schema=schema)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"Configuration validation error: {str(e)}")
            return False, [str(e)]
    
    def get_default_config(self, config_type, preset=None):
        """
        Get default configuration values.
        
        Args:
            config_type: Type of configuration
            preset: Preset name (e.g., 'low', 'medium', 'high', 'ultra')
            
        Returns:
            Default configuration dictionary
        """
        if config_type not in self.default_configs:
            self.logger.warning(f"No default configuration for '{config_type}'")
            return {}
        
        if preset and preset in self.default_configs[config_type]:
            return self.default_configs[config_type][preset]
        elif 'default' in self.default_configs[config_type]:
            return self.default_configs[config_type]['default']
        elif config_type == 'reconstruction':
            # For reconstruction, use 'medium' as fallback
            return self.default_configs[config_type]['medium']
        else:
            # Return first preset as fallback
            return next(iter(self.default_configs[config_type].values()))
    
    def validate_file_exists(self, file_path, required=True):
        """
        Check if a file exists and is accessible.
        
        Args:
            file_path: Path to the file
            required: Whether the file is required
            
        Returns:
            True if valid, False otherwise
        """
        if not file_path:
            return not required
        
        if not os.path.isfile(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False
        
        if not os.access(file_path, os.R_OK):
            self.logger.error(f"File is not readable: {file_path}")
            return False
        
        return True
    
    def validate_directory_exists(self, dir_path, create_if_missing=False, required=True):
        """
        Check if a directory exists and is accessible.
        
        Args:
            dir_path: Path to the directory
            create_if_missing: Whether to create the directory if it doesn't exist
            required: Whether the directory is required
            
        Returns:
            True if valid, False otherwise
        """
        if not dir_path:
            return not required
        
        if not os.path.exists(dir_path):
            if create_if_missing:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    self.logger.info(f"Created directory: {dir_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to create directory {dir_path}: {str(e)}")
                    return False
            else:
                self.logger.error(f"Directory not found: {dir_path}")
                return False
        
        if not os.path.isdir(dir_path):
            self.logger.error(f"Path is not a directory: {dir_path}")
            return False
        
        if not os.access(dir_path, os.R_OK | os.W_OK):
            self.logger.error(f"Directory is not accessible: {dir_path}")
            return False
        
        return True
    
    def validate_ini_file(self, ini_path, required_sections=None):
        """
        Validate an INI configuration file.
        
        Args:
            ini_path: Path to the INI file
            required_sections: List of required sections
            
        Returns:
            Tuple of (is_valid, config_object, errors)
        """
        if not self.validate_file_exists(ini_path, required=True):
            return False, None, ["INI file not found or not accessible"]
        
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)
            
            errors = []
            
            # Check required sections
            if required_sections:
                for section in required_sections:
                    if section not in config:
                        errors.append(f"Required section '{section}' not found in INI file")
            
            return len(errors) == 0, config, errors
            
        except Exception as e:
            self.logger.error(f"Error parsing INI file: {str(e)}")
            return False, None, [f"Error parsing INI file: {str(e)}"]
    
    def validate_json_file(self, json_path, schema_name=None):
        """
        Validate a JSON configuration file.
        
        Args:
            json_path: Path to the JSON file
            schema_name: Name of the schema to validate against
            
        Returns:
            Tuple of (is_valid, config_object, errors)
        """
        if not self.validate_file_exists(json_path, required=True):
            return False, None, ["JSON file not found or not accessible"]
        
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            if schema_name:
                is_valid, errors = self.validate_config(config, schema_name)
                return is_valid, config, errors
            else:
                return True, config, []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON file: {str(e)}")
            return False, None, [f"Error parsing JSON file: {str(e)}"]
        except Exception as e:
            self.logger.error(f"Error validating JSON file: {str(e)}")
            return False, None, [f"Error validating JSON file: {str(e)}"]
    
    def merge_configs(self, base_config, override_config):
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if not base_config:
            return override_config or {}
        
        if not override_config:
            return base_config
        
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # Override base value
                merged[key] = value
        
        return merged