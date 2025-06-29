"""
Error Recovery Module for Meshbuilder - FIXED VERSION
Handles error recovery strategies and graceful fallbacks
"""
import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import utilities with error handling
try:
    from utils.config_manager import ConfigManager
    from utils.advanced_logger import AdvancedLogger
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Basic logging setup
logger = logging.getLogger("MeshBuilder.ErrorRecovery")

class ErrorResult:
    """Represents an error recovery result"""
    
    def __init__(self, 
                action: str, 
                message: str, 
                data: Optional[Dict[str, Any]] = None):
        """
        Initialize with action and message
        
        Args:
            action: Action to take ('retry', 'fallback', 'abort', etc.)
            message: Message explaining the recovery action
            data: Optional data for recovery (e.g. alternative parameters)
        """
        self.action = action
        self.message = message
        self.data = data or {}
        
    def get(self, key: str, default=None):
        """
        Get a value from the result
        
        Args:
            key: Key to get
            default: Default value if key not found
            
        Returns:
            Value for key or default
        """
        if key == "action":
            return self.action
        elif key == "message":
            return self.message
        elif not self.data:
            return default
        else:
            return self.data.get(key, default)

class ErrorRecovery:
    """Handles error recovery and graceful fallbacks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Load config if not provided and available
        if config is None and CONFIG_AVAILABLE:
            try:
                config_manager = ConfigManager()
                config = config_manager.get_config_dict()
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
                config = {}
        
        self.config = config or {}
        
        # Initialize advanced logger if available
        if CONFIG_AVAILABLE:
            try:
                self.advanced_logger = AdvancedLogger()
            except Exception:
                self.advanced_logger = None
        else:
            self.advanced_logger = None
        
        self.error_handlers = {}
        self.register_default_handlers()
        
        # User-friendly error messages
        self.user_friendly_errors = {
            "feature_extraction_failed": "Could not analyze image features. Try using higher quality images with more visual details.",
            "dense_reconstruction_failed": "Failed to create detailed 3D point cloud. Try using more images with better overlap.",
            "mesh_generation_failed": "Could not create 3D mesh from point cloud. Try using a different mesh resolution.",
            "texturing_failed": "Failed to apply texture to the 3D model. Using a simple color texture instead.",
            "optimization_failed": "Failed to optimize mesh. The model may be larger than expected.",
            "export_failed": "Failed to export model in the requested format. Try using OBJ format instead.",
            "COLMAP not found": "COLMAP software not found. Please check the installation path in settings.",
            "CUDA out of memory": "Your GPU ran out of memory. Try processing fewer images or reducing quality settings.",
            "out of memory": "Your computer ran out of memory. Try processing fewer images or closing other applications.",
            "no valid reconstruction": "Could not create a valid 3D reconstruction. Try using images with more overlap.",
            "dataset_creation_failed": "Failed to create dataset. Check image quality and COLMAP installation.",
            "training_failed": "3D model training failed. Try with different quality settings.",
            "image_processing_failed": "Image preprocessing failed. Check image formats and quality."
        }
        
        # Error context for recovery
        self.error_context = {}
    
    def register_default_handlers(self):
        """Register default error handlers"""
        self.register_handler("feature_extraction_failed", self._handle_feature_extraction_error)
        self.register_handler("COLMAP not found", self._handle_colmap_not_found)
        self.register_handler("dense_reconstruction_failed", self._handle_dense_reconstruction_error)
        self.register_handler("mesh_generation_failed", self._handle_mesh_generation_error)
        self.register_handler("texturing_failed", self._handle_texturing_error)
        self.register_handler("optimization_failed", self._handle_optimization_error)
        self.register_handler("export_failed", self._handle_export_error)
        self.register_handler("CUDA out of memory", self._handle_cuda_out_of_memory)
        self.register_handler("out of memory", self._handle_out_of_memory)
        self.register_handler("dataset_creation_failed", self._handle_dataset_creation_error)
        self.register_handler("training_failed", self._handle_training_error)
        self.register_handler("image_processing_failed", self._handle_image_processing_error)
    
    def register_handler(self, 
                        error_type: str, 
                        handler: Callable[[Exception, Dict[str, Any]], ErrorResult]):
        """
        Register a handler for a specific error type
        
        Args:
            error_type: Error type string (partial match)
            handler: Handler function that takes (error, context) and returns ErrorResult
        """
        self.error_handlers[error_type] = handler
    
    def handle_error(self, 
                   error_type: str, 
                   error_details: Exception, 
                   context: Dict[str, Any]) -> Optional[ErrorResult]:
        """
        Handle an error with the appropriate handler
        
        Args:
            error_type: Error type string
            error_details: Exception object
            context: Context dictionary for recovery
            
        Returns:
            ErrorResult with recovery action or None if no handler
        """
        # Save context for recovery
        self.error_context = context.copy()
        
        # Log the error
        logger.error(f"Error: {error_type}")
        logger.error(f"Details: {str(error_details)}")
        
        # Use advanced logger if available
        if self.advanced_logger:
            self.advanced_logger.log_error(error_details, context)
        else:
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Find suitable handler
        for key, handler in self.error_handlers.items():
            if key.lower() in error_type.lower():
                logger.info(f"Found error handler for '{key}'")
                try:
                    result = handler(error_details, context)
                    if result:
                        logger.info(f"Recovery action: {result.action} - {result.message}")
                        return result
                except Exception as e:
                    logger.error(f"Error in error handler: {str(e)}")
        
        # Default handler if no specific handler found
        return self._handle_unknown_error(error_details, context)
    
    def get_user_friendly_message(self, error_type: str) -> str:
        """
        Get a user-friendly error message
        
        Args:
            error_type: Error type string
            
        Returns:
            User-friendly error message
        """
        for key, message in self.user_friendly_errors.items():
            if key.lower() in error_type.lower():
                return message
                
        # Default message if no match
        return f"An error occurred: {error_type}"
    
    def _handle_feature_extraction_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle feature extraction errors"""
        error_str = str(error)
        
        # Check for common feature extraction errors
        if "COLMAP executable not found" in error_str or "COLMAP not found" in error_str:
            return self._handle_colmap_not_found(error, context)
            
        if "CUDA out of memory" in error_str or "out of memory" in error_str:
            return self._handle_out_of_memory(error, context)
            
        if "Check failed: ExistsDir(*image_path)" in error_str:
            # Directory doesn't exist, try to create it
            image_path = context.get("image_path")
            if image_path:
                try:
                    os.makedirs(image_path, exist_ok=True)
                    return ErrorResult(
                        "retry", 
                        "Image directory created. Retrying feature extraction.",
                        {"fixed": True}
                    )
                except Exception:
                    pass
        
        # Try with different parameters
        return ErrorResult(
            "retry_with_params", 
            "Retrying feature extraction with alternative parameters.",
            {
                "params": {
                    "SiftExtraction.max_image_size": "2048",
                    "feature_matcher": "sequential",
                    "SiftExtraction.max_num_features": "8000"
                }
            }
        )
    
    def _handle_colmap_not_found(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle COLMAP not found errors"""
        return ErrorResult(
            "reconfigure", 
            "COLMAP not found. Please check the COLMAP path in settings.",
            {"error_type": "configuration"}
        )
    
    def _handle_dense_reconstruction_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle dense reconstruction errors"""
        error_str = str(error)
        
        if "CUDA out of memory" in error_str or "out of memory" in error_str:
            return self._handle_out_of_memory(error, context)
            
        # Try with lower density
        return ErrorResult(
            "retry_with_params", 
            "Retrying dense reconstruction with lower density.",
            {
                "params": {
                    "point_density": "low",
                    "PatchMatchStereo.window_radius": "1",
                    "PatchMatchStereo.max_image_size": "1024"
                }
            }
        )
    
    def _handle_mesh_generation_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle mesh generation errors"""
        # Try with simpler mesh generation
        return ErrorResult(
            "retry_with_params", 
            "Retrying mesh generation with simpler parameters.",
            {
                "params": {
                    "mesh_resolution": "low",
                    "smoothing": "0.3",
                    "target_triangles": "50000"
                }
            }
        )
    
    def _handle_texturing_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle texturing errors"""
        # Fall back to simple texturing
        return ErrorResult(
            "fallback", 
            "Using simple texturing as fallback.",
            {
                "method": "simple_texture",
                "params": {
                    "texture_resolution": "1024"
                }
            }
        )
    
    def _handle_optimization_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle optimization errors"""
        # Skip optimization
        return ErrorResult(
            "skip", 
            "Skipping mesh optimization due to error.",
            {
                "skip_stage": "optimization"
            }
        )
    
    def _handle_export_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle export errors"""
        # Fall back to OBJ format
        requested_format = context.get("format", "obj")
        
        if requested_format.lower() != "obj":
            return ErrorResult(
                "fallback", 
                f"Falling back to OBJ format instead of {requested_format}.",
                {
                    "format": "obj"
                }
            )
        else:
            return ErrorResult(
                "abort", 
                "Export failed. Could not save model.",
                {
                    "error_type": "fatal"
                }
            )
    
    def _handle_cuda_out_of_memory(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle CUDA out of memory errors"""
        # Reduce GPU memory usage
        return ErrorResult(
            "retry_with_params", 
            "GPU memory exceeded. Retrying with lower memory usage.",
            {
                "params": {
                    "use_gpu": "true",
                    "SiftExtraction.max_image_size": "1024",
                    "PatchMatchStereo.max_image_size": "1024",
                    "PatchMatchStereo.window_radius": "1",
                    "max_image_dimension": "2048"
                }
            }
        )
    
    def _handle_out_of_memory(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle out of memory errors"""
        # Reduce memory usage
        return ErrorResult(
            "retry_with_params", 
            "Memory exceeded. Retrying with lower memory usage.",
            {
                "params": {
                    "max_image_dimension": "1024",
                    "point_density": "low",
                    "mesh_resolution": "low",
                    "feature_matcher": "sequential"
                }
            }
        )
    
    def _handle_dataset_creation_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle dataset creation errors"""
        error_str = str(error)
        
        if "COLMAP" in error_str:
            return self._handle_colmap_not_found(error, context)
        
        # Try with reduced settings
        return ErrorResult(
            "retry_with_params",
            "Retrying dataset creation with reduced settings.",
            {
                "params": {
                    "feature_matcher": "sequential", 
                    "SiftExtraction.max_image_size": "1600",
                    "SiftExtraction.max_num_features": "8000"
                }
            }
        )
    
    def _handle_training_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle training errors"""
        # Try with reduced training parameters
        return ErrorResult(
            "retry_with_params",
            "Retrying training with reduced parameters.",
            {
                "params": {
                    "resolution": "1000",
                    "iterations": "10000"
                }
            }
        )
    
    def _handle_image_processing_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle image processing errors"""
        error_str = str(error)
        
        if "CUDA out of memory" in error_str or "out of memory" in error_str:
            return self._handle_out_of_memory(error, context)
        
        # Try with simpler processing
        return ErrorResult(
            "retry_with_params",
            "Retrying image processing with simpler settings.",
            {
                "params": {
                    "enhancement_mode": "balanced",
                    "max_image_dimension": "2048",
                    "use_gpu": "false"
                }
            }
        )
    
    def _handle_unknown_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResult:
        """Handle unknown errors"""
        # Just abort with message
        return ErrorResult(
            "abort", 
            f"Unknown error: {str(error)}",
            {
                "error_type": "unknown"
            }
        )


# Test function
def test_error_recovery():
    """Test the error recovery module"""
    try:
        print("Testing ErrorRecovery...")
        
        # Create error recovery
        error_recovery = ErrorRecovery()
        
        # Test error handling
        test_error = ValueError("Test error message")
        test_context = {
            "operation": "test_operation",
            "data_size": 100
        }
        
        result = error_recovery.handle_error(
            "feature_extraction_failed", 
            test_error, 
            test_context
        )
        
        if result:
            print(f"✓ Error handling works: {result.action} - {result.message}")
        
        # Test user-friendly messages
        friendly_msg = error_recovery.get_user_friendly_message("CUDA out of memory")
        print(f"✓ User-friendly message: {friendly_msg}")
        
        print("✓ ErrorRecovery test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ ErrorRecovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_error_recovery()