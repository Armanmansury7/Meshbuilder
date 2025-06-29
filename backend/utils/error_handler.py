"""
Error handling and recovery module for Meshbuilder with photorealistic support
"""
import os
import logging
import traceback
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("MeshBuilder.ErrorHandler")

class ErrorHandler:
    """Handles error recovery and fallbacks for the processing pipeline"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
    
    def handle_error(self, stage, error, context) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Handle error at a specific pipeline stage
        
        Args:
            stage: Pipeline stage where error occurred
            error: Exception that was raised
            context: Dictionary with context information
            
        Returns:
            Tuple of (can_recover, message, updated_settings)
        """
        error_str = str(error)
        error_type = type(error).__name__
        
        logger.error(f"Error in {stage} stage: {error_type}: {error_str}")
        
        # Log full traceback for debugging
        logger.error(traceback.format_exc())
        
        # Try to handle common errors
        if stage == "sparse_reconstruction":
            return self._handle_sparse_error(error_str, context)
        elif stage == "dense_reconstruction":
            return self._handle_dense_error(error_str, context)
        elif stage == "mesh_generation":
            return self._handle_mesh_error(error_str, context)
        elif stage == "texturing":
            return self._handle_texture_error(error_str, context)
        elif stage == "texture_enhancement":
            return self._handle_texture_enhancement_error(error_str, context)
        elif stage == "optimization":
            return self._handle_optimization_error(error_str, context)
        elif stage == "export":
            return self._handle_export_error(error_str, context)
        
        # Default: can't recover
        return False, f"Unhandled error in {stage} stage: {error_str}", {}
    
    def _handle_sparse_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in sparse reconstruction stage"""
        updated_settings = {}
        
        # Windows path error with spaces
        if "was unexpected at this time" in error_str or "No such file or directory" in error_str:
            logger.info("Detected Windows path handling issue, applying fix")
            updated_settings["fix_paths"] = True
            return True, "Retrying with fixed path handling", updated_settings
            
        # COLMAP execution error
        if "COLMAP executable not found" in error_str or "COLMAP verification failed" in error_str:
            logger.error("COLMAP not properly installed or configured")
            return False, "COLMAP installation issue detected. Please verify COLMAP is correctly installed.", {}
            
        # Out of memory error
        if "out of memory" in error_str.lower() or "cuda runtime error" in error_str.lower():
            logger.info("Detected out of memory error, reducing quality settings")
            updated_settings["max_image_dimension"] = 2048
            updated_settings["feature_matcher"] = "sequential"
            return True, "Retrying with reduced memory settings due to out of memory error", updated_settings
        
        # Feature extraction failure - try sequential matcher
        if "Feature extraction failed" in error_str and context.get("feature_matcher") == "exhaustive":
            logger.info("Feature extraction failed with exhaustive matcher, trying sequential")
            updated_settings["feature_matcher"] = "sequential"
            return True, "Retrying with sequential feature matcher", updated_settings
            
        return False, f"Unable to recover from sparse reconstruction error: {error_str}", {}
    
    def _handle_dense_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in dense reconstruction stage"""
        updated_settings = {}
        
        # Out of memory during dense reconstruction
        if "out of memory" in error_str.lower() or "cuda runtime error" in error_str.lower() or "failed allocation" in error_str.lower():
            logger.info("Dense reconstruction memory error, reducing density")
            updated_settings["point_density"] = "medium" if context.get("point_density") == "high" else "low"
            updated_settings["max_image_dimension"] = 1600  # Further reduce image size
            return True, "Retrying with lower point density due to memory constraints", updated_settings
        
        # Generic patch match error
        if "Patch match stereo failed" in error_str:
            logger.info("Patch match stereo failed, reducing settings")
            updated_settings["point_density"] = "low"
            updated_settings["max_image_dimension"] = 1600
            updated_settings["pms_quality"] = "low"  # Reduce PMS quality
            return True, "Retrying dense reconstruction with lower settings", updated_settings
        
        # Error in advanced PMS settings
        if "invalid parameter" in error_str.lower() or "parameter out of bounds" in error_str.lower():
            logger.info("PMS parameter error, trying with default settings")
            updated_settings["pms_quality"] = "normal"
            return True, "Retrying dense reconstruction with default PMS settings", updated_settings
            
        return False, f"Unable to recover from dense reconstruction error: {error_str}", {}
    
    def _handle_mesh_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in mesh generation stage"""
        updated_settings = {}
        
        # Out of memory during meshing
        if "out of memory" in error_str.lower() or "memory allocation" in error_str.lower():
            logger.info("Mesh generation memory error, reducing resolution")
            updated_settings["mesh_resolution"] = "low"
            return True, "Retrying with lower mesh resolution due to memory constraints", updated_settings
        
        # Insufficient points error
        if "insufficient points" in error_str.lower() or "empty point cloud" in error_str.lower():
            logger.error("Insufficient points for mesh generation")
            return False, "Not enough points to generate a mesh. Try adding more images with better overlap.", {}
        
        # Open3D error - often a point cloud issue
        if "Open3D error" in error_str or "RuntimeError" in error_str:
            logger.info("Open3D error, trying with simple meshing approach")
            updated_settings["mesh_resolution"] = "low"
            updated_settings["smoothing"] = 0.0
            return True, "Retrying with simplified meshing approach", updated_settings
        
        # Error with specific meshing algorithm
        if "poisson" in error_str.lower() or "reconstruction failed" in error_str.lower():
            # Try alternative meshing algorithm
            current_algo = context.get("meshing_algorithm", "poisson")
            if current_algo == "poisson":
                logger.info("Poisson meshing failed, trying BPA algorithm")
                updated_settings["meshing_algorithm"] = "bpa"
                return True, "Retrying with Ball-Pivoting Algorithm (BPA) for meshing", updated_settings
            else:
                logger.info("BPA meshing failed, falling back to poisson with low resolution")
                updated_settings["meshing_algorithm"] = "poisson"
                updated_settings["mesh_resolution"] = "low"
                return True, "Retrying with Poisson meshing at low resolution", updated_settings
            
        return False, f"Unable to recover from mesh generation error: {error_str}", {}
    
    def _handle_texture_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in texturing stage"""
        updated_settings = {}
        
        # Memory error with high resolution textures
        if "out of memory" in error_str.lower() or "memory error" in error_str.lower():
            current_res = context.get("texture_resolution", 4096)
            # Reduce texture resolution
            if current_res > 2048:
                logger.info(f"Texture resolution too high ({current_res}), reducing to 2048")
                updated_settings["texture_resolution"] = 2048
                return True, "Retrying with lower texture resolution (2048)", updated_settings
            elif current_res > 1024:
                logger.info(f"Texture resolution still too high ({current_res}), reducing to 1024")
                updated_settings["texture_resolution"] = 1024
                return True, "Retrying with lower texture resolution (1024)", updated_settings
            
        # COLMAP texturing error
        if "colmap" in error_str.lower() and "texturing" in error_str.lower():
            logger.info("COLMAP texturing failed, trying advanced internal texturing")
            updated_settings["use_internal_texturing"] = True
            return True, "Trying advanced internal texturing method", updated_settings
            
        # Generic texturing error - fall back to simpler method
        logger.info("Texturing error, falling back to simple texturing")
        updated_settings["skip_texturing"] = True
        updated_settings["color_enhance_mode"] = "none"  # Disable color enhancement for simple texturing
        return True, "Continuing with simple vertex coloring instead of full texturing", updated_settings
    
    def _handle_texture_enhancement_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in texture enhancement stage"""
        updated_settings = {}
        
        # Memory error during enhancement
        if "out of memory" in error_str.lower() or "memory error" in error_str.lower():
            current_res = context.get("texture_resolution", 4096)
            # Reduce texture resolution
            if current_res > 2048:
                logger.info(f"Enhancement texture resolution too high ({current_res}), reducing to 2048")
                updated_settings["texture_resolution"] = 2048
                return True, "Retrying enhancement with lower texture resolution (2048)", updated_settings
            elif current_res > 1024:
                logger.info(f"Enhancement texture resolution still too high ({current_res}), reducing to 1024")
                updated_settings["texture_resolution"] = 1024
                return True, "Retrying enhancement with lower texture resolution (1024)", updated_settings
        
        # Error with specific enhancement mode
        enhancement_mode = context.get("color_enhance_mode", "balanced")
        if enhancement_mode != "none":
            if enhancement_mode == "vibrant":
                # Try balanced mode instead
                logger.info("Vibrant enhancement failed, trying balanced mode")
                updated_settings["color_enhance_mode"] = "balanced"
                return True, "Retrying with balanced color enhancement", updated_settings
            elif enhancement_mode == "realistic":
                # Try balanced mode instead
                logger.info("Realistic enhancement failed, trying balanced mode")
                updated_settings["color_enhance_mode"] = "balanced"
                return True, "Retrying with balanced color enhancement", updated_settings
            elif enhancement_mode == "balanced":
                # Disable enhancement
                logger.info("Balanced enhancement failed, disabling enhancement")
                updated_settings["color_enhance_mode"] = "none"
                return True, "Continuing without color enhancement", updated_settings
        
        # Skip enhancement completely
        updated_settings["skip_enhancement"] = True
        updated_settings["color_enhance_mode"] = "none"
        return True, "Skipping texture enhancement", updated_settings
    
    def _handle_optimization_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in mesh optimization stage"""
        # Out of memory during optimization
        if "out of memory" in error_str.lower() or "memory error" in error_str.lower():
            current_faces = context.get("target_faces", 100000)
            # Try with fewer faces
            if current_faces > 50000:
                return True, "Retrying with fewer target faces", {"target_faces": 50000}
            elif current_faces > 25000:
                return True, "Retrying with fewer target faces", {"target_faces": 25000}
        
        # Skip optimization if it fails - not critical
        logger.info("Optimization error, skipping optimization")
        updated_settings = {"skip_optimization": True}
        return True, "Continuing with unoptimized mesh", updated_settings
    
    def _handle_export_error(self, error_str: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle errors in export stage"""
        updated_settings = {}
        
        # Try a different format if export fails
        export_format = context.get("export_format", "obj")
        
        # Memory error during export
        if "out of memory" in error_str.lower() or "memory error" in error_str.lower():
            # Fall back to a simpler format (OBJ is usually safest)
            logger.info(f"Memory error during {export_format} export, falling back to OBJ format")
            updated_settings["export_format"] = "obj"
            return True, "Memory error during export, falling back to OBJ format", updated_settings
        
        # Format-specific errors
        if export_format == "fbx" and "fbx" in error_str.lower():
            logger.info("FBX export failed, trying OBJ format")
            updated_settings["export_format"] = "obj"
            return True, "FBX export failed, falling back to OBJ format", updated_settings
        elif export_format == "gltf" and "gltf" in error_str.lower():
            logger.info("GLTF export failed, trying OBJ format")
            updated_settings["export_format"] = "obj"
            return True, "GLTF export failed, falling back to OBJ format", updated_settings
        elif export_format != "obj":
            logger.info(f"Export to {export_format} failed, trying OBJ format")
            updated_settings["export_format"] = "obj"
            return True, "Falling back to OBJ format", updated_settings
            
        return False, f"Unable to export model to {export_format} format", {}