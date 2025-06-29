"""
Memory Manager Module for Meshbuilder - FIXED VERSION
Handles adaptive memory optimization for large datasets with fallbacks
"""
import os
import sys
import tempfile
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import utilities with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Basic logging setup
logger = logging.getLogger("MeshBuilder.MemoryManager")

class MemoryManager:
    """Manages memory usage during processing with adaptive quality scaling"""
    
    def __init__(self, max_memory_gb: Optional[float] = None):
        """
        Initialize with maximum memory limit
        
        Args:
            max_memory_gb: Maximum memory usage in GB (default: 90% of system memory)
        """
        # Get total system memory with fallback
        if PSUTIL_AVAILABLE:
            self.system_memory = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        else:
            # Fallback: estimate based on common system configurations
            self.system_memory = 8.0  # Assume 8GB as conservative default
            logger.warning("psutil not available, assuming 8GB system memory")
        
        self.total_memory_gb = self.system_memory
        
        # If max_memory_gb is None, use 90% of system memory
        if max_memory_gb is None:
            self.max_memory_gb = self.system_memory * 0.9
        else:
            # Cap at 90% of system memory even if user requests more
            self.max_memory_gb = min(max_memory_gb, self.system_memory * 0.9)
            
        self.temp_files = []
        self.memory_mapped_arrays = []
        
        logger.info(f"Memory manager initialized with {self.max_memory_gb:.1f} GB limit")
        logger.info(f"System memory: {self.system_memory:.1f} GB")
        
        # Log current memory status if available
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            logger.info(f"Current memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
        
        # Define memory thresholds for different operations
        self.memory_thresholds = {
            "sparse_reconstruction": 2.0,  # Base GB required
            "feature_extraction": 1.5,
            "dense_reconstruction": 4.0,
            "mesh_generation": 6.0,
            "texture_mapping": 8.0,
            "gaussian_splatting": 4.0,
            "load_images": 1.2,
            "extract_frames": 1.5,
            "optimize_mesh": 2.0
        }
        
        # Calculate scaling factors based on available memory
        self.memory_scaling = self._calculate_scaling_factors()
    
    def _calculate_scaling_factors(self) -> Dict[str, float]:
        """
        Calculate scaling factors based on available memory
        
        Returns:
            Dictionary with scaling factors for different parameters
        """
        # Define base memory required for minimum quality
        base_memory_gb = 8.0
        
        # Calculate linear scaling factor (1.0 at base memory, higher with more memory)
        scaling_factor = max(1.0, self.max_memory_gb / base_memory_gb)
        
        # Calculate logarithmic scaling for more balanced progression
        log_scaling = max(1.0, 1.0 + math.log(scaling_factor, 2))
        
        # For very high memory systems, add an extra boost
        memory_boost = 1.0
        if self.max_memory_gb > 32.0:
            memory_boost = 1.2
        elif self.max_memory_gb > 64.0:
            memory_boost = 1.5
        
        logger.info(f"Memory scaling factor: {scaling_factor:.2f}, Log scaling: {log_scaling:.2f}, Boost: {memory_boost:.2f}")
        
        return {
            "point_density": min(4.0, log_scaling * memory_boost),
            "texture_resolution": min(5.0, scaling_factor * memory_boost),
            "mesh_resolution": min(4.0, log_scaling * memory_boost),
            "max_image_dimension": min(3.5, log_scaling ** 0.8 * memory_boost),
            "detail_level": min(3.0, log_scaling * memory_boost),
            "normal_estimation": min(3.0, log_scaling * memory_boost),
            "poisson_depth": min(3.0, log_scaling * 0.7 * memory_boost)
        }
    
    def check_memory_requirements(self, input_size: float, operation_type: str) -> bool:
        """
        Check if operation will fit in memory
        
        Args:
            input_size: Size of input data in GB
            operation_type: Type of operation for estimation
            
        Returns:
            True if operation fits in memory, False otherwise
        """
        required_memory = self._estimate_memory_requirement(input_size, operation_type)
        available_memory = self._get_available_memory()
        
        logger.info(f"Memory check for {operation_type}: " + 
                   f"Required {required_memory:.1f} GB, Available {available_memory:.1f} GB")
        
        return required_memory <= available_memory
    
    def _estimate_memory_requirement(self, input_size: float, operation_type: str) -> float:
        """
        Estimate memory requirements based on operation type
        
        Args:
            input_size: Size of input data in GB
            operation_type: Type of operation
            
        Returns:
            Estimated memory requirement in GB
        """
        # Get base memory requirement from thresholds
        base_requirement = self.memory_thresholds.get(operation_type, 1.5)
        
        # Calculate total requirement (base + scaled input size)
        # Use higher multipliers for more memory-intensive operations
        if operation_type in ["dense_reconstruction", "mesh_generation", "texture_mapping"]:
            return base_requirement + (input_size * 3.0)
        
        return base_requirement + (input_size * 1.5)
    
    def _get_available_memory(self) -> float:
        """
        Get available system memory
        
        Returns:
            Available memory in GB
        """
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            return mem.available / (1024**3)  # Convert to GB
        else:
            # Fallback: assume 70% of total memory is available
            return self.system_memory * 0.7
    
    def create_memory_mapped_array(self, shape: Tuple, dtype_name: str = 'float32'):
        """
        Create a memory-mapped array for large data
        
        Args:
            shape: Shape of array
            dtype_name: Data type name (e.g., 'float32', 'int32')
            
        Returns:
            Memory-mapped numpy array or None if numpy not available
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, cannot create memory-mapped array")
            return None
        
        import numpy as np
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        self.temp_files.append(temp_file.name)
        
        # Close file handle
        temp_file.close()
        
        # Calculate array size in GB
        dtype = np.dtype(dtype_name)
        element_size = dtype.itemsize
        total_elements = np.prod(shape)
        array_size_gb = total_elements * element_size / (1024**3)
        
        logger.info(f"Creating memory-mapped array of shape {shape}, size {array_size_gb:.2f} GB")
        
        try:
            # Create memory-mapped array
            mmap_array = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=shape)
            self.memory_mapped_arrays.append(mmap_array)
            
            return mmap_array
        except Exception as e:
            logger.error(f"Failed to create memory-mapped array: {e}")
            return None
    
    def create_chunked_processing(self, total_items: int, chunk_size: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Create processing chunks for large datasets
        
        Args:
            total_items: Total number of items to process
            chunk_size: Optional chunk size (default: auto-calculate)
            
        Returns:
            List of (start, end) tuples for each chunk
        """
        if chunk_size is None:
            # Get available memory
            available_memory = self._get_available_memory()
            
            # Assume each item requires about 10MB (adjust based on your data)
            item_memory_mb = 10
            
            # Calculate maximum items per chunk
            max_items = int((available_memory * 1024) / item_memory_mb)
            
            # Limit to a reasonable size
            chunk_size = min(max_items, 1000)
            chunk_size = max(chunk_size, 10)  # Minimum chunk size
        
        # Create chunks
        chunks = []
        for start in range(0, total_items, chunk_size):
            end = min(start + chunk_size, total_items)
            chunks.append((start, end))
            
        logger.info(f"Created {len(chunks)} processing chunks of size {chunk_size} " +
                   f"for {total_items} items")
        
        return chunks
    
    def optimize_for_operation(self, operation_type: str) -> Dict[str, Any]:
        """
        Get optimized parameters for an operation based on available memory
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Dictionary of optimized parameters
        """
        # Get current memory usage
        if PSUTIL_AVAILABLE:
            current_usage = psutil.virtual_memory().percent / 100.0
            available_memory_gb = self.max_memory_gb * (1.0 - current_usage)
        else:
            # Fallback: assume 70% available
            available_memory_gb = self.max_memory_gb * 0.7
        
        # Base requirement for this operation
        base_requirement = self.memory_thresholds.get(operation_type, 1.0)
        
        # Calculate memory ratio
        memory_ratio = available_memory_gb / base_requirement
        
        # If available memory is severely constrained, use lowest settings
        if available_memory_gb < 2.0 or memory_ratio < 1.0:
            logger.warning(f"Severe memory constraint for {operation_type}: {available_memory_gb:.2f} GB available")
            return self._get_low_memory_settings(operation_type)
        
        # For normal/high memory scenarios, use scaled settings
        return self._get_scaled_settings(operation_type, memory_ratio)
    
    def _get_low_memory_settings(self, operation: str) -> Dict[str, Any]:
        """
        Get minimal memory settings for constrained environments
        
        Args:
            operation: Operation type
            
        Returns:
            Dictionary with low memory settings
        """
        if operation == "sparse_reconstruction":
            return {
                "SiftExtraction.max_image_size": "1600",
                "feature_matcher": "sequential",
                "SiftExtraction.max_num_features": "5000",
                "SiftMatching.max_num_matches": "4096",
                "SiftMatching.min_num_inliers": "10"
            }
        elif operation == "dense_reconstruction":
            return {
                "point_density": "low",
                "PatchMatchStereo.max_image_size": "1024",
                "PatchMatchStereo.window_radius": "1",
                "PatchMatchStereo.num_samples": "5",
                "PatchMatchStereo.num_iterations": "3",
                "StereoFusion.min_num_pixels": "2"
            }
        elif operation == "mesh_generation":
            return {
                "mesh_resolution": "low",
                "target_faces": "50000",
                "poisson_depth": "8",
                "density_threshold": "0.02",
                "cleaning_strength": "high"
            }
        elif operation == "texture_mapping":
            return {
                "texture_resolution": "1024",
                "uv_mapping_method": "basic",
                "texture_blending": "simple"
            }
        elif operation == "load_images" or operation == "image_processing":
            return {
                "max_image_dimension": "2048",
                "enhancement_mode": "balanced",
                "use_gpu": "false",
                "max_workers": "2"
            }
        
        return {"memory_optimized": "true"}
    
    def _get_scaled_settings(self, operation: str, memory_ratio: float) -> Dict[str, Any]:
        """
        Get settings scaled according to available memory
        
        Args:
            operation: Operation type
            memory_ratio: Ratio of available memory to base requirement
            
        Returns:
            Dictionary with scaled settings
        """
        settings = {}
        
        if operation == "sparse_reconstruction":
            # Scale image size based on available memory
            max_img_size = int(2048 * self.memory_scaling["max_image_dimension"])
            settings["SiftExtraction.max_image_size"] = str(max_img_size)
            
            # Scale feature extraction settings
            base_features = 10000
            settings["SiftExtraction.max_num_features"] = str(int(base_features * self.memory_scaling["detail_level"]))
            
            # Only use exhaustive matching if we have enough memory
            if memory_ratio > 2.0:
                settings["feature_matcher"] = "exhaustive"
                settings["SiftMatching.max_num_matches"] = str(int(16384 * self.memory_scaling["detail_level"]))
                if memory_ratio > 4.0:
                    settings["SiftMatching.guided_matching"] = "true"
            else:
                settings["feature_matcher"] = "sequential"
                
        elif operation == "dense_reconstruction":
            # Scale point density based on memory
            if self.memory_scaling["point_density"] > 3.0:
                settings["point_density"] = "ultra"
                settings["PatchMatchStereo.window_radius"] = "3"
            elif self.memory_scaling["point_density"] > 2.0:
                settings["point_density"] = "high"
                settings["PatchMatchStereo.window_radius"] = "2"
            elif self.memory_scaling["point_density"] > 1.0:
                settings["point_density"] = "medium"
                settings["PatchMatchStereo.window_radius"] = "2"
            else:
                settings["point_density"] = "low"
                settings["PatchMatchStereo.window_radius"] = "1"
                
            # Scale max image size for stereo
            max_stereo_size = int(2000 * self.memory_scaling["max_image_dimension"])
            settings["PatchMatchStereo.max_image_size"] = str(max_stereo_size)
            
        elif operation == "mesh_generation":
            # Scale mesh resolution based on memory
            if self.memory_scaling["mesh_resolution"] > 3.0:
                settings["mesh_resolution"] = "ultra"
                settings["poisson_depth"] = "12"
            elif self.memory_scaling["mesh_resolution"] > 2.0:
                settings["mesh_resolution"] = "high"
                settings["poisson_depth"] = "10"
            elif self.memory_scaling["mesh_resolution"] > 1.0:
                settings["mesh_resolution"] = "medium"
                settings["poisson_depth"] = "9"
            else:
                settings["mesh_resolution"] = "low"
                settings["poisson_depth"] = "8"
                
            # Scale target face count based on memory
            base_faces = 100000
            target_faces = int(base_faces * self.memory_scaling["mesh_resolution"])
            settings["target_faces"] = str(target_faces)
            
        elif operation == "texture_mapping":
            # Scale texture resolution based on memory
            base_res = 2048
            res_scale = self.memory_scaling["texture_resolution"] ** 1.3
            texture_res = int(base_res * min(4.0, res_scale))
            
            # Ensure texture resolution is power of 2
            texture_powers = [1024, 2048, 4096, 8192, 16384]
            for power in texture_powers:
                if texture_res <= power:
                    texture_res = power
                    break
            
            settings["texture_resolution"] = str(texture_res)
            
            # Advanced UV mapping for high memory systems
            if self.memory_scaling["detail_level"] > 2.0:
                settings["uv_mapping_method"] = "advanced"
                settings["texture_blending"] = "multiband"
        
        elif operation in ["load_images", "image_processing"]:
            # Scale image processing settings
            max_dim = int(3072 * self.memory_scaling["max_image_dimension"])
            settings["max_image_dimension"] = str(max_dim)
            
            if memory_ratio > 2.0:
                settings["enhancement_mode"] = "realistic"
                settings["use_gpu"] = "true"
                settings["max_workers"] = "4"
            else:
                settings["enhancement_mode"] = "balanced"
                settings["use_gpu"] = "false"
                settings["max_workers"] = "2"
        
        logger.info(f"Generated scaled settings for {operation}: {settings}")
        return settings
    
    def get_best_quality_level(self) -> str:
        """
        Determine the best overall quality level based on available memory
        
        Returns:
            Quality level string: 'low', 'medium', 'high', or 'ultra'
        """
        # Get mean scaling factor across parameters
        scaling_values = list(self.memory_scaling.values())
        mean_scaling = sum(scaling_values) / len(scaling_values)
        
        # Determine quality level based on scaling
        if mean_scaling > 3.0:
            quality = "ultra"
        elif mean_scaling > 2.0:
            quality = "high"
        elif mean_scaling > 1.5:
            quality = "medium"
        else:
            quality = "low"
            
        logger.info(f"Determined best quality level: {quality} (scaling factor: {mean_scaling:.2f})")
        return quality
    
    def get_system_specs(self) -> Dict[str, Any]:
        """
        Get detailed system specifications
        
        Returns:
            Dictionary with system specs
        """
        specs = {
            "total_memory_gb": self.system_memory,
            "available_memory_gb": self._get_available_memory(),
            "scaling_factors": self.memory_scaling,
            "quality_level": self.get_best_quality_level()
        }
        
        # Add CPU info
        specs["cpu_count"] = os.cpu_count() or 4
        
        # Add detailed info if psutil available
        if PSUTIL_AVAILABLE:
            specs["memory_utilization"] = psutil.virtual_memory().percent
            specs["cpu_usage"] = psutil.cpu_percent()
        
        # Try to get GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                specs["gpu_available"] = True
                specs["gpu_name"] = torch.cuda.get_device_name(0)
                specs["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                specs["gpu_available"] = False
        except ImportError:
            specs["gpu_available"] = False
        
        return specs
    
    def cleanup(self):
        """Clean up temporary files and memory-mapped arrays"""
        # Close memory-mapped arrays
        for array in self.memory_mapped_arrays:
            try:
                del array
            except:
                pass
            
        self.memory_mapped_arrays = []
        
        # Remove temporary files
        for filename in self.temp_files:
            try:
                os.unlink(filename)
                logger.debug(f"Removed temporary file: {filename}")
            except:
                pass
                
        self.temp_files = []
        
        logger.info("Memory manager cleanup complete")


# Test function
def test_memory_manager():
    """Test the memory manager"""
    try:
        print("Testing MemoryManager...")
        
        # Create memory manager
        mm = MemoryManager()
        
        # Test system specs
        specs = mm.get_system_specs()
        print(f"✓ System specs: {specs['total_memory_gb']:.1f} GB, Quality: {specs['quality_level']}")
        
        # Test memory check
        can_handle = mm.check_memory_requirements(1.0, "sparse_reconstruction")
        print(f"✓ Memory check (1GB sparse): {'Pass' if can_handle else 'Fail'}")
        
        # Test optimization
        settings = mm.optimize_for_operation("sparse_reconstruction")
        print(f"✓ Optimization settings: {len(settings)} parameters")
        
        # Test chunking
        chunks = mm.create_chunked_processing(100)
        print(f"✓ Chunking: {len(chunks)} chunks for 100 items")
        
        # Test quality level
        quality = mm.get_best_quality_level()
        print(f"✓ Best quality level: {quality}")
        
        # Cleanup
        mm.cleanup()
        
        print("✓ MemoryManager test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ MemoryManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_memory_manager()