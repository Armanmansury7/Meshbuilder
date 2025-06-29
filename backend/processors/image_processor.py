"""
Image Processor Module for Meshbuilder - ENHANCED WITH CONSISTENCY FIXES
Handles image preprocessing for 3D reconstruction with integrated utilities
"""
import os
import sys
import logging
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import utilities with error handling
try:
    from utils.config_manager import ConfigManager
    from utils.error_recovery import ErrorRecovery, ErrorResult
    from utils.memory_manager import MemoryManager
    from utils.advanced_logger import AdvancedLogger
    from utils.resource_monitor import ResourceMonitor
    UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some utilities not available: {e}")
    UTILITIES_AVAILABLE = False

# Import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Image processing will be limited.")

# Import NumPy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some image operations will be limited.")

# Import PyTorch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Suppress warnings from the deep learning libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("MeshBuilder.ImageProcessor")

class ImageProcessor:
    """Handles image preprocessing operations for 3D reconstruction with integrated utilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration and utility managers
        
        Args:
            config: Configuration dictionary (optional)
        """
        # FIX 1: Properly initialize cancellation flag FIRST
        self.should_cancel = False
        
        # Load config if not provided
        if config is None and UTILITIES_AVAILABLE:
            try:
                config_manager = ConfigManager()
                self.config = config_manager.get_config_dict()
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
                self.config = self._get_default_config()
        else:
            self.config = config or self._get_default_config()
        
        # Initialize utility managers if available
        if UTILITIES_AVAILABLE:
            try:
                self.error_recovery = ErrorRecovery(self.config)
                self.memory_manager = MemoryManager()
                self.advanced_logger = AdvancedLogger()
                self.resource_monitor = ResourceMonitor()
                logger.info("Utility managers initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize utility managers: {e}")
                self._init_fallback_utilities()
        else:
            self._init_fallback_utilities()
        
        # Check dependencies
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV is required for image processing")
            raise ImportError("OpenCV is required but not available. Install with: pip install opencv-python")
        
        if not NUMPY_AVAILABLE:
            logger.error("NumPy is required for image processing")
            raise ImportError("NumPy is required but not available. Install with: pip install numpy")
        
        # Check GPU availability
        self.use_gpu = False
        self.gpu_info = self._check_gpu_availability()
        
        # Set GPU usage based on config and availability
        gpu_config = self._get_boolean_config("Processing.use_gpu", True)
        if gpu_config and self.gpu_info['available']:
            self.use_gpu = True
            logger.info(f"GPU acceleration enabled: {self.gpu_info['details']}")
        else:
            logger.info("Using CPU for image processing")
        
        # Initialize deep learning models if enabled
        self.dl_models = {}
        dl_config = self._get_boolean_config("Processing.use_deep_enhancement", False)
        self.dl_enabled = dl_config and TORCH_AVAILABLE
        
        if self.dl_enabled:
            try:
                self._initialize_dl_models()
            except Exception as e:
                logger.error(f"Failed to initialize deep learning models: {str(e)}")
                logger.warning("Deep enhancement will be disabled")
                self.dl_enabled = False
        
        logger.info("ImageProcessor initialized successfully")
    
    def cancel_processing(self):
        """Cancel current processing operation"""
        self.should_cancel = True
        logger.info("Image processing cancellation requested")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "Processing": {
                "use_gpu": True,
                "max_image_dimension": 4096,
                "enhance_images": True,
                "quality_threshold": 0.5,
                "max_workers": 4,
                "use_deep_enhancement": False
            },
            "DeepEnhancement": {
                "use_realesrgan": True,
                "use_gfpgan": True,
                "tile_size": 400,
                "half_precision": True,
                "mode": "both"
            }
        }
    
    def _get_numeric_config(self, key_path: str, default_value: Union[int, float]) -> Union[int, float]:
        """
        FIX 2: Enhanced safely get numeric configuration values with robust type conversion
        
        Args:
            key_path: Dot-separated path to config value (e.g., "Processing.max_workers")
            default_value: Default value to return if key not found
            
        Returns:
            Numeric value (int or float)
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    logger.debug(f"Config key not found: {key_path}, using default: {default_value}")
                    return default_value
            
            if value is None:
                return default_value
            
            # Enhanced type conversion handling
            if isinstance(value, str):
                try:
                    # Handle string representations of numbers
                    if isinstance(default_value, int):
                        # Try int conversion first
                        if '.' in value:
                            # Handle decimal strings that should be int
                            return int(float(value))
                        else:
                            return int(value)
                    else:
                        return float(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert '{value}' to {type(default_value).__name__} for {key_path}: {e}")
                    return default_value
            
            # Handle numeric types
            if isinstance(value, (int, float)):
                if isinstance(default_value, int):
                    return int(value)
                else:
                    return float(value)
            
            # Handle other types
            logger.warning(f"Unexpected config type {type(value)} for {key_path}, using default: {default_value}")
            return default_value
            
        except Exception as e:
            logger.warning(f"Error parsing config value for {key_path}: {e}, using default: {default_value}")
            return default_value
    
    def _get_boolean_config(self, key_path: str, default_value: bool) -> bool:
        """
        FIX 2: Enhanced safely get boolean configuration values with robust conversion
        
        Args:
            key_path: Dot-separated path to config value (e.g., "Processing.use_gpu")
            default_value: Default value to return if key not found
            
        Returns:
            Boolean value
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    logger.debug(f"Config key not found: {key_path}, using default: {default_value}")
                    return default_value
            
            if value is None:
                return default_value
            
            # Enhanced boolean conversion
            if isinstance(value, str):
                # Handle various string representations
                value_lower = value.lower().strip()
                if value_lower in ('true', '1', 'yes', 'on', 'enabled', 'enable'):
                    return True
                elif value_lower in ('false', '0', 'no', 'off', 'disabled', 'disable'):
                    return False
                else:
                    logger.warning(f"Unknown boolean string '{value}' for {key_path}, using default: {default_value}")
                    return default_value
            
            # Handle numeric types (0 = False, non-zero = True)
            if isinstance(value, (int, float)):
                return bool(value)
            
            # Handle actual booleans
            if isinstance(value, bool):
                return value
            
            # Default case
            logger.warning(f"Unexpected config type {type(value)} for {key_path}, using default: {default_value}")
            return default_value
            
        except Exception as e:
            logger.warning(f"Error parsing config value for {key_path}: {e}, using default: {default_value}")
            return default_value
    
    def _safe_callback(self, callback: Optional[Callable], message: str, progress: int):
        """
        FIX 3: Safe callback execution with consistent signature
        
        Args:
            callback: Callback function
            message: Progress message
            progress: Progress percentage (0-100, or -1 for error)
        """
        if callback:
            try:
                callback(message, progress)
            except Exception as e:
                logger.warning(f"Callback execution failed: {e}")
    
    def _init_fallback_utilities(self):
        """Initialize fallback utilities when main utilities are not available"""
        self.error_recovery = None
        self.memory_manager = None
        self.advanced_logger = None
        self.resource_monitor = None
        logger.warning("Running with fallback utilities - some features may be limited")
    
    def _initialize_dl_models(self):
        """Initialize deep learning models for image enhancement"""
        logger.info("Initializing deep learning models...")
        
        # This is a placeholder for deep learning model initialization
        # In a real implementation, you would load models like Real-ESRGAN, GFPGAN, etc.
        # For now, we'll just log that it's not implemented
        logger.warning("Deep learning model initialization not implemented in this version")
        logger.info("To enable deep learning enhancement, install required packages:")
        logger.info("pip install torch basicsr realesrgan gfpgan facexlib")
        
        self.dl_enabled = False
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """
        Check for GPU availability for image processing
        
        Returns:
            Dictionary with GPU information
        """
        gpu_info = {
            'available': False,
            'opencv_cuda': False,
            'details': {}
        }
        
        try:
            # Check if OpenCV has CUDA support
            if OPENCV_AVAILABLE:
                cv2_info = cv2.getBuildInformation()
                
                if 'CUDA:YES' in cv2_info or 'CUDA: YES' in cv2_info:
                    gpu_info['opencv_cuda'] = True
                    
                    # Try to set CUDA device with better error handling
                    try:
                        cv2.setUseOptimized(True)
                        if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                            if cuda_devices > 0:
                                cv2.cuda.setDevice(0)
                                gpu_info['available'] = True
                                logger.info(f"OpenCV detected {cuda_devices} CUDA device(s)")
                    except Exception as e:
                        logger.warning(f"Failed to set OpenCV CUDA device: {str(e)}")
                        gpu_info['opencv_cuda'] = False
            
            # Check for PyTorch CUDA
            if TORCH_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        gpu_info['cuda_available'] = True
                        gpu_info['available'] = True
                        gpu_info['device_count'] = torch.cuda.device_count()
                        gpu_info['device_name'] = torch.cuda.get_device_name(0)
                        
                        # Get memory information
                        try:
                            gpu_info['memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                        except Exception as e:
                            logger.debug(f"Could not get GPU memory info: {e}")
                            gpu_info['memory'] = 0
                except Exception as e:
                    logger.warning(f"Error checking PyTorch CUDA: {e}")
            
            # Store details for logging
            gpu_info['details'] = {
                'opencv_cuda': gpu_info['opencv_cuda'],
                'cuda_available': gpu_info.get('cuda_available', False),
                'device_name': gpu_info.get('device_name', 'Unknown'),
                'memory_gb': gpu_info.get('memory', 0)
            }
        
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}")
        
        return gpu_info
    
    def preprocess_images(self, 
                         image_paths: List[str], 
                         output_dir: Path,
                         quality_threshold: Optional[float] = None,
                         enhancement_mode: str = "balanced",
                         use_gpu: Optional[bool] = None,
                         deep_enhance: Optional[bool] = None,
                         callback: Optional[Callable[[str, int], None]] = None) -> List[str]:
        """
        Preprocess images for 3D reconstruction with enhanced algorithms and error recovery
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save processed images
            quality_threshold: Threshold for image quality (0.0-1.0)
            enhancement_mode: Color enhancement mode ("balanced", "vibrant", "realistic")
            use_gpu: Whether to use GPU acceleration (overrides default)
            deep_enhance: Whether to use deep learning enhancement (overrides default)
            callback: Progress callback function (message: str, progress: int)
            
        Returns:
            List of paths to processed images
        """
        # FIX 1: Reset cancellation flag with direct access
        self.should_cancel = False
        
        # Use defaults from config if not specified
        if quality_threshold is None:
            quality_threshold = self._get_numeric_config("Processing.quality_threshold", 0.5)
        
        # Determine GPU usage
        use_gpu = self.use_gpu if use_gpu is None else use_gpu
        
        # Determine deep enhancement usage
        deep_enhance = self.dl_enabled if deep_enhance is None else deep_enhance
        if deep_enhance and not self.dl_enabled:
            logger.warning("Deep enhancement requested but not available, will use standard enhancement")
            deep_enhance = False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_paths = []
        
        # Start resource monitoring if available
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        logger.info(f"Preprocessing {len(image_paths)} images with enhancement mode: {enhancement_mode}")
        logger.info(f"Deep learning enhancement: {'Enabled' if deep_enhance else 'Disabled'}")
        
        # FIX 3: Use safe callback with consistent signature
        callback_msg = f"Preprocessing {len(image_paths)} images"
        if deep_enhance:
            callback_msg += " with deep enhancement"
        self._safe_callback(callback, callback_msg, 5)
        
        # Get memory-optimized settings if memory manager is available
        if self.memory_manager:
            # Check if we have enough memory for the operation
            input_size_gb = self._estimate_input_size(image_paths)
            if not self.memory_manager.check_memory_requirements(input_size_gb, "image_processing"):
                logger.warning("Insufficient memory for optimal processing, applying optimizations")
                # Get memory-optimized settings
                optimized_settings = self.memory_manager.optimize_for_operation("image_processing")
                # Apply optimizations
                if "max_image_dimension" in optimized_settings:
                    try:
                        self.config["Processing"]["max_image_dimension"] = int(optimized_settings["max_image_dimension"])
                    except (ValueError, TypeError):
                        logger.warning("Invalid max_image_dimension in optimization settings")
                if "max_workers" in optimized_settings:
                    try:
                        self.config["Processing"]["max_workers"] = int(optimized_settings["max_workers"])
                    except (ValueError, TypeError):
                        logger.warning("Invalid max_workers in optimization settings")
                if "use_gpu" in optimized_settings:
                    try:
                        use_gpu = str(optimized_settings["use_gpu"]).lower() == "true"
                    except (AttributeError, TypeError):
                        logger.warning("Invalid use_gpu in optimization settings")
        
        # FIX 2: Get max workers with enhanced type conversion
        max_workers = self._get_numeric_config("Processing.max_workers", 4)
        
        # Process images with error recovery
        max_retries = 3
        for attempt in range(max_retries):
            try:
                processed_paths = self._process_images_batch(
                    image_paths, output_dir, quality_threshold, enhancement_mode,
                    use_gpu, deep_enhance, max_workers, callback
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                # Check if it was a cancellation
                if "cancelled" in str(e).lower():
                    logger.info("Image processing cancelled by user")
                    return []
                
                logger.error(f"Image processing attempt {attempt + 1} failed: {e}")
                
                if self.error_recovery and attempt < max_retries - 1:
                    # Try error recovery
                    recovery_result = self.error_recovery.handle_error(
                        "image_processing_failed", e, {
                            "input_images": len(image_paths),
                            "enhancement_mode": enhancement_mode,
                            "use_gpu": use_gpu
                        }
                    )
                    
                    if recovery_result and recovery_result.action == "retry_with_params":
                        # Apply recovery parameters
                        recovery_params = recovery_result.data.get("params", {})
                        logger.info(f"Applying recovery parameters: {recovery_params}")
                        
                        # Update settings based on recovery
                        if "enhancement_mode" in recovery_params:
                            enhancement_mode = recovery_params["enhancement_mode"]
                        if "use_gpu" in recovery_params:
                            use_gpu = str(recovery_params["use_gpu"]).lower() == "true"
                        if "max_image_dimension" in recovery_params:
                            try:
                                self.config["Processing"]["max_image_dimension"] = int(recovery_params["max_image_dimension"])
                            except (ValueError, TypeError):
                                logger.warning("Invalid max_image_dimension in recovery params")
                        
                        continue  # Retry with new parameters
                
                if attempt == max_retries - 1:
                    logger.error("All image processing attempts failed")
                    if self.error_recovery:
                        user_msg = self.error_recovery.get_user_friendly_message("image_processing_failed")
                        logger.error(f"User message: {user_msg}")
                    raise e
        
        # Stop resource monitoring if available
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            # Log resource recommendations
            recommendations = self.resource_monitor.get_recommendation()
            for resource, recommendation in recommendations.items():
                logger.info(f"Resource recommendation ({resource}): {recommendation}")
        
        # Log results
        passed_count = len(processed_paths)
        filtered_count = len(image_paths) - passed_count
        
        logger.info(f"Image preprocessing complete: {passed_count} images passed, {filtered_count} filtered out")
        
        # FIX 3: Use safe callback
        self._safe_callback(callback, f"Preprocessed {passed_count} images, filtered {filtered_count} low-quality images", 100)
        
        return processed_paths
    
    def _estimate_input_size(self, image_paths: List[str]) -> float:
        """Estimate input size in GB"""
        total_size = 0
        sample_size = min(10, len(image_paths))  # Sample first 10 images
        
        for i, img_path in enumerate(image_paths[:sample_size]):
            try:
                if os.path.exists(img_path):
                    total_size += os.path.getsize(img_path)
            except Exception as e:
                logger.debug(f"Could not get size for {img_path}: {e}")
        
        if sample_size > 0:
            avg_size = total_size / sample_size
            estimated_total = avg_size * len(image_paths)
            return estimated_total / (1024**3)  # Convert to GB
        
        return 1.0  # Default estimate
    
    def _process_images_batch(self, 
                             image_paths: List[str], 
                             output_dir: Path,
                             quality_threshold: float,
                             enhancement_mode: str,
                             use_gpu: bool,
                             deep_enhance: bool,
                             max_workers: Union[int, float],
                             callback: Optional[Callable]) -> List[str]:
        """Process images in batch with parallel processing"""
        
        # FIX 1: Check for cancellation with direct access
        if self.should_cancel:
            raise RuntimeError("Processing cancelled by user")
        
        processed_paths = []
        
        # Determine batch size based on memory constraints
        if deep_enhance:
            batch_size = min(20, len(image_paths))
        else:
            batch_size = min(100, len(image_paths))
        
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        # Process images in batches
        total_processed = 0
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            # FIX 1: Check for cancellation in loop with direct access
            if self.should_cancel:
                raise RuntimeError("Processing cancelled by user")
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_image_paths = image_paths[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_image_paths)} images")
            
            # FIX 3: Use safe callback
            if callback:
                progress = 5 + (batch_idx / total_batches) * 90
                self._safe_callback(callback, f"Processing image batch {batch_idx+1}/{total_batches}", int(progress))
            
            # Use parallel processing for the batch
            batch_results = []
            
            # FIX 2: Enhanced max_workers validation and conversion
            try:
                max_workers = int(max_workers) if not isinstance(max_workers, int) else max_workers
                max_workers = max(1, min(max_workers, 32))  # Ensure it's between 1 and 32
            except (ValueError, TypeError):
                logger.warning(f"Invalid max_workers value: {max_workers}, using default of 4")
                max_workers = 4
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit processing tasks
                future_to_path = {
                    executor.submit(
                        self._process_single_image, 
                        idx + start_idx,
                        img_path, 
                        output_dir, 
                        quality_threshold, 
                        enhancement_mode,
                        use_gpu,
                        deep_enhance
                    ): img_path 
                    for idx, img_path in enumerate(batch_image_paths)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    # FIX 1: Check for cancellation during processing with direct access
                    if self.should_cancel:
                        # Cancel all remaining futures
                        for f in future_to_path:
                            f.cancel()
                        raise RuntimeError("Processing cancelled by user")
                    
                    img_path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                            total_processed += 1
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {str(e)}")
            
            # Add batch results to overall results
            processed_paths.extend(batch_results)
            
            # Free up memory
            import gc
            gc.collect()
            
            # Explicitly release CUDA memory if using GPU with better error handling
            if use_gpu and TORCH_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"Could not clear CUDA cache: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Image processing took {elapsed_time:.2f} seconds ({elapsed_time/len(image_paths):.2f} seconds per image)")
        
        return processed_paths
    
    def _process_single_image(self, 
                             idx: int, 
                             img_path: str, 
                             output_dir: Path, 
                             quality_threshold: float,
                             enhancement_mode: str,
                             use_gpu: bool,
                             deep_enhance: bool) -> Optional[str]:
        """
        FIX 4: Process a single image with enhanced error handling
        
        Args:
            idx: Image index
            img_path: Path to the image
            output_dir: Directory to save processed image
            quality_threshold: Quality threshold
            enhancement_mode: Enhancement mode
            use_gpu: Whether to use GPU
            deep_enhance: Whether to use deep learning enhancement
            
        Returns:
            Path to processed image if successful, None otherwise
        """
        try:
            # FIX 1: Check for cancellation with direct access
            if self.should_cancel:
                return None
            
            # Enhanced error message for file reading
            if not os.path.exists(img_path):
                logger.warning(f"Image file does not exist: {img_path}")
                return None
            
            # Read the image with better error handling
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not read image (unsupported format or corrupted): {img_path}")
                    return None
            except Exception as e:
                logger.error(f"Error reading image {img_path}: {e}")
                return None
            
            # Validate image dimensions
            if len(img.shape) != 3 or img.shape[2] != 3:
                logger.warning(f"Image has unexpected dimensions {img.shape}: {img_path}")
                return None
            
            # Check image quality
            try:
                quality_score = self._assess_image_quality(img)
                if quality_score < quality_threshold:
                    logger.warning(f"Image {img_path} below quality threshold ({quality_score:.2f} < {quality_threshold})")
                    return None
            except Exception as e:
                logger.error(f"Error assessing image quality for {img_path}: {e}")
                return None
            
            # Resize if needed - FIX 2: Use enhanced numeric config retrieval
            max_dimension = self._get_numeric_config("Processing.max_image_dimension", 4096)
            if max(img.shape[0], img.shape[1]) > max_dimension:
                try:
                    img = self._resize_image(img, int(max_dimension))
                except Exception as e:
                    logger.error(f"Error resizing image {img_path}: {e}")
                    return None
            
            # Standard enhancement if enabled
            if self._get_boolean_config("Processing.enhance_images", True):
                try:
                    img = self._enhance_image(img, enhancement_mode)
                except Exception as e:
                    logger.warning(f"Error enhancing image {img_path}: {e}, using original")
                    # Continue with original image
            
            # Apply deep learning enhancement if enabled
            if deep_enhance:
                logger.debug(f"Applying deep enhancement to image {idx}")
                try:
                    img = self._apply_deep_enhancement(img)
                    
                    # Resize again if the deep enhancement upscaled the image
                    if max(img.shape[0], img.shape[1]) > max_dimension:
                        img = self._resize_image(img, int(max_dimension))
                except Exception as e:
                    logger.warning(f"Error in deep enhancement for {img_path}: {e}, using standard image")
                    # Continue with standard enhanced image
            
            # Save processed image
            try:
                img_filename = f"processed_{idx:04d}.jpg"
                output_path = output_dir / img_filename
                
                # Use high quality JPEG settings
                success = cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    logger.error(f"Failed to save processed image: {output_path}")
                    return None
                
                return str(output_path)
            except Exception as e:
                logger.error(f"Error saving processed image {img_path}: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error processing image {img_path}: {str(e)}")
            return None
    
    def _apply_deep_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply deep learning-based enhancement to an image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Enhanced image
        """
        if not self.dl_models or not self.dl_enabled:
            logger.debug("Deep enhancement requested but models not initialized")
            return img
        
        try:
            # Placeholder for deep learning enhancement
            # In a real implementation, this would use models like Real-ESRGAN, GFPGAN, etc.
            logger.debug("Deep learning enhancement not implemented in this version")
            return img
            
        except Exception as e:
            logger.error(f"Error in deep enhancement: {str(e)}")
            # Return original image if enhancement fails
            return img
    
    def _assess_image_quality(self, img: np.ndarray) -> float:
        """
        Assess image quality using multiple metrics
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Quality score (0.0-1.0)
        """
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Calculate sharpness using Laplacian variance
            try:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_score = min(1.0, max(0.0, laplacian_var / 1000))
            except Exception:
                blur_score = 0.5
            
            # Calculate contrast score
            try:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist_norm = hist / (gray.shape[0] * gray.shape[1])
                contrast_score = np.std(hist_norm) * 100
                contrast_score = min(1.0, max(0.0, contrast_score / 0.1))
            except Exception:
                contrast_score = 0.5
            
            # Calculate exposure score
            try:
                mean_val = np.mean(gray)
                exposure_score = 1.0
                if mean_val < 40:  # Too dark
                    exposure_score = mean_val / 40.0
                elif mean_val > 220:  # Too bright
                    exposure_score = (255 - mean_val) / 35.0
            except Exception:
                exposure_score = 0.5
            
            # Combined quality score
            quality_score = (blur_score * 0.5 + 
                            contrast_score * 0.3 + 
                            exposure_score * 0.2)
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Error assessing image quality: {str(e)}")
            return 0.5  # Return neutral score if assessment fails
    
    def _resize_image(self, img: np.ndarray, max_dimension: int) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            img: Input image
            max_dimension: Maximum dimension (width or height)
            
        Returns:
            Resized image
        """
        try:
            h, w = img.shape[:2]
            if max(h, w) <= max_dimension:
                return img
                
            # Calculate new dimensions
            if h > w:
                new_h = max_dimension
                new_w = int(w * (max_dimension / h))
            else:
                new_w = max_dimension
                new_h = int(h * (max_dimension / w))
                
            # Ensure dimensions are at least 1
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            # Resize using area interpolation for downsampling
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return resized
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return img  # Return original on error
    
    def _enhance_image(self, img: np.ndarray, enhancement_mode: str = "balanced") -> np.ndarray:
        """
        FIX 4: Enhance image for better feature detection with improved error handling
        
        Args:
            img: Input image
            enhancement_mode: Enhancement mode ("balanced", "vibrant", "realistic")
            
        Returns:
            Enhanced image
        """
        try:
            if enhancement_mode == "vibrant":
                return self._apply_vibrant_enhancement(img)
            elif enhancement_mode == "realistic":
                return self._apply_realistic_enhancement(img)
            else:  # balanced (default)
                return self._apply_balanced_enhancement(img)
        except Exception as e:
            logger.warning(f"Error in image enhancement ({enhancement_mode}): {str(e)}")
            return img  # Return original if enhancement fails
    
    def _apply_balanced_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply balanced enhancement with error handling"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            lab = cv2.merge((l, a, b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply subtle bilateral filter for noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Error in balanced enhancement: {e}")
            return img
    
    def _apply_vibrant_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply vibrant color enhancement with error handling"""
        try:
            # Convert to float for processing
            img_float = img.astype(np.float32) / 255.0
            
            # Convert to HSV
            hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
            
            # Enhance saturation (boost by 30%)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 1)
            
            # Enhance value (contrast boost by 20%, brightness by 5%)
            hsv[:, :, 2] = np.clip(0.5 + (hsv[:, :, 2] - 0.5) * 1.2 + 0.05, 0, 1)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Convert back to 8-bit
            return (enhanced * 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Error in vibrant enhancement: {e}")
            return img
    
    def _apply_realistic_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply realistic enhancement with error handling"""
        try:
            # Convert to float for processing
            img_float = img.astype(np.float32) / 255.0
            
            # Convert to HSV
            hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
            
            # Subtle saturation boost (15%)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 1)
            
            # Refined contrast boost (15%)
            hsv[:, :, 2] = np.clip(0.5 + (hsv[:, :, 2] - 0.5) * 1.15, 0, 1)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Convert back to 8-bit
            return (enhanced * 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Error in realistic enhancement: {e}")
            return img


# Test function
def test_image_processor():
    """Test the image processor with enhanced consistency checks"""
    try:
        print("Testing ImageProcessor...")
        
        # Create image processor
        processor = ImageProcessor()
        
        # Test that cancellation flag is properly initialized
        assert hasattr(processor, 'should_cancel'), "should_cancel not initialized"
        assert processor.should_cancel == False, "should_cancel not properly initialized"
        print(f"✓ Cancellation flag properly initialized: {processor.should_cancel}")
        
        # Test GPU availability check
        gpu_info = processor._check_gpu_availability()
        print(f"✓ GPU availability: {gpu_info['available']}")
        
        # Test configuration helpers with various inputs
        # Test numeric config
        max_workers = processor._get_numeric_config("Processing.max_workers", 4)
        assert isinstance(max_workers, (int, float)), f"max_workers should be numeric, got {type(max_workers)}"
        print(f"✓ Numeric config test: max_workers = {max_workers} (type: {type(max_workers).__name__})")
        
        # Test boolean config
        use_gpu = processor._get_boolean_config("Processing.use_gpu", True)
        assert isinstance(use_gpu, bool), f"use_gpu should be boolean, got {type(use_gpu)}"
        print(f"✓ Boolean config test: use_gpu = {use_gpu} (type: {type(use_gpu).__name__})")
        
        # Test callback safety
        test_callback_called = False
        def test_callback(message, progress):
            nonlocal test_callback_called
            assert isinstance(message, str), "Message should be string"
            assert isinstance(progress, int), "Progress should be int"
            test_callback_called = True
        
        processor._safe_callback(test_callback, "test message", 50)
        assert test_callback_called, "Callback was not called"
        print("✓ Safe callback test passed")
        
        # Test image quality assessment (with dummy image)
        if NUMPY_AVAILABLE:
            dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            quality = processor._assess_image_quality(dummy_img)
            assert 0.0 <= quality <= 1.0, f"Quality score should be 0-1, got {quality}"
            print(f"✓ Image quality assessment: {quality:.2f}")
        
        # Test image enhancement (with dummy image)
        if NUMPY_AVAILABLE:
            enhanced = processor._enhance_image(dummy_img, "balanced")
            assert enhanced.shape == dummy_img.shape, "Enhanced image shape mismatch"
            print(f"✓ Image enhancement: {enhanced.shape}")
        
        # Test cancellation
        processor.cancel_processing()
        assert processor.should_cancel == True, "Cancellation flag not set"
        print(f"✓ Cancellation flag set: {processor.should_cancel}")
        
        print("✓ ImageProcessor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ ImageProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_image_processor()