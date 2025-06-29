"""
Quality Manager for MeshBuilder 3D Reconstruction Pipeline
Centralizes quality level definitions and parameter management across all components
"""
import logging
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union, List

logger = logging.getLogger("MeshBuilder.QualityManager")

class QualityLevel(Enum):
    """Quality levels for 3D reconstruction pipeline"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    VERY_HIGH = "very_high"
    PHOTOREALISTIC = "photorealistic"

@dataclass
class QualitySettings:
    """Complete quality settings for the entire 3D reconstruction pipeline"""
    # Basic info
    name: str
    level: QualityLevel
    description: str
    
    # 3D Gaussian Splatting parameters
    iterations: int
    resolution: int
    
    # Mesh reconstruction parameters
    target_triangles: int
    poisson_depth: int
    voxel_size: float
    
    # System requirements
    estimated_time_minutes: int
    memory_requirement_gb: float
    gpu_memory_requirement_gb: float
    
    # Processing options
    enhance_with_trimesh: bool
    remove_outliers: bool
    use_deep_enhancement: bool

@dataclass
class SystemSpecs:
    """System specifications for quality recommendations"""
    ram_gb: float
    gpu_memory_gb: float
    cpu_cores: int
    gpu_available: bool
    gpu_name: str = "Unknown"

class QualityManager:
    """Manages quality levels and parameters for the entire 3D reconstruction pipeline"""
    
    # Comprehensive quality parameter mappings
    QUALITY_DEFINITIONS = {
        QualityLevel.FAST: QualitySettings(
            name="Fast",
            level=QualityLevel.FAST,
            description="Fast preview quality - good for testing and quick results",
            # 3DGS parameters
            iterations=5000,
            resolution=1000,
            # Mesh parameters
            target_triangles=25000,
            poisson_depth=7,
            voxel_size=0.02,
            # System requirements
            estimated_time_minutes=30,
            memory_requirement_gb=8,
            gpu_memory_requirement_gb=4,
            # Processing options
            enhance_with_trimesh=False,
            remove_outliers=True,
            use_deep_enhancement=False
        ),
        
        QualityLevel.BALANCED: QualitySettings(
            name="Balanced",
            level=QualityLevel.BALANCED,
            description="Balanced quality and speed - recommended for most use cases",
            # 3DGS parameters
            iterations=10000,
            resolution=1200,
            # Mesh parameters
            target_triangles=50000,
            poisson_depth=8,
            voxel_size=0.015,
            # System requirements
            estimated_time_minutes=60,
            memory_requirement_gb=12,
            gpu_memory_requirement_gb=6,
            # Processing options
            enhance_with_trimesh=True,
            remove_outliers=True,
            use_deep_enhancement=False
        ),
        
        QualityLevel.HIGH: QualitySettings(
            name="High",
            level=QualityLevel.HIGH,
            description="High quality results - good for final outputs and presentations",
            # 3DGS parameters
            iterations=15000,
            resolution=1200,
            # Mesh parameters
            target_triangles=100000,
            poisson_depth=9,
            voxel_size=0.01,
            # System requirements
            estimated_time_minutes=90,
            memory_requirement_gb=16,
            gpu_memory_requirement_gb=8,
            # Processing options
            enhance_with_trimesh=True,
            remove_outliers=True,
            use_deep_enhancement=False
        ),
        
        QualityLevel.VERY_HIGH: QualitySettings(
            name="Very High",
            level=QualityLevel.VERY_HIGH,
            description="Very high quality - for professional work and detailed models",
            # 3DGS parameters
            iterations=20000,
            resolution=1400,
            # Mesh parameters
            target_triangles=150000,
            poisson_depth=9,
            voxel_size=0.008,
            # System requirements
            estimated_time_minutes=150,
            memory_requirement_gb=20,
            gpu_memory_requirement_gb=10,
            # Processing options
            enhance_with_trimesh=True,
            remove_outliers=True,
            use_deep_enhancement=True
        ),
        
        QualityLevel.PHOTOREALISTIC: QualitySettings(
            name="Photorealistic",
            level=QualityLevel.PHOTOREALISTIC,
            description="Maximum quality - photorealistic results with highest detail",
            # 3DGS parameters
            iterations=30000,
            resolution=1600,
            # Mesh parameters
            target_triangles=200000,
            poisson_depth=10,
            voxel_size=0.005,
            # System requirements
            estimated_time_minutes=240,
            memory_requirement_gb=24,
            gpu_memory_requirement_gb=12,
            # Processing options
            enhance_with_trimesh=True,
            remove_outliers=True,
            use_deep_enhancement=True
        )
    }
    
    def __init__(self):
        """Initialize the quality manager"""
        self.system_specs = self._analyze_system()
        logger.info(f"Quality Manager initialized")
        logger.info(f"System: {self.system_specs.ram_gb:.1f}GB RAM, "
                   f"{self.system_specs.gpu_memory_gb:.1f}GB GPU, "
                   f"{self.system_specs.cpu_cores} CPU cores")
    
    def _analyze_system(self) -> SystemSpecs:
        """
        Analyze current system specifications
        
        Returns:
            SystemSpecs object with current system information
        """
        specs = SystemSpecs(
            ram_gb=0,
            gpu_memory_gb=0,
            cpu_cores=1,
            gpu_available=False
        )
        
        try:
            # Get RAM information
            memory = psutil.virtual_memory()
            specs.ram_gb = memory.total / (1024**3)
            
            # Get CPU cores
            specs.cpu_cores = psutil.cpu_count(logical=False) or 1
            
            # Check GPU availability and memory
            try:
                import torch
                if torch.cuda.is_available():
                    specs.gpu_available = True
                    specs.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    specs.gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"GPU detected: {specs.gpu_name}")
            except ImportError:
                logger.info("PyTorch not available for GPU detection")
            
        except Exception as e:
            logger.warning(f"Error analyzing system: {e}")
        
        return specs
    
    def get_quality_settings(self, quality_level: Union[str, QualityLevel]) -> QualitySettings:
        """
        Get quality settings for a specified quality level
        
        Args:
            quality_level: Quality level (string or enum)
            
        Returns:
            QualitySettings object with all parameters
        """
        if isinstance(quality_level, str):
            try:
                quality_level = QualityLevel(quality_level.lower())
            except ValueError:
                logger.warning(f"Unknown quality level: {quality_level}, using BALANCED")
                quality_level = QualityLevel.BALANCED
        
        return self.QUALITY_DEFINITIONS[quality_level]
    
    def get_all_quality_levels(self) -> Dict[str, QualitySettings]:
        """
        Get all available quality levels and their settings
        
        Returns:
            Dictionary mapping quality names to settings
        """
        return {level.value: settings for level, settings in self.QUALITY_DEFINITIONS.items()}
    
    def estimate_processing_time(self, 
                               quality_level: Union[str, QualityLevel],
                               num_images: int = 50,
                               system_specs: Optional[SystemSpecs] = None) -> Dict[str, Any]:
        """
        Estimate total processing time for the entire pipeline
        
        Args:
            quality_level: Quality level to estimate for
            num_images: Number of input images
            system_specs: System specifications (uses current system if None)
            
        Returns:
            Dictionary with detailed time estimates
        """
        settings = self.get_quality_settings(quality_level)
        specs = system_specs or self.system_specs
        
        # Base time from quality settings
        base_minutes = settings.estimated_time_minutes
        
        # Factors that affect processing time
        
        # Image count factor (more images = more time, but not linear)
        image_factor = max(0.7, min(2.0, num_images / 50))
        
        # GPU factor (better GPU = faster processing)
        if specs.gpu_available and specs.gpu_memory_gb > 0:
            # GPU memory adequacy factor
            required_gpu = settings.gpu_memory_requirement_gb
            available_gpu = specs.gpu_memory_gb
            
            if available_gpu >= required_gpu * 1.5:
                gpu_factor = 1.2  # Plenty of GPU memory = faster
            elif available_gpu >= required_gpu:
                gpu_factor = 1.0  # Adequate GPU memory
            elif available_gpu >= required_gpu * 0.7:
                gpu_factor = 0.8  # Marginal GPU memory = slower
            else:
                gpu_factor = 0.5  # Insufficient GPU memory = much slower
        else:
            gpu_factor = 0.3  # No GPU = much slower (CPU only)
        
        # RAM factor (more RAM = better performance)
        required_ram = settings.memory_requirement_gb
        available_ram = specs.ram_gb
        
        if available_ram >= required_ram * 1.5:
            ram_factor = 1.2  # Plenty of RAM
        elif available_ram >= required_ram:
            ram_factor = 1.0  # Adequate RAM
        elif available_ram >= required_ram * 0.8:
            ram_factor = 0.9  # Marginal RAM
        else:
            ram_factor = 0.7  # Low RAM
        
        # CPU factor (more cores = better for some operations)
        cpu_factor = min(1.3, 1.0 + (specs.cpu_cores - 4) * 0.05)  # Benefit caps at ~8 cores
        cpu_factor = max(0.8, cpu_factor)  # Minimum factor for low core count
        
        # Calculate final estimate
        # Formula: base_time * image_factor / (gpu_factor * ram_factor * cpu_factor)
        performance_factor = gpu_factor * ram_factor * cpu_factor
        adjusted_minutes = base_minutes * image_factor / performance_factor
        
        # Add breakdown for different pipeline stages
        stage_breakdown = {
            "image_processing": adjusted_minutes * 0.15,      # 15% - Image preprocessing
            "dataset_building": adjusted_minutes * 0.20,     # 20% - COLMAP processing  
            "gaussian_splatting": adjusted_minutes * 0.50,   # 50% - 3DGS training
            "mesh_conversion": adjusted_minutes * 0.10,      # 10% - Point cloud to mesh
            "finalization": adjusted_minutes * 0.05         # 5% - Final processing
        }
        
        return {
            "quality_level": settings.name,
            "num_images": num_images,
            "base_minutes": base_minutes,
            "estimated_minutes": max(15, adjusted_minutes),  # Minimum 15 minutes
            "estimated_hours": max(15, adjusted_minutes) / 60,
            "stage_breakdown": stage_breakdown,
            "factors": {
                "image_factor": image_factor,
                "gpu_factor": gpu_factor,
                "ram_factor": ram_factor,
                "cpu_factor": cpu_factor,
                "performance_factor": performance_factor
            },
            "system_adequate": self.validate_quality_for_system(quality_level, specs),
            "bottlenecks": self._identify_bottlenecks(settings, specs)
        }
    
    def recommend_quality_level(self, 
                               system_specs: Optional[SystemSpecs] = None,
                               target_time_hours: Optional[float] = None,
                               num_images: int = 50) -> Dict[str, Any]:
        """
        Recommend the best quality level for the given system and constraints
        
        Args:
            system_specs: System specifications (uses current system if None)
            target_time_hours: Maximum acceptable processing time in hours
            num_images: Number of images to process
            
        Returns:
            Dictionary with recommendations
        """
        specs = system_specs or self.system_specs
        
        recommendations = {
            "recommended_quality": QualityLevel.FAST,
            "recommended_settings": None,
            "alternatives": [],
            "system_analysis": {
                "ram_gb": specs.ram_gb,
                "gpu_memory_gb": specs.gpu_memory_gb,
                "gpu_available": specs.gpu_available,
                "cpu_cores": specs.cpu_cores
            },
            "reasoning": []
        }
        
        # Check each quality level from highest to lowest
        compatible_levels = []
        
        for quality in [QualityLevel.PHOTOREALISTIC, QualityLevel.VERY_HIGH, 
                       QualityLevel.HIGH, QualityLevel.BALANCED, QualityLevel.FAST]:
            
            settings = self.get_quality_settings(quality)
            is_compatible = self.validate_quality_for_system(quality, specs)
            time_estimate = self.estimate_processing_time(quality, num_images, specs)
            
            level_info = {
                "quality": quality,
                "settings": settings,
                "compatible": is_compatible,
                "estimated_hours": time_estimate["estimated_hours"],
                "bottlenecks": time_estimate["bottlenecks"]
            }
            
            # Check time constraint
            if target_time_hours:
                level_info["within_time_limit"] = time_estimate["estimated_hours"] <= target_time_hours
            else:
                level_info["within_time_limit"] = True
            
            if is_compatible and level_info["within_time_limit"]:
                compatible_levels.append(level_info)
        
        # Select the highest compatible quality
        if compatible_levels:
            best_level = compatible_levels[0]  # Highest quality that's compatible
            recommendations["recommended_quality"] = best_level["quality"]
            recommendations["recommended_settings"] = best_level["settings"]
            
            # Add reasoning
            if best_level["quality"] == QualityLevel.PHOTOREALISTIC:
                recommendations["reasoning"].append("System can handle maximum quality")
            elif len(compatible_levels) == len(self.QUALITY_DEFINITIONS):
                recommendations["reasoning"].append("System can handle all quality levels")
            else:
                recommendations["reasoning"].append(f"System optimized for {best_level['quality'].value} quality")
            
            # Add alternatives (other compatible levels)
            recommendations["alternatives"] = compatible_levels[1:4]  # Next 3 options
        else:
            # Fallback to FAST if nothing is compatible
            recommendations["recommended_quality"] = QualityLevel.FAST
            recommendations["recommended_settings"] = self.get_quality_settings(QualityLevel.FAST)
            recommendations["reasoning"].append("System has limited resources, using fastest setting")
        
        # Add specific recommendations based on system analysis
        if specs.ram_gb < 12:
            recommendations["reasoning"].append("Consider upgrading RAM for better quality options")
        
        if not specs.gpu_available or specs.gpu_memory_gb < 6:
            recommendations["reasoning"].append("GPU upgrade recommended for higher quality levels")
        
        if specs.cpu_cores < 4:
            recommendations["reasoning"].append("More CPU cores would improve processing speed")
        
        return recommendations
    
    def validate_quality_for_system(self, 
                                  quality_level: Union[str, QualityLevel],
                                  system_specs: Optional[SystemSpecs] = None) -> bool:
        """
        Validate if a quality level is suitable for the given system
        
        Args:
            quality_level: Quality level to validate
            system_specs: System specifications (uses current system if None)
            
        Returns:
            True if system can handle the quality level, False otherwise
        """
        settings = self.get_quality_settings(quality_level)
        specs = system_specs or self.system_specs
        
        # Check RAM requirement
        ram_ok = specs.ram_gb >= settings.memory_requirement_gb * 0.8  # 80% threshold
        
        # Check GPU memory requirement (if GPU available)
        if specs.gpu_available:
            gpu_ok = specs.gpu_memory_gb >= settings.gpu_memory_requirement_gb * 0.7  # 70% threshold
        else:
            # If no GPU, only allow FAST and BALANCED quality
            gpu_ok = quality_level in [QualityLevel.FAST, QualityLevel.BALANCED]
        
        return ram_ok and gpu_ok
    
    def _identify_bottlenecks(self, settings: QualitySettings, specs: SystemSpecs) -> List[str]:
        """
        Identify potential performance bottlenecks
        
        Args:
            settings: Quality settings
            specs: System specifications
            
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        
        # RAM bottleneck
        if specs.ram_gb < settings.memory_requirement_gb:
            shortage = settings.memory_requirement_gb - specs.ram_gb
            bottlenecks.append(f"RAM shortage: need {shortage:.1f}GB more")
        elif specs.ram_gb < settings.memory_requirement_gb * 1.2:
            bottlenecks.append("RAM is marginal for this quality level")
        
        # GPU bottleneck
        if not specs.gpu_available:
            bottlenecks.append("No GPU detected - processing will be very slow")
        elif specs.gpu_memory_gb < settings.gpu_memory_requirement_gb:
            shortage = settings.gpu_memory_requirement_gb - specs.gpu_memory_gb
            bottlenecks.append(f"GPU memory shortage: need {shortage:.1f}GB more")
        elif specs.gpu_memory_gb < settings.gpu_memory_requirement_gb * 1.2:
            bottlenecks.append("GPU memory is marginal for this quality level")
        
        # CPU bottleneck
        if specs.cpu_cores < 4:
            bottlenecks.append("Low CPU core count may slow processing")
        
        return bottlenecks
    
    def get_quality_comparison(self) -> Dict[str, Any]:
        """
        Get a comparison of all quality levels
        
        Returns:
            Dictionary with quality level comparison data
        """
        comparison = {
            "levels": [],
            "system_recommendations": self.recommend_quality_level()
        }
        
        for quality in QualityLevel:
            settings = self.get_quality_settings(quality)
            time_estimate = self.estimate_processing_time(quality)
            
            level_data = {
                "name": settings.name,
                "level": quality.value,
                "description": settings.description,
                "iterations": settings.iterations,
                "resolution": settings.resolution,
                "target_triangles": settings.target_triangles,
                "estimated_hours": time_estimate["estimated_hours"],
                "memory_gb": settings.memory_requirement_gb,
                "gpu_memory_gb": settings.gpu_memory_requirement_gb,
                "compatible": self.validate_quality_for_system(quality),
                "bottlenecks": time_estimate["bottlenecks"]
            }
            
            comparison["levels"].append(level_data)
        
        return comparison
    
    def export_quality_settings(self, quality_level: Union[str, QualityLevel]) -> Dict[str, Any]:
        """
        Export quality settings in a format suitable for other components
        
        Args:
            quality_level: Quality level to export
            
        Returns:
            Dictionary with settings for different pipeline components
        """
        settings = self.get_quality_settings(quality_level)
        
        return {
            "general": {
                "quality_level": settings.level.value,
                "name": settings.name,
                "description": settings.description
            },
            "image_processing": {
                "use_deep_enhancement": settings.use_deep_enhancement,
                "enhancement_mode": "realistic" if settings.level in [QualityLevel.HIGH, QualityLevel.VERY_HIGH, QualityLevel.PHOTOREALISTIC] else "balanced"
            },
            "gaussian_splatting": {
                "iterations": settings.iterations,
                "resolution": settings.resolution
            },
            "mesh_conversion": {
                "target_triangles": settings.target_triangles,
                "poisson_depth": settings.poisson_depth,
                "voxel_size": settings.voxel_size,
                "enhance_with_trimesh": settings.enhance_with_trimesh,
                "remove_outliers": settings.remove_outliers
            },
            "system": {
                "memory_requirement_gb": settings.memory_requirement_gb,
                "gpu_memory_requirement_gb": settings.gpu_memory_requirement_gb,
                "estimated_time_minutes": settings.estimated_time_minutes
            }
        }


def test_quality_manager():
    """Test the quality manager functionality"""
    try:
        print("Testing Quality Manager...")
        
        # Create quality manager
        manager = QualityManager()
        
        print(f"\n=== System Analysis ===")
        print(f"RAM: {manager.system_specs.ram_gb:.1f} GB")
        print(f"GPU Memory: {manager.system_specs.gpu_memory_gb:.1f} GB") 
        print(f"CPU Cores: {manager.system_specs.cpu_cores}")
        print(f"GPU Available: {manager.system_specs.gpu_available}")
        
        # Test quality settings
        print(f"\n=== Quality Settings ===")
        for quality in QualityLevel:
            settings = manager.get_quality_settings(quality)
            print(f"{settings.name}: {settings.iterations} iter, {settings.resolution}px, "
                  f"{settings.target_triangles} triangles")
        
        # Test time estimation
        print(f"\n=== Time Estimation ===")
        time_est = manager.estimate_processing_time(QualityLevel.HIGH, 50)
        print(f"HIGH quality (50 images): {time_est['estimated_hours']:.1f} hours")
        print(f"Bottlenecks: {time_est['bottlenecks']}")
        
        # Test recommendations
        print(f"\n=== Recommendations ===")
        rec = manager.recommend_quality_level()
        print(f"Recommended: {rec['recommended_quality'].value}")
        print(f"Reasoning: {rec['reasoning']}")
        
        # Test validation
        print(f"\n=== Quality Validation ===")
        for quality in QualityLevel:
            valid = manager.validate_quality_for_system(quality)
            print(f"{quality.value}: {'✓' if valid else '✗'}")
        
        print(f"\n✓ Quality Manager test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quality Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_quality_manager()
