# =================================================================
# backend/processors/__init__.py
# =================================================================
"""
MeshBuilder Processors Package
Core processing modules for the 3D reconstruction pipeline
"""

# Import processing modules with error handling
try:
    from backend.processors.image_processor import ImageProcessor
except ImportError as e:
    print(f"Warning: Could not import ImageProcessor: {e}")
    ImageProcessor = None

try:
    from backend.processors.video_processor import VideoProcessor
except ImportError as e:
    print(f"Warning: Could not import VideoProcessor: {e}")
    VideoProcessor = None

try:
    from backend.processors.dataset_builder import DatasetBuilder
except ImportError as e:
    print(f"Warning: Could not import DatasetBuilder: {e}")
    DatasetBuilder = None

try:
    from backend.processors.splat_trainer import SplatTrainer
except ImportError as e:
    print(f"Warning: Could not import SplatTrainer: {e}")
    SplatTrainer = None

try:
    from backend.processors.point_cloud_converter import PointCloudToMeshConverter
except ImportError as e:
    print(f"Warning: Could not import PointCloudToMeshConverter: {e}")
    PointCloudToMeshConverter = None

# List available components
available_components = []
for component in ['ImageProcessor', 'VideoProcessor', 'DatasetBuilder', 'SplatTrainer', 'PointCloudToMeshConverter']:
    if globals().get(component) is not None:
        available_components.append(component)

__all__ = available_components

# =================================================================
# backend/utils/__init__.py  
# =================================================================
"""
MeshBuilder Utilities Package
Core utility modules for configuration, logging, error handling, etc.
"""

# Import utilities with error handling
try:
    from backend.utils.config_manager import ConfigManager
except ImportError as e:
    print(f"Warning: Could not import ConfigManager: {e}")
    ConfigManager = None

try:
    from backend.utils.advanced_logger import AdvancedLogger
except ImportError as e:
    print(f"Warning: Could not import AdvancedLogger: {e}")
    AdvancedLogger = None

try:
    from backend.utils.error_recovery import ErrorRecovery
except ImportError as e:
    print(f"Warning: Could not import ErrorRecovery: {e}")
    ErrorRecovery = None

try:
    from backend.utils.memory_manager import MemoryManager
except ImportError as e:
    print(f"Warning: Could not import MemoryManager: {e}")
    MemoryManager = None

try:
    from backend.utils.resource_monitor import ResourceMonitor
except ImportError as e:
    print(f"Warning: Could not import ResourceMonitor: {e}")
    ResourceMonitor = None

try:
    from backend.utils.project_manager import ProjectManager
except ImportError as e:
    print(f"Warning: Could not import ProjectManager: {e}")
    ProjectManager = None

try:
    from backend.utils.batch_processor import BatchProcessor
except ImportError as e:
    print(f"Warning: Could not import BatchProcessor: {e}")
    BatchProcessor = None

try:
    from backend.utils.parallel_processor import ParallelProcessor
except ImportError as e:
    print(f"Warning: Could not import ParallelProcessor: {e}")
    ParallelProcessor = None

try:
    from backend.utils.file_handler import FileHandler
except ImportError as e:
    print(f"Warning: Could not import FileHandler: {e}")
    FileHandler = None

try:
    from backend.utils.settings_manager import SettingsManager
except ImportError as e:
    print(f"Warning: Could not import SettingsManager: {e}")
    SettingsManager = None

try:
    from backend.utils.process_manager import ProcessManager
except ImportError as e:
    print(f"Warning: Could not import ProcessManager: {e}")
    ProcessManager = None

try:
    from backend.utils.optimization_manager import OptimizationManager
except ImportError as e:
    print(f"Warning: Could not import OptimizationManager: {e}")
    OptimizationManager = None

try:
    from backend.utils.file_io_optimizer import FileIOOptimizer
except ImportError as e:
    print(f"Warning: Could not import FileIOOptimizer: {e}")
    FileIOOptimizer = None

try:
    from backend.utils.config_validator import ConfigValidator
except ImportError as e:
    print(f"Warning: Could not import ConfigValidator: {e}")
    ConfigValidator = None

try:
    from backend.utils.error_handler import ErrorHandler
except ImportError as e:
    print(f"Warning: Could not import ErrorHandler: {e}")
    ErrorHandler = None

# List available components
available_components = []
all_components = ['ConfigManager', 'AdvancedLogger', 'ErrorRecovery', 'MemoryManager', 
                 'ResourceMonitor', 'ProjectManager', 'BatchProcessor', 'ParallelProcessor',
                 'FileHandler', 'SettingsManager', 'ProcessManager', 'OptimizationManager',
                 'FileIOOptimizer', 'ConfigValidator', 'ErrorHandler']

for component in all_components:
    if globals().get(component) is not None:
        available_components.append(component)

__all__ = available_components