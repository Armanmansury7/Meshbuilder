"""
Advanced Logger Module for Meshbuilder - FIXED VERSION
Provides enhanced logging functionality with proper import handling
"""
import os
import sys
import logging
import traceback
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Basic imports only - no dependencies on other backend modules yet

class AdvancedLogger:
    """Enhanced logging with context and error tracking"""
    
    def __init__(self, 
                log_dir: Optional[str] = None, 
                app_name: str = "MeshBuilder",
                verbose: bool = False):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for log files (default: ~/.meshbuilder/logs)
            app_name: Application name
            verbose: Whether to enable verbose logging
        """
        # Set up log directory
        if log_dir is None:
            log_dir = os.path.join(os.path.expanduser("~"), ".meshbuilder", "logs")
            
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.app_name = app_name
        self.verbose = verbose
        
        # Create timestamp-based log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{app_name.lower()}_{timestamp}.log")
        
        # Configure root logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        self.logger.addHandler(console_handler)
        
        # Error logging
        self.errors = []
        
        self.logger.info(f"{app_name} logging initialized")
        self.logger.info(f"Log file: {self.log_file}")
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name (optional)
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f"{self.app_name}.{name}")
        return self.logger
    
    def log_system_info(self):
        """Log system information"""
        try:
            import platform
            
            self.logger.info("=== System Information ===")
            self.logger.info(f"OS: {platform.system()} {platform.release()} {platform.version()}")
            self.logger.info(f"Python: {platform.python_version()}")
            self.logger.info(f"CPU: {platform.processor()}")
            
            # Memory info (with fallback)
            try:
                import psutil
                mem = psutil.virtual_memory()
                self.logger.info(f"Memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
                
                # Disk info
                disk = psutil.disk_usage('/')
                self.logger.info(f"Disk: {disk.total / (1024**3):.1f} GB total, {disk.free / (1024**3):.1f} GB free")
            except ImportError:
                self.logger.info("Memory/Disk info: psutil not available")
            
            # GPU info (with fallback)
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                    self.logger.info(f"CUDA: {torch.version.cuda}")
                    self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
                else:
                    self.logger.info("GPU: None")
            except ImportError:
                self.logger.info("GPU: Unknown (torch not available)")
                
            self.logger.info("=== End System Information ===")
            
        except Exception as e:
            self.logger.warning(f"Could not log complete system info: {e}")
    
    def log_command(self, command: str, args: Optional[List[str]] = None):
        """
        Log an external command execution
        
        Args:
            command: Command being executed
            args: Command arguments
        """
        if args:
            args_str = " ".join(str(arg) for arg in args)
            self.logger.info(f"Running command: {command} {args_str}")
        else:
            self.logger.info(f"Running command: {command}")
            
    def log_error(self, 
                 error: Exception, 
                 context: Optional[Dict[str, Any]] = None,
                 include_traceback: bool = True):
        """
        Log an error with context
        
        Args:
            error: Exception object
            context: Context for the error
            include_traceback: Whether to include traceback
        """
        error_str = str(error)
        error_type = error.__class__.__name__
        
        self.logger.error(f"Error: {error_type}: {error_str}")
        
        if context:
            try:
                context_str = json.dumps(context, default=str, indent=2)
                self.logger.error(f"Error context: {context_str}")
            except Exception:
                self.logger.error(f"Error context: {context}")
            
        if include_traceback:
            tb = traceback.format_exc()
            self.logger.error(f"Traceback:\n{tb}")
            
        # Store error for later reference
        error_info = {
            "time": datetime.datetime.now().isoformat(),
            "type": error_type,
            "message": error_str,
            "context": context,
            "traceback": tb if include_traceback else None
        }
        
        self.errors.append(error_info)
    
    def log_performance(self, operation: str, duration: float):
        """
        Log performance metric
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        self.logger.info(f"Performance: {operation} took {duration:.2f} seconds")
    
    def get_recent_logs(self, lines: int = 50) -> List[str]:
        """
        Get recent log entries for display in UI
        
        Args:
            lines: Number of lines to retrieve
            
        Returns:
            List of recent log lines
        """
        if not os.path.exists(self.log_file):
            return []
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
            return all_lines[-lines:]
        except Exception as e:
            self.logger.warning(f"Could not read recent logs: {e}")
            return []
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get list of errors
        
        Returns:
            List of error dictionaries
        """
        return self.errors
    
    def export_logs(self, output_path: Optional[str] = None) -> str:
        """
        Export logs to file
        
        Args:
            output_path: Path to save logs (default: use timestamp)
            
        Returns:
            Path to exported logs
        """
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.log_dir, f"{self.app_name.lower()}_export_{timestamp}.zip")
            
        try:
            import zipfile
            
            with zipfile.ZipFile(output_path, 'w') as zipf:
                # Add main log file
                if os.path.exists(self.log_file):
                    zipf.write(self.log_file, os.path.basename(self.log_file))
                
                # Add errors as JSON
                if self.errors:
                    errors_json = json.dumps(self.errors, indent=2, default=str)
                    errors_path = os.path.join(self.log_dir, "errors.json")
                    
                    with open(errors_path, 'w') as f:
                        f.write(errors_json)
                        
                    zipf.write(errors_path, "errors.json")
                    os.remove(errors_path)  # Clean up
                    
                # Add basic system info
                try:
                    import platform
                    
                    sys_info = {
                        "os": {
                            "system": platform.system(),
                            "release": platform.release(),
                            "version": platform.version()
                        },
                        "python": platform.python_version(),
                        "cpu": platform.processor(),
                    }
                    
                    # Add memory info if psutil available
                    try:
                        import psutil
                        sys_info["memory"] = {
                            "total": psutil.virtual_memory().total,
                            "available": psutil.virtual_memory().available
                        }
                        sys_info["disk"] = {
                            "total": psutil.disk_usage('/').total,
                            "free": psutil.disk_usage('/').free
                        }
                    except ImportError:
                        pass
                    
                    # Add GPU info if torch available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            sys_info["gpu"] = {
                                "name": torch.cuda.get_device_name(0),
                                "cuda_version": torch.version.cuda,
                                "memory": torch.cuda.get_device_properties(0).total_memory
                            }
                    except ImportError:
                        sys_info["gpu"] = "Unknown (torch not available)"
                        
                    sys_info_json = json.dumps(sys_info, indent=2, default=str)
                    sys_info_path = os.path.join(self.log_dir, "system_info.json")
                    
                    with open(sys_info_path, 'w') as f:
                        f.write(sys_info_json)
                        
                    zipf.write(sys_info_path, "system_info.json")
                    os.remove(sys_info_path)  # Clean up
                    
                except Exception as e:
                    self.logger.warning(f"Could not add system info to export: {str(e)}")
            
            self.logger.info(f"Logs exported to {output_path}")
            return output_path
            
        except ImportError:
            self.logger.error("zipfile module not available, cannot export logs")
            return ""
        except Exception as e:
            self.logger.error(f"Error exporting logs: {str(e)}")
            return ""
    
    def close(self):
        """Close the logger and clean up handlers"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# Test function
def test_advanced_logger():
    """Test the advanced logger"""
    try:
        print("Testing AdvancedLogger...")
        
        # Create logger
        logger = AdvancedLogger()
        
        # Test basic logging
        log = logger.get_logger("test")
        log.info("Test message")
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.log_error(e, {"test_context": "test_value"})
        
        # Test system info
        logger.log_system_info()
        
        print("✓ AdvancedLogger test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ AdvancedLogger test failed: {e}")
        return False


if __name__ == "__main__":
    test_advanced_logger()