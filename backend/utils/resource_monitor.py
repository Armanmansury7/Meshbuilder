"""
Resource Monitor Module for Meshbuilder
Tracks system resource usage during processing
"""
import os
import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional

logger = logging.getLogger("MeshBuilder.ResourceMonitor")

class ResourceMonitor:
    """Monitors system resources during processing"""
    
    def __init__(self, interval: float = 2.0):
        """
        Initialize with monitoring interval
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource usage history
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.disk_usage = []
        
        # Peak usage
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
        self.peak_gpu = 0.0
        self.peak_disk_io = 0.0
        
        # GPU monitoring
        self.has_gpu = False
        self.initialize_gpu_monitoring()
    
    def initialize_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                logger.info(f"GPU monitoring enabled for {torch.cuda.get_device_name(0)}")
                
                # Try to initialize NVML for more detailed GPU metrics
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    self.nvml_available = True
                    device_count = pynvml.nvmlDeviceGetCount()
                    logger.info(f"NVML initialized with {device_count} devices")
                except:
                    self.nvml_available = False
            else:
                logger.info("No CUDA GPU available for monitoring")
        except ImportError:
            logger.info("torch not available, GPU monitoring disabled")
            self.has_gpu = False
            self.nvml_available = False
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
            
        # Reset metrics
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.disk_usage = []
        
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
        self.peak_gpu = 0.0
        self.peak_disk_io = 0.0
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.interval * 2)
            
        logger.info("Resource monitoring stopped")
        self._log_resource_summary()
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        prev_disk_io = psutil.disk_io_counters()
        last_time = time.time()
        
        while self.monitoring:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_usage.append(cpu_percent)
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.memory_usage.append(memory_percent)
                self.peak_memory = max(self.peak_memory, memory_percent)
                
                # Get disk I/O
                curr_time = time.time()
                time_delta = curr_time - last_time
                curr_disk_io = psutil.disk_io_counters()
                
                if prev_disk_io and curr_disk_io:
                    # Calculate I/O rate in MB/s
                    read_mbps = (curr_disk_io.read_bytes - prev_disk_io.read_bytes) / (1024 * 1024) / time_delta
                    write_mbps = (curr_disk_io.write_bytes - prev_disk_io.write_bytes) / (1024 * 1024) / time_delta
                    
                    disk_io = read_mbps + write_mbps
                    self.disk_usage.append(disk_io)
                    self.peak_disk_io = max(self.peak_disk_io, disk_io)
                    
                prev_disk_io = curr_disk_io
                last_time = curr_time
                
                # Get GPU usage if available
                if self.has_gpu:
                    try:
                        if self.nvml_available:
                            import pynvml
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_percent = util.gpu
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_memory_percent = mem_info.used / mem_info.total * 100
                        else:
                            import torch
                            # Less accurate method using torch
                            gpu_percent = 0.0
                            if torch.cuda.is_available():
                                # Allocate small tensor to measure memory usage
                                torch.cuda.synchronize()
                                allocated = torch.cuda.memory_allocated()
                                reserved = torch.cuda.memory_reserved()
                                total = torch.cuda.get_device_properties(0).total_memory
                                gpu_memory_percent = allocated / total * 100
                                gpu_percent = reserved / total * 100
                                
                        self.gpu_usage.append(gpu_percent)
                        self.peak_gpu = max(self.peak_gpu, gpu_percent)
                    except Exception as e:
                        logger.warning(f"Error monitoring GPU: {str(e)}")
                        if self.has_gpu:
                            # Fallback to estimation
                            self.gpu_usage.append(50.0 if self.cpu_usage[-1] > 80 else 30.0)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
            
            # Sleep until next interval
            time.sleep(self.interval)
    
    def _log_resource_summary(self):
        """Log resource usage summary"""
        if not self.cpu_usage:
            return
            
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
        avg_disk = sum(self.disk_usage) / len(self.disk_usage) if self.disk_usage else 0
        
        logger.info(f"Resource usage summary:")
        logger.info(f"CPU - Avg: {avg_cpu:.1f}%, Peak: {self.peak_cpu:.1f}%")
        logger.info(f"Memory - Avg: {avg_memory:.1f}%, Peak: {self.peak_memory:.1f}%")
        
        if self.has_gpu:
            logger.info(f"GPU - Avg: {avg_gpu:.1f}%, Peak: {self.peak_gpu:.1f}%")
            
        logger.info(f"Disk I/O - Avg: {avg_disk:.1f} MB/s, Peak: {self.peak_disk_io:.1f} MB/s")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get resource usage summary
        
        Returns:
            Dictionary of resource statistics
        """
        summary = {}
        
        if self.cpu_usage:
            summary["cpu"] = {
                "current": self.cpu_usage[-1] if self.cpu_usage else 0,
                "average": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
                "peak": self.peak_cpu,
                "history": self.cpu_usage
            }
            
        if self.memory_usage:
            summary["memory"] = {
                "current": self.memory_usage[-1] if self.memory_usage else 0,
                "average": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                "peak": self.peak_memory,
                "history": self.memory_usage
            }
            
        if self.gpu_usage:
            summary["gpu"] = {
                "current": self.gpu_usage[-1] if self.gpu_usage else 0,
                "average": sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0,
                "peak": self.peak_gpu,
                "history": self.gpu_usage
            }
            
        if self.disk_usage:
            summary["disk"] = {
                "current": self.disk_usage[-1] if self.disk_usage else 0,
                "average": sum(self.disk_usage) / len(self.disk_usage) if self.disk_usage else 0,
                "peak": self.peak_disk_io,
                "history": self.disk_usage
            }
            
        return summary
    
    def get_recommendation(self) -> Dict[str, str]:
        """
        Get resource optimization recommendations
        
        Returns:
            Dictionary of resource recommendations
        """
        recommendations = {}
        
        # CPU recommendations
        if self.peak_cpu > 90:
            recommendations["cpu"] = "CPU usage is very high. Consider reducing quality settings or using a more powerful CPU."
        elif self.peak_cpu > 75:
            recommendations["cpu"] = "CPU usage is high but acceptable. For faster processing, reduce quality settings."
            
        # Memory recommendations
        if self.peak_memory > 90:
            recommendations["memory"] = "Memory usage is very high. Consider reducing max images or point cloud density."
        elif self.peak_memory > 75:
            recommendations["memory"] = "Memory usage is high but acceptable. For larger projects, increase system memory."
            
        # GPU recommendations
        if self.has_gpu and self.peak_gpu > 90:
            recommendations["gpu"] = "GPU usage is very high. Consider reducing quality settings or using a more powerful GPU."
        elif self.has_gpu and self.peak_gpu > 75:
            recommendations["gpu"] = "GPU usage is high but acceptable. For faster processing, reduce quality settings."
        elif not self.has_gpu:
            recommendations["gpu"] = "No GPU detected. Processing will be significantly faster with a compatible NVIDIA GPU."
            
        # Disk recommendations
        if self.peak_disk_io > 100:  # 100 MB/s
            recommendations["disk"] = "Disk I/O is very high. Using an SSD will significantly improve performance."
            
        return recommendations