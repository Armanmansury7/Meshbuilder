import os
import sys
import time
import numpy as np
import multiprocessing
from functools import wraps
import psutil
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.advanced_logger import AdvancedLogger
from utils.resource_monitor import ResourceMonitor

class OptimizationManager:
    """
    Manages performance optimization strategies.
    Handles dynamic workload adjustment, parallel processing, and task scheduling.
    """
    def __init__(self):
        self.logger = AdvancedLogger().get_logger(self.__class__.__name__)
        self.resource_monitor = ResourceMonitor()
        self.optimal_threads = min(multiprocessing.cpu_count(), 8)
        self.performance_stats = {}
    
    def profile_execution(self, func):
        """
        Decorator to profile function execution time and resource usage.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_mem = self.resource_monitor.get_memory_usage()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_mem = self.resource_monitor.get_memory_usage()
            
            # Record performance stats
            exec_time = end_time - start_time
            mem_usage = end_mem - start_mem
            
            func_name = func.__name__
            self.performance_stats[func_name] = {
                'last_execution_time': exec_time,
                'last_memory_usage': mem_usage,
                'call_count': self.performance_stats.get(func_name, {}).get('call_count', 0) + 1,
                'average_execution_time': self._update_average(
                    self.performance_stats.get(func_name, {}).get('average_execution_time', 0),
                    self.performance_stats.get(func_name, {}).get('call_count', 0),
                    exec_time
                )
            }
            
            self.logger.debug(f"Function {func_name} executed in {exec_time:.2f}s with {mem_usage:.2f}MB memory usage")
            return result
        
        return wrapper
    
    def _update_average(self, prev_avg, count, new_value):
        """Update running average with new value."""
        if count == 0:
            return new_value
        return (prev_avg * count + new_value) / (count + 1)
    
    def get_optimal_chunk_size(self, data_size, min_chunk_size=10, max_chunks=None):
        """
        Calculate optimal chunk size for parallel processing based on 
        available resources and data size.
        
        Args:
            data_size: Total size of data to process
            min_chunk_size: Minimum size of each chunk
            max_chunks: Maximum number of chunks
            
        Returns:
            Optimal chunk size
        """
        # Get available processors
        max_processors = min(self.optimal_threads, 
                             psutil.cpu_count(logical=False) * 2)
        
        # Check memory headroom
        mem_headroom = self.resource_monitor.get_memory_headroom()
        
        # Adjust processors based on memory headroom
        if mem_headroom < 0.2:  # Less than 20% memory available
            max_processors = max(1, int(max_processors * 0.5))
        elif mem_headroom < 0.4:  # Less than 40% memory available
            max_processors = max(1, int(max_processors * 0.75))
        
        # Calculate chunks
        if max_chunks is None:
            max_chunks = max_processors * 4
        
        chunks = min(max_chunks, max_processors * 2, data_size // min_chunk_size)
        chunks = max(1, chunks)  # At least 1 chunk
        
        # Calculate chunk size
        chunk_size = max(min_chunk_size, data_size // chunks)
        
        self.logger.debug(f"Optimal chunk size: {chunk_size} for data size: {data_size} " 
                          f"using {chunks} chunks on {max_processors} processors")
        
        return chunk_size
    
    def parallel_execute(self, func, data_list, chunk_size=None, max_workers=None, **kwargs):
        """
        Execute a function on data items in parallel.
        
        Args:
            func: Function to execute
            data_list: List of data items to process
            chunk_size: Size of data chunks (calculated automatically if None)
            max_workers: Maximum number of worker threads/processes
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            List of results
        """
        if not data_list:
            return []
        
        if max_workers is None:
            max_workers = min(self.optimal_threads, psutil.cpu_count(logical=False) * 2)
            
            # Adjust based on memory availability
            mem_headroom = self.resource_monitor.get_memory_headroom()
            if mem_headroom < 0.2:
                max_workers = max(1, int(max_workers * 0.5))
        
        # Calculate chunk size if not provided
        if chunk_size is None:
            chunk_size = self.get_optimal_chunk_size(len(data_list))
        
        # Create chunks of data
        chunks = [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]
        
        self.logger.info(f"Parallel execution with {len(chunks)} chunks using {max_workers} workers")
        
        # Process chunks in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk, func, chunk, kwargs): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    self.logger.debug(f"Completed chunk {chunk_idx+1}/{len(chunks)}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                    raise
        
        return results
    
    def _process_chunk(self, func, chunk, kwargs):
        """Process a single chunk of data."""
        return [func(item, **kwargs) for item in chunk]
    
    def adaptive_parallel_execute(self, func, data_list, resource_intensive=False, **kwargs):
        """
        Adaptively determine the best parallelization strategy based on 
        data size and system resources.
        
        Args:
            func: Function to execute
            data_list: List of data items to process
            resource_intensive: Whether the operation is resource intensive
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            List of results
        """
        data_size = len(data_list)
        
        # Check system resources
        cpu_usage = self.resource_monitor.get_cpu_usage()
        mem_headroom = self.resource_monitor.get_memory_headroom()
        
        # Determine execution strategy
        if data_size < 10 or (resource_intensive and data_size < 100):
            # For small data sets or resource-intensive operations with moderate data,
            # process sequentially
            self.logger.info(f"Using sequential execution for {data_size} items")
            return [func(item, **kwargs) for item in data_list]
            
        elif cpu_usage > 80 or mem_headroom < 0.15:
            # System is under heavy load, use conservative parallelism
            max_workers = max(1, min(2, self.optimal_threads // 2))
            chunk_size = max(10, data_size // max_workers)
            self.logger.info(f"Using conservative parallel execution with {max_workers} workers")
            return self.parallel_execute(func, data_list, chunk_size, max_workers, **kwargs)
            
        else:
            # System has resources available, use optimal parallelism
            max_workers = self.optimal_threads
            if resource_intensive:
                max_workers = max(1, max_workers // 2)
            self.logger.info(f"Using optimal parallel execution with {max_workers} workers")
            return self.parallel_execute(func, data_list, None, max_workers, **kwargs)
    
    def get_performance_report(self):
        """
        Generate a performance report of profiled functions.
        
        Returns:
            Dictionary with performance metrics
        """
        report = {
            'function_stats': self.performance_stats,
            'system_resources': {
                'cpu_usage': self.resource_monitor.get_cpu_usage(),
                'memory_usage': self.resource_monitor.get_memory_usage(),
                'memory_headroom': self.resource_monitor.get_memory_headroom(),
                'disk_usage': self.resource_monitor.get_disk_usage(),
                'optimal_threads': self.optimal_threads
            }
        }
        return report