"""
Parallel Processor Module for Meshbuilder
Handles parallel processing tasks
"""
import os
import sys
import time
import logging
import threading
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

logger = logging.getLogger("MeshBuilder.ParallelProcessor")

class ParallelProcessor:
    """Handles parallel processing operations"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize with maximum number of workers
        
        Args:
            max_workers: Maximum number of worker processes/threads
        """
        # If max_workers is None, use CPU count - 1 (leave one core free)
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            
        self.max_workers = max_workers
        
        logger.info(f"Parallel processor initialized with {self.max_workers} workers")
    
    def process_in_parallel(self, 
                           items: List[Any], 
                           process_func: Callable[[Any], Any],
                           use_threads: bool = False,
                           chunk_size: Optional[int] = None) -> List[Any]:
        """
        Process items in parallel
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            use_threads: Whether to use threads instead of processes
            chunk_size: Optional chunk size for processing
            
        Returns:
            List of processed results
        """
        if not items:
            return []
            
        # Use appropriate executor
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        # Process in chunks if specified
        if chunk_size is not None and chunk_size > 0:
            return self._process_in_chunks(items, process_func, executor_class, chunk_size)
        
        # Process all items at once
        results = []
        
        logger.info(f"Processing {len(items)} items in parallel with {self.max_workers} workers")
        start_time = time.time()
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(process_func, item): i for i, item in enumerate(items)}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                item_index = future_to_item[future]
                try:
                    result = future.result()
                    results.append((item_index, result))
                except Exception as e:
                    logger.error(f"Error processing item {item_index}: {str(e)}")
                    results.append((item_index, None))
        
        # Sort results by original index
        sorted_results = [r[1] for r in sorted(results, key=lambda x: x[0])]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        
        return sorted_results
    
    def _process_in_chunks(self, 
                          items: List[Any], 
                          process_func: Callable[[Any], Any],
                          executor_class,
                          chunk_size: int) -> List[Any]:
        """
        Process items in chunks
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            executor_class: Executor class to use
            chunk_size: Chunk size for processing
            
        Returns:
            List of processed results
        """
        all_results = []
        
        # Create chunks
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunks.append(chunk)
            
        logger.info(f"Processing {len(items)} items in {len(chunks)} chunks of size {chunk_size}")
        
        # Process each chunk
        for chunk_index, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_index + 1}/{len(chunks)}")
            
            chunk_start_time = time.time()
            chunk_results = []
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit chunk tasks
                future_to_item = {executor.submit(process_func, item): i for i, item in enumerate(chunk)}
                
                # Collect chunk results
                for future in concurrent.futures.as_completed(future_to_item):
                    item_index = future_to_item[future]
                    try:
                        result = future.result()
                        chunk_results.append((item_index, result))
                    except Exception as e:
                        logger.error(f"Error processing item {item_index} in chunk {chunk_index + 1}: {str(e)}")
                        chunk_results.append((item_index, None))
            
            # Sort chunk results by original index
            sorted_chunk_results = [r[1] for r in sorted(chunk_results, key=lambda x: x[0])]
            all_results.extend(sorted_chunk_results)
            
            chunk_elapsed_time = time.time() - chunk_start_time
            logger.info(f"Processed chunk {chunk_index + 1}/{len(chunks)} in {chunk_elapsed_time:.2f} seconds")
        
        return all_results
    
    def process_batch_tasks(self, 
                           tasks: List[Dict[str, Any]], 
                           process_func: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        """
        Process a batch of independent tasks
        
        Args:
            tasks: List of task dictionaries
            process_func: Function to process each task
            
        Returns:
            Dictionary of task IDs to results
        """
        if not tasks:
            return {}
            
        results = {}
        
        logger.info(f"Processing batch of {len(tasks)} tasks")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_func, task): task.get('id', i) 
                             for i, task in enumerate(tasks)}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results[task_id] = {'status': 'completed', 'result': result}
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {str(e)}")
                    results[task_id] = {'status': 'failed', 'error': str(e)}
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def process_image_batch(self, 
                          images: List[str], 
                          process_func: Callable[[str], Any]) -> Dict[str, Any]:
        """
        Process a batch of images in parallel
        
        Args:
            images: List of image paths
            process_func: Function to process each image
            
        Returns:
            Dictionary of image paths to results
        """
        if not images:
            return {}
            
        results = {}
        
        logger.info(f"Processing batch of {len(images)} images")
        start_time = time.time()
        
        # Images often benefit from thread-based parallelism due to I/O
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {executor.submit(process_func, image): image for image in images}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    results[image_path] = {'status': 'completed', 'result': result}
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {str(e)}")
                    results[image_path] = {'status': 'failed', 'error': str(e)}
        
        elapsed_time = time.time() - start_time
        logger.info(f"Image batch processing completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def process_point_cloud(self, 
                          points: List[Tuple[float, float, float]], 
                          process_func: Callable[[List[Tuple[float, float, float]]], Any],
                          num_chunks: int = 4) -> Any:
        """
        Process a large point cloud by splitting it into chunks
        
        Args:
            points: List of 3D points (x, y, z)
            process_func: Function to process point chunks
            num_chunks: Number of chunks to split into
            
        Returns:
            Combined processing result
        """
        if not points:
            return None
            
        logger.info(f"Processing point cloud with {len(points)} points in {num_chunks} chunks")
        start_time = time.time()
        
        # Split points into chunks
        chunk_size = len(points) // num_chunks
        if chunk_size == 0:
            chunk_size = 1
            num_chunks = len(points)
            
        chunks = []
        for i in range(0, len(points), chunk_size):
            end = min(i + chunk_size, len(points))
            chunks.append(points[i:end])
            
        logger.info(f"Created {len(chunks)} point chunks with ~{chunk_size} points each")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(process_func, chunk): i for i, chunk in enumerate(chunks)}
            
            # Collect results as they complete
            chunk_results = []
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results.append((chunk_index, result))
                    logger.debug(f"Processed point chunk {chunk_index + 1}/{len(chunks)}")
                except Exception as e:
                    logger.error(f"Error processing point chunk {chunk_index + 1}: {str(e)}")
                    chunk_results.append((chunk_index, None))
        
        # Sort results by chunk index
        sorted_results = [r[1] for r in sorted(chunk_results, key=lambda x: x[0])]
        
        # Combine results (implementation depends on specific result type)
        # This is a stub - actual implementation should be provided
        combined_result = self._combine_point_cloud_results(sorted_results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Point cloud processing completed in {elapsed_time:.2f} seconds")
        
        return combined_result
    
    def _combine_point_cloud_results(self, results: List[Any]) -> Any:
        """
        Combine point cloud processing results
        
        Args:
            results: List of results for each chunk
            
        Returns:
            Combined result
        """
        # This is a stub - actual implementation should be provided
        # Would typically combine point clouds, meshes, etc.
        logger.warning("Point cloud result combination is a placeholder - implement based on specific result type")
        
        # Filter out None results (failed chunks)
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return None
            
        # Simple case: if results are lists, concatenate them
        if all(isinstance(r, list) for r in valid_results):
            combined = []
            for r in valid_results:
                combined.extend(r)
            return combined
            
        # If results are dictionaries, merge them
        if all(isinstance(r, dict) for r in valid_results):
            combined = {}
            for r in valid_results:
                combined.update(r)
            return combined
            
        # Default: return the first result (not ideal)
        return valid_results[0]
    
    @staticmethod
    def chunks(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Split items into chunks of specified size
        
        Args:
            items: List of items
            chunk_size: Size of each chunk
            
        Returns:
            List of chunks
        """
        result = []
        for i in range(0, len(items), chunk_size):
            end = min(i + chunk_size, len(items))
            result.append(items[i:end])
        return result