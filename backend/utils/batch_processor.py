"""
Batch Processor Module for MeshBuilder
Handles batch processing of multiple 3D reconstruction jobs
"""
import os
import time
import threading
import logging
import queue
import json
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

logger = logging.getLogger("MeshBuilder.BatchProcessor")

class BatchJob:
    """Represents a single job in the batch processor"""
    
    def __init__(self, job_id: str, job_config: Dict[str, Any]):
        """
        Initialize a batch job
        
        Args:
            job_id: Unique job identifier
            job_config: Job configuration dictionary
        """
        self.job_id = job_id
        self.config = job_config
        self.status = "queued"
        self.queue_time = time.time()
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "queue_time": self.queue_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": (self.end_time - self.start_time) if self.end_time and self.start_time else None,
            "queue_wait": (self.start_time - self.queue_time) if self.start_time else (time.time() - self.queue_time),
            "result": self.result,
            "error": self.error,
            "config": {k: v for k, v in self.config.items() if k != "config" and k != "settings"}
        }
    
    def __str__(self) -> str:
        """String representation of job"""
        status_str = f"Job {self.job_id} ({self.status})"
        if self.start_time and self.end_time:
            status_str += f" - Completed in {self.end_time - self.start_time:.1f}s"
        elif self.start_time:
            status_str += f" - Running for {time.time() - self.start_time:.1f}s"
        else:
            status_str += f" - Queued for {time.time() - self.queue_time:.1f}s"
        return status_str

class BatchProcessor:
    """Handles batch processing of multiple 3D reconstruction jobs"""
    
    def __init__(self, config=None, max_parallel=1, results_dir=None):
        """
        Initialize batch processor
        
        Args:
            config: Configuration dictionary
            max_parallel: Maximum number of parallel jobs (default: 1)
            results_dir: Directory to store results (default: temp)
        """
        self.config = config or {}
        self.max_parallel = max_parallel
        self.results_dir = results_dir or os.path.join(os.path.expanduser("~"), "MeshBuilder", "BatchResults")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.job_queue = queue.Queue()
        self.active_jobs = {}  # job_id -> BatchJob
        self.completed_jobs = []  # List of completed BatchJob objects
        self.failed_jobs = []  # List of failed BatchJob objects
        
        self.is_processing = False
        self._stop_event = threading.Event()
        self._worker_threads = []
        self._job_history_path = os.path.join(self.results_dir, "job_history.json")
        
        # Load job history if exists
        self._load_job_history()
    
    def add_job(self, job_id: str, job_config: Dict[str, Any]) -> str:
        """
        Add a job to the processing queue
        
        Args:
            job_id: Unique job identifier (or None to auto-generate)
            job_config: Job configuration dictionary
            
        Returns:
            Job ID
        """
        # Generate job ID if not provided
        if not job_id:
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.completed_jobs) + len(self.failed_jobs) + self.job_queue.qsize() + len(self.active_jobs)}"
        
        # Create job object
        job = BatchJob(job_id, job_config)
        
        # Add to queue
        self.job_queue.put(job)
        logger.info(f"Added job {job_id} to queue. Queue size: {self.job_queue.qsize()}")
        
        return job_id
    
    def start_processing(self, processor_function: Callable, callback: Callable = None):
        """
        Start batch processing
        
        Args:
            processor_function: Function to process each job
            callback: Optional callback function to call after each job
        """
        if self.is_processing:
            logger.warning("Batch processing already in progress")
            return
            
        self.is_processing = True
        self._stop_event.clear()
        
        # Determine number of worker threads
        num_workers = min(self.max_parallel, max(1, os.cpu_count() - 1)) if self.max_parallel > 1 else 1
        
        # Start worker threads
        self._worker_threads = []
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._process_queue,
                args=(processor_function, callback, i),
                daemon=True,
                name=f"BatchWorker-{i}"
            )
            self._worker_threads.append(worker)
            worker.start()
        
        logger.info(f"Batch processing started with {num_workers} worker threads")
    
    def stop_processing(self):
        """Stop batch processing"""
        if not self.is_processing:
            return
            
        logger.info("Stopping batch processing")
        self._stop_event.set()
        
        # Wait for all workers to finish
        for i, worker in enumerate(self._worker_threads):
            if worker.is_alive():
                logger.debug(f"Waiting for worker {i} to finish...")
                worker.join(timeout=5.0)
        
        self._worker_threads = []
        self.is_processing = False
        logger.info("Batch processing stopped")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current status of the job queue
        
        Returns:
            Dictionary with queue status
        """
        return {
            "queued": self.job_queue.qsize(),
            "active": len(self.active_jobs),
            "completed": len(self.completed_jobs),
            "failed": len(self.failed_jobs),
            "is_processing": self.is_processing,
            "worker_count": len(self._worker_threads)
        }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None if not found
        """
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job.to_dict()
        
        # Check failed jobs
        for job in self.failed_jobs:
            if job.job_id == job_id:
                return job.to_dict()
        
        # Check queue
        queue_list = list(self.job_queue.queue)
        for job in queue_list:
            if job.job_id == job_id:
                return job.to_dict()
        
        return None
    
    def get_all_jobs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all jobs grouped by status
        
        Returns:
            Dictionary with jobs grouped by status
        """
        # Get queued jobs
        queued_jobs = []
        queue_list = list(self.job_queue.queue)
        for job in queue_list:
            queued_jobs.append(job.to_dict())
        
        # Get active jobs
        active_jobs = [job.to_dict() for job in self.active_jobs.values()]
        
        # Get completed and failed jobs
        completed_jobs = [job.to_dict() for job in self.completed_jobs]
        failed_jobs = [job.to_dict() for job in self.failed_jobs]
        
        return {
            "queued": queued_jobs,
            "active": active_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs
        }
    
    def clear_completed_jobs(self):
        """Clear the list of completed jobs"""
        self.completed_jobs = []
        self._save_job_history()
        logger.info("Cleared completed jobs")
    
    def clear_failed_jobs(self):
        """Clear the list of failed jobs"""
        self.failed_jobs = []
        self._save_job_history()
        logger.info("Cleared failed jobs")
    
    def _process_queue(self, processor_function, callback, worker_id):
        """
        Worker function to process the job queue
        
        Args:
            processor_function: Function to process jobs
            callback: Callback function for job completion
            worker_id: Worker thread identifier
        """
        logger.info(f"Worker {worker_id} started")
        
        while not self._stop_event.is_set():
            try:
                # Get the next job from the queue
                try:
                    job = self.job_queue.get(block=True, timeout=1.0)
                except queue.Empty:
                    continue
                
                # Start the job
                logger.info(f"Worker {worker_id} starting job {job.job_id}")
                job.status = "processing"
                job.start_time = time.time()
                self.active_jobs[job.job_id] = job
                
                # Process the job
                try:
                    # Prepare job config for processor
                    # This allows the processor to get input_files, output_path, etc. directly
                    processor_args = job.config.copy()
                    
                    result = processor_function(processor_args)
                    
                    # Mark as completed
                    job.status = "completed"
                    job.end_time = time.time()
                    job.result = result
                    
                    self.completed_jobs.append(job)
                    logger.info(f"Worker {worker_id} completed job {job.job_id} successfully")
                    
                    if callback:
                        callback(job.job_id, True, result)
                        
                except Exception as e:
                    # Mark as failed
                    logger.error(f"Worker {worker_id} - Job {job.job_id} failed: {str(e)}")
                    job.status = "failed"
                    job.end_time = time.time()
                    job.error = str(e)
                    
                    self.failed_jobs.append(job)
                    
                    if callback:
                        callback(job.job_id, False, str(e))
                
                # Remove from active jobs
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                
                # Mark queue task as done
                self.job_queue.task_done()
                
                # Save job history
                self._save_job_history()
                
            except Exception as e:
                logger.error(f"Error in batch processing worker {worker_id}: {str(e)}")
                time.sleep(1.0)
                
        logger.info(f"Worker {worker_id} stopped")
    
    def _save_job_history(self):
        """Save job history to file"""
        try:
            history = {
                "timestamp": datetime.now().isoformat(),
                "jobs": {
                    "completed": [job.to_dict() for job in self.completed_jobs],
                    "failed": [job.to_dict() for job in self.failed_jobs]
                }
            }
            
            with open(self._job_history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving job history: {str(e)}")
    
    def _load_job_history(self):
        """Load job history from file"""
        if not os.path.exists(self._job_history_path):
            return
            
        try:
            with open(self._job_history_path, 'r') as f:
                history = json.load(f)
            
            # Recreate job objects from history
            if "jobs" in history:
                if "completed" in history["jobs"]:
                    for job_dict in history["jobs"]["completed"]:
                        job = BatchJob(job_dict["job_id"], job_dict.get("config", {}))
                        job.status = "completed"
                        job.queue_time = job_dict.get("queue_time", 0)
                        job.start_time = job_dict.get("start_time", 0)
                        job.end_time = job_dict.get("end_time", 0)
                        job.result = job_dict.get("result")
                        self.completed_jobs.append(job)
                
                if "failed" in history["jobs"]:
                    for job_dict in history["jobs"]["failed"]:
                        job = BatchJob(job_dict["job_id"], job_dict.get("config", {}))
                        job.status = "failed"
                        job.queue_time = job_dict.get("queue_time", 0)
                        job.start_time = job_dict.get("start_time", 0)
                        job.end_time = job_dict.get("end_time", 0)
                        job.error = job_dict.get("error")
                        self.failed_jobs.append(job)
            
            logger.info(f"Loaded job history: {len(self.completed_jobs)} completed, {len(self.failed_jobs)} failed")
            
        except Exception as e:
            logger.error(f"Error loading job history: {str(e)}")