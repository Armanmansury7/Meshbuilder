import os
import sys
import io
import shutil
import mmap
import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.advanced_logger import AdvancedLogger

class FileIOOptimizer:
    """
    Optimizes file I/O operations for large files and batch processing.
    Provides memory-mapped file access, chunked reading/writing, and background I/O.
    """
    def __init__(self, temp_dir=None, max_workers=4):
        self.logger = AdvancedLogger().get_logger(self.__class__.__name__)
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_workers = max_workers
        self.io_executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_mmaps = {}
        self._background_tasks = {}
        self._bg_task_counter = 0
        self._lock = threading.Lock()
    
    def __del__(self):
        """Cleanup resources on deletion."""
        self.close_all_mmaps()
        self.io_executor.shutdown(wait=False)
    
    def read_file_chunked(self, file_path, chunk_size=1024*1024):
        """
        Read a file in chunks to minimize memory usage.
        
        Args:
            file_path: Path to the file
            chunk_size: Size of each chunk in bytes
            
        Yields:
            Chunks of file data
        """
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            self.logger.error(f"Error reading file in chunks: {str(e)}")
            raise
    
    def write_file_chunked(self, file_path, data_generator, chunk_size=1024*1024):
        """
        Write data to a file in chunks.
        
        Args:
            file_path: Path to the file
            data_generator: Generator yielding data chunks
            chunk_size: Size of each chunk in bytes
            
        Returns:
            Total bytes written
        """
        try:
            bytes_written = 0
            with open(file_path, 'wb') as f:
                for chunk in data_generator:
                    f.write(chunk)
                    bytes_written += len(chunk)
            return bytes_written
        except Exception as e:
            self.logger.error(f"Error writing file in chunks: {str(e)}")
            raise
    
    def get_memory_mapped_file(self, file_path, access='r'):
        """
        Get a memory-mapped file object for efficient random access.
        
        Args:
            file_path: Path to the file
            access: Access mode ('r' for read-only, 'w+' for read-write)
            
        Returns:
            Memory-mapped file object
        """
        try:
            if file_path in self._active_mmaps:
                return self._active_mmaps[file_path]
            
            access_mode = mmap.ACCESS_READ if access == 'r' else mmap.ACCESS_WRITE
            
            with open(file_path, 'rb' if access == 'r' else 'r+b') as f:
                # Get file size
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(0)
                
                if size == 0:
                    self.logger.warning(f"Cannot memory-map empty file: {file_path}")
                    return None
                
                # Create memory mapping
                mm = mmap.mmap(f.fileno(), size, access=access_mode)
                self._active_mmaps[file_path] = mm
                return mm
                
        except Exception as e:
            self.logger.error(f"Error creating memory-mapped file: {str(e)}")
            return None
    
    def close_mmap(self, file_path):
        """Close a memory-mapped file."""
        with self._lock:
            if file_path in self._active_mmaps:
                try:
                    self._active_mmaps[file_path].close()
                    del self._active_mmaps[file_path]
                except Exception as e:
                    self.logger.error(f"Error closing memory-mapped file: {str(e)}")
    
    def close_all_mmaps(self):
        """Close all memory-mapped files."""
        with self._lock:
            for path, mm in list(self._active_mmaps.items()):
                try:
                    mm.close()
                except Exception:
                    pass
            self._active_mmaps.clear()
    
    def read_async(self, file_path, callback=None):
        """
        Read a file asynchronously in the background.
        
        Args:
            file_path: Path to the file
            callback: Function to call with the file data when read is complete
            
        Returns:
            Task ID for the background operation
        """
        with self._lock:
            task_id = self._bg_task_counter
            self._bg_task_counter += 1
        
        def _read_task():
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                if callback:
                    callback(data)
                
                with self._lock:
                    if task_id in self._background_tasks:
                        self._background_tasks[task_id]['status'] = 'completed'
                        self._background_tasks[task_id]['result'] = data
                
                return data
            except Exception as e:
                self.logger.error(f"Error in async read: {str(e)}")
                
                with self._lock:
                    if task_id in self._background_tasks:
                        self._background_tasks[task_id]['status'] = 'error'
                        self._background_tasks[task_id]['error'] = str(e)
                
                if callback:
                    callback(None, error=str(e))
                
                raise
        
        future = self.io_executor.submit(_read_task)
        
        with self._lock:
            self._background_tasks[task_id] = {
                'future': future,
                'file_path': file_path,
                'operation': 'read',
                'status': 'running'
            }
        
        return task_id
    
    def write_async(self, file_path, data, callback=None, overwrite=True):
        """
        Write data to a file asynchronously in the background.
        
        Args:
            file_path: Path to the file
            data: Data to write
            callback: Function to call when write is complete
            overwrite: Whether to overwrite existing file
            
        Returns:
            Task ID for the background operation
        """
        if not overwrite and os.path.exists(file_path):
            error = f"File already exists: {file_path}"
            self.logger.error(error)
            if callback:
                callback(False, error=error)
            return None
        
        with self._lock:
            task_id = self._bg_task_counter
            self._bg_task_counter += 1
        
        def _write_task():
            try:
                # Create parent directories if needed
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                
                # Write to a temporary file first
                temp_file = f"{file_path}.tmp"
                with open(temp_file, 'wb') as f:
                    f.write(data)
                
                # Move temporary file to target location (atomic operation)
                shutil.move(temp_file, file_path)
                
                if callback:
                    callback(True)
                
                with self._lock:
                    if task_id in self._background_tasks:
                        self._background_tasks[task_id]['status'] = 'completed'
                
                return True
            except Exception as e:
                self.logger.error(f"Error in async write: {str(e)}")
                
                # Clean up temporary file if needed
                try:
                    if os.path.exists(f"{file_path}.tmp"):
                        os.remove(f"{file_path}.tmp")
                except Exception:
                    pass
                
                with self._lock:
                    if task_id in self._background_tasks:
                        self._background_tasks[task_id]['status'] = 'error'
                        self._background_tasks[task_id]['error'] = str(e)
                
                if callback:
                    callback(False, error=str(e))
                
                raise
        
        future = self.io_executor.submit(_write_task)
        
        with self._lock:
            self._background_tasks[task_id] = {
                'future': future,
                'file_path': file_path,
                'operation': 'write',
                'status': 'running'
            }
        
        return task_id
    
    def copy_file_async(self, source_path, target_path, callback=None, overwrite=True):
        """
        Copy a file asynchronously in the background.
        
        Args:
            source_path: Path to the source file
            target_path: Path to the target file
            callback: Function to call when copy is complete
            overwrite: Whether to overwrite existing target file
            
        Returns:
            Task ID for the background operation
        """
        if not os.path.exists(source_path):
            error = f"Source file does not exist: {source_path}"
            self.logger.error(error)
            if callback:
                callback(False, error=error)
            return None
        
        if not overwrite and os.path.exists(target_path):
            error = f"Target file already exists: {target_path}"
            self.logger.error(error)
            if callback:
                callback(False, error=error)
            return None
        
        with self._lock:
            task_id = self._bg_task_counter
            self._bg_task_counter += 1
        
        def _copy_task():
            try:
                # Create parent directories if needed
                os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
                
                # For large files, copy in chunks
                file_size = os.path.getsize(source_path)
                
                if file_size > 100 * 1024 * 1024:  # Over 100 MB
                    # Copy in chunks
                    with open(source_path, 'rb') as src, open(f"{target_path}.tmp", 'wb') as dst:
                        copied = 0
                        chunk_size = 4 * 1024 * 1024  # 4 MB chunks
                        
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            
                            dst.write(chunk)
                            copied += len(chunk)
                            
                            # Update status (approximate progress)
                            with self._lock:
                                if task_id in self._background_tasks:
                                    self._background_tasks[task_id]['progress'] = copied / file_size
                    
                    # Move temporary file to target location
                    shutil.move(f"{target_path}.tmp", target_path)
                else:
                    # Use standard copy for small files
                    shutil.copy2(source_path, target_path)
                
                if callback:
                    callback(True)
                
                with self._lock:
                    if task_id in self._background_tasks:
                        self._background_tasks[task_id]['status'] = 'completed'
                
                return True
            except Exception as e:
                self.logger.error(f"Error in async copy: {str(e)}")
                
                # Clean up temporary file if needed
                try:
                    if os.path.exists(f"{target_path}.tmp"):
                        os.remove(f"{target_path}.tmp")
                except Exception:
                    pass
                
                with self._lock:
                    if task_id in self._background_tasks:
                        self._background_tasks[task_id]['status'] = 'error'
                        self._background_tasks[task_id]['error'] = str(e)
                
                if callback:
                    callback(False, error=str(e))
                
                raise
        
        future = self.io_executor.submit(_copy_task)
        
        with self._lock:
            self._background_tasks[task_id] = {
                'future': future,
                'source_path': source_path,
                'target_path': target_path,
                'operation': 'copy',
                'status': 'running',
                'progress': 0
            }
        
        return task_id
    
    def get_task_status(self, task_id):
        """
        Get the status of a background task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task status information
        """
        with self._lock:
            if task_id not in self._background_tasks:
                return {'status': 'not_found'}
            
            task_info = self._background_tasks[task_id].copy()
            
            # Remove future to make it serializable
            if 'future' in task_info:
                del task_info['future']
            
            return task_info
    
    def wait_for_task(self, task_id, timeout=None):
        """
        Wait for a background task to complete.
        
        Args:
            task_id: ID of the task
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Task result or None if task failed or timed out
        """
        with self._lock:
            if task_id not in self._background_tasks:
                return None
            
            future = self._background_tasks[task_id]['future']
        
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            self.logger.error(f"Error waiting for task {task_id}: {str(e)}")
            return None
    
    def cancel_task(self, task_id):
        """
        Cancel a background task if it's still running.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self._lock:
            if task_id not in self._background_tasks:
                return False
            
            task_info = self._background_tasks[task_id]
            
            if task_info['status'] != 'running':
                return False
            
            future = task_info['future']
            
            # Attempt to cancel the future
            cancelled = future.cancel()
            
            if cancelled:
                task_info['status'] = 'cancelled'
            
            return cancelled
    
    def cleanup_temp_files(self, prefix='meshbuildr_'):
        """
        Clean up temporary files created by the application.
        
        Args:
            prefix: Prefix of temporary files to clean up
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        try:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    if file.startswith(prefix):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except Exception as e:
                            self.logger.warning(f"Could not remove temp file {file_path}: {str(e)}")
            
            return cleaned_count
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {str(e)}")
            return cleaned_count
    
    def create_temp_file(self, content=None, suffix=None, prefix='meshbuildr_', dir=None):
        """
        Create a temporary file.
        
        Args:
            content: Content to write to the file (optional)
            suffix: File suffix (extension)
            prefix: File prefix
            dir: Directory to create the file in (defaults to temp_dir)
            
        Returns:
            Path to the created temporary file
        """
        try:
            temp_dir = dir or self.temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            
            if content is not None:
                # Create named temporary file with content
                fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=temp_dir)
                
                try:
                    # Write content
                    with os.fdopen(fd, 'wb') as f:
                        if isinstance(content, str):
                            f.write(content.encode('utf-8'))
                        else:
                            f.write(content)
                except Exception as e:
                    self.logger.error(f"Error writing to temp file: {str(e)}")
                    os.close(fd)
                    os.unlink(temp_path)
                    raise
            else:
                # Just create a named temporary file
                fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=temp_dir)
                os.close(fd)
            
            return temp_path
        except Exception as e:
            self.logger.error(f"Error creating temp file: {str(e)}")
            raise
    
    def create_temp_dir(self, prefix='meshbuildr_', dir=None):
        """
        Create a temporary directory.
        
        Args:
            prefix: Directory prefix
            dir: Parent directory (defaults to temp_dir)
            
        Returns:
            Path to the created temporary directory
        """
        try:
            parent_dir = dir or self.temp_dir
            os.makedirs(parent_dir, exist_ok=True)
            
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)
            return temp_dir
        except Exception as e:
            self.logger.error(f"Error creating temp directory: {str(e)}")
            raise