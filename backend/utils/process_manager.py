import os
import sys
import subprocess
import threading
import queue
import time
import signal
import psutil
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.advanced_logger import AdvancedLogger
from utils.resource_monitor import ResourceMonitor

class ProcessState(Enum):
    """Enum for process states."""
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    TERMINATED = "terminated"

class ProcessInfo:
    """Class to store information about a managed process."""
    def __init__(self, process_id, name, command, priority):
        self.id = process_id
        self.name = name
        self.command = command
        self.priority = priority
        self.process = None
        self.state = ProcessState.PENDING
        self.pid = None
        self.start_time = None
        self.end_time = None
        self.exit_code = None
        self.stdout = []
        self.stderr = []
        self.resource_usage = []
        self.error_message = None

class ProcessManager:
    """
    Manages external processes and subprocesses for MeshBuildr.
    Handles process creation, monitoring, resource allocation, and graceful termination.
    """
    def __init__(self, max_processes=4, monitoring_interval=1.0):
        self.logger = AdvancedLogger().get_logger(self.__class__.__name__)
        self.resource_monitor = ResourceMonitor()
        self.max_processes = max_processes
        self.monitoring_interval = monitoring_interval
        
        self.processes = {}
        self.process_queue = queue.PriorityQueue()
        self.process_counter = 0
        
        self._next_process_id = 1
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Start monitoring thread
        self._start_monitoring()
    
    def __del__(self):
        """Clean up on deletion."""
        self.shutdown()
    
    def _start_monitoring(self):
        """Start the monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_processes(self):
        """Monitor running processes and resource usage."""
        while not self._stop_monitoring.is_set():
            try:
                # Check if we can start any queued processes
                self._check_process_queue()
                
                # Update status of running processes
                self._update_process_status()
                
                # Wait for next monitoring interval
                self._stop_monitoring.wait(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {str(e)}")
    
    def _check_process_queue(self):
        """Check if any queued processes can be started."""
        with self._lock:
            running_count = sum(1 for p in self.processes.values() 
                             if p.state == ProcessState.RUNNING)
            
            # Check if we can start more processes
            while running_count < self.max_processes and not self.process_queue.empty():
                # Get the next process from the queue
                _, process_id = self.process_queue.get()
                
                if process_id in self.processes:
                    # Start the process
                    self._start_process(process_id)
                    running_count += 1
                else:
                    self.logger.warning(f"Process {process_id} not found in queue")
    
    def _update_process_status(self):
        """Update the status of running processes."""
        with self._lock:
            for process_id, process_info in list(self.processes.items()):
                if process_info.state != ProcessState.RUNNING:
                    continue
                
                # Check if process is still running
                if process_info.process.poll() is not None:
                    # Process has finished
                    process_info.end_time = time.time()
                    process_info.exit_code = process_info.process.returncode
                    
                    # Get remaining output
                    stdout, stderr = process_info.process.communicate()
                    if stdout:
                        process_info.stdout.append(stdout.decode('utf-8', errors='replace'))
                    if stderr:
                        process_info.stderr.append(stderr.decode('utf-8', errors='replace'))
                    
                    # Set process state
                    if process_info.exit_code == 0:
                        process_info.state = ProcessState.FINISHED
                        self.logger.info(f"Process {process_info.name} (ID: {process_id}) finished successfully")
                    else:
                        process_info.state = ProcessState.FAILED
                        error_msg = f"Process failed with exit code {process_info.exit_code}"
                        process_info.error_message = error_msg
                        self.logger.error(f"Process {process_info.name} (ID: {process_id}): {error_msg}")
                else:
                    # Process is still running - collect output
                    self._collect_process_output(process_info)
                    
                    # Collect resource usage
                    try:
                        if process_info.pid:
                            process = psutil.Process(process_info.pid)
                            cpu_percent = process.cpu_percent(interval=None)
                            memory_info = process.memory_info()
                            
                            usage = {
                                'timestamp': time.time(),
                                'cpu_percent': cpu_percent,
                                'memory_rss': memory_info.rss,
                                'memory_vms': memory_info.vms
                            }
                            process_info.resource_usage.append(usage)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                        self.logger.debug(f"Could not collect resource usage for process {process_id}: {str(e)}")
    
    def _collect_process_output(self, process_info):
        """Collect non-blocking output from a running process."""
        if not process_info.process:
            return
        
        # Read from stdout without blocking
        try:
            while True:
                line = process_info.process.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace')
                process_info.stdout.append(decoded_line)
                self.logger.debug(f"[{process_info.name}] {decoded_line.strip()}")
        except (IOError, ValueError, AttributeError):
            pass
        
        # Read from stderr without blocking
        try:
            while True:
                line = process_info.process.stderr.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace')
                process_info.stderr.append(decoded_line)
                self.logger.debug(f"[{process_info.name}] ERROR: {decoded_line.strip()}")
        except (IOError, ValueError, AttributeError):
            pass
    
    def _next_id(self):
        """Get the next process ID."""
        with self._lock:
            process_id = self._next_process_id
            self._next_process_id += 1
            return process_id
    
    def create_process(self, name, command, priority=1, env=None, cwd=None, shell=False):
        """
        Create a new process (queued for execution).
        
        Args:
            name: Process name/description
            command: Command to execute (list or string)
            priority: Priority (lower numbers = higher priority)
            env: Environment variables dictionary
            cwd: Working directory
            shell: Whether to run in a shell
            
        Returns:
            Process ID
        """
        with self._lock:
            process_id = self._next_id()
            
            # Create process info
            process_info = ProcessInfo(process_id, name, command, priority)
            self.processes[process_id] = process_info
            
            # Store additional execution parameters
            process_info.env = env
            process_info.cwd = cwd
            process_info.shell = shell
            
            # Add to queue
            self.process_queue.put((priority, process_id))
            
            self.logger.info(f"Created process {name} (ID: {process_id})")
            
            return process_id
    
    def _start_process(self, process_id):
        """
        Start a process that has been queued.
        
        Args:
            process_id: ID of the process to start
        """
        with self._lock:
            if process_id not in self.processes:
                self.logger.error(f"Process {process_id} not found")
                return False
            
            process_info = self.processes[process_id]
            
            if process_info.state != ProcessState.PENDING:
                self.logger.warning(f"Process {process_id} is not in pending state")
                return False
            
            try:
                # Start the subprocess
                self.logger.info(f"Starting process {process_info.name} (ID: {process_id})")
                
                # Configure process startup
                startup_info = None
                if sys.platform == 'win32':
                    startup_info = subprocess.STARTUPINFO()
                    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startup_info.wShowWindow = subprocess.SW_HIDE
                
                process = subprocess.Popen(
                    process_info.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    env=process_info.env,
                    cwd=process_info.cwd,
                    shell=process_info.shell,
                    startupinfo=startup_info,
                    bufsize=1,
                    universal_newlines=False
                )
                
                process_info.process = process
                process_info.pid = process.pid
                process_info.state = ProcessState.RUNNING
                process_info.start_time = time.time()
                
                self.logger.info(f"Process {process_info.name} (ID: {process_id}) started with PID {process_info.pid}")
                
                return True
            except Exception as e:
                process_info.state = ProcessState.FAILED
                process_info.error_message = str(e)
                self.logger.error(f"Error starting process {process_info.name} (ID: {process_id}): {str(e)}")
                return False
    
    def run_process_sync(self, name, command, timeout=None, env=None, cwd=None, shell=False):
        """
        Run a process synchronously (blocking).
        
        Args:
            name: Process name/description
            command: Command to execute (list or string)
            timeout: Maximum time to wait (in seconds)
            env: Environment variables dictionary
            cwd: Working directory
            shell: Whether to run in a shell
            
        Returns:
            Tuple of (success, stdout, stderr, exit_code)
        """
        self.logger.info(f"Running synchronous process: {name}")
        
        try:
            # Configure process startup
            startup_info = None
            if sys.platform == 'win32':
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup_info.wShowWindow = subprocess.SW_HIDE
            
            # Run the process
            completed_process = subprocess.run(
                command,
                capture_output=True,
                env=env,
                cwd=cwd,
                shell=shell,
                timeout=timeout,
                startupinfo=startup_info
            )
            
            # Process results
            stdout = completed_process.stdout.decode('utf-8', errors='replace')
            stderr = completed_process.stderr.decode('utf-8', errors='replace')
            exit_code = completed_process.returncode
            success = exit_code == 0
            
            if success:
                self.logger.info(f"Process {name} completed successfully")
            else:
                self.logger.error(f"Process {name} failed with exit code {exit_code}")
                self.logger.debug(f"Process stderr: {stderr}")
            
            return success, stdout, stderr, exit_code
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Process {name} timed out after {timeout} seconds")
            return False, "", "Process timed out", -1
        except Exception as e:
            self.logger.error(f"Error running process {name}: {str(e)}")
            return False, "", str(e), -1
    
    def terminate_process(self, process_id, force=False, timeout=5):
        """
        Terminate a running process.
        
        Args:
            process_id: ID of the process to terminate
            force: Whether to force termination
            timeout: Timeout for graceful termination
            
        Returns:
            True if process was terminated, False otherwise
        """
        with self._lock:
            if process_id not in self.processes:
                self.logger.error(f"Process {process_id} not found")
                return False
            
            process_info = self.processes[process_id]
            
            if process_info.state != ProcessState.RUNNING:
                self.logger.warning(f"Process {process_id} is not running")
                self.logger.warning(f"Process {process_id} is not running")
                return False
            
            if not process_info.process:
                self.logger.warning(f"Process {process_id} has no subprocess")
                return False
            
            try:
                # Get process by PID to handle termination properly
                if process_info.pid:
                    try:
                        proc = psutil.Process(process_info.pid)
                        
                        if force:
                            # Kill immediately
                            proc.kill()
                        else:
                            # Try graceful termination first
                            proc.terminate()
                            
                            # Wait for process to terminate
                            try:
                                proc.wait(timeout=timeout)
                            except psutil.TimeoutExpired:
                                # Force kill if timeout
                                self.logger.warning(f"Forcing termination of process {process_id} after timeout")
                                proc.kill()
                    except psutil.NoSuchProcess:
                        # Process already gone
                        pass
                
                # Wait for process to terminate
                process_info.process.wait(timeout=1.0)
                
                # Update process info
                process_info.state = ProcessState.TERMINATED
                process_info.end_time = time.time()
                process_info.exit_code = process_info.process.returncode
                
                # Get remaining output
                stdout, stderr = process_info.process.communicate()
                if stdout:
                    process_info.stdout.append(stdout.decode('utf-8', errors='replace'))
                if stderr:
                    process_info.stderr.append(stderr.decode('utf-8', errors='replace'))
                
                self.logger.info(f"Process {process_info.name} (ID: {process_id}) terminated")
                return True
                
            except Exception as e:
                self.logger.error(f"Error terminating process {process_id}: {str(e)}")
                return False
    
    def get_process_info(self, process_id):
        """
        Get information about a process.
        
        Args:
            process_id: ID of the process
            
        Returns:
            Dictionary with process information or None if not found
        """
        with self._lock:
            if process_id not in self.processes:
                return None
            
            process_info = self.processes[process_id]
            
            # Create a copy of the process info for return
            info = {
                'id': process_info.id,
                'name': process_info.name,
                'state': process_info.state.value,
                'pid': process_info.pid,
                'start_time': process_info.start_time,
                'end_time': process_info.end_time,
                'exit_code': process_info.exit_code,
                'stdout': ''.join(process_info.stdout),
                'stderr': ''.join(process_info.stderr),
                'error_message': process_info.error_message
            }
            
            # Add resource usage summary if available
            if process_info.resource_usage:
                resource_summary = {
                    'last_cpu_percent': process_info.resource_usage[-1]['cpu_percent'],
                    'last_memory_mb': process_info.resource_usage[-1]['memory_rss'] / (1024 * 1024),
                    'max_memory_mb': max(usage['memory_rss'] for usage in process_info.resource_usage) / (1024 * 1024),
                    'avg_cpu_percent': sum(usage['cpu_percent'] for usage in process_info.resource_usage) / len(process_info.resource_usage)
                }
                info['resources'] = resource_summary
            
            return info
    
    def wait_for_process(self, process_id, timeout=None):
        """
        Wait for a process to complete.
        
        Args:
            process_id: ID of the process to wait for
            timeout: Maximum time to wait (in seconds)
            
        Returns:
            True if process completed successfully, False otherwise
        """
        if timeout:
            end_time = time.time() + timeout
        
        while True:
            with self._lock:
                if process_id not in self.processes:
                    self.logger.error(f"Process {process_id} not found")
                    return False
                
                process_info = self.processes[process_id]
                
                if process_info.state == ProcessState.FINISHED:
                    return True
                elif process_info.state in (ProcessState.FAILED, ProcessState.TERMINATED):
                    return False
                elif process_info.state == ProcessState.PENDING:
                    # Process still in queue
                    pass
                elif process_info.state == ProcessState.RUNNING:
                    if process_info.process:
                        # Check if process is still running
                        exit_code = process_info.process.poll()
                        if exit_code is not None:
                            # Process has finished
                            process_info.end_time = time.time()
                            process_info.exit_code = exit_code
                            
                            # Get remaining output
                            stdout, stderr = process_info.process.communicate()
                            if stdout:
                                process_info.stdout.append(stdout.decode('utf-8', errors='replace'))
                            if stderr:
                                process_info.stderr.append(stderr.decode('utf-8', errors='replace'))
                            
                            # Set process state
                            if process_info.exit_code == 0:
                                process_info.state = ProcessState.FINISHED
                                self.logger.info(f"Process {process_info.name} (ID: {process_id}) finished successfully")
                                return True
                            else:
                                process_info.state = ProcessState.FAILED
                                error_msg = f"Process failed with exit code {process_info.exit_code}"
                                process_info.error_message = error_msg
                                self.logger.error(f"Process {process_info.name} (ID: {process_id}): {error_msg}")
                                return False
            
            # Check timeout
            if timeout and time.time() > end_time:
                self.logger.warning(f"Timeout waiting for process {process_id}")
                return False
            
            # Wait a bit before checking again
            time.sleep(0.1)
    
    def shutdown(self):
        """
        Shutdown the process manager and terminate all processes.
        """
        self.logger.info("Shutting down process manager")
        
        # Stop monitoring thread
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        # Terminate all running processes
        with self._lock:
            for process_id, process_info in list(self.processes.items()):
                if process_info.state == ProcessState.RUNNING:
                    try:
                        self.terminate_process(process_id, force=True)
                    except Exception as e:
                        self.logger.error(f"Error terminating process {process_id} during shutdown: {str(e)}")
        
        self.logger.info("Process manager shutdown complete")