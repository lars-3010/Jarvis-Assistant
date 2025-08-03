"""
Async Task Queue System for Background Processing.

This module provides a comprehensive task queue system for handling background
tasks with priority support, retry logic, and distributed processing capabilities.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import pickle
from datetime import datetime, timezone, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq

from jarvis.core.events import Event, EventBus, get_event_bus, publish_event
from jarvis.core.service_registry import ServiceRegistry, get_service_registry
from jarvis.core.interfaces import IMetrics
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import TaskError, SerializationError

logger = setup_logging(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Represents the result of a task execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task in the queue."""
    task_id: str
    func_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    scheduled_at: Optional[float] = None
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority first, then earlier scheduled)."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        
        self_time = self.scheduled_at or self.created_at
        other_time = other.scheduled_at or other.created_at
        return self_time < other_time
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if not self.expires_at:
            return False
        return time.time() > self.expires_at
    
    def should_run_now(self) -> bool:
        """Check if task should run now."""
        if self.scheduled_at:
            return time.time() >= self.scheduled_at
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'func_name': self.func_name,
            'args': self.args,
            'kwargs': self.kwargs,
            'priority': self.priority.value,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout,
            'scheduled_at': self.scheduled_at,
            'expires_at': self.expires_at,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'status': self.status.value,
            'attempts': self.attempts,
            'last_error': self.last_error,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            task_id=data['task_id'],
            func_name=data['func_name'],
            args=tuple(data.get('args', [])),
            kwargs=data.get('kwargs', {}),
            priority=TaskPriority(data.get('priority', TaskPriority.NORMAL.value)),
            max_retries=data.get('max_retries', 3),
            retry_delay=data.get('retry_delay', 1.0),
            timeout=data.get('timeout'),
            scheduled_at=data.get('scheduled_at'),
            expires_at=data.get('expires_at'),
            created_at=data.get('created_at', time.time()),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            status=TaskStatus(data.get('status', TaskStatus.PENDING.value)),
            attempts=data.get('attempts', 0),
            last_error=data.get('last_error'),
            metadata=data.get('metadata', {})
        )


class ITaskStore(ABC):
    """Abstract interface for task persistence."""
    
    @abstractmethod
    async def store_task(self, task: Task) -> bool:
        """Store a task."""
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        pass
    
    @abstractmethod
    async def update_task(self, task: Task) -> bool:
        """Update a task."""
        pass
    
    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        pass
    
    @abstractmethod
    async def get_pending_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Get pending tasks."""
        pass
    
    @abstractmethod
    async def get_failed_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Get failed tasks."""
        pass


class MemoryTaskStore(ITaskStore):
    """In-memory task store implementation."""
    
    def __init__(self, max_tasks: int = 10000):
        self.max_tasks = max_tasks
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
    
    async def store_task(self, task: Task) -> bool:
        """Store a task in memory."""
        async with self._lock:
            try:
                self._tasks[task.task_id] = task
                
                # Maintain size limit
                if len(self._tasks) > self.max_tasks:
                    # Remove oldest completed/failed tasks
                    old_tasks = [
                        (task_id, task) for task_id, task in self._tasks.items()
                        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                    ]
                    old_tasks.sort(key=lambda x: x[1].completed_at or x[1].created_at)
                    
                    for task_id, _ in old_tasks[:len(self._tasks) - self.max_tasks]:
                        del self._tasks[task_id]
                
                return True
            except Exception as e:
                logger.error(f"Failed to store task {task.task_id}: {e}")
                return False
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def update_task(self, task: Task) -> bool:
        """Update a task."""
        async with self._lock:
            if task.task_id in self._tasks:
                self._tasks[task.task_id] = task
                return True
            return False
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False
    
    async def get_pending_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Get pending tasks."""
        async with self._lock:
            pending_tasks = [
                task for task in self._tasks.values()
                if task.status == TaskStatus.PENDING
            ]
            pending_tasks.sort(key=lambda t: (t.priority.value, t.created_at), reverse=True)
            
            if limit:
                pending_tasks = pending_tasks[:limit]
            
            return pending_tasks
    
    async def get_failed_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Get failed tasks."""
        async with self._lock:
            failed_tasks = [
                task for task in self._tasks.values()
                if task.status == TaskStatus.FAILED
            ]
            failed_tasks.sort(key=lambda t: t.completed_at or t.created_at, reverse=True)
            
            if limit:
                failed_tasks = failed_tasks[:limit]
            
            return failed_tasks


class TaskWorker:
    """Individual task worker for processing tasks."""
    
    def __init__(
        self,
        worker_id: str,
        task_registry: Dict[str, Callable],
        metrics: Optional[IMetrics] = None
    ):
        """Initialize the task worker.
        
        Args:
            worker_id: Unique worker identifier
            task_registry: Registry of available task functions
            metrics: Optional metrics collector
        """
        self.worker_id = worker_id
        self.task_registry = task_registry
        self.metrics = metrics
        self._current_task: Optional[Task] = None
        self._running = False
        
        logger.debug(f"Task worker {worker_id} initialized")
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult with execution details
        """
        if task.func_name not in self.task_registry:
            error_msg = f"Task function '{task.func_name}' not found in registry"
            logger.error(error_msg)
            return TaskResult(success=False, error=error_msg)
        
        func = self.task_registry[task.func_name]
        self._current_task = task
        
        start_time = time.time()
        
        try:
            # Execute with timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_function(func, task.args, task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_function(func, task.args, task.kwargs)
            
            duration = time.time() - start_time
            
            # Record metrics
            if self.metrics:
                self.metrics.record_counter(
                    "task_queue.tasks_completed",
                    tags={
                        "worker_id": self.worker_id,
                        "func_name": task.func_name,
                        "priority": task.priority.name
                    }
                )
                
                self.metrics.record_histogram(
                    "task_queue.task_duration",
                    duration,
                    tags={
                        "worker_id": self.worker_id,
                        "func_name": task.func_name
                    }
                )
            
            return TaskResult(
                success=True,
                result=result,
                duration=duration,
                metadata={"worker_id": self.worker_id}
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"Task timed out after {task.timeout}s"
            logger.warning(f"Task {task.task_id} timed out: {error_msg}")
            
            if self.metrics:
                self.metrics.record_counter(
                    "task_queue.tasks_timeout",
                    tags={
                        "worker_id": self.worker_id,
                        "func_name": task.func_name
                    }
                )
            
            return TaskResult(
                success=False,
                error=error_msg,
                duration=duration,
                metadata={"worker_id": self.worker_id}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Task {task.task_id} failed: {error_msg}")
            
            if self.metrics:
                self.metrics.record_counter(
                    "task_queue.tasks_failed",
                    tags={
                        "worker_id": self.worker_id,
                        "func_name": task.func_name,
                        "error_type": type(e).__name__
                    }
                )
            
            return TaskResult(
                success=False,
                error=error_msg,
                duration=duration,
                metadata={"worker_id": self.worker_id}
            )
        
        finally:
            self._current_task = None
    
    async def _execute_function(self, func: Callable, args: tuple, kwargs: Dict[str, Any]) -> Any:
        """Execute a function with proper async handling."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run in thread pool for CPU-bound tasks
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    def get_current_task(self) -> Optional[Task]:
        """Get the currently executing task.
        
        Returns:
            Current task or None if idle
        """
        return self._current_task


class AsyncTaskQueue:
    """Async task queue with priority support and background processing."""
    
    def __init__(
        self,
        max_workers: int = 10,
        task_store: Optional[ITaskStore] = None,
        event_bus: Optional[EventBus] = None,
        metrics: Optional[IMetrics] = None
    ):
        """Initialize the task queue.
        
        Args:
            max_workers: Maximum number of concurrent workers
            task_store: Optional task persistence store
            event_bus: Optional event bus for notifications
            metrics: Optional metrics collector
        """
        self.max_workers = max_workers
        self.task_store = task_store or MemoryTaskStore()
        self.event_bus = event_bus or get_event_bus()
        self.metrics = metrics
        
        self._task_registry: Dict[str, Callable] = {}
        self._workers: List[TaskWorker] = []
        self._task_queue: List[Task] = []  # Priority queue
        self._scheduled_tasks: List[Task] = []  # Future tasks
        self._running_tasks: Dict[str, Task] = {}
        self._queue_lock = asyncio.Lock()
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Initialize workers
        for i in range(max_workers):
            worker = TaskWorker(f"worker-{i}", self._task_registry, metrics)
            self._workers.append(worker)
        
        logger.info(f"Task queue initialized with {max_workers} workers")
    
    def register_task_function(self, name: str, func: Callable):
        """Register a task function.
        
        Args:
            name: Name to register the function under
            func: Function to register
        """
        self._task_registry[name] = func
        logger.debug(f"Registered task function: {name}")
    
    def task(self, name: Optional[str] = None):
        """Decorator to register task functions.
        
        Args:
            name: Optional name for the task (defaults to function name)
        """
        def decorator(func):
            task_name = name or func.__name__
            self.register_task_function(task_name, func)
            return func
        return decorator
    
    async def enqueue(
        self,
        func_name: str,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        scheduled_at: Optional[float] = None,
        expires_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Enqueue a task for execution.
        
        Args:
            func_name: Name of the registered function to execute
            *args: Positional arguments for the function
            priority: Task priority
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            timeout: Task timeout in seconds
            scheduled_at: Unix timestamp to schedule execution
            expires_at: Unix timestamp when task expires
            metadata: Optional task metadata
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            scheduled_at=scheduled_at,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Store task
        await self.task_store.store_task(task)
        
        async with self._queue_lock:
            if scheduled_at and scheduled_at > time.time():
                # Schedule for future execution
                heapq.heappush(self._scheduled_tasks, task)
            else:
                # Add to immediate execution queue
                heapq.heappush(self._task_queue, task)
        
        # Publish event
        await publish_event(
            "task.enqueued",
            {
                'task_id': task_id,
                'func_name': func_name,
                'priority': priority.name,
                'scheduled_at': scheduled_at
            },
            source="task_queue"
        )
        
        # Record metrics
        if self.metrics:
            self.metrics.record_counter(
                "task_queue.tasks_enqueued",
                tags={
                    "func_name": func_name,
                    "priority": priority.name
                }
            )
        
        logger.debug(f"Enqueued task {task_id}: {func_name}")
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        async with self._queue_lock:
            # Check if task is running
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.completed_at = time.time()
                await self.task_store.update_task(task)
                
                # Remove from running tasks
                del self._running_tasks[task_id]
                
                await publish_event(
                    "task.cancelled",
                    {'task_id': task_id},
                    source="task_queue"
                )
                
                return True
            
            # Check queued tasks
            for i, task in enumerate(self._task_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
                    await self.task_store.update_task(task)
                    
                    self._task_queue.pop(i)
                    heapq.heapify(self._task_queue)
                    
                    await publish_event(
                        "task.cancelled",
                        {'task_id': task_id},
                        source="task_queue"
                    )
                    
                    return True
            
            # Check scheduled tasks
            for i, task in enumerate(self._scheduled_tasks):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
                    await self.task_store.update_task(task)
                    
                    self._scheduled_tasks.pop(i)
                    heapq.heapify(self._scheduled_tasks)
                    
                    await publish_event(
                        "task.cancelled",
                        {'task_id': task_id},
                        source="task_queue"
                    )
                    
                    return True
        
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task instance or None if not found
        """
        return await self.task_store.get_task(task_id)
    
    async def start(self):
        """Start the task queue processing."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_tasks())
        self._scheduler_task = asyncio.create_task(self._schedule_tasks())
        
        logger.info("Task queue started")
    
    async def stop(self):
        """Stop the task queue processing."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel processor tasks
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running tasks to complete
        while self._running_tasks:
            logger.info(f"Waiting for {len(self._running_tasks)} tasks to complete...")
            await asyncio.sleep(1)
        
        logger.info("Task queue stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "running": self._running,
            "max_workers": self.max_workers,
            "registered_functions": list(self._task_registry.keys()),
            "queued_tasks": len(self._task_queue),
            "scheduled_tasks": len(self._scheduled_tasks),
            "running_tasks": len(self._running_tasks),
            "worker_status": [
                {
                    "worker_id": worker.worker_id,
                    "current_task": worker.get_current_task().task_id if worker.get_current_task() else None
                }
                for worker in self._workers
            ]
        }
    
    async def _process_tasks(self):
        """Background task processing loop."""
        while self._running:
            try:
                await self._process_next_task()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_next_task(self):
        """Process the next available task."""
        # Get next task from queue
        task = None
        async with self._queue_lock:
            if self._task_queue:
                task = heapq.heappop(self._task_queue)
        
        if not task:
            return
        
        # Check if task has expired
        if task.is_expired():
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            task.last_error = "Task expired"
            await self.task_store.update_task(task)
            return
        
        # Find available worker
        available_worker = None
        for worker in self._workers:
            if worker.get_current_task() is None:
                available_worker = worker
                break
        
        if not available_worker:
            # Put task back in queue
            async with self._queue_lock:
                heapq.heappush(self._task_queue, task)
            return
        
        # Execute task
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.attempts += 1
        await self.task_store.update_task(task)
        
        # Add to running tasks
        self._running_tasks[task.task_id] = task
        
        await publish_event(
            "task.started",
            {
                'task_id': task.task_id,
                'func_name': task.func_name,
                'worker_id': available_worker.worker_id
            },
            source="task_queue"
        )
        
        try:
            result = await available_worker.execute_task(task)
            
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                await self.task_store.update_task(task)
                
                await publish_event(
                    "task.completed",
                    {
                        'task_id': task.task_id,
                        'func_name': task.func_name,
                        'duration': result.duration
                    },
                    source="task_queue"
                )
            else:
                await self._handle_task_failure(task, result.error)
        
        except Exception as e:
            await self._handle_task_failure(task, str(e))
        
        finally:
            # Remove from running tasks
            self._running_tasks.pop(task.task_id, None)
    
    async def _handle_task_failure(self, task: Task, error: str):
        """Handle task failure with retry logic."""
        task.last_error = error
        
        if task.attempts < task.max_retries:
            # Retry task
            task.status = TaskStatus.RETRYING
            task.scheduled_at = time.time() + (task.retry_delay * task.attempts)
            await self.task_store.update_task(task)
            
            async with self._queue_lock:
                heapq.heappush(self._scheduled_tasks, task)
            
            await publish_event(
                "task.retrying",
                {
                    'task_id': task.task_id,
                    'func_name': task.func_name,
                    'attempt': task.attempts,
                    'max_retries': task.max_retries,
                    'error': error
                },
                source="task_queue"
            )
        else:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            await self.task_store.update_task(task)
            
            await publish_event(
                "task.failed",
                {
                    'task_id': task.task_id,
                    'func_name': task.func_name,
                    'error': error,
                    'attempts': task.attempts
                },
                source="task_queue"
            )
    
    async def _schedule_tasks(self):
        """Background task scheduling loop."""
        while self._running:
            try:
                current_time = time.time()
                
                async with self._queue_lock:
                    # Move ready scheduled tasks to main queue
                    ready_tasks = []
                    remaining_tasks = []
                    
                    for task in self._scheduled_tasks:
                        if task.should_run_now():
                            ready_tasks.append(task)
                        else:
                            remaining_tasks.append(task)
                    
                    self._scheduled_tasks = remaining_tasks
                    heapq.heapify(self._scheduled_tasks)
                    
                    for task in ready_tasks:
                        heapq.heappush(self._task_queue, task)
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task scheduling loop: {e}")
                await asyncio.sleep(1)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global task queue instance
_global_task_queue: Optional[AsyncTaskQueue] = None


def get_task_queue(
    max_workers: int = 10,
    task_store: Optional[ITaskStore] = None,
    event_bus: Optional[EventBus] = None,
    metrics: Optional[IMetrics] = None
) -> AsyncTaskQueue:
    """Get the global task queue instance.
    
    Args:
        max_workers: Maximum number of concurrent workers
        task_store: Optional task persistence store
        event_bus: Optional event bus
        metrics: Optional metrics collector
        
    Returns:
        Global task queue instance
    """
    global _global_task_queue
    if _global_task_queue is None:
        _global_task_queue = AsyncTaskQueue(max_workers, task_store, event_bus, metrics)
    return _global_task_queue


def reset_task_queue() -> None:
    """Reset the global task queue (mainly for testing)."""
    global _global_task_queue
    if _global_task_queue:
        asyncio.create_task(_global_task_queue.stop())
    _global_task_queue = None


# Convenience functions
async def enqueue_task(
    func_name: str,
    *args,
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs
) -> str:
    """Convenience function to enqueue a task.
    
    Args:
        func_name: Name of the function to execute
        *args: Positional arguments
        priority: Task priority
        **kwargs: Keyword arguments
        
    Returns:
        Task ID
    """
    task_queue = get_task_queue()
    return await task_queue.enqueue(func_name, *args, priority=priority, **kwargs)


def task_function(name: Optional[str] = None):
    """Decorator to register a task function.
    
    Args:
        name: Optional name for the task
    """
    task_queue = get_task_queue()
    return task_queue.task(name)