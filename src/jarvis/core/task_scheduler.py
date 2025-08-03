"""
Task Scheduler for Recurring and Scheduled Tasks.

This module provides a comprehensive task scheduler for handling recurring tasks,
cron-like scheduling, and time-based automation.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import re
from enum import Enum

from jarvis.core.task_queue import AsyncTaskQueue, TaskPriority, get_task_queue
from jarvis.core.events import get_event_bus, publish_event
from jarvis.core.interfaces import IMetrics
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import TaskError, ConfigurationError

logger = setup_logging(__name__)


class ScheduleType(Enum):
    """Types of schedules."""
    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    schedule_id: str
    task_name: str
    func_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    schedule_type: ScheduleType = ScheduleType.INTERVAL
    interval_seconds: Optional[float] = None
    cron_expression: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_runs: Optional[int] = None
    priority: TaskPriority = TaskPriority.NORMAL
    timezone: str = "UTC"
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_run_now(self) -> bool:
        """Check if task should run now."""
        if not self.enabled:
            return False
        
        current_time = datetime.now(timezone.utc)
        
        # Check if task has expired
        if self.end_date and current_time > self.end_date:
            return False
        
        # Check if max runs exceeded
        if self.max_runs and self.run_count >= self.max_runs:
            return False
        
        # Check if it's time to run
        if self.next_run_at and current_time >= self.next_run_at:
            return True
        
        return False
    
    def calculate_next_run(self):
        """Calculate the next run time based on schedule."""
        current_time = datetime.now(timezone.utc)
        
        if self.schedule_type == ScheduleType.ONCE:
            # One-time task
            if self.run_count == 0:
                self.next_run_at = self.start_date or current_time
            else:
                self.next_run_at = None
                
        elif self.schedule_type == ScheduleType.INTERVAL:
            # Interval-based scheduling
            if self.interval_seconds:
                if self.last_run_at:
                    self.next_run_at = self.last_run_at + timedelta(seconds=self.interval_seconds)
                else:
                    self.next_run_at = self.start_date or current_time
                    
        elif self.schedule_type == ScheduleType.CRON:
            # Cron-like scheduling
            if self.cron_expression:
                self.next_run_at = self._calculate_cron_next_run(current_time)
    
    def _calculate_cron_next_run(self, current_time: datetime) -> Optional[datetime]:
        """Calculate next run time for cron expression."""
        # Basic cron parser (minute hour day month weekday)
        try:
            parts = self.cron_expression.strip().split()
            if len(parts) != 5:
                logger.error(f"Invalid cron expression: {self.cron_expression}")
                return None
            
            minute, hour, day, month, weekday = parts
            
            # Start from next minute
            next_time = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            # Simple implementation - find next matching time
            for _ in range(60 * 24 * 7):  # Search for a week
                if self._matches_cron_time(next_time, minute, hour, day, month, weekday):
                    return next_time
                next_time += timedelta(minutes=1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating cron next run: {e}")
            return None
    
    def _matches_cron_time(
        self,
        dt: datetime,
        minute: str,
        hour: str,
        day: str,
        month: str,
        weekday: str
    ) -> bool:
        """Check if datetime matches cron expression."""
        return (
            self._matches_cron_field(dt.minute, minute, 0, 59) and
            self._matches_cron_field(dt.hour, hour, 0, 23) and
            self._matches_cron_field(dt.day, day, 1, 31) and
            self._matches_cron_field(dt.month, month, 1, 12) and
            self._matches_cron_field(dt.weekday(), weekday, 0, 6)
        )
    
    def _matches_cron_field(self, value: int, pattern: str, min_val: int, max_val: int) -> bool:
        """Check if value matches cron field pattern."""
        if pattern == "*":
            return True
        
        if pattern.isdigit():
            return value == int(pattern)
        
        if "/" in pattern:
            base, step = pattern.split("/")
            if base == "*":
                return value % int(step) == 0
            else:
                return value >= int(base) and (value - int(base)) % int(step) == 0
        
        if "-" in pattern:
            start, end = pattern.split("-")
            return int(start) <= value <= int(end)
        
        if "," in pattern:
            values = [int(v.strip()) for v in pattern.split(",")]
            return value in values
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'schedule_id': self.schedule_id,
            'task_name': self.task_name,
            'func_name': self.func_name,
            'args': self.args,
            'kwargs': self.kwargs,
            'schedule_type': self.schedule_type.value,
            'interval_seconds': self.interval_seconds,
            'cron_expression': self.cron_expression,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'max_runs': self.max_runs,
            'priority': self.priority.value,
            'timezone': self.timezone,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat(),
            'last_run_at': self.last_run_at.isoformat() if self.last_run_at else None,
            'next_run_at': self.next_run_at.isoformat() if self.next_run_at else None,
            'run_count': self.run_count,
            'failure_count': self.failure_count,
            'metadata': self.metadata
        }


class TaskScheduler:
    """Task scheduler for recurring and scheduled tasks."""
    
    def __init__(
        self,
        task_queue: Optional[AsyncTaskQueue] = None,
        metrics: Optional[IMetrics] = None
    ):
        """Initialize the task scheduler.
        
        Args:
            task_queue: Optional task queue instance
            metrics: Optional metrics collector
        """
        self.task_queue = task_queue or get_task_queue()
        self.metrics = metrics
        self.event_bus = get_event_bus()
        
        self._scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        logger.info("Task scheduler initialized")
    
    def schedule_interval(
        self,
        task_name: str,
        func_name: str,
        interval_seconds: float,
        *args,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        enabled: bool = True,
        **kwargs
    ) -> str:
        """Schedule a task to run at regular intervals.
        
        Args:
            task_name: Name for the scheduled task
            func_name: Name of the function to execute
            interval_seconds: Interval between runs in seconds
            *args: Positional arguments for the function
            start_date: When to start running (defaults to now)
            end_date: When to stop running
            max_runs: Maximum number of runs
            priority: Task priority
            enabled: Whether the schedule is enabled
            **kwargs: Keyword arguments for the function
            
        Returns:
            Schedule ID
        """
        import uuid
        schedule_id = str(uuid.uuid4())
        
        scheduled_task = ScheduledTask(
            schedule_id=schedule_id,
            task_name=task_name,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=interval_seconds,
            start_date=start_date,
            end_date=end_date,
            max_runs=max_runs,
            priority=priority,
            enabled=enabled
        )
        
        scheduled_task.calculate_next_run()
        self._scheduled_tasks[schedule_id] = scheduled_task
        
        logger.info(f"Scheduled interval task {task_name} every {interval_seconds}s")
        return schedule_id
    
    def schedule_cron(
        self,
        task_name: str,
        func_name: str,
        cron_expression: str,
        *args,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timezone: str = "UTC",
        enabled: bool = True,
        **kwargs
    ) -> str:
        """Schedule a task using cron expression.
        
        Args:
            task_name: Name for the scheduled task
            func_name: Name of the function to execute
            cron_expression: Cron expression (minute hour day month weekday)
            *args: Positional arguments for the function
            start_date: When to start running
            end_date: When to stop running
            max_runs: Maximum number of runs
            priority: Task priority
            timezone: Timezone for the schedule
            enabled: Whether the schedule is enabled
            **kwargs: Keyword arguments for the function
            
        Returns:
            Schedule ID
        """
        import uuid
        schedule_id = str(uuid.uuid4())
        
        scheduled_task = ScheduledTask(
            schedule_id=schedule_id,
            task_name=task_name,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            start_date=start_date,
            end_date=end_date,
            max_runs=max_runs,
            priority=priority,
            timezone=timezone,
            enabled=enabled
        )
        
        scheduled_task.calculate_next_run()
        self._scheduled_tasks[schedule_id] = scheduled_task
        
        logger.info(f"Scheduled cron task {task_name}: {cron_expression}")
        return schedule_id
    
    def schedule_once(
        self,
        task_name: str,
        func_name: str,
        run_at: datetime,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        enabled: bool = True,
        **kwargs
    ) -> str:
        """Schedule a task to run once at a specific time.
        
        Args:
            task_name: Name for the scheduled task
            func_name: Name of the function to execute
            run_at: When to run the task
            *args: Positional arguments for the function
            priority: Task priority
            enabled: Whether the schedule is enabled
            **kwargs: Keyword arguments for the function
            
        Returns:
            Schedule ID
        """
        import uuid
        schedule_id = str(uuid.uuid4())
        
        scheduled_task = ScheduledTask(
            schedule_id=schedule_id,
            task_name=task_name,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            schedule_type=ScheduleType.ONCE,
            start_date=run_at,
            max_runs=1,
            priority=priority,
            enabled=enabled
        )
        
        scheduled_task.calculate_next_run()
        self._scheduled_tasks[schedule_id] = scheduled_task
        
        logger.info(f"Scheduled one-time task {task_name} at {run_at}")
        return schedule_id
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a scheduled task.
        
        Args:
            schedule_id: Schedule ID to enable
            
        Returns:
            True if enabled successfully, False otherwise
        """
        if schedule_id in self._scheduled_tasks:
            self._scheduled_tasks[schedule_id].enabled = True
            self._scheduled_tasks[schedule_id].calculate_next_run()
            logger.info(f"Enabled schedule {schedule_id}")
            return True
        return False
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a scheduled task.
        
        Args:
            schedule_id: Schedule ID to disable
            
        Returns:
            True if disabled successfully, False otherwise
        """
        if schedule_id in self._scheduled_tasks:
            self._scheduled_tasks[schedule_id].enabled = False
            logger.info(f"Disabled schedule {schedule_id}")
            return True
        return False
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a scheduled task.
        
        Args:
            schedule_id: Schedule ID to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if schedule_id in self._scheduled_tasks:
            del self._scheduled_tasks[schedule_id]
            logger.info(f"Removed schedule {schedule_id}")
            return True
        return False
    
    def get_schedule(self, schedule_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID.
        
        Args:
            schedule_id: Schedule ID to get
            
        Returns:
            ScheduledTask or None if not found
        """
        return self._scheduled_tasks.get(schedule_id)
    
    def list_schedules(self, enabled_only: bool = False) -> List[ScheduledTask]:
        """List all scheduled tasks.
        
        Args:
            enabled_only: Whether to return only enabled schedules
            
        Returns:
            List of scheduled tasks
        """
        schedules = list(self._scheduled_tasks.values())
        
        if enabled_only:
            schedules = [s for s in schedules if s.enabled]
        
        return sorted(schedules, key=lambda s: s.next_run_at or datetime.max.replace(tzinfo=timezone.utc))
    
    async def start(self):
        """Start the task scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task scheduler stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_schedules = len(self._scheduled_tasks)
        enabled_schedules = sum(1 for s in self._scheduled_tasks.values() if s.enabled)
        
        next_runs = [
            s.next_run_at for s in self._scheduled_tasks.values()
            if s.enabled and s.next_run_at
        ]
        
        return {
            "running": self._running,
            "total_schedules": total_schedules,
            "enabled_schedules": enabled_schedules,
            "disabled_schedules": total_schedules - enabled_schedules,
            "next_run_at": min(next_runs).isoformat() if next_runs else None,
            "schedules_by_type": {
                schedule_type.value: sum(
                    1 for s in self._scheduled_tasks.values()
                    if s.schedule_type == schedule_type
                )
                for schedule_type in ScheduleType
            }
        }
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check all scheduled tasks
                for schedule_id, scheduled_task in list(self._scheduled_tasks.items()):
                    if scheduled_task.should_run_now():
                        await self._execute_scheduled_task(scheduled_task)
                
                # Sleep for a short interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(10)
    
    async def _execute_scheduled_task(self, scheduled_task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            # Enqueue the task
            task_id = await self.task_queue.enqueue(
                scheduled_task.func_name,
                *scheduled_task.args,
                priority=scheduled_task.priority,
                metadata={
                    'schedule_id': scheduled_task.schedule_id,
                    'scheduled_task_name': scheduled_task.task_name,
                    'scheduled_run': True
                },
                **scheduled_task.kwargs
            )
            
            # Update schedule stats
            scheduled_task.last_run_at = datetime.now(timezone.utc)
            scheduled_task.run_count += 1
            
            # Calculate next run
            scheduled_task.calculate_next_run()
            
            # Publish event
            await publish_event(
                "scheduler.task_executed",
                {
                    'schedule_id': scheduled_task.schedule_id,
                    'task_name': scheduled_task.task_name,
                    'func_name': scheduled_task.func_name,
                    'task_id': task_id,
                    'run_count': scheduled_task.run_count
                },
                source="task_scheduler"
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_counter(
                    "scheduler.tasks_executed",
                    tags={
                        'task_name': scheduled_task.task_name,
                        'schedule_type': scheduled_task.schedule_type.value
                    }
                )
            
            logger.debug(f"Executed scheduled task {scheduled_task.task_name}")
            
        except Exception as e:
            scheduled_task.failure_count += 1
            logger.error(f"Failed to execute scheduled task {scheduled_task.task_name}: {e}")
            
            await publish_event(
                "scheduler.task_failed",
                {
                    'schedule_id': scheduled_task.schedule_id,
                    'task_name': scheduled_task.task_name,
                    'error': str(e),
                    'failure_count': scheduled_task.failure_count
                },
                source="task_scheduler"
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global task scheduler instance
_global_task_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler(
    task_queue: Optional[AsyncTaskQueue] = None,
    metrics: Optional[IMetrics] = None
) -> TaskScheduler:
    """Get the global task scheduler instance.
    
    Args:
        task_queue: Optional task queue instance
        metrics: Optional metrics collector
        
    Returns:
        Global task scheduler instance
    """
    global _global_task_scheduler
    if _global_task_scheduler is None:
        _global_task_scheduler = TaskScheduler(task_queue, metrics)
    return _global_task_scheduler


def reset_task_scheduler() -> None:
    """Reset the global task scheduler (mainly for testing)."""
    global _global_task_scheduler
    if _global_task_scheduler:
        asyncio.create_task(_global_task_scheduler.stop())
    _global_task_scheduler = None


# Convenience decorators and functions
def scheduled_task(
    interval: Optional[float] = None,
    cron: Optional[str] = None,
    name: Optional[str] = None,
    priority: TaskPriority = TaskPriority.NORMAL
):
    """Decorator to mark a function as a scheduled task.
    
    Args:
        interval: Interval in seconds for recurring execution
        cron: Cron expression for scheduling
        name: Optional task name
        priority: Task priority
    """
    def decorator(func):
        task_name = name or func.__name__
        func_name = func.__name__
        
        # Register the function with task queue
        task_queue = get_task_queue()
        task_queue.register_task_function(func_name, func)
        
        # Schedule the task
        scheduler = get_task_scheduler()
        
        if interval:
            schedule_id = scheduler.schedule_interval(
                task_name, func_name, interval, priority=priority
            )
        elif cron:
            schedule_id = scheduler.schedule_cron(
                task_name, func_name, cron, priority=priority
            )
        else:
            raise ConfigurationError("Either interval or cron must be specified")
        
        func._schedule_id = schedule_id
        return func
    
    return decorator


# Predefined schedule helpers
def every_minute():
    """Schedule to run every minute."""
    return scheduled_task(interval=60)


def every_hour():
    """Schedule to run every hour."""
    return scheduled_task(interval=3600)


def every_day():
    """Schedule to run every day."""
    return scheduled_task(interval=86400)


def daily_at(hour: int, minute: int = 0):
    """Schedule to run daily at specific time."""
    return scheduled_task(cron=f"{minute} {hour} * * *")


def weekly_at(weekday: int, hour: int, minute: int = 0):
    """Schedule to run weekly at specific time."""
    return scheduled_task(cron=f"{minute} {hour} * * {weekday}")


def monthly_at(day: int, hour: int, minute: int = 0):
    """Schedule to run monthly at specific time."""
    return scheduled_task(cron=f"{minute} {hour} {day} * *")