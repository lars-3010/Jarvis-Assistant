"""
Progress tracking and performance monitoring utilities.

This module provides comprehensive progress tracking with time estimates,
memory usage monitoring, and performance metrics collection.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque

from jarvis.utils.logging import setup_logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = setup_logging(__name__)


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a specific time."""
    timestamp: float
    items_processed: int
    items_total: int
    memory_usage_mb: float
    processing_rate: float  # items per second
    estimated_completion: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    start_time: float
    end_time: Optional[float] = None
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    
    # Memory metrics
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    
    # Performance metrics
    average_rate: float = 0.0  # items per second
    peak_rate: float = 0.0
    current_rate: float = 0.0
    
    # Time estimates
    estimated_completion: Optional[datetime] = None
    estimated_remaining_seconds: float = 0.0
    
    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    
    # Additional metrics
    batch_count: int = 0
    average_batch_size: float = 0.0
    average_batch_time: float = 0.0


class ProgressTracker:
    """Comprehensive progress tracker with time estimates and performance monitoring."""

    def __init__(self, total_items: int, description: str = "Processing",
                 update_interval: float = 1.0, history_size: int = 100):
        """Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the processing task
            update_interval: Minimum interval between progress updates (seconds)
            history_size: Number of progress snapshots to keep for rate calculation
        """
        self.total_items = total_items
        self.description = description
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Progress tracking
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_log_time = self.start_time
        
        # Progress history for rate calculation
        self.progress_history = deque(maxlen=history_size)
        
        # Performance metrics
        self.metrics = PerformanceMetrics(
            start_time=self.start_time,
            total_items=total_items
        )
        
        # Memory monitoring
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.metrics.initial_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        else:
            self.process = None
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        
        logger.info(f"Progress tracker initialized: {total_items} items to process")

    def update(self, processed_count: int, failed_count: int = 0, 
               force_update: bool = False, context: str = ""):
        """Update progress with current counts.
        
        Args:
            processed_count: Number of items processed so far
            failed_count: Number of items that failed processing
            force_update: Force update even if interval hasn't passed
            context: Additional context for logging
        """
        current_time = time.time()
        
        # Update counters
        self.processed_items = processed_count
        self.failed_items = failed_count
        self.metrics.processed_items = processed_count
        self.metrics.failed_items = failed_count
        
        # Update memory usage
        self._update_memory_metrics()
        
        # Check if we should update
        if not force_update and (current_time - self.last_update_time) < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Calculate progress metrics
        self._calculate_progress_metrics(current_time)
        
        # Create progress snapshot
        snapshot = ProgressSnapshot(
            timestamp=current_time,
            items_processed=processed_count,
            items_total=self.total_items,
            memory_usage_mb=self.metrics.current_memory_mb,
            processing_rate=self.metrics.current_rate,
            estimated_completion=self.metrics.estimated_completion
        )
        
        self.progress_history.append(snapshot)
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.get_progress_info())
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Log progress periodically
        if (current_time - self.last_log_time) >= 5.0:  # Log every 5 seconds
            self._log_progress(context)
            self.last_log_time = current_time

    def increment(self, processed_delta: int = 1, failed_delta: int = 0, 
                 context: str = ""):
        """Increment progress counters.
        
        Args:
            processed_delta: Number of newly processed items
            failed_delta: Number of newly failed items
            context: Additional context for logging
        """
        self.update(
            self.processed_items + processed_delta,
            self.failed_items + failed_delta,
            context=context
        )

    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function.
        
        Args:
            callback: Function to call on progress updates
        """
        self.progress_callbacks.append(callback)

    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary with progress information
        """
        total_processed = self.processed_items + self.failed_items
        progress_percent = (total_processed / self.total_items * 100) if self.total_items > 0 else 0
        
        return {
            "description": self.description,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "total_items": self.total_items,
            "progress_percent": progress_percent,
            "current_rate": self.metrics.current_rate,
            "average_rate": self.metrics.average_rate,
            "estimated_completion": self.metrics.estimated_completion,
            "estimated_remaining_seconds": self.metrics.estimated_remaining_seconds,
            "elapsed_seconds": time.time() - self.start_time,
            "memory_usage_mb": self.metrics.current_memory_mb,
            "peak_memory_mb": self.metrics.peak_memory_mb,
            "success_rate": self.metrics.success_rate,
            "error_rate": self.metrics.error_rate
        }

    def get_performance_summary(self) -> PerformanceMetrics:
        """Get comprehensive performance summary.
        
        Returns:
            PerformanceMetrics object with all metrics
        """
        # Update final metrics
        self.metrics.end_time = time.time()
        self._calculate_final_metrics()
        
        return self.metrics

    def finish(self, context: str = ""):
        """Mark processing as finished and log final summary.
        
        Args:
            context: Additional context for logging
        """
        self.metrics.end_time = time.time()
        self._calculate_final_metrics()
        
        # Force final update
        self.update(self.processed_items, self.failed_items, force_update=True, context=context)
        
        # Log final summary
        self._log_final_summary()

    def _update_memory_metrics(self):
        """Update memory usage metrics."""
        if not PSUTIL_AVAILABLE or not self.process:
            return
        
        try:
            current_memory = self.process.memory_info().rss / (1024 * 1024)
            self.metrics.current_memory_mb = current_memory
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, current_memory)
            
            # Update system memory usage
            system_memory = psutil.virtual_memory()
            self.metrics.memory_usage_percent = system_memory.percent
            
            # Update CPU usage
            self.metrics.cpu_usage_percent = self.process.cpu_percent()
            
        except Exception as e:
            logger.debug(f"Failed to update memory metrics: {e}")

    def _calculate_progress_metrics(self, current_time: float):
        """Calculate progress and performance metrics.
        
        Args:
            current_time: Current timestamp
        """
        elapsed_time = current_time - self.start_time
        total_processed = self.processed_items + self.failed_items
        
        # Calculate rates
        if elapsed_time > 0:
            self.metrics.average_rate = total_processed / elapsed_time
        
        # Calculate current rate from recent history
        if len(self.progress_history) >= 2:
            recent_snapshots = list(self.progress_history)[-10:]  # Last 10 snapshots
            if len(recent_snapshots) >= 2:
                time_diff = recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
                items_diff = (recent_snapshots[-1].items_processed + 
                            (self.failed_items - sum(s.items_processed for s in recent_snapshots[:-1])))
                items_diff -= recent_snapshots[0].items_processed
                
                if time_diff > 0:
                    self.metrics.current_rate = items_diff / time_diff
                    self.metrics.peak_rate = max(self.metrics.peak_rate, self.metrics.current_rate)
        
        # Calculate time estimates
        remaining_items = self.total_items - total_processed
        if self.metrics.current_rate > 0 and remaining_items > 0:
            self.metrics.estimated_remaining_seconds = remaining_items / self.metrics.current_rate
            self.metrics.estimated_completion = datetime.now() + timedelta(
                seconds=self.metrics.estimated_remaining_seconds
            )
        
        # Calculate quality metrics
        if total_processed > 0:
            self.metrics.success_rate = self.processed_items / total_processed
            self.metrics.error_rate = self.failed_items / total_processed

    def _calculate_final_metrics(self):
        """Calculate final performance metrics."""
        if self.metrics.end_time:
            total_time = self.metrics.end_time - self.metrics.start_time
            total_processed = self.processed_items + self.failed_items
            
            if total_time > 0:
                self.metrics.average_rate = total_processed / total_time
            
            if total_processed > 0:
                self.metrics.success_rate = self.processed_items / total_processed
                self.metrics.error_rate = self.failed_items / total_processed

    def _log_progress(self, context: str = ""):
        """Log current progress information.
        
        Args:
            context: Additional context for logging
        """
        total_processed = self.processed_items + self.failed_items
        progress_percent = (total_processed / self.total_items * 100) if self.total_items > 0 else 0
        
        log_msg = f"{self.description}: {total_processed}/{self.total_items} ({progress_percent:.1f}%)"
        
        if context:
            log_msg += f" - {context}"
        
        log_msg += f" | Rate: {self.metrics.current_rate:.1f} items/sec"
        
        if self.metrics.estimated_remaining_seconds > 0:
            remaining_time = timedelta(seconds=int(self.metrics.estimated_remaining_seconds))
            log_msg += f" | ETA: {remaining_time}"
        
        log_msg += f" | Memory: {self.metrics.current_memory_mb:.1f}MB"
        
        if self.failed_items > 0:
            log_msg += f" | Errors: {self.failed_items} ({self.metrics.error_rate:.1%})"
        
        logger.info(log_msg)

    def _log_final_summary(self):
        """Log final processing summary."""
        total_time = self.metrics.end_time - self.metrics.start_time if self.metrics.end_time else 0
        total_processed = self.processed_items + self.failed_items
        
        logger.info(f"{self.description} completed:")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  Items processed: {self.processed_items}")
        logger.info(f"  Items failed: {self.failed_items}")
        logger.info(f"  Success rate: {self.metrics.success_rate:.1%}")
        logger.info(f"  Average rate: {self.metrics.average_rate:.2f} items/sec")
        logger.info(f"  Peak rate: {self.metrics.peak_rate:.2f} items/sec")
        logger.info(f"  Peak memory: {self.metrics.peak_memory_mb:.1f}MB")
        
        if self.metrics.current_memory_mb > self.metrics.initial_memory_mb:
            memory_increase = self.metrics.current_memory_mb - self.metrics.initial_memory_mb
            logger.info(f"  Memory increase: {memory_increase:.1f}MB")


class BatchProgressTracker:
    """Progress tracker specialized for batch processing."""

    def __init__(self, total_batches: int, total_items: int, description: str = "Batch processing"):
        """Initialize batch progress tracker.
        
        Args:
            total_batches: Total number of batches to process
            total_items: Total number of items across all batches
            description: Description of the processing task
        """
        self.total_batches = total_batches
        self.total_items = total_items
        self.description = description
        
        self.completed_batches = 0
        self.processed_items = 0
        self.failed_items = 0
        
        self.batch_times = []
        self.batch_sizes = []
        
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        logger.info(f"Batch progress tracker initialized: {total_batches} batches, {total_items} total items")

    def update_batch(self, batch_index: int, batch_size: int, batch_time: float,
                    successful_items: int, failed_items: int):
        """Update progress with completed batch information.
        
        Args:
            batch_index: Index of the completed batch
            batch_size: Size of the completed batch
            batch_time: Time taken to process the batch
            successful_items: Number of successful items in the batch
            failed_items: Number of failed items in the batch
        """
        self.completed_batches += 1
        self.processed_items += successful_items
        self.failed_items += failed_items
        
        self.batch_times.append(batch_time)
        self.batch_sizes.append(batch_size)
        
        # Log progress periodically
        current_time = time.time()
        if (current_time - self.last_log_time) >= 5.0:  # Log every 5 seconds
            self._log_batch_progress()
            self.last_log_time = current_time

    def get_batch_summary(self) -> Dict[str, Any]:
        """Get batch processing summary.
        
        Returns:
            Dictionary with batch processing metrics
        """
        total_time = time.time() - self.start_time
        total_processed = self.processed_items + self.failed_items
        
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        
        batches_per_second = self.completed_batches / total_time if total_time > 0 else 0
        items_per_second = total_processed / total_time if total_time > 0 else 0
        
        # Estimate remaining time
        remaining_batches = self.total_batches - self.completed_batches
        estimated_remaining_time = remaining_batches * avg_batch_time if avg_batch_time > 0 else 0
        
        return {
            "description": self.description,
            "completed_batches": self.completed_batches,
            "total_batches": self.total_batches,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "total_items": self.total_items,
            "batch_progress_percent": (self.completed_batches / self.total_batches * 100) if self.total_batches > 0 else 0,
            "item_progress_percent": (total_processed / self.total_items * 100) if self.total_items > 0 else 0,
            "average_batch_time": avg_batch_time,
            "average_batch_size": avg_batch_size,
            "batches_per_second": batches_per_second,
            "items_per_second": items_per_second,
            "estimated_remaining_seconds": estimated_remaining_time,
            "success_rate": self.processed_items / total_processed if total_processed > 0 else 0,
            "error_rate": self.failed_items / total_processed if total_processed > 0 else 0
        }

    def finish(self):
        """Mark batch processing as finished and log final summary."""
        summary = self.get_batch_summary()
        
        logger.info(f"{self.description} completed:")
        logger.info(f"  Batches processed: {summary['completed_batches']}/{summary['total_batches']}")
        logger.info(f"  Items processed: {summary['processed_items']}")
        logger.info(f"  Items failed: {summary['failed_items']}")
        logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        logger.info(f"  Average batch time: {summary['average_batch_time']:.2f}s")
        logger.info(f"  Average batch size: {summary['average_batch_size']:.1f}")
        logger.info(f"  Processing rate: {summary['items_per_second']:.2f} items/sec")

    def _log_batch_progress(self):
        """Log current batch progress."""
        summary = self.get_batch_summary()
        
        log_msg = (f"{self.description}: {summary['completed_batches']}/{summary['total_batches']} batches "
                  f"({summary['batch_progress_percent']:.1f}%) | "
                  f"{summary['processed_items'] + summary['failed_items']}/{summary['total_items']} items "
                  f"({summary['item_progress_percent']:.1f}%) | "
                  f"Rate: {summary['items_per_second']:.1f} items/sec")
        
        if summary['estimated_remaining_seconds'] > 0:
            remaining_time = timedelta(seconds=int(summary['estimated_remaining_seconds']))
            log_msg += f" | ETA: {remaining_time}"
        
        if summary['failed_items'] > 0:
            log_msg += f" | Errors: {summary['failed_items']} ({summary['error_rate']:.1%})"
        
        logger.info(log_msg)