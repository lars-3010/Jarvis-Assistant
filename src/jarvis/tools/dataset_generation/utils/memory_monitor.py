"""
Memory monitoring utilities for dataset generation.

This module provides memory monitoring and management capabilities
to optimize performance and prevent memory overflow during dataset generation.
"""

import gc
import time
from typing import Optional

from jarvis.utils.logging import setup_logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = setup_logging(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage during dataset generation."""

    def __init__(self, threshold_mb: int = 1000, warning_threshold_mb: int = 800):
        """Initialize memory monitor.
        
        Args:
            threshold_mb: Memory threshold in MB for triggering cleanup
            warning_threshold_mb: Memory threshold in MB for warnings
        """
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.warning_threshold_bytes = warning_threshold_mb * 1024 * 1024
        self.peak_usage = 0.0
        self.start_time = time.time()
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss
        else:
            self.process = None
            self.initial_memory = 0
            logger.warning("psutil not available - memory monitoring will be limited")

    def get_current_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Current memory usage in MB
        """
        if not PSUTIL_AVAILABLE or not self.process:
            return 0.0
            
        try:
            current_bytes = self.process.memory_info().rss
            current_mb = current_bytes / (1024 * 1024)
            self.peak_usage = max(self.peak_usage, current_mb)
            return current_mb
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB.
        
        Returns:
            Peak memory usage in MB
        """
        return self.peak_usage

    def get_memory_increase(self) -> float:
        """Get memory increase since initialization in MB.
        
        Returns:
            Memory increase in MB
        """
        if not PSUTIL_AVAILABLE or not self.process:
            return 0.0
            
        try:
            current_bytes = self.process.memory_info().rss
            increase_mb = (current_bytes - self.initial_memory) / (1024 * 1024)
            return max(0.0, increase_mb)
        except Exception as e:
            logger.warning(f"Failed to calculate memory increase: {e}")
            return 0.0

    def should_cleanup(self) -> bool:
        """Check if memory cleanup should be triggered.
        
        Returns:
            True if cleanup should be performed
        """
        if not PSUTIL_AVAILABLE:
            return False
            
        current_usage = self.get_current_usage()
        return current_usage * 1024 * 1024 > self.threshold_bytes

    def should_warn(self) -> bool:
        """Check if memory warning should be issued.
        
        Returns:
            True if warning should be issued
        """
        if not PSUTIL_AVAILABLE:
            return False
            
        current_usage = self.get_current_usage()
        return current_usage * 1024 * 1024 > self.warning_threshold_bytes

    def force_cleanup(self) -> float:
        """Force garbage collection and return memory freed.
        
        Returns:
            Memory freed in MB
        """
        if not PSUTIL_AVAILABLE:
            gc.collect()
            return 0.0
            
        try:
            memory_before = self.get_current_usage()
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to allow cleanup
            time.sleep(0.1)
            
            memory_after = self.get_current_usage()
            memory_freed = max(0.0, memory_before - memory_after)
            
            if memory_freed > 0:
                logger.debug(f"Garbage collection freed {memory_freed:.1f}MB")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"Failed to perform memory cleanup: {e}")
            return 0.0

    def get_system_memory_info(self) -> dict:
        """Get system memory information.
        
        Returns:
            Dictionary with system memory stats
        """
        if not PSUTIL_AVAILABLE:
            return {
                "available_mb": 0,
                "total_mb": 0,
                "percent_used": 0.0,
                "available": False
            }
            
        try:
            memory = psutil.virtual_memory()
            return {
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024),
                "percent_used": memory.percent,
                "available": True
            }
        except Exception as e:
            logger.warning(f"Failed to get system memory info: {e}")
            return {
                "available_mb": 0,
                "total_mb": 0,
                "percent_used": 0.0,
                "available": False
            }

    def log_memory_status(self, context: str = ""):
        """Log current memory status.
        
        Args:
            context: Context description for the log
        """
        current_usage = self.get_current_usage()
        memory_increase = self.get_memory_increase()
        system_info = self.get_system_memory_info()
        
        log_msg = f"Memory status"
        if context:
            log_msg += f" ({context})"
        log_msg += f": Current: {current_usage:.1f}MB, Increase: {memory_increase:.1f}MB, Peak: {self.peak_usage:.1f}MB"
        
        if system_info["available"]:
            log_msg += f", System: {system_info['percent_used']:.1f}% used"
        
        if self.should_warn():
            logger.warning(log_msg)
        else:
            logger.debug(log_msg)


class AdaptiveBatchSizer:
    """Dynamically adjust batch sizes based on memory usage and performance."""

    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 1, max_batch_size: int = 128):
        """Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.memory_history = []

    def adjust_batch_size(self, memory_monitor: MemoryMonitor, 
                         processing_time: float, items_processed: int) -> int:
        """Adjust batch size based on memory usage and performance.
        
        Args:
            memory_monitor: Memory monitor instance
            processing_time: Time taken to process last batch
            items_processed: Number of items processed in last batch
            
        Returns:
            New batch size
        """
        current_memory = memory_monitor.get_current_usage()
        
        # Record performance metrics
        if items_processed > 0:
            items_per_second = items_processed / max(processing_time, 0.001)
            self.performance_history.append(items_per_second)
            self.memory_history.append(current_memory)
            
            # Keep only recent history
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
                self.memory_history = self.memory_history[-10:]

        # Adjust based on memory pressure
        if memory_monitor.should_cleanup():
            # Reduce batch size if memory is high
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.7))
            logger.info(f"Reducing batch size due to memory pressure: {self.current_batch_size} -> {new_size}")
            self.current_batch_size = new_size
            
        elif current_memory < memory_monitor.warning_threshold_bytes / (1024 * 1024):
            # Increase batch size if memory is low and performance is good
            if len(self.performance_history) >= 3:
                recent_performance = sum(self.performance_history[-3:]) / 3
                if recent_performance > 0:  # Performance is reasonable
                    new_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
                    if new_size > self.current_batch_size:
                        logger.debug(f"Increasing batch size: {self.current_batch_size} -> {new_size}")
                        self.current_batch_size = new_size

        return self.current_batch_size

    def get_recommended_batch_size(self, total_items: int, available_memory_mb: float) -> int:
        """Get recommended batch size based on total items and available memory.
        
        Args:
            total_items: Total number of items to process
            available_memory_mb: Available memory in MB
            
        Returns:
            Recommended batch size
        """
        # Estimate memory per item (rough heuristic)
        estimated_memory_per_item = 0.5  # MB per item (conservative estimate)
        
        # Calculate max batch size based on memory
        memory_based_max = int(available_memory_mb * 0.3 / estimated_memory_per_item)  # Use 30% of available memory
        memory_based_max = max(self.min_batch_size, min(memory_based_max, self.max_batch_size))
        
        # Calculate reasonable batch size based on total items
        if total_items < 100:
            item_based_size = max(1, total_items // 10)
        elif total_items < 1000:
            item_based_size = max(10, total_items // 50)
        else:
            item_based_size = max(20, min(100, total_items // 100))
        
        # Use the smaller of the two constraints
        recommended_size = min(memory_based_max, item_based_size)
        
        logger.debug(f"Recommended batch size: {recommended_size} "
                    f"(memory-based: {memory_based_max}, item-based: {item_based_size})")
        
        return recommended_size

    def reset(self):
        """Reset batch sizer to initial state."""
        self.current_batch_size = self.initial_batch_size
        self.performance_history.clear()
        self.memory_history.clear()


class PerformanceTracker:
    """Track performance metrics during dataset generation."""

    def __init__(self):
        """Initialize performance tracker."""
        self.start_time = time.time()
        self.batch_times = []
        self.batch_sizes = []
        self.memory_snapshots = []
        self.error_counts = []

    def record_batch(self, batch_size: int, processing_time: float, 
                    memory_usage: float, error_count: int = 0):
        """Record batch processing metrics.
        
        Args:
            batch_size: Size of the processed batch
            processing_time: Time taken to process the batch
            memory_usage: Memory usage in MB
            error_count: Number of errors in the batch
        """
        self.batch_times.append(processing_time)
        self.batch_sizes.append(batch_size)
        self.memory_snapshots.append(memory_usage)
        self.error_counts.append(error_count)

    def get_performance_summary(self) -> dict:
        """Get performance summary statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        total_time = time.time() - self.start_time
        total_items = sum(self.batch_sizes)
        total_errors = sum(self.error_counts)
        
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        avg_memory = sum(self.memory_snapshots) / len(self.memory_snapshots) if self.memory_snapshots else 0
        max_memory = max(self.memory_snapshots) if self.memory_snapshots else 0
        
        items_per_second = total_items / total_time if total_time > 0 else 0
        error_rate = total_errors / total_items if total_items > 0 else 0
        
        return {
            "total_time_seconds": total_time,
            "total_items_processed": total_items,
            "total_errors": total_errors,
            "items_per_second": items_per_second,
            "error_rate": error_rate,
            "average_batch_time": avg_batch_time,
            "average_batch_size": avg_batch_size,
            "average_memory_mb": avg_memory,
            "peak_memory_mb": max_memory,
            "batches_processed": len(self.batch_times)
        }

    def log_performance_summary(self):
        """Log performance summary."""
        summary = self.get_performance_summary()
        
        logger.info(f"Performance Summary:")
        logger.info(f"  Total time: {summary['total_time_seconds']:.2f}s")
        logger.info(f"  Items processed: {summary['total_items_processed']}")
        logger.info(f"  Processing rate: {summary['items_per_second']:.2f} items/sec")
        logger.info(f"  Error rate: {summary['error_rate']:.2%}")
        logger.info(f"  Average batch size: {summary['average_batch_size']:.1f}")
        logger.info(f"  Average memory usage: {summary['average_memory_mb']:.1f}MB")
        logger.info(f"  Peak memory usage: {summary['peak_memory_mb']:.1f}MB")
        logger.info(f"  Batches processed: {summary['batches_processed']}")