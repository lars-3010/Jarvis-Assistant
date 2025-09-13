"""
Metrics collection and reporting for Jarvis Assistant.
"""

import functools
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from jarvis.core.interfaces import IMetrics
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class JarvisMetrics(IMetrics):
    """Collects and provides performance metrics for Jarvis Assistant services and tools."""

    def __init__(self):
        self._metrics: dict[str, Any] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0,
            "last_recorded": 0.0
        })
        self._start_time = time.time()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        logger.info("JarvisMetrics initialized.")

    def record_metric(self, name: str, duration: float = 0.0, is_error: bool = False) -> None:
        """
        Records a single metric event.

        Args:
            name: The name of the metric (e.g., "semantic_search_time", "neo4j_connection_errors").
            duration: The time taken for the operation, if applicable.
            is_error: True if the operation resulted in an error.
        """
        with self._lock:
            metric = self._metrics[name]
            metric["count"] += 1
            metric["total_time"] += duration
            if is_error:
                metric["errors"] += 1
            metric["last_recorded"] = time.time()
            logger.debug(f"Metric recorded: {name}, duration={duration:.4f}s, error={is_error}")

    def get_metrics(self) -> dict[str, Any]:
        """
        Returns a snapshot of all collected metrics.
        """
        with self._lock:
            snapshot = {
                "uptime_seconds": time.time() - self._start_time,
                "timestamp": time.time(),
                "metrics": {}
            }
            for name, data in self._metrics.items():
                metric_data = data.copy()
                if metric_data["count"] > 0:
                    metric_data["avg_time"] = metric_data["total_time"] / metric_data["count"]
                else:
                    metric_data["avg_time"] = 0.0
                snapshot["metrics"][name] = metric_data
            logger.debug("Metrics snapshot generated.")
            return snapshot

    def reset_metrics(self) -> None:
        """
        Resets all collected metrics.
        """
        with self._lock:
            self._metrics.clear()
            self._start_time = time.time()
            logger.info("JarvisMetrics reset.")

    def record_tool_execution(self, tool_name: str, duration: float, success: bool) -> None:
        """
        Records metrics specifically for MCP tool executions.
        """
        metric_base_name = f"mcp_tool_{tool_name}"
        self.record_metric(f"{metric_base_name}_executions", duration=duration)
        if not success:
            self.record_metric(f"{metric_base_name}_errors")
        logger.debug(f"Tool execution metric: {tool_name}, duration={duration:.4f}s, success={success}")

    def record_service_operation(self, service_name: str, operation_name: str, duration: float, success: bool) -> None:
        """
        Records metrics for service-level operations.
        """
        metric_name = f"service_{service_name}_{operation_name}"
        self.record_metric(f"{metric_name}_total", duration=duration)
        if not success:
            self.record_metric(f"{metric_name}_errors")
        logger.debug(f"Service operation metric: {service_name}.{operation_name}, duration={duration:.4f}s, success={success}")

    def record_cache_stats(self, cache_name: str, stats: dict[str, Any]) -> None:
        """
        Records and updates cache-related metrics.
        """
        with self._lock:
            for key, value in stats.items():
                self._metrics[f"cache_{cache_name}_{key}"]["value"] = value
                self._metrics[f"cache_{cache_name}_{key}"]["last_recorded"] = time.time()

    def record_counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        """Record a counter metric."""
        metric_key = name
        if tags:
            tag_str = "_".join(f"{k}_{v}" for k, v in tags.items())
            metric_key = f"{name}_{tag_str}"

        with self._lock:
            self._metrics[metric_key]["count"] += value
            self._metrics[metric_key]["last_recorded"] = time.time()

    def record_gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        metric_key = name
        if tags:
            tag_str = "_".join(f"{k}_{v}" for k, v in tags.items())
            metric_key = f"{name}_{tag_str}"

        with self._lock:
            self._metrics[metric_key]["value"] = value
            self._metrics[metric_key]["last_recorded"] = time.time()

    def record_histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        metric_key = name
        if tags:
            tag_str = "_".join(f"{k}_{v}" for k, v in tags.items())
            metric_key = f"{name}_{tag_str}"

        with self._lock:
            metric = self._metrics[metric_key]
            metric["count"] += 1
            metric["total_time"] += value
            if "values" not in metric:
                metric["values"] = []
            metric["values"].append(value)
            metric["last_recorded"] = time.time()

    @contextmanager
    def time_operation(self, operation_name: str, error_on_exception: bool = True) -> Generator[None, None, None]:
        """
        Context manager for timing operations with automatic error recording.
        
        Args:
            operation_name: Name of the operation being timed
            error_on_exception: Whether to record an error if an exception occurs
            
        Example:
            with metrics.time_operation("database_query"):
                # Operation code here
                result = database.query(sql)
        """
        start_time = time.time()
        success = True

        try:
            yield
        except Exception:
            success = False
            if error_on_exception:
                self.record_metric(f"{operation_name}_errors")
            raise
        finally:
            duration = time.time() - start_time
            self.record_metric(f"{operation_name}_duration", duration=duration)
            if not success and error_on_exception:
                self.record_metric(f"{operation_name}_failures")
            logger.debug(f"Operation {operation_name} completed in {duration:.4f}s, success={success}")

    def time_function(self, metric_name: str | None = None, include_args: bool = False):
        """
        Decorator for automatically timing function execution.
        
        Args:
            metric_name: Custom metric name (defaults to function name)
            include_args: Whether to include function arguments in metric name
            
        Example:
            @metrics.time_function("mcp_tool_search")
            async def handle_search(query: str):
                # Function automatically timed
                return search_results
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = metric_name or f"function_{func.__name__}"
                start_time = time.time()
                success = True

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception:
                    success = False
                    self.record_metric(f"{name}_errors")
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_metric(f"{name}_executions", duration=duration)
                    if not success:
                        self.record_metric(f"{name}_failures")

                    # Optional argument logging for debugging
                    if include_args and args:
                        logger.debug(f"Function {name} called with args: {args[:2]}...")  # Limit for privacy

                    logger.debug(f"Function {name} completed in {duration:.4f}s, success={success}")

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = metric_name or f"function_{func.__name__}"
                start_time = time.time()
                success = True

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    success = False
                    self.record_metric(f"{name}_errors")
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_metric(f"{name}_executions", duration=duration)
                    if not success:
                        self.record_metric(f"{name}_failures")

                    if include_args and args:
                        logger.debug(f"Function {name} called with args: {args[:2]}...")

                    logger.debug(f"Function {name} completed in {duration:.4f}s, success={success}")

            # Return appropriate wrapper based on function type
            return async_wrapper if functools.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def safe_record(self, metric_name: str, duration: float = 0.0, is_error: bool = False) -> None:
        """
        Safely record a metric with error handling to prevent metrics from breaking core functionality.
        
        Args:
            metric_name: Name of the metric to record
            duration: Duration of the operation
            is_error: Whether this represents an error condition
        """
        try:
            self.record_metric(metric_name, duration, is_error)
        except (KeyError, TypeError, ValueError) as e:
            # Handle expected metric recording errors
            logger.warning(f"Failed to record metric {metric_name}: {e}")
        except Exception as e:
            # Never let metrics collection break the application, but log unexpected errors
            logger.error(f"Unexpected error recording metric {metric_name}: {e}")
            # Don't re-raise as metrics should not break the application
