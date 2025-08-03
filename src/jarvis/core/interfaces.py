"""
Service interfaces for dependency injection and modularity.

This module defines abstract base classes for all major services in the system,
enabling loose coupling and easier testing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence, Set, Callable
import torch
from jarvis.models.document import SearchResult


class IVectorDatabase(ABC):
    """Abstract interface for vector database operations."""
    
    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
    
    @abstractmethod
    def num_notes(self) -> int:
        """Get the total number of notes in the database."""
        pass
    
    @abstractmethod
    def get_most_recent_seen_timestamp(self, vault_name: str) -> Optional[float]:
        """Get the most recent seen timestamp for a vault."""
        pass
    
    @abstractmethod
    def store_note(
        self,
        path: Path,
        vault_name: str,
        last_modified: float,
        embedding: List[float],
        checksum: Optional[str] = None
    ) -> bool:
        """Store a note in the database."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
        vault_name: Optional[str] = None
    ) -> Sequence[tuple[str, Path, float]]:
        """Search for notes similar to a query embedding."""
        pass
    
    @abstractmethod
    def get_note_by_path(self, vault_name: str, path: Path) -> Optional[dict]:
        """Get a specific note by vault name and path."""
        pass
    
    @abstractmethod
    def delete_note(self, vault_name: str, path: Path) -> bool:
        """Delete a note from the database."""
        pass
    
    @abstractmethod
    def get_vault_stats(self, vault_name: str) -> dict:
        """Get statistics for a specific vault."""
        pass


class IGraphDatabase(ABC):
    """Abstract interface for graph database operations."""
    
    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the graph database connection is healthy."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the graph database connection."""
        pass
    
    @abstractmethod
    def create_or_update_note(self, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a note with all its relationships."""
        pass
    
    @abstractmethod
    def get_note_graph(self, path: str, depth: int = 2) -> Dict[str, Any]:
        """Get a knowledge graph centered on a specific note."""
        pass


class IVaultReader(ABC):
    """Abstract interface for vault file operations."""
    
    @abstractmethod
    def read_file(self, relative_path: str) -> Tuple[str, Dict[str, Any]]:
        """Read a file and return its content and metadata."""
        pass
    
    @abstractmethod
    def search_vault(self, query: str, search_content: bool = False, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for files in the vault."""
        pass
    
    @abstractmethod
    def list_files(self, extension: str = ".md") -> List[Path]:
        """List all files in the vault with specified extension."""
        pass


class IVectorEncoder(ABC):
    """Abstract interface for vector encoding operations."""
    
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Encode text into a vector representation."""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple texts into vector representations."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the encoding model."""
        pass


class IVectorSearcher(ABC):
    """Abstract interface for vector search operations."""
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        vault_name: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Perform semantic search."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the search model."""
        pass
    
    @abstractmethod
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        pass
    
    @abstractmethod
    def get_vault_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get vault statistics."""
        pass
    
    @abstractmethod
    def validate_vaults(self) -> Dict[str, bool]:
        """Validate vault accessibility."""
        pass


class IHealthChecker(ABC):
    """Abstract interface for health checking operations."""
    
    @abstractmethod
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        pass
    
    @abstractmethod
    def check_vector_database(self) -> bool:
        """Check vector database health."""
        pass
    
    @abstractmethod
    def check_graph_database(self) -> bool:
        """Check graph database health."""
        pass
    
    @abstractmethod
    def check_vault_access(self) -> Dict[str, bool]:
        """Check vault accessibility."""
        pass


class IMetrics(ABC):
    """Abstract interface for metrics collection."""
    
    @abstractmethod
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        pass
    
    @abstractmethod
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        pass
    
    @abstractmethod
    def time_operation(self, operation_name: str, error_on_exception: bool = True):
        """Context manager for timing operations."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        pass


class IServiceRegistry(ABC):
    """Abstract interface for service registry operations."""
    
    @abstractmethod
    def register_service(self, service_name: str, instance: Any, replace_existing: bool = False) -> bool:
        """Register a service instance."""
        pass
    
    @abstractmethod
    def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""
        pass
    
    @abstractmethod
    def discover_service(
        self,
        service_name: str,
        tags: Optional[Set[str]] = None,
        healthy_only: bool = True
    ) -> List[Any]:
        """Discover service instances."""
        pass
    
    @abstractmethod
    def get_service_instance(
        self,
        service_name: str,
        strategy: str = "round_robin",
        tags: Optional[Set[str]] = None
    ) -> Optional[Any]:
        """Get a service instance using load balancing."""
        pass
    
    @abstractmethod
    def update_service_health(
        self,
        service_name: str,
        instance_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update service health status."""
        pass
    
    @abstractmethod
    def subscribe_to_service_changes(
        self,
        service_name: str,
        callback: Callable[[str, str, Any], None]
    ) -> str:
        """Subscribe to service changes."""
        pass
    
    @abstractmethod
    def unsubscribe_from_service_changes(
        self,
        service_name: str,
        subscription_id: str
    ) -> bool:
        """Unsubscribe from service changes."""
        pass
    
    @abstractmethod
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service registry statistics."""
        pass


class ITaskQueue(ABC):
    """Abstract interface for task queue operations."""
    
    @abstractmethod
    def register_task_function(self, name: str, func: Callable) -> None:
        """Register a task function."""
        pass
    
    @abstractmethod
    async def enqueue(
        self,
        func_name: str,
        *args,
        priority: str = "normal",
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Enqueue a task for execution."""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[Any]:
        """Get task status."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the task queue processing."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the task queue processing."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        pass


class ITaskScheduler(ABC):
    """Abstract interface for task scheduler operations."""
    
    @abstractmethod
    def schedule_interval(
        self,
        task_name: str,
        func_name: str,
        interval_seconds: float,
        *args,
        **kwargs
    ) -> str:
        """Schedule a task to run at regular intervals."""
        pass
    
    @abstractmethod
    def schedule_cron(
        self,
        task_name: str,
        func_name: str,
        cron_expression: str,
        *args,
        **kwargs
    ) -> str:
        """Schedule a task using cron expression."""
        pass
    
    @abstractmethod
    def schedule_once(
        self,
        task_name: str,
        func_name: str,
        run_at: Any,
        *args,
        **kwargs
    ) -> str:
        """Schedule a task to run once at a specific time."""
        pass
    
    @abstractmethod
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a scheduled task."""
        pass
    
    @abstractmethod
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a scheduled task."""
        pass
    
    @abstractmethod
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a scheduled task."""
        pass
    
    @abstractmethod
    def list_schedules(self, enabled_only: bool = False) -> List[Any]:
        """List all scheduled tasks."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the task scheduler."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the task scheduler."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        pass


class IVaultAnalyticsService(ABC):
    """Abstract interface for vault analytics operations."""
    
    @abstractmethod
    async def get_vault_context(self, vault_name: str = "default") -> Dict[str, Any]:
        """Generate comprehensive vault overview with structured data."""
        pass
    
    @abstractmethod
    async def analyze_quality_distribution(self, vault_name: str = "default") -> Dict[str, Any]:
        """Analyze content quality patterns across the vault."""
        pass
    
    @abstractmethod
    async def map_knowledge_domains(self, vault_name: str = "default") -> Dict[str, Any]:
        """Identify and map knowledge domains with connection analysis."""
        pass
    
    @abstractmethod
    async def assess_note_quality(self, note_path: str, vault_name: str = "default") -> Dict[str, Any]:
        """Assess quality of a specific note."""
        pass
    
    @abstractmethod
    async def get_analytics_cache_status(self) -> Dict[str, Any]:
        """Get current cache status and freshness indicators."""
        pass
    
    @abstractmethod
    async def invalidate_cache(self, vault_name: Optional[str] = None) -> bool:
        """Invalidate analytics cache for a vault or all vaults."""
        pass
    
    @abstractmethod
    async def get_recommendations(self, vault_name: str = "default", limit: int = 10) -> Dict[str, Any]:
        """Get actionable recommendations for vault improvement."""
        pass