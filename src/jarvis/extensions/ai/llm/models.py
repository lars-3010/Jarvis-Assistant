"""
LLM response models and data structures.

This module provides additional models and utilities for LLM operations
that complement the core interfaces.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .interfaces import ModelInfo, TaskType


class PromptTemplate(BaseModel):
    """Template for constructing prompts."""
    name: str
    template: str
    variables: list[str] = []
    task_type: TaskType = TaskType.GENERAL
    model_preference: str | None = None
    default_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator('variables')
    @classmethod
    def validate_variables(cls, v):
        """Validate that all variables in template are listed."""
        # Basic validation - could be enhanced with Jinja2 parsing
        return v

    model_config = ConfigDict(use_enum_values=True)


class ConversationMessage(BaseModel):
    """Message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationHistory(BaseModel):
    """History of a conversation."""
    messages: list[ConversationMessage] = []
    max_messages: int = 50
    total_tokens: int = 0

    def add_message(self, role: str, content: str, metadata: dict[str, Any] = None):
        """Add a message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Trim if exceeding max messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context_string(self, include_system: bool = True) -> str:
        """Get conversation as context string."""
        context_parts = []
        for msg in self.messages:
            if msg.role == "system" and not include_system:
                continue
            context_parts.append(f"{msg.role}: {msg.content}")
        return "\n\n".join(context_parts)


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for a model."""
    model_name: str
    task_type: TaskType

    # Response time metrics
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0

    # Success metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Quality metrics
    avg_confidence: float = 0.0
    avg_tokens_per_second: float = 0.0

    # Resource usage
    avg_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0

    # Timestamps
    first_request: datetime | None = None
    last_request: datetime | None = None
    last_updated: datetime = Field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

    def update_metrics(
        self,
        response_time_ms: float,
        success: bool,
        confidence: float | None = None,
        tokens_per_second: float | None = None,
        memory_usage_mb: float | None = None
    ):
        """Update metrics with new request data."""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        # Update response time metrics
        self.avg_response_time_ms = (
            (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms)
            / self.total_requests
        )
        self.min_response_time_ms = min(self.min_response_time_ms, response_time_ms)
        self.max_response_time_ms = max(self.max_response_time_ms, response_time_ms)

        # Update quality metrics
        if confidence is not None:
            self.avg_confidence = (
                (self.avg_confidence * (self.total_requests - 1) + confidence)
                / self.total_requests
            )

        if tokens_per_second is not None:
            self.avg_tokens_per_second = (
                (self.avg_tokens_per_second * (self.total_requests - 1) + tokens_per_second)
                / self.total_requests
            )

        # Update resource usage
        if memory_usage_mb is not None:
            self.avg_memory_usage_mb = (
                (self.avg_memory_usage_mb * (self.total_requests - 1) + memory_usage_mb)
                / self.total_requests
            )
            self.peak_memory_usage_mb = max(self.peak_memory_usage_mb, memory_usage_mb)

        # Update timestamps
        now = datetime.now()
        if self.first_request is None:
            self.first_request = now
        self.last_request = now
        self.last_updated = now

    model_config = ConfigDict(use_enum_values=True)


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider."""
    provider_name: str
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Model configuration
    available_models: list[ModelInfo] = []
    default_model: str | None = None
    model_preferences: dict[TaskType, str] = Field(default_factory=dict)

    # Resource limits
    max_concurrent_requests: int = 2
    max_memory_gb: float = 8.0

    # Performance tuning
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

    model_config = ConfigDict(use_enum_values=True)


class LLMServiceStatus(BaseModel):
    """Status of LLM service."""
    is_healthy: bool = True
    provider: str = ""
    active_models: list[str] = []

    # Resource usage
    memory_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0

    # Performance metrics
    active_requests: int = 0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0

    # Error information
    last_error: str | None = None
    error_count: int = 0

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    last_request_at: datetime | None = None
    last_check_at: datetime = Field(default_factory=datetime.now)


class TaskRoutingRule(BaseModel):
    """Rule for routing tasks to models."""
    task_type: TaskType
    model_name: str
    conditions: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    enabled: bool = True

    # Performance requirements
    max_response_time_ms: float | None = None
    min_success_rate: float | None = None

    # Resource requirements
    max_memory_gb: float | None = None
    min_quality_score: float | None = None

    model_config = ConfigDict(use_enum_values=True)


class ModelRoutingConfig(BaseModel):
    """Configuration for model routing."""
    routing_rules: list[TaskRoutingRule] = []
    fallback_model: str | None = None

    # Routing strategy
    strategy: str = "performance"  # "performance", "quality", "cost"

    # Performance thresholds
    max_response_time_ms: float = 10000.0
    min_success_rate: float = 0.95

    # Quality preferences
    quality_weight: float = 0.4
    speed_weight: float = 0.4
    resource_weight: float = 0.2

    # Health check settings
    health_check_interval_seconds: int = 60
    model_failure_threshold: int = 3

    def get_routing_rule(self, task_type: TaskType) -> TaskRoutingRule | None:
        """Get routing rule for a task type."""
        enabled_rules = [r for r in self.routing_rules if r.enabled and r.task_type == task_type]
        if not enabled_rules:
            return None

        # Return highest priority rule
        return max(enabled_rules, key=lambda r: r.priority)

    def add_routing_rule(self, rule: TaskRoutingRule):
        """Add a routing rule."""
        self.routing_rules.append(rule)

    def remove_routing_rule(self, task_type: TaskType, model_name: str):
        """Remove a routing rule."""
        self.routing_rules = [
            r for r in self.routing_rules
            if not (r.task_type == task_type and r.model_name == model_name)
        ]

    model_config = ConfigDict(use_enum_values=True)


class LLMError(Exception):
    """Base exception for LLM operations."""

    def __init__(self, message: str, model_name: str = None, task_type: TaskType = None):
        super().__init__(message)
        self.message = message
        self.model_name = model_name
        self.task_type = task_type


class ModelUnavailableError(LLMError):
    """Exception raised when a model is not available."""
    pass


class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out."""
    pass


class LLMRateLimitError(LLMError):
    """Exception raised when rate limit is exceeded."""
    pass


class LLMValidationError(LLMError):
    """Exception raised when input validation fails."""
    pass
