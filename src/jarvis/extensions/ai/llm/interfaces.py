"""
LLM service interfaces for AI extension.

This module defines the abstract interfaces for LLM services with
configurable model routing and task-specific optimization.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class TaskType(str, Enum):
    """Supported task types for LLM operations."""
    GENERAL = "general"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    QUICK_ANSWER = "quick_answer"
    GENERATE = "generate"
    EXTRACT = "extract"
    CLASSIFY = "classify"


class LLMResponseStatus(str, Enum):
    """Status of LLM response."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    MODEL_UNAVAILABLE = "model_unavailable"


class ModelInfo(BaseModel):
    """Information about an LLM model."""
    name: str
    provider: str
    size: str  # e.g., "7b", "13b", "70b"
    quantization: str | None = None  # e.g., "q4", "q8"
    context_window: int = 4096
    max_tokens: int = 2048
    estimated_memory_gb: float = 0.0
    specialized_for: list[TaskType] = Field(default_factory=list)
    download_size_mb: int | None = None
    is_available: bool = False

    model_config = ConfigDict(use_enum_values=True)


class LLMResponse(BaseModel):
    """Response from LLM service."""
    text: str
    model_used: str
    task_type: TaskType
    status: LLMResponseStatus = LLMResponseStatus.SUCCESS
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Performance metrics
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_time_ms: float | None = None

    # Quality metrics
    confidence_score: float | None = None
    temperature_used: float | None = None

    model_config = ConfigDict(use_enum_values=True)


class AnalysisResult(BaseModel):
    """Result of content analysis."""
    answer: str
    key_points: list[str] = Field(default_factory=list)
    sentiment: str | None = None
    topics: list[str] = Field(default_factory=list)
    confidence: float | None = None
    sources_analyzed: int = 0

    # Additional analysis metadata
    analysis_method: str = "llm"
    processing_time_ms: float | None = None


class StreamingResponse(BaseModel):
    """Streaming response chunk."""
    chunk: str
    is_complete: bool = False
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Configuration for LLM requests."""
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = Field(default_factory=list)
    stream: bool = False
    timeout_seconds: int = 30

    # Task-specific overrides
    task_type: TaskType | None = None
    context_window_override: int | None = None


class ILLMService(ABC):
    """Abstract interface for LLM services with configurable model routing."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        config: LLMConfig | None = None
    ) -> LLMResponse:
        """Generate text using the LLM.
        
        Args:
            prompt: The prompt to generate from
            context: Optional context to include
            config: Optional configuration overrides
            
        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        context: str | None = None,
        config: LLMConfig | None = None
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Generate text using streaming.
        
        Args:
            prompt: The prompt to generate from
            context: Optional context to include
            config: Optional configuration overrides
            
        Yields:
            StreamingResponse chunks
        """
        pass

    @abstractmethod
    async def summarize(
        self,
        text: str,
        style: str = "bullet",
        max_length: int | None = None
    ) -> LLMResponse:
        """Summarize text content.
        
        Args:
            text: Text to summarize
            style: Summary style (bullet, paragraph, outline)
            max_length: Maximum summary length
            
        Returns:
            LLMResponse with summary
        """
        pass

    @abstractmethod
    async def analyze(
        self,
        content: list[str],
        question: str,
        analysis_type: str = "general"
    ) -> AnalysisResult:
        """Analyze content and answer questions.
        
        Args:
            content: List of content to analyze
            question: Question to answer
            analysis_type: Type of analysis (general, sentiment, topics)
            
        Returns:
            AnalysisResult with analysis
        """
        pass

    @abstractmethod
    async def quick_answer(
        self,
        question: str,
        context: str | None = None
    ) -> LLMResponse:
        """Get quick answer to a question.
        
        Args:
            question: Question to answer
            context: Optional context
            
        Returns:
            LLMResponse with quick answer
        """
        pass

    @abstractmethod
    async def extract_information(
        self,
        text: str,
        extraction_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract structured information from text.
        
        Args:
            text: Text to extract from
            extraction_schema: Schema defining what to extract
            
        Returns:
            Extracted information dictionary
        """
        pass

    @abstractmethod
    async def classify_text(
        self,
        text: str,
        categories: list[str]
    ) -> dict[str, float]:
        """Classify text into categories.
        
        Args:
            text: Text to classify
            categories: List of category names
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        pass

    @abstractmethod
    def get_model_for_task(self, task_type: TaskType) -> str:
        """Get the best model for a specific task type.
        
        Args:
            task_type: Type of task to get model for
            
        Returns:
            Model name
        """
        pass

    @abstractmethod
    def get_model_info(self, model_name: str) -> ModelInfo | None:
        """Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo or None if not found
        """
        pass

    @abstractmethod
    def list_available_models(self) -> list[ModelInfo]:
        """List all available models.
        
        Returns:
            List of ModelInfo objects
        """
        pass

    @abstractmethod
    async def download_model(self, model_name: str) -> bool:
        """Download a model if not available locally.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            True if download successful
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check health of LLM service.
        
        Returns:
            Health status dictionary
        """
        pass

    @abstractmethod
    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics.
        
        Returns:
            Resource usage dictionary
        """
        pass


class IModelRouter(ABC):
    """Abstract interface for model routing logic."""

    @abstractmethod
    def select_model(
        self,
        task_type: TaskType,
        context_length: int = 0,
        quality_preference: str = "balanced"
    ) -> str:
        """Select the best model for a task.
        
        Args:
            task_type: Type of task
            context_length: Length of context in tokens
            quality_preference: "speed", "quality", or "balanced"
            
        Returns:
            Model name
        """
        pass

    @abstractmethod
    def get_fallback_model(self, failed_model: str, task_type: TaskType) -> str | None:
        """Get a fallback model if the primary fails.
        
        Args:
            failed_model: Name of the failed model
            task_type: Type of task
            
        Returns:
            Fallback model name or None
        """
        pass

    @abstractmethod
    def update_model_performance(
        self,
        model_name: str,
        task_type: TaskType,
        response_time: float,
        success: bool
    ) -> None:
        """Update model performance metrics.
        
        Args:
            model_name: Name of the model
            task_type: Type of task
            response_time: Response time in milliseconds
            success: Whether the request succeeded
        """
        pass
