"""
LLM services module for AI extension.

This module provides local LLM integration with configurable model routing,
prompt templates, and task-specific optimization.
"""

from .interfaces import (
    AnalysisResult,
    ILLMService,
    IModelRouter,
    LLMConfig,
    LLMResponse,
    LLMResponseStatus,
    ModelInfo,
    StreamingResponse,
    TaskType,
)
from .models import (
    ConversationHistory,
    LLMError,
    LLMProviderConfig,
    LLMRateLimitError,
    LLMServiceStatus,
    LLMTimeoutError,
    ModelPerformanceMetrics,
    ModelRoutingConfig,
    ModelUnavailableError,
    PromptTemplate,
    TaskRoutingRule,
)
from .ollama import OllamaClient
from .prompts import (
    PromptTemplateManager,
    PromptVariables,
    get_template_manager,
    render_prompt,
    render_prompt_string,
)
from .router import ModelRouter

__all__ = [
    # Core interfaces
    "ILLMService",
    "IModelRouter",
    "TaskType",
    "LLMResponse",
    "LLMResponseStatus",
    "AnalysisResult",
    "StreamingResponse",
    "LLMConfig",
    "ModelInfo",

    # Models and data structures
    "PromptTemplate",
    "ConversationHistory",
    "ModelPerformanceMetrics",
    "LLMProviderConfig",
    "LLMServiceStatus",
    "TaskRoutingRule",
    "ModelRoutingConfig",
    "PromptVariables",

    # Implementations
    "OllamaClient",
    "ModelRouter",
    "PromptTemplateManager",

    # Utilities
    "get_template_manager",
    "render_prompt",
    "render_prompt_string",

    # Exceptions
    "LLMError",
    "ModelUnavailableError",
    "LLMTimeoutError",
    "LLMRateLimitError"
]
