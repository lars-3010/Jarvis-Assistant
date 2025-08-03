"""
LLM services module for AI extension.

This module provides local LLM integration with configurable model routing,
prompt templates, and task-specific optimization.
"""

from .interfaces import (
    ILLMService, IModelRouter, TaskType, LLMResponse, LLMResponseStatus,
    AnalysisResult, StreamingResponse, LLMConfig, ModelInfo
)
from .models import (
    PromptTemplate, ConversationHistory, ModelPerformanceMetrics,
    LLMProviderConfig, LLMServiceStatus, TaskRoutingRule, ModelRoutingConfig,
    LLMError, ModelUnavailableError, LLMTimeoutError, LLMRateLimitError
)
from .ollama import OllamaClient
from .router import ModelRouter
from .prompts import (
    PromptTemplateManager, PromptVariables, get_template_manager,
    render_prompt, render_prompt_string
)

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