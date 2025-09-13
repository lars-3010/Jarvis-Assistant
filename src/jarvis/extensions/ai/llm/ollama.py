"""
Ollama client implementation for LLM services.

This module provides the Ollama-specific implementation of the ILLMService
interface with support for local model management and streaming.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from jarvis.utils.logging import setup_logging

from .interfaces import (
    AnalysisResult,
    ILLMService,
    LLMConfig,
    LLMResponse,
    LLMResponseStatus,
    ModelInfo,
    StreamingResponse,
    TaskType,
)
from .models import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    ModelPerformanceMetrics,
    ModelUnavailableError,
)

logger = setup_logging(__name__)


class OllamaClient(ILLMService):
    """Ollama client implementation of ILLMService."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        model_config: dict[str, Any] = None
    ):
        """Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            model_config: Model configuration mapping
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.model_config = model_config or {}

        # Model management
        self.available_models: dict[str, ModelInfo] = {}
        self.performance_metrics: dict[str, ModelPerformanceMetrics] = {}

        # Session management
        self.session: aiohttp.ClientSession | None = None
        self.is_healthy = False

        # Task-specific model routing
        self.task_model_mapping = {
            TaskType.SUMMARIZE: "mistral:7b-instruct-q4_K_M",
            TaskType.ANALYZE: "llama3:8b-instruct-q8_0",
            TaskType.QUICK_ANSWER: "tinyllama:1.1b",
            TaskType.GENERAL: "mistral:7b-instruct-q4_K_M",
            TaskType.GENERATE: "llama3:8b-instruct-q8_0",
            TaskType.EXTRACT: "mistral:7b-instruct-q4_K_M",
            TaskType.CLASSIFY: "mistral:7b-instruct-q4_K_M"
        }

        # Default configurations per task
        self.task_configs = {
            TaskType.SUMMARIZE: {"temperature": 0.3, "max_tokens": 500},
            TaskType.ANALYZE: {"temperature": 0.5, "max_tokens": 1000},
            TaskType.QUICK_ANSWER: {"temperature": 0.2, "max_tokens": 200},
            TaskType.GENERAL: {"temperature": 0.7, "max_tokens": 1000},
            TaskType.GENERATE: {"temperature": 0.8, "max_tokens": 2000},
            TaskType.EXTRACT: {"temperature": 0.1, "max_tokens": 800},
            TaskType.CLASSIFY: {"temperature": 0.1, "max_tokens": 100}
        }

        logger.info(f"Initialized Ollama client: {base_url}")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        stream: bool = False
    ) -> Any:
        """Make HTTP request to Ollama API."""
        session = await self._ensure_session()
        url = f"{self.base_url}/{endpoint}"

        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method,
                    url,
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        if stream:
                            return response
                        else:
                            return await response.json()
                    elif response.status == 404:
                        raise ModelUnavailableError(f"Model not found: {url}")
                    elif response.status == 429:
                        raise LLMRateLimitError(f"Rate limit exceeded: {url}")
                    else:
                        error_text = await response.text()
                        raise LLMError(f"Request failed: {response.status} - {error_text}")

            except TimeoutError:
                if attempt == self.max_retries:
                    raise LLMTimeoutError(f"Request timed out after {self.timeout_seconds}s")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)

        raise LLMError("Max retries exceeded")

    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        config: LLMConfig | None = None
    ) -> LLMResponse:
        """Generate text using Ollama."""
        start_time = time.time()

        # Determine task type and model
        task_type = config.task_type if config else TaskType.GENERAL
        model_name = self.get_model_for_task(task_type)

        # Build full prompt
        full_prompt = self._build_prompt(prompt, context, task_type)

        # Get configuration
        request_config = self._get_request_config(task_type, config)

        try:
            # Make request to Ollama
            request_data = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": request_config.get("temperature", 0.7),
                    "num_predict": request_config.get("max_tokens", 1000),
                    "top_p": request_config.get("top_p", 0.9),
                    "stop": request_config.get("stop_sequences", [])
                }
            }

            response_data = await self._make_request("POST", "api/generate", request_data)

            response_time = (time.time() - start_time) * 1000

            # Create response
            response = LLMResponse(
                text=response_data.get("response", ""),
                model_used=model_name,
                task_type=task_type,
                status=LLMResponseStatus.SUCCESS,
                response_time_ms=response_time,
                metadata={
                    "total_duration": response_data.get("total_duration", 0),
                    "load_duration": response_data.get("load_duration", 0),
                    "prompt_eval_count": response_data.get("prompt_eval_count", 0),
                    "eval_count": response_data.get("eval_count", 0),
                    "eval_duration": response_data.get("eval_duration", 0)
                }
            )

            # Calculate token information
            response.input_tokens = response_data.get("prompt_eval_count", 0)
            response.output_tokens = response_data.get("eval_count", 0)

            # Update performance metrics
            self._update_performance_metrics(model_name, task_type, response_time, True)

            return response

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(model_name, task_type, response_time, False)

            error_response = LLMResponse(
                text="",
                model_used=model_name,
                task_type=task_type,
                status=LLMResponseStatus.ERROR,
                error_message=str(e),
                response_time_ms=response_time
            )

            if isinstance(e, LLMTimeoutError):
                error_response.status = LLMResponseStatus.TIMEOUT
            elif isinstance(e, ModelUnavailableError):
                error_response.status = LLMResponseStatus.MODEL_UNAVAILABLE
            elif isinstance(e, LLMRateLimitError):
                error_response.status = LLMResponseStatus.RATE_LIMITED

            return error_response

    async def generate_stream(
        self,
        prompt: str,
        context: str | None = None,
        config: LLMConfig | None = None
    ) -> AsyncGenerator[StreamingResponse, None]:
        """Generate text using streaming."""
        task_type = config.task_type if config else TaskType.GENERAL
        model_name = self.get_model_for_task(task_type)

        full_prompt = self._build_prompt(prompt, context, task_type)
        request_config = self._get_request_config(task_type, config)

        request_data = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": request_config.get("temperature", 0.7),
                "num_predict": request_config.get("max_tokens", 1000),
                "top_p": request_config.get("top_p", 0.9)
            }
        }

        try:
            response = await self._make_request("POST", "api/generate", request_data, stream=True)

            chunk_index = 0
            async for line in response.content:
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))

                        if chunk_data.get("done", False):
                            yield StreamingResponse(
                                chunk="",
                                is_complete=True,
                                chunk_index=chunk_index,
                                metadata=chunk_data
                            )
                            break

                        chunk_text = chunk_data.get("response", "")
                        if chunk_text:
                            yield StreamingResponse(
                                chunk=chunk_text,
                                is_complete=False,
                                chunk_index=chunk_index,
                                metadata={"model": model_name}
                            )
                            chunk_index += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield StreamingResponse(
                chunk="",
                is_complete=True,
                chunk_index=0,
                metadata={"error": str(e)}
            )

    async def summarize(
        self,
        text: str,
        style: str = "bullet",
        max_length: int | None = None
    ) -> LLMResponse:
        """Summarize text content."""
        prompt = self._build_summarize_prompt(text, style, max_length)

        config = LLMConfig(
            task_type=TaskType.SUMMARIZE,
            temperature=0.3,
            max_tokens=max_length or 500
        )

        return await self.generate(prompt, config=config)

    async def analyze(
        self,
        content: list[str],
        question: str,
        analysis_type: str = "general"
    ) -> AnalysisResult:
        """Analyze content and answer questions."""
        start_time = time.time()

        # Combine content
        combined_content = "\n\n".join(content)

        # Build analysis prompt
        prompt = self._build_analyze_prompt(combined_content, question, analysis_type)

        config = LLMConfig(
            task_type=TaskType.ANALYZE,
            temperature=0.5,
            max_tokens=1000
        )

        response = await self.generate(prompt, config=config)

        # Parse response into AnalysisResult
        result = AnalysisResult(
            answer=response.text,
            sources_analyzed=len(content),
            processing_time_ms=(time.time() - start_time) * 1000,
            analysis_method="ollama_llm"
        )

        # Extract key points (simple implementation)
        if "Key points:" in response.text:
            lines = response.text.split("\n")
            key_points = [
                line.strip("- ").strip()
                for line in lines
                if line.strip().startswith("- ")
            ]
            result.key_points = key_points

        return result

    async def quick_answer(
        self,
        question: str,
        context: str | None = None
    ) -> LLMResponse:
        """Get quick answer to a question."""
        prompt = f"Answer concisely: {question}"

        config = LLMConfig(
            task_type=TaskType.QUICK_ANSWER,
            temperature=0.2,
            max_tokens=200
        )

        return await self.generate(prompt, context=context, config=config)

    async def extract_information(
        self,
        text: str,
        extraction_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract structured information from text."""
        prompt = self._build_extraction_prompt(text, extraction_schema)

        config = LLMConfig(
            task_type=TaskType.EXTRACT,
            temperature=0.1,
            max_tokens=800
        )

        response = await self.generate(prompt, config=config)

        # Parse JSON response (simplified)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse extracted information", "raw_response": response.text}

    async def classify_text(
        self,
        text: str,
        categories: list[str]
    ) -> dict[str, float]:
        """Classify text into categories."""
        prompt = self._build_classification_prompt(text, categories)

        config = LLMConfig(
            task_type=TaskType.CLASSIFY,
            temperature=0.1,
            max_tokens=100
        )

        response = await self.generate(prompt, config=config)

        # Parse classification response (simplified)
        results = {}
        for category in categories:
            if category.lower() in response.text.lower():
                results[category] = 0.8  # Simplified scoring
            else:
                results[category] = 0.2

        return results

    def get_model_for_task(self, task_type: TaskType) -> str:
        """Get the best model for a specific task type."""
        return self.task_model_mapping.get(task_type, "mistral:7b-instruct-q4_K_M")

    def get_model_info(self, model_name: str) -> ModelInfo | None:
        """Get information about a model."""
        return self.available_models.get(model_name)

    def list_available_models(self) -> list[ModelInfo]:
        """List all available models."""
        return list(self.available_models.values())

    async def download_model(self, model_name: str) -> bool:
        """Download a model if not available locally."""
        try:
            request_data = {"name": model_name}
            await self._make_request("POST", "api/pull", request_data)

            # Refresh model list
            await self._refresh_model_list()
            return True

        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Check health of Ollama service."""
        try:
            # Test basic connectivity
            await self._make_request("GET", "api/version")

            # Refresh model list
            await self._refresh_model_list()

            self.is_healthy = True

            return {
                "status": "healthy",
                "base_url": self.base_url,
                "available_models": len(self.available_models),
                "model_names": list(self.available_models.keys())
            }

        except Exception as e:
            self.is_healthy = False
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.base_url
            }

    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        return {
            "active_models": len(self.available_models),
            "performance_metrics": {
                name: {
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                    "success_rate": metrics.success_rate,
                    "total_requests": metrics.total_requests
                }
                for name, metrics in self.performance_metrics.items()
            }
        }

    def _build_prompt(self, prompt: str, context: str | None, task_type: TaskType) -> str:
        """Build full prompt with context and task-specific formatting."""
        if context:
            return f"Context: {context}\n\nTask: {prompt}"
        return prompt

    def _build_summarize_prompt(self, text: str, style: str, max_length: int | None) -> str:
        """Build prompt for summarization."""
        length_instruction = f" in {max_length} words or less" if max_length else ""

        style_instructions = {
            "bullet": "Use bullet points for key information.",
            "paragraph": "Write in paragraph form.",
            "outline": "Use an outline format with main points and sub-points."
        }

        style_instruction = style_instructions.get(style, "")

        return f"""Summarize the following text{length_instruction}. {style_instruction}

Text to summarize:
{text}

Summary:"""

    def _build_analyze_prompt(self, content: str, question: str, analysis_type: str) -> str:
        """Build prompt for analysis."""
        return f"""Analyze the following content and answer the question.

Content:
{content}

Question: {question}

Analysis type: {analysis_type}

Please provide:
1. A clear answer to the question
2. Key points from the content
3. Supporting evidence

Answer:"""

    def _build_extraction_prompt(self, text: str, schema: dict[str, Any]) -> str:
        """Build prompt for information extraction."""
        return f"""Extract the following information from the text and return as JSON:

Schema: {json.dumps(schema, indent=2)}

Text:
{text}

JSON:"""

    def _build_classification_prompt(self, text: str, categories: list[str]) -> str:
        """Build prompt for text classification."""
        return f"""Classify the following text into one or more of these categories: {', '.join(categories)}

Text:
{text}

Classification:"""

    def _get_request_config(self, task_type: TaskType, config: LLMConfig | None) -> dict[str, Any]:
        """Get request configuration for task type."""
        # Start with task-specific defaults
        request_config = self.task_configs.get(task_type, {}).copy()

        # Apply user config overrides
        if config:
            if config.temperature is not None:
                request_config["temperature"] = config.temperature
            if config.max_tokens is not None:
                request_config["max_tokens"] = config.max_tokens
            if config.top_p is not None:
                request_config["top_p"] = config.top_p
            if config.stop_sequences:
                request_config["stop_sequences"] = config.stop_sequences

        return request_config

    def _update_performance_metrics(
        self,
        model_name: str,
        task_type: TaskType,
        response_time: float,
        success: bool
    ) -> None:
        """Update performance metrics for a model."""
        key = f"{model_name}_{task_type.value}"

        if key not in self.performance_metrics:
            self.performance_metrics[key] = ModelPerformanceMetrics(
                model_name=model_name,
                task_type=task_type
            )

        self.performance_metrics[key].update_metrics(response_time, success)

    async def _refresh_model_list(self) -> None:
        """Refresh the list of available models."""
        try:
            response = await self._make_request("GET", "api/tags")

            self.available_models.clear()

            for model_data in response.get("models", []):
                model_name = model_data.get("name", "")

                # Parse model info
                model_info = ModelInfo(
                    name=model_name,
                    provider="ollama",
                    size=self._extract_model_size(model_name),
                    quantization=self._extract_quantization(model_name),
                    is_available=True
                )

                self.available_models[model_name] = model_info

        except Exception as e:
            logger.error(f"Failed to refresh model list: {e}")

    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size from model name."""
        parts = model_name.split(":")
        if len(parts) > 1:
            size_part = parts[1].lower()
            if "7b" in size_part:
                return "7b"
            elif "13b" in size_part:
                return "13b"
            elif "70b" in size_part:
                return "70b"
            elif "1.1b" in size_part:
                return "1.1b"
        return "unknown"

    def _extract_quantization(self, model_name: str) -> str | None:
        """Extract quantization from model name."""
        if "q4" in model_name.lower():
            return "q4"
        elif "q8" in model_name.lower():
            return "q8"
        elif "q16" in model_name.lower():
            return "q16"
        return None

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("Ollama client closed")
