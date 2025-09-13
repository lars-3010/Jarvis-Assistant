"""
AI Extension main entry point.

This module provides the main AI extension class that implements
the IExtension interface for Jarvis Assistant.
"""

import time
from typing import Any

from jarvis.core.container import ServiceContainer
from jarvis.extensions.errors import ExtensionInitializationError, ExtensionToolError
from jarvis.extensions.interfaces import (
    ExtensionHealth,
    ExtensionMetadata,
    ExtensionStatus,
    IExtension,
    MCPTool,
)
import logging
from mcp import types

# Phase 1: LLM imports
from .llm import (
    ILLMService,
    LLMError,
    LLMProviderConfig,
    ModelRouter,
    ModelRoutingConfig,
    OllamaClient,
    PromptTemplateManager,
    get_template_manager,
)

logger = logging.getLogger(__name__)


class AIExtension(IExtension):
    """AI Extension for Jarvis Assistant.
    
    Provides AI capabilities including LLM integration, GraphRAG,
    and workflow orchestration in future phases.
    """

    def __init__(self):
        """Initialize the AI extension."""
        self.container: ServiceContainer = None
        self.initialized = False
        self.last_error: str = None

        # Phase tracking
        self.llm_enabled = False
        self.graphrag_enabled = False
        self.workflows_enabled = False

        # Phase 1: LLM services
        self.llm_service: ILLMService | None = None
        self.model_router: ModelRouter | None = None
        self.prompt_manager: PromptTemplateManager | None = None

        logger.info("AI Extension created")

    def get_metadata(self) -> ExtensionMetadata:
        """Return AI extension metadata."""
        return ExtensionMetadata(
            name="ai",
            version="0.1.0",
            description="AI capabilities extension with LLM, GraphRAG, and workflows",
            author="Jarvis Assistant",
            dependencies=[],  # No extension dependencies
            required_services=["vector_searcher", "vault_reader"],  # Core services needed
            optional_services=["graph_database", "metrics"],  # Nice to have
            configuration_schema={
                "type": "object",
                "properties": {
                    "llm_provider": {"type": "string", "enum": ["ollama", "huggingface"]},
                    "models": {"type": "array", "items": {"type": "string"}},
                    "max_memory_gb": {"type": "integer", "minimum": 1},
                    "timeout_seconds": {"type": "integer", "minimum": 5},
                    "graphrag_enabled": {"type": "boolean"},
                    "workflows_enabled": {"type": "boolean"}
                },
                "required": ["llm_provider"]
            }
        )

    async def initialize(self, container: ServiceContainer) -> None:
        """Initialize the AI extension with core services."""
        if self.initialized:
            logger.warning("AI Extension already initialized")
            return

        logger.info("Initializing AI Extension")

        try:
            self.container = container

            # Get settings from container
            settings = self.container.settings

            # Check if AI extension is enabled
            if not settings.ai_extension_enabled:
                raise ExtensionInitializationError(
                    "AI extension is disabled in settings",
                    extension_name="ai"
                )

            # Validate required services are available
            required_services = ["vector_searcher", "vault_reader"]
            for service_name in required_services:
                logger.debug(f"Checking for required service: {service_name}")
                # Note: Full service validation would check container registry

            # Phase 1: Initialize LLM capabilities
            if hasattr(settings, 'ai_llm_provider') and settings.ai_llm_provider:
                logger.info(f"LLM provider configured: {settings.ai_llm_provider}")
                await self._initialize_llm_services(settings)
                self.llm_enabled = True

            # Phase 2: Initialize GraphRAG (placeholder for now)
            if getattr(settings, 'ai_graphrag_enabled', False):
                logger.info("GraphRAG capabilities will be enabled in Phase 2")
                # TODO Phase 2: Initialize GraphRAG service
                self.graphrag_enabled = False  # Not implemented yet

            # Phase 3: Initialize workflows (placeholder for now)
            if getattr(settings, 'ai_workflows_enabled', False):
                logger.info("Workflow capabilities will be enabled in Phase 3")
                # TODO Phase 3: Initialize workflow service
                self.workflows_enabled = False  # Not implemented yet

            self.initialized = True
            logger.info("AI Extension initialized successfully")

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize AI Extension: {e}")
            raise ExtensionInitializationError(
                f"AI extension initialization failed: {e!s}",
                extension_name="ai",
                cause=e
            )

    async def shutdown(self) -> None:
        """Shutdown the AI extension."""
        if not self.initialized:
            logger.debug("AI Extension not initialized, nothing to shutdown")
            return

        logger.info("Shutting down AI Extension")

        try:
            # TODO Phase 3: Shutdown workflow services
            if self.workflows_enabled:
                logger.debug("Shutting down workflow services")
                self.workflows_enabled = False

            # TODO Phase 2: Shutdown GraphRAG services
            if self.graphrag_enabled:
                logger.debug("Shutting down GraphRAG services")
                self.graphrag_enabled = False

            # Phase 1: Shutdown LLM services
            if self.llm_enabled:
                logger.debug("Shutting down LLM services")
                await self._shutdown_llm_services()
                self.llm_enabled = False

            self.initialized = False
            self.container = None

            logger.info("AI Extension shutdown complete")

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error during AI Extension shutdown: {e}")

    def get_tools(self) -> list[MCPTool]:
        """Return MCP tools provided by the AI extension."""
        tools = []

        # Phase 0: Placeholder tool to verify extension system works
        tools.append(MCPTool(
            name="ai-test",
            description="Test tool to verify AI extension is working",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Test message"}
                },
                "required": ["message"]
            },
            handler=self._handle_ai_test
        ))

        # Phase 1: Add LLM tools
        if self.llm_enabled and self.llm_service:
            tools.extend(self._get_llm_tools())

        # TODO Phase 2: Add GraphRAG tools
        if self.graphrag_enabled:
            # tools.append(graphrag_search_tool)
            pass

        # TODO Phase 3: Add workflow tools
        if self.workflows_enabled:
            # tools.append(workflow_execute_tool)
            pass

        return tools

    def get_health_status(self) -> ExtensionHealth:
        """Return current health status of the AI extension."""
        if not self.initialized:
            return ExtensionHealth(
                status=ExtensionStatus.INACTIVE,
                message="AI extension not initialized"
            )

        if self.last_error:
            return ExtensionHealth(
                status=ExtensionStatus.ERROR,
                message="AI extension has errors",
                error_details=self.last_error,
                last_check=time.time()
            )

        # Check component health
        components_healthy = True
        resource_usage = {}

        # Phase 1: Check LLM service health
        if self.llm_enabled and self.llm_service:
            try:
                health_info = await self.llm_service.health_check()
                llm_healthy = health_info.get("status") == "healthy"
                components_healthy &= llm_healthy

                # Get resource usage
                resource_info = await self.llm_service.get_resource_usage()
                resource_usage.update(resource_info)
            except Exception as e:
                logger.error(f"LLM health check failed: {e}")
                components_healthy = False
                resource_usage["llm_error"] = str(e)

        # TODO Phase 2: Check GraphRAG health
        if self.graphrag_enabled:
            # graphrag_healthy = check_graphrag_health()
            # components_healthy &= graphrag_healthy
            pass

        status = ExtensionStatus.ACTIVE if components_healthy else ExtensionStatus.ERROR

        return ExtensionHealth(
            status=status,
            message="AI extension operational" if components_healthy else "Some AI components have issues",
            last_check=time.time(),
            dependencies_healthy=components_healthy,
            resource_usage=resource_usage
        )

    async def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Handle tool calls from the MCP server."""
        if not self.initialized:
            raise ExtensionToolError(
                "AI extension not initialized",
                tool_name=tool_name,
                extension_name="ai"
            )

        try:
            if tool_name == "ai-test":
                return await self._handle_ai_test(arguments)
            # Phase 1: LLM tool handlers
            elif tool_name == "llm-summarize":
                return await self._handle_llm_summarize(arguments)
            elif tool_name == "llm-analyze":
                return await self._handle_llm_analyze(arguments)
            elif tool_name == "llm-quick-answer":
                return await self._handle_llm_quick_answer(arguments)
            else:
                raise ExtensionToolError(
                    f"Unknown tool: {tool_name}",
                    tool_name=tool_name,
                    extension_name="ai"
                )

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"AI tool {tool_name} failed: {e}")
            raise ExtensionToolError(
                f"Tool execution failed: {e!s}",
                tool_name=tool_name,
                extension_name="ai",
                cause=e
            )

    async def _handle_ai_test(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Handle the ai-test tool call."""
        message = arguments.get("message", "No message provided")

        response = f"AI Extension Test Response: {message}\n\n"
        response += "Extension Status:\n"
        response += f"- Initialized: {self.initialized}\n"
        response += f"- LLM Enabled: {self.llm_enabled}\n"
        response += f"- GraphRAG Enabled: {self.graphrag_enabled}\n"
        response += f"- Workflows Enabled: {self.workflows_enabled}\n"

        if self.container:
            response += "- Container Available: True\n"
            response += f"- Settings Available: {hasattr(self.container, 'settings')}\n"
        else:
            response += "- Container Available: False\n"

        return [types.TextContent(type="text", text=response)]

    async def _initialize_llm_services(self, settings) -> None:
        """Initialize LLM services for Phase 1."""
        logger.info("Initializing LLM services")

        try:
            # Create LLM provider configuration
            provider_config = LLMProviderConfig(
                provider_name=settings.ai_llm_provider,
                timeout_seconds=settings.ai_timeout_seconds,
                max_concurrent_requests=2,
                max_memory_gb=settings.ai_max_memory_gb
            )

            # Create model routing configuration
            routing_config = ModelRoutingConfig(
                strategy="performance",
                max_response_time_ms=settings.ai_timeout_seconds * 1000,
                min_success_rate=0.9
            )

            # Initialize model router
            self.model_router = ModelRouter(routing_config, provider_config)

            # Initialize LLM service based on provider
            if settings.ai_llm_provider == "ollama":
                self.llm_service = OllamaClient(
                    base_url="http://localhost:11434",
                    timeout_seconds=settings.ai_timeout_seconds,
                    max_retries=3
                )
            else:
                raise ExtensionInitializationError(
                    f"Unsupported LLM provider: {settings.ai_llm_provider}",
                    extension_name="ai"
                )

            # Initialize prompt manager
            self.prompt_manager = get_template_manager()

            # Health check
            health_info = await self.llm_service.health_check()
            if health_info.get("status") != "healthy":
                logger.warning(f"LLM service health check failed: {health_info}")

            logger.info("LLM services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LLM services: {e}")
            raise ExtensionInitializationError(
                f"LLM service initialization failed: {e!s}",
                extension_name="ai",
                cause=e
            )

    async def _shutdown_llm_services(self) -> None:
        """Shutdown LLM services."""
        logger.info("Shutting down LLM services")

        if self.llm_service:
            try:
                if hasattr(self.llm_service, 'close'):
                    await self.llm_service.close()
            except Exception as e:
                logger.error(f"Error closing LLM service: {e}")

        self.llm_service = None
        self.model_router = None
        self.prompt_manager = None

    def _get_llm_tools(self) -> list[MCPTool]:
        """Get LLM-specific MCP tools."""
        tools = []

        # LLM Summarize tool
        tools.append(MCPTool(
            name="llm-summarize",
            description="Summarize text using LLM",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "style": {"type": "string", "enum": ["bullet", "paragraph", "outline"], "default": "bullet"},
                    "max_length": {"type": "integer", "description": "Maximum length in words"}
                },
                "required": ["text"]
            },
            handler=self._handle_llm_summarize
        ))

        # LLM Analyze tool
        tools.append(MCPTool(
            name="llm-analyze",
            description="Analyze content and answer questions using LLM",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "array", "items": {"type": "string"}, "description": "Content to analyze"},
                    "question": {"type": "string", "description": "Question to answer"},
                    "analysis_type": {"type": "string", "enum": ["general", "sentiment", "topics"], "default": "general"}
                },
                "required": ["content", "question"]
            },
            handler=self._handle_llm_analyze
        ))

        # LLM Quick Answer tool
        tools.append(MCPTool(
            name="llm-quick-answer",
            description="Get quick answers to questions using LLM",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question to answer"},
                    "context": {"type": "string", "description": "Optional context"}
                },
                "required": ["question"]
            },
            handler=self._handle_llm_quick_answer
        ))

        return tools

    async def _handle_llm_summarize(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Handle LLM summarize tool call."""
        if not self.llm_service:
            raise ExtensionToolError("LLM service not initialized", "llm-summarize", "ai")

        text = arguments.get("text", "")
        style = arguments.get("style", "bullet")
        max_length = arguments.get("max_length")

        try:
            response = await self.llm_service.summarize(text, style, max_length)

            result = f"**Summary ({style} style):**\n\n{response.text}"
            if response.response_time_ms:
                result += f"\n\n*Response time: {response.response_time_ms:.0f}ms*"

            return [types.TextContent(type="text", text=result)]

        except LLMError as e:
            error_msg = f"LLM summarization failed: {e!s}"
            return [types.TextContent(type="text", text=error_msg)]

    async def _handle_llm_analyze(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Handle LLM analyze tool call."""
        if not self.llm_service:
            raise ExtensionToolError("LLM service not initialized", "llm-analyze", "ai")

        content = arguments.get("content", [])
        question = arguments.get("question", "")
        analysis_type = arguments.get("analysis_type", "general")

        try:
            result = await self.llm_service.analyze(content, question, analysis_type)

            response = f"**Analysis ({analysis_type}):**\n\n{result.answer}"

            if result.key_points:
                response += "\n\n**Key Points:**\n"
                for point in result.key_points:
                    response += f"• {point}\n"

            if result.processing_time_ms:
                response += f"\n*Processing time: {result.processing_time_ms:.0f}ms*"

            return [types.TextContent(type="text", text=response)]

        except LLMError as e:
            error_msg = f"LLM analysis failed: {e!s}"
            return [types.TextContent(type="text", text=error_msg)]

    async def _handle_llm_quick_answer(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Handle LLM quick answer tool call."""
        if not self.llm_service:
            raise ExtensionToolError("LLM service not initialized", "llm-quick-answer", "ai")

        question = arguments.get("question", "")
        context = arguments.get("context")

        try:
            response = await self.llm_service.quick_answer(question, context)

            result = f"**Answer:** {response.text}"
            if response.response_time_ms:
                result += f"\n\n*Response time: {response.response_time_ms:.0f}ms*"

            return [types.TextContent(type="text", text=result)]

        except LLMError as e:
            error_msg = f"LLM quick answer failed: {e!s}"
            return [types.TextContent(type="text", text=error_msg)]


# Extension entry point
Extension = AIExtension
