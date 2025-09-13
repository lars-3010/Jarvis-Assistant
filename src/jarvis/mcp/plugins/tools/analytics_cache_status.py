"""
Analytics Cache Status Plugin for MCP Tools.

This plugin provides information about the analytics cache status,
including hit rates, entry counts, and freshness indicators.
"""

import json
import time
from typing import Any

from jarvis.core.interfaces import IVaultAnalyticsService
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.utils.errors import PluginError, ServiceError
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class AnalyticsCacheStatusPlugin(UtilityPlugin):
    """Plugin for checking analytics cache status."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "analytics-cache-status"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Get current analytics cache status and performance metrics"

    @property
    def version(self) -> str:
        """Get the plugin version."""
        return "1.0.0"

    @property
    def author(self) -> str:
        """Get the plugin author."""
        return "Jarvis Assistant"

    @property
    def tags(self) -> list[str]:
        """Get plugin tags."""
        return ["analytics", "cache", "monitoring", "utility"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVaultAnalyticsService]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import UtilitySchemaConfig, create_utility_schema

        # Create standardized utility schema (add debug flag via additional_properties)
        schema_config = UtilitySchemaConfig(
            supported_formats=["json"],
            additional_properties={
                "debug": {
                    "type": "boolean",
                    "description": "Include detailed cache statistics and debugging information",
                    "default": False
                }
            },
        )

        input_schema = create_utility_schema(schema_config)

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute analytics cache status check."""
        output_format = arguments.get("format", "json")
        debug = arguments.get("debug", False)

        start_time = time.time()

        try:
            # Get analytics service
            analytics_service = self.container.get(IVaultAnalyticsService) if self.container else None
            if not analytics_service:
                raise PluginError("Analytics service not available")

            # Get cache status
            cache_status = await analytics_service.get_analytics_cache_status()

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Return JSON format if requested
            if output_format == "json":
                json_data = {
                    "cache_status": cache_status,
                    "execution_time_ms": execution_time_ms,
                    "debug": debug,
                    "timestamp": time.time()
                }

                if debug:
                    json_data["debug_info"] = {
                        "service_available": analytics_service is not None,
                        "container_available": self.container is not None
                    }

                return [types.TextContent(type="text", text=json.dumps(json_data, indent=2))]

            # Default markdown format
            response_lines = [
                "## 📊 Analytics Cache Status\n",
                f"**Cache Hit Rate**: {cache_status.get('cache_hit_rate', 0):.1%}",
                f"**Total Cache Entries**: {cache_status.get('total_entries', 0):,}",
                f"**Cache Size (MB)**: {cache_status.get('total_size_mb', 0):.1f}",
                f"**Last Updated**: {cache_status.get('last_updated_str', 'Unknown')}",
                ""
            ]

            # Add level-specific information
            levels = cache_status.get('levels', {})
            if levels:
                response_lines.append("### Cache Levels:")
                for level, info in levels.items():
                    response_lines.append(f"- **Level {level}**: {info.get('entry_count', 0)} entries, {info.get('hit_rate', 0):.1%} hit rate")
                response_lines.append("")

            # Add freshness information
            if cache_status.get('freshness_indicators'):
                response_lines.append("### Freshness Indicators:")
                freshness = cache_status['freshness_indicators']
                for indicator, value in freshness.items():
                    response_lines.append(f"- **{indicator.replace('_', ' ').title()}**: {value}")
                response_lines.append("")

            # Add debug information if requested
            if debug:
                response_lines.extend([
                    "### Debug Information:",
                    f"- **Execution Time**: {execution_time_ms}ms",
                    f"- **Analytics Service**: {'Available' if analytics_service else 'Unavailable'}",
                    f"- **Container**: {'Available' if self.container else 'Unavailable'}",
                    f"- **Timestamp**: {time.time():.3f}",
                    ""
                ])

            return [types.TextContent(type="text", text="\n".join(response_lines))]

        except ServiceError as e:
            logger.error(f"Analytics cache status error: {e}")
            return [types.TextContent(
                type="text",
                text=f"Cache status error: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error in analytics cache status: {e}")
            raise PluginError(f"Cache status check failed: {e!s}") from e
