"""
Analytics Cache Invalidation Plugin for MCP Tools.

This plugin provides functionality to invalidate analytics cache entries
for specific vaults or globally, triggering fresh analysis on next request.
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


class AnalyticsInvalidateCachePlugin(UtilityPlugin):
    """Plugin for invalidating analytics cache."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "analytics-invalidate-cache"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Invalidate analytics cache entries to force fresh analysis on next request"

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
        return ["analytics", "cache", "invalidation", "utility"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVaultAnalyticsService]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import UtilitySchemaConfig, create_utility_schema

        # Create standardized utility schema (add vault_name and debug via additional_properties)
        schema_config = UtilitySchemaConfig(
            supported_formats=["json", "markdown"],
            additional_properties={
                "vault_name": {
                    "type": "string",
                    "description": "Name of the vault to invalidate cache for. If not provided, invalidates all vaults",
                    "default": None
                },
                "debug": {
                    "type": "boolean",
                    "description": "Include detailed information about the invalidation process",
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
        """Execute analytics cache invalidation."""
        vault_name = arguments.get("vault_name")
        output_format = arguments.get("format", "markdown")
        debug = arguments.get("debug", False)

        start_time = time.time()

        try:
            # Get analytics service
            analytics_service = self.container.get(IVaultAnalyticsService) if self.container else None
            if not analytics_service:
                raise PluginError("Analytics service not available")

            # Publish event before invalidation
            from jarvis.core.events import publish_event
            await publish_event(
                "analytics.cache_invalidation_requested",
                {
                    "vault_name": vault_name,
                    "invalidation_scope": "selective" if vault_name else "global",
                    "timestamp": start_time,
                    "user_request": True
                },
                source="analytics_cache_invalidation"
            )

            # Perform cache invalidation
            success = await analytics_service.invalidate_cache(vault_name)

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Publish completion event
            await publish_event(
                "analytics.cache_invalidation_completed",
                {
                    "vault_name": vault_name,
                    "invalidation_scope": "selective" if vault_name else "global",
                    "success": success,
                    "processing_time_ms": execution_time_ms,
                    "timestamp": time.time()
                },
                source="analytics_cache_invalidation"
            )

            # Return JSON format if requested
            if output_format == "json":
                json_data = {
                    "invalidation_success": success,
                    "vault_name": vault_name,
                    "scope": "selective" if vault_name else "global",
                    "execution_time_ms": execution_time_ms,
                    "debug": debug,
                    "timestamp": time.time()
                }

                if debug:
                    json_data["debug_info"] = {
                        "service_available": analytics_service is not None,
                        "container_available": self.container is not None,
                        "invalidation_type": "selective" if vault_name else "global"
                    }

                return [types.TextContent(type="text", text=json.dumps(json_data, indent=2))]

            # Default markdown format
            scope_description = f"vault '{vault_name}'" if vault_name else "all vaults"
            status_emoji = "✅" if success else "❌"
            status_text = "successful" if success else "failed"

            response_lines = [
                "## 🗑️ Analytics Cache Invalidation\n",
                f"{status_emoji} **Status**: Cache invalidation {status_text}",
                f"**Scope**: {scope_description}",
                f"**Execution Time**: {execution_time_ms}ms",
                ""
            ]

            if success:
                response_lines.extend([
                    "### What happens next?",
                    "- Next analytics requests will perform fresh analysis",
                    "- Cache will be rebuilt automatically on demand",
                    "- This may result in slower response times initially",
                    ""
                ])
            else:
                response_lines.extend([
                    "### Troubleshooting:",
                    "- Verify the analytics service is running",
                    "- Check if the vault name is correct (if specified)",
                    "- Ensure no other processes are accessing the cache",
                    ""
                ])

            # Add debug information if requested
            if debug:
                response_lines.extend([
                    "### Debug Information:",
                    f"- **Execution Time**: {execution_time_ms}ms",
                    f"- **Analytics Service**: {'Available' if analytics_service else 'Unavailable'}",
                    f"- **Container**: {'Available' if self.container else 'Unavailable'}",
                    f"- **Invalidation Type**: {'Selective' if vault_name else 'Global'}",
                    f"- **Timestamp**: {time.time():.3f}",
                    ""
                ])

            return [types.TextContent(type="text", text="\n".join(response_lines))]

        except ServiceError as e:
            logger.error(f"Analytics cache invalidation error: {e}")
            return [types.TextContent(
                type="text",
                text=f"Cache invalidation error: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error in analytics cache invalidation: {e}")
            raise PluginError(f"Cache invalidation failed: {e!s}") from e
