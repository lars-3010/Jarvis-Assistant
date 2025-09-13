"""
Health Status Plugin for MCP Tools.

This plugin provides system health monitoring capabilities
for all Jarvis Assistant services and components.
"""

import json
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IHealthChecker
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.mcp.structured import health_status_to_json
from jarvis.utils.errors import PluginError

# Lazy import to avoid circular dependencies
import logging
from mcp import types

logger = logging.getLogger(__name__)


class HealthStatusPlugin(UtilityPlugin):
    """Plugin for system health monitoring."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "get-health-status"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Get the health status of all Jarvis Assistant services"

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
        return ["health", "monitoring", "system", "diagnostics"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IHealthChecker]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import create_health_schema

        # Create standardized health check schema
        input_schema = create_health_schema()

        # Keep `format` field but restrict to JSON for uniformity
        if "format" in input_schema["properties"]:
            input_schema["properties"]["format"] = {
                "type": "string",
                "enum": ["json"],
                "default": "json",
                "description": "Response format"
            }

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute health status check."""
        try:
            if not self.container:
                raise PluginError("Service container not available")

            health_checker = self.container.get(IHealthChecker)
            if not health_checker:
                raise PluginError("Health checker service not available")

            # Get overall health status
            health_status = health_checker.get_overall_health()

            response_lines = ["# 🏥 Jarvis Assistant Health Status\n"]

            # Overall status
            overall_status = health_status.get('overall_status', 'unknown')
            status_emoji = "✅" if overall_status == "healthy" else "❌" if overall_status == "unhealthy" else "⚠️"
            response_lines.append(f"**Overall Status:** {status_emoji} {overall_status.upper()}\n")

            # Individual service status
            services = health_status.get('services', {})
            if services:
                response_lines.append("## Service Status")
                for service_name, status in services.items():
                    emoji = "✅" if status else "❌"
                    response_lines.append(f"- **{service_name}:** {emoji} {'Healthy' if status else 'Unhealthy'}")
                response_lines.append("")

            # Database status
            databases = health_status.get('databases', {})
            if databases:
                response_lines.append("## Database Status")
                for db_name, status in databases.items():
                    emoji = "✅" if status else "❌"
                    response_lines.append(f"- **{db_name}:** {emoji} {'Connected' if status else 'Disconnected'}")
                response_lines.append("")

            # Vault status
            vaults = health_status.get('vaults', {})
            if vaults:
                response_lines.append("## Vault Status")
                for vault_name, status in vaults.items():
                    emoji = "✅" if status else "❌"
                    response_lines.append(f"- **{vault_name}:** {emoji} {'Accessible' if status else 'Inaccessible'}")
                response_lines.append("")

            # Additional health metrics
            metrics = health_status.get('metrics', {})
            if metrics:
                response_lines.append("## Health Metrics")
                for metric_name, value in metrics.items():
                    response_lines.append(f"- **{metric_name}:** {value}")
                response_lines.append("")

            # Timestamp
            timestamp = health_status.get('timestamp')
            if timestamp:
                response_lines.append(f"*Health check performed at: {timestamp}*")

            payload = health_status_to_json(health_status)
            payload["correlation_id"] = str(uuid4())
            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except Exception as e:
            logger.error(f"Health status check error: {e}")
            return [types.TextContent(
                type="text",
                text=f"❌ Health status check failed: {e!s}"
            )]
