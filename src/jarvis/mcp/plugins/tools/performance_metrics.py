"""
Performance Metrics Plugin for MCP Tools.

This plugin provides performance monitoring and statistics
for MCP tools and system services.
"""

import json
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IMetrics
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.mcp.structured import performance_metrics_to_json
from jarvis.utils.errors import PluginError

# Lazy import to avoid circular dependencies
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class PerformanceMetricsPlugin(UtilityPlugin):
    """Plugin for performance metrics monitoring."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "get-performance-metrics"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Get performance metrics and statistics for MCP tools and services"

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
        return ["metrics", "performance", "monitoring", "statistics"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IMetrics]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import create_metrics_schema

        # Create standardized metrics schema
        input_schema = create_metrics_schema()

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
        """Execute performance metrics collection."""
        filter_prefix = arguments.get("filter_prefix")
        reset_after_read = arguments.get("reset_after_read", False)

        try:
            if not self.container:
                raise PluginError("Service container not available")

            metrics_service = self.container.get(IMetrics)
            if not metrics_service:
                return [types.TextContent(
                    type="text",
                    text="📊 Performance metrics are not enabled or available."
                )]

            # Get all metrics
            all_metrics = metrics_service.get_metrics()

            # Filter metrics if prefix specified
            if filter_prefix:
                filtered_metrics = {
                    key: value for key, value in all_metrics.items()
                    if key.startswith(filter_prefix)
                }
            else:
                filtered_metrics = all_metrics

            response_lines = ["# 📊 Performance Metrics\n"]

            if not filtered_metrics:
                filter_msg = f" (filtered by '{filter_prefix}')" if filter_prefix else ""
                response_lines.append(f"No metrics available{filter_msg}.")
                return [types.TextContent(type="text", text="\n".join(response_lines))]

            # Group metrics by category
            mcp_tool_metrics = {}
            system_metrics = {}
            other_metrics = {}

            for key, value in filtered_metrics.items():
                if key.startswith('mcp_tool_'):
                    mcp_tool_metrics[key] = value
                elif key.startswith('system_') or key.startswith('service_'):
                    system_metrics[key] = value
                else:
                    other_metrics[key] = value

            # Display MCP tool metrics
            if mcp_tool_metrics:
                response_lines.append("## 🔧 MCP Tool Metrics")
                for metric_name, metric_value in sorted(mcp_tool_metrics.items()):
                    display_name = metric_name.replace('mcp_tool_', '').replace('_', ' ').title()
                    response_lines.append(f"- **{display_name}:** {metric_value}")
                response_lines.append("")

            # Display system metrics
            if system_metrics:
                response_lines.append("## ⚙️ System Metrics")
                for metric_name, metric_value in sorted(system_metrics.items()):
                    display_name = metric_name.replace('system_', '').replace('service_', '').replace('_', ' ').title()
                    response_lines.append(f"- **{display_name}:** {metric_value}")
                response_lines.append("")

            # Display other metrics
            if other_metrics:
                response_lines.append("## 📈 Other Metrics")
                for metric_name, metric_value in sorted(other_metrics.items()):
                    display_name = metric_name.replace('_', ' ').title()
                    response_lines.append(f"- **{display_name}:** {metric_value}")
                response_lines.append("")

            # Add summary statistics
            total_metrics = len(filtered_metrics)
            response_lines.append("## 📋 Summary")
            response_lines.append(f"- **Total Metrics:** {total_metrics}")
            if filter_prefix:
                response_lines.append(f"- **Filter Applied:** `{filter_prefix}`")
            if reset_after_read:
                response_lines.append("- **Metrics Reset:** Yes (after this reading)")

            # Reset metrics if requested
            if reset_after_read:
                try:
                    metrics_service.reset_metrics()
                    response_lines.append("\n*Metrics have been reset.*")
                except Exception as e:
                    logger.warning(f"Failed to reset metrics: {e}")
                    response_lines.append(f"\n*Warning: Failed to reset metrics: {e}*")

            payload = performance_metrics_to_json(all_metrics, filter_prefix, reset_after_read)
            payload["correlation_id"] = str(uuid4())
            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return [types.TextContent(
                type="text",
                text=f"❌ Performance metrics retrieval failed: {e!s}"
            )]
