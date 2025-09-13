"""
List Vaults Plugin for MCP Tools.

This plugin provides the ability to list all available vaults
with their statistics and status information.
"""

import json
import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IVectorSearcher
from jarvis.mcp.plugins.base import UtilityPlugin
from jarvis.mcp.schemas import UtilitySchemaConfig, create_utility_schema
from jarvis.mcp.structured import list_vaults_to_json
from jarvis.utils.errors import PluginError, ServiceError
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class ListVaultsPlugin(UtilityPlugin):
    """Plugin for listing available vaults and their statistics."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "list-vaults"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "List all available vaults and their statistics"

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
        return ["vault", "management", "statistics", "utility"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVectorSearcher]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        schema_config = UtilitySchemaConfig(
            supported_formats=["json"],
            additional_properties={},
        )
        input_schema = create_utility_schema(schema_config)

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema,
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute vault listing."""
        try:
            # Get vector searcher service
            if not self.container:
                raise PluginError("Service container not available")

            searcher = self.container.get(IVectorSearcher)
            if not searcher:
                raise PluginError("Vector searcher service not available")

            start_time = time.time()

            # Get vault statistics and validation
            vault_stats = searcher.get_vault_stats()
            validation = searcher.validate_vaults()

            output_format = arguments.get("format", "json").lower()
            response_lines = ["# Available Vaults\n"]

            # Note: The current interface doesn't expose vault paths directly
            # We'll work with what's available from the searcher

            # If we have vault stats, show them
            if vault_stats:
                for vault_name, stats in vault_stats.items():
                    is_valid = validation.get(vault_name, False)

                    status = "✅ Available" if is_valid else "❌ Unavailable"
                    note_count = stats.get('note_count', 0)

                    response_lines.append(f"## {vault_name}")
                    response_lines.append(f"- **Status:** {status}")
                    response_lines.append(f"- **Notes:** {note_count}")

                    if stats.get('latest_modified'):
                        try:
                            latest = datetime.fromtimestamp(stats['latest_modified']).isoformat()
                            response_lines.append(f"- **Last Modified:** {latest}")
                        except (ValueError, OSError) as e:
                            logger.warning(f"Invalid timestamp for vault {vault_name}: {e}")

                    response_lines.append("")
            else:
                response_lines.append("No vault statistics available.")
                response_lines.append("")

            # Add model and system information
            try:
                model_info = searcher.get_model_info()
                response_lines.append("## Search Configuration")

                encoder_info = model_info.get('encoder_info', {})
                response_lines.append(f"- **Model:** {encoder_info.get('model_name', 'Unknown')}")
                response_lines.append(f"- **Device:** {encoder_info.get('device', 'Unknown')}")
                response_lines.append(f"- **Total Notes:** {model_info.get('database_note_count', 0)}")

                # Add additional system stats if available
                search_stats = searcher.get_search_stats()
                if search_stats:
                    response_lines.append(f"- **Total Searches:** {search_stats.get('total_searches', 0)}")
                    response_lines.append(f"- **Average Response Time:** {search_stats.get('avg_response_time_ms', 0):.2f}ms")

            except Exception as e:
                logger.warning(f"Could not retrieve model info: {e}")
                response_lines.append("## Search Configuration")
                response_lines.append("- Model information unavailable")

            payload = list_vaults_to_json(
                vault_stats,
                validation,
                model_info if 'model_info' in locals() else None,
                search_stats if 'search_stats' in locals() else None,
            )
            payload["correlation_id"] = str(uuid4())

            if output_format == "json":
                return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]
            else:
                return [types.TextContent(type="text", text="\n".join(response_lines))]

        except ServiceError as e:
            logger.error(f"Error listing vaults: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error listing vaults: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error listing vaults: {e}")
            raise PluginError(f"Vault listing failed: {e!s}") from e
