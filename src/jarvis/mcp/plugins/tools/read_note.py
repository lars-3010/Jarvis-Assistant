"""
Read Note Plugin for MCP Tools.

This plugin provides the ability to read the content of specific notes
from vaults with metadata information.
"""

import json
from typing import Any

from jarvis.core.interfaces import IVaultReader
from jarvis.mcp.plugins.base import VaultPlugin

# Lazy import to avoid circular dependencies
from jarvis.mcp.structured import read_note_to_json
from jarvis.utils.errors import PluginError, ServiceError
import logging
from mcp import types

logger = logging.getLogger(__name__)


class ReadNotePlugin(VaultPlugin):
    """Plugin for reading note content from vaults."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "read-note"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Read the content of a specific note from a vault"

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
        return ["vault", "files", "read", "content"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVaultReader]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import VaultSchemaConfig, create_vault_schema

        # Create standardized vault schema with custom configuration
        schema_config = VaultSchemaConfig(
            path_required=True,
            enable_vault_selection=True,
            supported_formats=["json"]
        )

        # Generate standardized schema
        input_schema = create_vault_schema(schema_config)

        # Customize path description for reading notes
        input_schema["properties"]["path"]["description"] = "Path to the note relative to vault root"

        # Customize vault description
        input_schema["properties"]["vault"]["description"] = "Vault name (uses first available if not specified)"

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute note reading."""
        path = arguments.get("path", "").strip()
        vault_name = arguments.get("vault")
        output_format = arguments.get("format", "json")

        # Validate input
        if not path:
            return [types.TextContent(
                type="text",
                text="Error: Path parameter is required and cannot be empty"
            )]

        try:
            # Get vault reader service
            if not self.container:
                raise PluginError("Service container not available")

            # For now, get the primary vault reader
            # In a future enhancement, we could support multiple vault readers
            vault_reader = self.container.get(IVaultReader)
            if not vault_reader:
                raise PluginError("Vault reader service not available")

            # Note: The current IVaultReader interface doesn't support vault selection
            # This is a limitation that could be addressed in future versions
            if vault_name:
                logger.warning(f"Vault selection '{vault_name}' requested but not supported by current interface")

            # Read the file
            try:
                content, metadata = vault_reader.read_file(path)
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error reading note '{path}': {e!s}"
                )]

            # Return JSON format if requested
            if output_format == "json":
                # Format last_modified timestamp
                last_modified = None
                if 'modified_formatted' in metadata:
                    last_modified = metadata['modified_formatted']
                elif 'modified' in metadata:
                    try:
                        from datetime import datetime
                        last_modified = datetime.fromtimestamp(metadata['modified']).isoformat()
                    except:
                        pass

                json_data = read_note_to_json(
                    path=path,
                    content=content,
                    vault_name=vault_name,
                    size_bytes=metadata.get('size', len(content.encode('utf-8'))),
                    last_modified=last_modified,
                    metadata=metadata,
                )
                return [types.TextContent(type="text", text=json.dumps(json_data, indent=2))]

            # Fallback to JSON even if markdown requested (schema is JSON-only)
            last_modified = None
            if 'modified_formatted' in metadata:
                last_modified = metadata['modified_formatted']
            elif 'modified' in metadata:
                try:
                    from datetime import datetime
                    last_modified = datetime.fromtimestamp(metadata['modified']).isoformat()
                except:
                    pass

            json_data = read_note_to_json(
                path=path,
                content=content,
                vault_name=vault_name,
                size_bytes=metadata.get('size', len(content.encode('utf-8'))),
                last_modified=last_modified,
                metadata=metadata,
            )
            return [types.TextContent(type="text", text=json.dumps(json_data, indent=2))]

        except ServiceError as e:
            logger.error(f"Error reading note {path}: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error reading note: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error reading note {path}: {e}")
            raise PluginError(f"Note reading failed: {e!s}") from e
