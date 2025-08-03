"""
Read Note Plugin for MCP Tools.

This plugin provides the ability to read the content of specific notes
from vaults with metadata information.
"""

from typing import Dict, Any, List, Type

from mcp import types
from jarvis.mcp.plugins.base import VaultPlugin
from jarvis.core.interfaces import IVaultReader
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import PluginError, ServiceError

logger = setup_logging(__name__)


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
    def tags(self) -> List[str]:
        """Get plugin tags."""
        return ["vault", "files", "read", "content"]
    
    def get_required_services(self) -> List[Type]:
        """Get required service interfaces."""
        return [IVaultReader]
    
    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition."""
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the note relative to vault root"
                    },
                    "vault": {
                        "type": "string",
                        "description": "Vault name (uses first available if not specified)"
                    }
                },
                "required": ["path"]
            }
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute note reading."""
        path = arguments.get("path", "").strip()
        vault_name = arguments.get("vault")
        
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
                    text=f"Error reading note '{path}': {str(e)}"
                )]
            
            # Format response with metadata
            response_lines = [
                f"# {metadata.get('path', path)}",
                "",
                f"**Size:** {metadata.get('size', 0)} bytes",
                f"**Modified:** {metadata.get('modified_formatted', 'Unknown')}",
                "",
                "---",
                "",
                content
            ]
            
            response = "\n".join(response_lines)
            
            return [types.TextContent(type="text", text=response)]
            
        except ServiceError as e:
            logger.error(f"Error reading note {path}: {e}")
            return [types.TextContent(
                type="text", 
                text=f"Error reading note: {str(e)}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error reading note {path}: {e}")
            raise PluginError(f"Note reading failed: {str(e)}") from e