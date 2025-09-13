# Extension Development Guide

*Added in Phase 0 - Extension Foundation*

This guide covers developing extensions for Jarvis Assistant using the plugin architecture introduced in Phase 0. Extensions enable adding new capabilities like AI functionality while maintaining system modularity and reliability.

## Quick Navigation

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Extension Architecture](#extension-architecture)
- [Development Workflow](#development-workflow)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Testing Extensions](#testing-extensions)
- [Deployment](#deployment)

---

## Overview

### What are Extensions?

Extensions are self-contained modules that extend Jarvis functionality through:
- **MCP Tools**: New tools available to Claude Desktop
- **Service Integration**: Access to core services via dependency injection
- **Configuration Management**: Extension-specific settings and validation
- **Health Monitoring**: Status reporting and error handling

### Extension Benefits

✅ **Optional**: Extensions can be enabled/disabled without affecting core functionality  
✅ **Isolated**: Extension failures don't crash the core system  
✅ **Modular**: Clean interfaces and dependency management  
✅ **Testable**: Independent testing and validation  
✅ **Configurable**: Environment-based configuration with validation  

---

## Getting Started

### Prerequisites

- Jarvis Assistant core system installed
- Python 3.11+ with type hints support
- Understanding of async/await patterns
- Familiarity with dependency injection concepts

### Enabling Extensions

```bash
# Enable extension system
export JARVIS_EXTENSIONS_ENABLED=true

# Auto-load specific extensions
export JARVIS_EXTENSIONS_AUTO_LOAD=my-extension
```

### Creating Your First Extension

1. **Create Extension Directory**
```bash
mkdir -p src/jarvis/extensions/my-extension
cd src/jarvis/extensions/my-extension
```

2. **Create Extension Entry Point**
```python
# src/jarvis/extensions/my-extension/main.py
from jarvis.extensions.interfaces import IExtension, ExtensionMetadata, ExtensionHealth, ExtensionStatus, MCPTool
from jarvis.core.container import ServiceContainer
import mcp.types as types

class MyExtension(IExtension):
    def __init__(self):
        self.initialized = False
    
    def get_metadata(self) -> ExtensionMetadata:
        return ExtensionMetadata(
            name="my-extension",
            version="1.0.0",
            description="My custom extension",
            author="Your Name",
            dependencies=[],
            required_services=["vault_reader"],
            optional_services=["metrics"]
        )
    
    async def initialize(self, container: ServiceContainer) -> None:
        self.container = container
        self.vault_reader = container.get(IVaultReader)
        self.initialized = True
    
    async def shutdown(self) -> None:
        self.initialized = False
    
    def get_tools(self) -> List[MCPTool]:
        return [
            MCPTool(
                name="my-tool",
                description="Example tool",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                },
                handler=self._handle_my_tool
            )
        ]
    
    def get_health_status(self) -> ExtensionHealth:
        return ExtensionHealth(
            status=ExtensionStatus.ACTIVE if self.initialized else ExtensionStatus.INACTIVE,
            message="Extension operational" if self.initialized else "Not initialized"
        )
    
    async def handle_tool_call(self, tool_name: str, arguments: dict) -> List[types.TextContent]:
        if tool_name == "my-tool":
            return await self._handle_my_tool(arguments)
        raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _handle_my_tool(self, arguments: dict) -> List[types.TextContent]:
        message = arguments.get("message", "Hello from my extension!")
        return [types.TextContent(type="text", text=f"Extension Response: {message}")]

# Extension entry point
Extension = MyExtension
```

3. **Test Your Extension**
```python
# Test script
from jarvis.extensions import ExtensionManager
from jarvis.core.container import ServiceContainer
from jarvis.utils.config import get_settings

async def test_extension():
    settings = get_settings()
    settings.extensions_enabled = True
    settings.extensions_auto_load = ["my-extension"]
    
    container = ServiceContainer(settings)
    manager = ExtensionManager(settings, container)
    
    async with manager.managed_lifecycle():
        tools = manager.get_all_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        result = await manager.handle_tool_call("my-tool", {"message": "Test!"})
        print(f"Tool result: {result}")

# Run test
import asyncio
asyncio.run(test_extension())
```

---

## Extension Architecture

### Core Interfaces

#### IExtension
Main interface that all extensions must implement:

```python
class IExtension(ABC):
    @abstractmethod
    def get_metadata(self) -> ExtensionMetadata
    
    @abstractmethod
    async def initialize(self, container: ServiceContainer) -> None
    
    @abstractmethod
    async def shutdown(self) -> None
    
    @abstractmethod
    def get_tools(self) -> List[MCPTool]
    
    @abstractmethod
    def get_health_status(self) -> ExtensionHealth
    
    @abstractmethod
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> List[types.TextContent]
```

#### ExtensionMetadata
Describes extension properties and requirements:

```python
@dataclass
class ExtensionMetadata:
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = []          # Other extensions needed
    required_services: List[str] = []     # Core services needed
    optional_services: List[str] = []     # Services that enhance functionality
    configuration_schema: Dict[str, Any] = {}  # JSON schema for config
```

#### MCPTool
Defines tools exposed to Claude Desktop:

```python
@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON schema for parameters
    handler: Callable           # Function to handle tool calls
```

### Dependency Injection

Extensions receive a `ServiceContainer` that provides access to core services:

```python
async def initialize(self, container: ServiceContainer) -> None:
    # Get required services
    self.vault_reader = container.get(IVaultReader)
    self.vector_searcher = container.get(IVectorSearcher)
    
    # Get optional services with fallback
    try:
        self.metrics = container.get(IMetrics)
    except ServiceNotFoundError:
        self.metrics = None
```

### Available Core Services

- **IVaultReader**: Read files from Obsidian vaults
- **IVectorSearcher**: Semantic search capabilities  
- **IGraphDatabase**: Graph database operations (if enabled)
- **IHealthChecker**: System health monitoring
- **IMetrics**: Performance metrics collection

---

## Development Workflow

### 1. Planning Phase

- Define extension purpose and scope
- Identify required and optional services
- Design tool interfaces and schemas
- Plan configuration requirements

### 2. Implementation Phase

```bash
# Create extension structure
mkdir -p src/jarvis/extensions/your-extension/{tools,services,tests}

# Implement core extension
touch src/jarvis/extensions/your-extension/main.py
touch src/jarvis/extensions/your-extension/config.py
touch src/jarvis/extensions/your-extension/tools/your_tool.py
```

### 3. Testing Phase

```python
# Unit tests
pytest src/jarvis/extensions/your-extension/tests/

# Integration tests with core system
pytest resources/tests/integration/test_extensions.py -k your-extension

# Manual testing with Claude Desktop
uv run jarvis mcp --vault /path/to/vault --extensions your-extension
```

### 4. Documentation Phase

- Add tool documentation to extension metadata
- Update configuration examples
- Create usage examples

---

## API Reference

### Extension Lifecycle

```python
# Extension loading order
1. discover_extensions()     # Find available extensions
2. load_extension()         # Import and instantiate
3. initialize()             # Provide dependencies
4. register_extension()     # Add to registry
5. get_tools()             # Collect MCP tools

# Extension shutdown order  
1. unregister_extension()   # Remove from registry
2. shutdown()              # Clean up resources
3. unload_extension()      # Remove from memory
```

### Tool Handler Pattern

```python
async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Route tool calls to appropriate handlers."""
    handlers = {
        "tool-one": self._handle_tool_one,
        "tool-two": self._handle_tool_two,
    }
    
    handler = handlers.get(tool_name)
    if not handler:
        raise ExtensionToolError(f"Unknown tool: {tool_name}", tool_name, self.get_metadata().name)
    
    try:
        return await handler(arguments)
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        raise ExtensionToolError(f"Tool execution failed: {e}", tool_name, self.get_metadata().name, e)
```

### Error Handling

```python
from jarvis.extensions.errors import (
    ExtensionError,
    ExtensionLoadError,
    ExtensionInitializationError,
    ExtensionToolError,
    ExtensionConfigurationError
)

# Use specific error types
if not required_config:
    raise ExtensionConfigurationError("Missing required config", self.name)

if service_unavailable:
    raise ExtensionInitializationError("Required service not available", self.name)
```

---

## Best Practices

### 1. Graceful Degradation

```python
def get_health_status(self) -> ExtensionHealth:
    issues = []
    
    # Check dependencies
    if not self.required_service:
        issues.append("Required service unavailable")
    
    # Check resources
    if self.memory_usage > self.max_memory:
        issues.append(f"High memory usage: {self.memory_usage}MB")
    
    status = ExtensionStatus.ERROR if issues else ExtensionStatus.ACTIVE
    
    return ExtensionHealth(
        status=status,
        message="; ".join(issues) if issues else "Operational",
        dependencies_healthy=len(issues) == 0
    )
```

### 2. Resource Management

```python
async def shutdown(self) -> None:
    """Clean shutdown with resource cleanup."""
    if hasattr(self, 'background_task'):
        self.background_task.cancel()
        
    if hasattr(self, 'http_session'):
        await self.http_session.close()
        
    if hasattr(self, 'file_handles'):
        for handle in self.file_handles:
            handle.close()
    
    self.initialized = False
```

### 3. Configuration Management

```python
def get_configuration_schema(self) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "minLength": 1},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
            "enabled_features": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["basic"]
            }
        },
        "required": ["api_key"],
        "additionalProperties": False
    }

def validate_configuration(self, config: Dict[str, Any]) -> bool:
    # Custom validation beyond JSON schema
    if config.get("timeout", 30) < 5:
        logger.warning("Timeout is very low, may cause issues")
    
    return True
```

### 4. Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

# Use structured logging
logger.info("Extension initialized", extra={
    "extension": self.name,
    "version": self.version,
    "tools_count": len(self.get_tools())
})

# Log errors with context
logger.error("Tool execution failed", extra={
    "tool_name": tool_name,
    "extension": self.name,
    "error": str(e)
}, exc_info=True)
```

---

## Testing Extensions

### Unit Tests

```python
# tests/test_my_extension.py
import pytest
from unittest.mock import Mock, AsyncMock
from jarvis.extensions.my_extension.main import MyExtension

@pytest.fixture
async def extension():
    ext = MyExtension()
    container = Mock()
    container.get.return_value = Mock()  # Mock services
    await ext.initialize(container)
    return ext

@pytest.mark.asyncio
async def test_tool_execution(extension):
    result = await extension.handle_tool_call("my-tool", {"message": "test"})
    assert len(result) == 1
    assert "test" in result[0].text

def test_metadata(extension):
    metadata = extension.get_metadata()
    assert metadata.name == "my-extension"
    assert metadata.version
```

### Integration Tests

```python
# Test with real extension manager
@pytest.mark.asyncio
async def test_extension_lifecycle():
    settings = get_test_settings()
    settings.extensions_enabled = True
    
    container = ServiceContainer(settings)
    manager = ExtensionManager(settings, container)
    
    # Test loading
    extension = await manager.load_extension("my-extension")
    assert manager.is_extension_loaded("my-extension")
    
    # Test tools
    tools = manager.get_all_tools()
    tool_names = [tool.name for tool in tools]
    assert "my-tool" in tool_names
    
    # Test execution
    result = await manager.handle_tool_call("my-tool", {"message": "test"})
    assert result
    
    # Test unloading
    await manager.unload_extension("my-extension")
    assert not manager.is_extension_loaded("my-extension")
```

---

## Deployment

### Development Deployment

```bash
# Enable extensions in development
export JARVIS_EXTENSIONS_ENABLED=true
export JARVIS_EXTENSIONS_AUTO_LOAD=my-extension

# Start with extension
uv run jarvis mcp --vault /path/to/vault --watch
```

### Production Deployment

```bash
# Production configuration
export JARVIS_EXTENSIONS_ENABLED=true
export JARVIS_EXTENSIONS_AUTO_LOAD=ai,production-tools
export JARVIS_AI_EXTENSION_ENABLED=true
export JARVIS_AI_MAX_MEMORY_GB=16

# Run with health monitoring
uv run jarvis mcp --vault /path/to/vault
```

### Extension Distribution

1. **Package Structure**
```
my-extension/
├── main.py              # Extension entry point
├── config.py            # Configuration helpers
├── tools/               # Tool implementations
├── services/            # Extension services
├── tests/               # Test suite
├── README.md            # Usage documentation
└── requirements.txt     # Additional dependencies
```

2. **Installation Script**
```bash
#!/bin/bash
# install-extension.sh
EXTENSION_NAME="my-extension"
EXTENSION_DIR="src/jarvis/extensions/$EXTENSION_NAME"

# Copy extension files
cp -r $EXTENSION_NAME $EXTENSION_DIR

# Install dependencies
if [ -f "$EXTENSION_NAME/requirements.txt" ]; then
    uv add -r $EXTENSION_NAME/requirements.txt
fi

echo "Extension $EXTENSION_NAME installed successfully"
echo "Enable with: export JARVIS_EXTENSIONS_AUTO_LOAD=$EXTENSION_NAME"
```

---

## Advanced Topics

### Creating Extension Services

```python
# my-extension/services/my_service.py
class MyExtensionService:
    def __init__(self, vault_reader, config):
        self.vault_reader = vault_reader
        self.config = config
    
    async def process_data(self, data):
        # Extension-specific logic
        return processed_data

# In main.py
async def initialize(self, container: ServiceContainer) -> None:
    vault_reader = container.get(IVaultReader)
    config = self.get_config()
    
    self.my_service = MyExtensionService(vault_reader, config)
```

### Inter-Extension Communication

```python
# Extension A
class ExtensionA(IExtension):
    def provide_data(self):
        return {"key": "value"}

# Extension B  
class ExtensionB(IExtension):
    async def initialize(self, container: ServiceContainer) -> None:
        # Access another extension via registry
        self.extension_a = container.get_extension("extension-a")
        
    def use_data(self):
        if self.extension_a:
            data = self.extension_a.provide_data()
            return self.process(data)
```

### Custom Configuration

```python
# my-extension/config.py
from jarvis.utils.config import JarvisSettings

def get_extension_config(settings: JarvisSettings) -> dict:
    return settings.extensions_config.get("my-extension", {})

def validate_extension_config(config: dict) -> bool:
    required_keys = ["api_key", "endpoint"]
    return all(key in config for key in required_keys)
```

---

## Next Steps

- [Testing Strategy](testing-strategy.md) - Comprehensive testing guide
- [Configuration Reference](../06-reference/configuration-reference.md) - All configuration options
- [API Reference](../06-reference/api-reference.md) - Complete API documentation
- [AI Implementation Roadmap](../01-overview/ai-implementation-roadmap.md) - AI extension development
