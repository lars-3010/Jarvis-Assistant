# MCP Integration

## Model Context Protocol Implementation and Tool Architecture

## MCP Protocol Overview

The Model Context Protocol (MCP) is a standardized communication protocol that enables AI systems like Claude Desktop to interact with external tools and data sources through a well-defined interface.

## Protocol Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Claude        │     │   MCP Server    │     │   Tool          │
│   Desktop       │     │   (Jarvis)      │     │   Implementation │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                        │                        │
         │ JSON-RPC over stdio    │                        │
         │◄──────────────────────►│                        │
         │                        │ Function calls         │
         │                        │◄──────────────────────►│
         │                        │                        │
         │ Tool responses         │                        │
         │◄──────────────────────►│                        │
```

### Communication Flow

1. **Initialization**: Claude Desktop starts MCP server process
2. **Discovery**: Server advertises available tools and capabilities
3. **Tool Invocation**: Claude sends tool requests via JSON-RPC
4. **Processing**: Server routes requests to appropriate tool implementations
5. **Response**: Server returns structured responses to Claude

## Server Implementation

### Core MCP Server Architecture

The MCP server supports both traditional and container-aware architectures:

```python
def create_mcp_server(
    vaults: Dict[str, Path],
    database_path: Path,
    settings: Optional[JarvisSettings] = None
) -> Server:
    """Create and configure the MCP server with dependency injection (default)."""
    server = Server("jarvis-assistant")
    
    # Always use container-aware context (traditional path removed)
    logger.info("Using container-aware MCP server context (default)")
    context = ContainerAwareMCPServerContext(vaults, database_path, settings)
    
    # Register MCP protocol handlers
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available MCP tools with dynamic capability detection."""
        return await context.get_available_tools()
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        """Handle tool execution with service container integration."""
        return await context.execute_tool(name, arguments or {})
```

### Container-Aware Context

The `ContainerAwareMCPServerContext` leverages the Service Container for dependency management:

```python
class ContainerAwareMCPServerContext:
    def __init__(self, vaults: Dict[str, Path], database_path: Path, settings: JarvisSettings):
        self.settings = settings
        
        # Initialize service container with database initialization
        self.container = ServiceContainer(settings)
        self._initialize_database(database_path)
        self.container.configure_default_services()
        
        # Get services through dependency injection
        self.vector_searcher = self.container.get(IVectorSearcher)
        self.graph_database = self.container.get(IGraphDatabase)
        self.vault_reader = self.container.get(IVaultReader)
        self.health_checker = self.container.get(IHealthChecker)
    
    def _initialize_database(self, database_path: Path):
        """Ensure database is ready before service initialization."""
        initializer = DatabaseInitializer(database_path, self.settings)
        if not initializer.ensure_database_exists():
            raise ServiceUnavailableError("Database initialization failed")
```

### Enhanced Error Handling Integration

The MCP server integrates comprehensive error handling with user-friendly messaging:

```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Handle tool execution with enhanced error handling."""
    try:
        # Execute tool logic
        results = await execute_tool(name, arguments or {})
        return results
        
    except DatabaseError as db_error:
        logger.error(f"Database error in tool {name}: {db_error}")
        
        # Format enhanced database error for MCP response
        error_handler = DatabaseErrorHandler(context.database_path)
        formatted_error = error_handler.format_error_for_user(db_error)
        
        return [
            types.TextContent(
                type="text",
                text=f"Database Error in {name}:\n\n{formatted_error}"
            )
        ]
    
    except JarvisError as e:
        logger.error(f"Jarvis error in tool {name}: {e}")
        
        # Provide enhanced error information if available
        error_text = f"❌ Internal error in {name}: {str(e)}"
        
        if hasattr(e, 'suggestions') and e.suggestions:
            error_text += "\n\n💡 Troubleshooting Steps:"
            for i, suggestion in enumerate(e.suggestions, 1):
                error_text += f"\n   {i}. {suggestion}"
        
        return [types.TextContent(type="text", text=error_text)]
```

### Real-World Usage Example

Here's how the MCP server actually starts up in production:

```python
# From src/jarvis/mcp/server.py - actual implementation
def run_mcp_server():
    """Run the MCP server with comprehensive initialization."""
    settings = get_settings()
    database_path = settings.get_vector_db_path()
    
    # Step 1: Initialize database using DatabaseInitializer
    logger.info("💾 Initializing database for MCP server")
    initializer = DatabaseInitializer(database_path, settings)
    
    if not initializer.ensure_database_exists():
        logger.error("❌ Database initialization failed")
        return False
    
    # Step 2: Create MCP server with container-aware context
    server = create_mcp_server(
        vaults={"default": Path(settings.vault_path)},
        database_path=database_path,
        settings=settings
    )
    
    # Step 3: Start server with error handling
    try:
        logger.info("🚀 Starting MCP server")
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("👋 MCP server stopped by user")
    except Exception as e:
        logger.error(f"❌ MCP server failed: {e}")
        return False
    
    return True
```
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute tool with comprehensive error handling and metrics."""
        try:
            # Get metrics service if available
            metrics = self.container.get(IMetrics) if self.settings.metrics_enabled else None
            
            # Execute with metrics tracking
            if metrics:
                with metrics.time_operation(f"mcp_tool_{name}"):
                    return await self._route_tool_request(name, arguments)
            else:
                return await self._route_tool_request(name, arguments)
                
        except ServiceUnavailableError as e:
            return self._handle_service_degradation(name, e)
        except Exception as e:
            logger.error(f"Tool execution failed: {name}: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def list_tools(self) -> List[Tool]:
        """List all available tools (using schema helpers)."""
        from jarvis.mcp.schemas import (
            SearchSchemaConfig,
            AnalyticsSchemaConfig,
            VaultSchemaConfig,
            UtilitySchemaConfig,
            create_search_schema,
            create_analytics_schema,
            create_vault_schema,
            create_utility_schema,
        )

        search_semantic_schema = create_search_schema(
            SearchSchemaConfig(
                query_required=True,
                enable_similarity_threshold=True,
                enable_vault_selection=True,
                default_limit=10,
                max_limit=50,
                supported_formats=["json"],
            )
        )

        # Graph search with custom fields via utility schema extension
        search_graph_schema = create_utility_schema(
            UtilitySchemaConfig(
                additional_properties={
                    "query_note_path": {"type": "string", "description": "Starting note path"},
                    "depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 5},
                }
            )
        )
        search_graph_schema["required"] = ["query_note_path"]

        search_vault_schema = create_search_schema(
            SearchSchemaConfig(
                enable_content_search=True,
                enable_vault_selection=True,
                default_limit=20,
                max_limit=50,
                supported_formats=["json"],
            )
        )

        read_note_schema = create_vault_schema(
            VaultSchemaConfig(path_required=True, enable_vault_selection=True, supported_formats=["json"])
        )

        list_vaults_schema = create_utility_schema()

        return [
            Tool(name="search-semantic", description="Search vault content using semantic similarity", inputSchema=search_semantic_schema),
            Tool(name="search-graph", description="Search for related notes using graph relationships", inputSchema=search_graph_schema),
            Tool(name="search-vault", description="Search vault using traditional keyword matching", inputSchema=search_vault_schema),
            Tool(name="read-note", description="Read content of a specific note", inputSchema=read_note_schema),
            Tool(name="list-vaults", description="List all available vaults and their statistics", inputSchema=list_vaults_schema),
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool invocation"""
        try:
            # Route to appropriate tool handler
            handler = self._get_tool_handler(name)
            result = await handler(arguments)
            
            # Format response
            return self._format_tool_response(result)
            
        except Exception as e:
            # Error handling
            return [TextContent(
                type="text",
                text=f"Error executing tool {name}: {str(e)}"
            )]
    
    def _get_tool_handler(self, tool_name: str) -> Callable:
        """Get handler function for tool"""
        handlers = {
            "search-semantic": self._handle_semantic_search,
            "search-graph": self._handle_graph_search,
            "search-vault": self._handle_vault_search,
            "read-note": self._handle_read_note,
            "list-vaults": self._handle_list_vaults
        }
        
        if tool_name not in handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return handlers[tool_name]
```

### Tool Handler Implementation

```python
class ToolHandlers:
    def __init__(self, services: Dict[str, Any]):
        self.services = services
        self.validator = ParameterValidator()
    
    async def _handle_semantic_search(self, args: Dict[str, Any]) -> SearchResult:
        """Handle semantic search tool"""
        # Validate parameters
        validated_args = self.validator.validate_semantic_search_args(args)
        
        # Execute search
        results = await self.services['vector'].search(
            query=validated_args['query'],
            similarity_threshold=validated_args.get('similarity_threshold', 0.7),
            limit=validated_args.get('limit', 10),
            vault_filter=validated_args.get('vault')
        )
        
        return SearchResult(
            tool_name="search-semantic",
            query=validated_args['query'],
            results=results,
            total_results=len(results),
            execution_time=time.time() - start_time
        )
    
    async def _handle_graph_search(self, args: Dict[str, Any]) -> GraphSearchResult:
        """Handle graph search tool"""
        validated_args = self.validator.validate_graph_search_args(args)
        
        # Execute graph search
        graph_result = await self.services['graph'].search(
            start_node=validated_args['query_note_path'],
            depth=validated_args.get('depth', 2)
        )
        
        return GraphSearchResult(
            tool_name="search-graph",
            start_node=validated_args['query_note_path'],
            depth=validated_args.get('depth', 2),
            nodes=graph_result.nodes,
            relationships=graph_result.relationships,
            paths=graph_result.paths
        )
    
    async def _handle_vault_search(self, args: Dict[str, Any]) -> VaultSearchResult:
        """Handle vault search tool"""
        validated_args = self.validator.validate_vault_search_args(args)
        
        # Execute vault search
        results = await self.services['vault'].search(
            query=validated_args['query'],
            search_content=validated_args.get('search_content', True),
            limit=validated_args.get('limit', 20)
        )
        
        return VaultSearchResult(
            tool_name="search-vault",
            query=validated_args['query'],
            results=results,
            search_content=validated_args.get('search_content', True)
        )
    
    async def _handle_read_note(self, args: Dict[str, Any]) -> NoteContent:
        """Handle read note tool"""
        validated_args = self.validator.validate_read_note_args(args)
        
        # Read note content
        note_content = await self.services['vault'].read_note(
            path=validated_args['path'],
            vault=validated_args.get('vault')
        )
        
        return NoteContent(
            tool_name="read-note",
            path=validated_args['path'],
            content=note_content.content,
            metadata=note_content.metadata,
            size=note_content.size,
            modified_at=note_content.modified_at
        )
    
    async def _handle_list_vaults(self, args: Dict[str, Any]) -> VaultList:
        """Handle list vaults tool"""
        # Get vault statistics
        vault_stats = await self.services['vault'].get_vault_statistics()
        
        return VaultList(
            tool_name="list-vaults",
            vaults=vault_stats.vaults,
            total_vaults=len(vault_stats.vaults),
            total_notes=vault_stats.total_notes
        )
```

## Parameter Validation

### Input Validation Framework

```python
class ParameterValidator:
    def __init__(self):
        self.schemas = self._load_schemas()
    
    def validate_semantic_search_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate semantic search parameters"""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "vault": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"}
            },
            "required": ["query"],
            "additionalProperties": False
        }
        
        return self._validate_against_schema(args, schema)
    
    def validate_graph_search_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate graph search parameters"""
        schema = {
            "type": "object",
            "properties": {
                "query_note_path": {"type": "string", "minLength": 1},
                "depth": {"type": "integer", "minimum": 1, "maximum": 5}
            },
            "required": ["query_note_path"],
            "additionalProperties": False
        }
        
        return self._validate_against_schema(args, schema)
    
    def _validate_against_schema(self, args: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate arguments against JSON schema"""
        try:
            jsonschema.validate(args, schema)
            return args
        except jsonschema.ValidationError as e:
            raise ParameterValidationError(f"Invalid parameters: {e.message}")
```

## Response Formatting

### Structured Response Format

```python
class ResponseFormatter:
    def __init__(self):
        self.max_content_length = 10000
        self.max_results_per_response = 50
    
    def format_semantic_search_response(self, result: SearchResult) -> List[TextContent]:
        """Format semantic search results for MCP"""
        content = []
        
        # Summary
        summary = f"Found {len(result.results)} results for '{result.query}'"
        if result.execution_time:
            summary += f" (took {result.execution_time:.2f}s)"
        
        content.append(TextContent(type="text", text=summary))
        
        # Results
        for i, search_result in enumerate(result.results, 1):
            if i > self.max_results_per_response:
                content.append(TextContent(
                    type="text",
                    text=f"... and {len(result.results) - i + 1} more results"
                ))
                break
            
            # Format individual result
            result_text = self._format_search_result(search_result, i)
            content.append(TextContent(type="text", text=result_text))
        
        return content
    
    def format_graph_search_response(self, result: GraphSearchResult) -> List[TextContent]:
        """Format graph search results for MCP"""
        content = []
        
        # Summary
        summary = f"Graph search from '{result.start_node}' (depth {result.depth})"
        summary += f"\nFound {len(result.nodes)} nodes and {len(result.relationships)} relationships"
        content.append(TextContent(type="text", text=summary))
        
        # Nodes
        if result.nodes:
            nodes_text = "Connected Notes:\n"
            for node in result.nodes[:10]:  # Limit display
                nodes_text += f"• {node.title} ({node.path})\n"
            
            if len(result.nodes) > 10:
                nodes_text += f"... and {len(result.nodes) - 10} more nodes"
            
            content.append(TextContent(type="text", text=nodes_text))
        
        # Key relationships
        if result.relationships:
            rel_text = "Key Relationships:\n"
            for rel in result.relationships[:10]:
                rel_text += f"• {rel.source_title} → {rel.target_title} ({rel.type})\n"
            
            content.append(TextContent(type="text", text=rel_text))
        
        return content
    
    def _format_search_result(self, result: Any, index: int) -> str:
        """Format individual search result"""
        # Truncate content if too long
        content = result.content
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        formatted = f"{index}. {result.file_path}"
        
        # Add similarity score if available
        if hasattr(result, 'similarity_score'):
            formatted += f" (Score: {result.similarity_score:.2f})"
        
        formatted += f"\n{content}\n"
        
        return formatted
```

## Error Handling

### MCP Error Mapping

```python
class MCPErrorHandler:
    def __init__(self):
        self.error_mappings = {
            ParameterValidationError: -32602,  # Invalid params
            ServiceUnavailableError: -32603,   # Internal error
            FileNotFoundError: -32601,         # Method not found
            TimeoutError: -32003,              # Request timeout
            PermissionError: -32000,           # Server error
        }
    
    def handle_error(self, error: Exception) -> List[TextContent]:
        """Convert Python exceptions to MCP error responses"""
        error_type = type(error)
        
        if error_type in self.error_mappings:
            error_code = self.error_mappings[error_type]
            error_message = str(error)
        else:
            error_code = -32603  # Internal error
            error_message = f"Unexpected error: {str(error)}"
        
        return [TextContent(
            type="text",
            text=f"Error (Code {error_code}): {error_message}"
        )]
    
    def handle_service_degradation(self, service_name: str, fallback_result: Any) -> List[TextContent]:
        """Handle graceful service degradation"""
        warning = f"⚠️ {service_name} service is currently unavailable. Using fallback method."
        
        content = [TextContent(type="text", text=warning)]
        
        if fallback_result:
            content.extend(self._format_fallback_result(fallback_result))
        
        return content
```

## Resource Management

### MCP Resources

```python
class MCPResourceManager:
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
    
    async def list_resources(self) -> List[Resource]:
        """List available MCP resources"""
        return [
            Resource(
                uri="vault://recent-notes",
                mimeType="application/json",
                name="Recent Notes",
                description="Recently modified notes in the vault"
            ),
            Resource(
                uri="vault://vault-stats",
                mimeType="application/json",
                name="Vault Statistics",
                description="Statistics about the current vault"
            ),
            Resource(
                uri="vault://search-index",
                mimeType="application/json",
                name="Search Index Status",
                description="Status of search indexes"
            )
        ]
    
    async def read_resource(self, uri: str) -> str:
        """Read specific resource content"""
        if uri == "vault://recent-notes":
            return await self._get_recent_notes()
        elif uri == "vault://vault-stats":
            return await self._get_vault_stats()
        elif uri == "vault://search-index":
            return await self._get_index_status()
        else:
            raise ResourceNotFoundError(f"Resource not found: {uri}")
    
    async def _get_recent_notes(self) -> str:
        """Get recently modified notes"""
        # Implementation to get recent notes
        recent_notes = await self.vault_service.get_recent_notes(limit=10)
        return json.dumps({
            "recent_notes": [
                {
                    "path": note.path,
                    "title": note.title,
                    "modified_at": note.modified_at.isoformat(),
                    "size": note.size
                }
                for note in recent_notes
            ]
        })
```

## Performance Optimization

### Connection Pooling

```python
class MCPConnectionManager:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connection_pool = asyncio.Queue(maxsize=max_connections)
        self.active_connections = 0
    
    async def get_connection(self):
        """Get connection from pool"""
        if self.connection_pool.empty() and self.active_connections < self.max_connections:
            # Create new connection
            connection = await self._create_connection()
            self.active_connections += 1
            return connection
        else:
            # Wait for available connection
            return await self.connection_pool.get()
    
    async def return_connection(self, connection):
        """Return connection to pool"""
        await self.connection_pool.put(connection)
```

### Request Batching

```python
class BatchProcessor:
    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.batch_timer = None
    
    async def add_request(self, request: Dict[str, Any]) -> Any:
        """Add request to batch"""
        future = asyncio.Future()
        self.pending_requests.append((request, future))
        
        # Start batch timer if not already running
        if not self.batch_timer:
            self.batch_timer = asyncio.create_task(self._process_batch_after_timeout())
        
        # Process batch if full
        if len(self.pending_requests) >= self.batch_size:
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process current batch of requests"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests
        self.pending_requests = []
        
        # Cancel timer
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Process batch
        results = await self._execute_batch([req for req, _ in batch])
        
        # Return results to futures
        for (_, future), result in zip(batch, results):
            future.set_result(result)
```

## Testing and Validation

### MCP Protocol Testing

```python
class MCPProtocolTester:
    def __init__(self, server_instance):
        self.server = server_instance
    
    async def test_tool_discovery(self):
        """Test tool discovery protocol"""
        tools = await self.server.list_tools()
        
        assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}"
        
        expected_tools = {"search-semantic", "search-graph", "search-vault", "read-note", "list-vaults"}
        actual_tools = {tool.name for tool in tools}
        
        assert expected_tools == actual_tools, f"Tool mismatch: {expected_tools} != {actual_tools}"
    
    async def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        valid_result = await self.server.call_tool("search-semantic", {"query": "test"})
        assert valid_result is not None
        
        # Test invalid parameters
        with pytest.raises(ParameterValidationError):
            await self.server.call_tool("search-semantic", {"invalid": "param"})
    
    async def test_error_handling(self):
        """Test error handling"""
        # Test with non-existent tool
        result = await self.server.call_tool("non-existent-tool", {})
        assert "Error" in result[0].text
        
        # Test with invalid file path
        result = await self.server.call_tool("read-note", {"path": "/non/existent/file.md"})
        assert "Error" in result[0].text
```

## For More Detail

- **Component Interaction**: [Component Interaction](component-interaction.md)
- **Data Flow**: [Data Flow Architecture](data-flow.md)
- **API Reference**: [API Reference](../06-reference/api-reference.md)
- **Setup Guide**: [Installation Guide](../03-getting-started/detailed-installation.md)
