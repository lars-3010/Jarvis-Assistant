# Architectural Evolution

This document tracks how the Jarvis Assistant architecture has evolved across versions, highlighting key design decisions and their rationale.

## Version 4.0 Architecture: Extension-Based AI Integration

**Date**: 2025-07-14  
**Status**: Planning Phase

### Key Architectural Changes

#### 1. Extension System Architecture

**Problem**: Adding AI capabilities without affecting core system stability  
**Solution**: Optional extension system with dynamic loading

```
src/jarvis/
├── extensions/               # NEW: Extension system
│   ├── loader.py            # Extension loading logic
│   ├── interfaces.py        # Extension interfaces
│   └── ai/                  # AI Extension Package
│       ├── main.py          # Extension entry point
│       ├── llm/             # LLM services
│       ├── graphrag/        # GraphRAG services
│       ├── workflows/       # Workflow orchestration
│       └── tools/           # AI-specific MCP tools
```

**Benefits**:
- Zero impact on core system when disabled
- Independent development and testing
- Granular feature control
- Resource efficiency

#### 2. Configurable Model Routing System

**Problem**: Different tasks need different models for optimal performance  
**Solution**: Task-specific model routing with JarvisSettings configuration

```python
class ILLMService:
    def get_model_for_task(self, task_type: str) -> str
    def generate(self, prompt: str, task_type: str = "general") -> LLMResponse
```

**Configuration**:
```yaml
ai:
  llm:
    task_models:
      summarize: "mistral:7b-q4"      # Fast summarization
      analyze: "llama3:8b-q8"         # Complex reasoning
      quick_answer: "tinyllama:1.1b"  # Ultra-fast queries
```

#### 3. Agent-Based Quality Systems

**Problem**: Manual quality assessment and improvement  
**Solution**: Intelligent agents for automated quality management

- **DuplicationDetectionAgent**: Semantic similarity clustering
- **AtomicBoundaryAnalyzer**: Wiki/teaching/linkability tests
- **VisualEnhancementEngine**: Diagram/table opportunity detection
- **VaultHealthMonitor**: Continuous health assessment

#### 4. Enhanced GraphRAG Pipeline

**Problem**: Limited context assembly and source attribution  
**Solution**: Multi-phase GraphRAG with quality analysis

1. **Query Understanding**: LLM-powered intent extraction
2. **Multi-Modal Retrieval**: Vector + graph + keyword + property search
3. **Quality Analysis**: Duplication + atomicity + enhancement detection
4. **Context Assembly**: Intelligent ranking and deduplication
5. **Generation**: Structured output with confidence scoring

---

## Version 3.0 Architecture: Production MCP Tools

**Date**: Previous  
**Status**: Stable

### Core Architecture Achievements

#### 1. Dual Database System

**Design**: DuckDB (vector) + Neo4j (graph) with graceful degradation

```python
class ServiceContainer:
    def __init__(self):
        self.vector_searcher = DuckDBVectorSearcher()
        self.graph_db = Neo4jGraphDB()  # Optional
        self.vault_reader = VaultReader()
```

**Benefits**:
- Semantic search with DuckDB performance
- Graph relationships when Neo4j available
- Graceful fallback to vector-only search

#### 2. MCP Protocol Integration

**Design**: 8 specialized MCP tools with consistent interfaces

```python
class MCPTool:
    def __init__(self, name: str, description: str)
    async def execute(self, arguments: Dict[str, Any]) -> Any
```

**Tools Implemented**:
- search-semantic, search-vault, search-graph, search-combined
- read-note, list-vaults, health-status, performance-metrics

#### 3. Service-Oriented Architecture

**Design**: Clean separation of concerns with dependency injection

```
src/jarvis/services/
├── vector_search.py    # DuckDB vector operations
├── graph_db.py         # Neo4j graph operations
├── vault_reader.py     # File system operations
└── container.py        # Dependency injection
```

#### 4. Local-First Philosophy

**Design**: No external dependencies, privacy-focused

- All processing happens locally
- No cloud API calls
- Offline-capable indexing and search
- User data never leaves the system

---

## Architecture Evolution Timeline

### Phase 0: Extension Foundation (COMPLETE)

**Key Addition**: Extension system with dynamic loading

**Before**:
```python
# Monolithic MCP server
class MCPServer:
    def __init__(self):
        self.tools = [SearchTool(), VaultTool(), ...]
```

**After**:
```python
# Extensible MCP server
class MCPServer:
    def __init__(self):
        self.core_tools = [SearchTool(), VaultTool(), ...]
        self.extensions = ExtensionLoader().load_enabled()
        self.tools = self.core_tools + self.get_extension_tools()
```

### Phase 1: LLM Integration (ENHANCED)

**Key Addition**: Task-specific model routing

**Architecture**:
```python
class LLMService:
    def __init__(self, config: LLMConfig):
        self.model_router = ModelRouter(config.task_models)
        self.ollama_client = OllamaClient()
    
    async def generate(self, prompt: str, task_type: str) -> LLMResponse:
        model = self.model_router.get_model(task_type)
        return await self.ollama_client.generate(prompt, model)
```

### Phase 2: GraphRAG & Quality Agents (EXPANDED)

**Key Addition**: Multi-phase retrieval with quality analysis

**Architecture**:
```python
class EnhancedGraphRAGService:
    def __init__(self):
        self.retrieval_pipeline = MultiModalRetriever()
        self.quality_analyzer = QualityAnalyzer()
        self.context_assembler = ContextAssembler()
        self.generation_engine = GenerationEngine()
```

### Phase 3: Agent Systems (EXPANDED)

**Key Addition**: Intelligent workflow automation

**Architecture**:
```python
class AgentOrchestrator:
    def __init__(self):
        self.agents = [
            BridgeDiscoveryAgent(),
            QualityProgressionAgent(),
            VaultHealthMonitor()
        ]
        self.workflow_engine = WorkflowEngine()
```

### Phase 4: Base Integration (NEW)

**Key Addition**: External knowledge base connectivity

**Architecture**:
```python
class BaseInteractionService:
    def __init__(self):
        self.connectors = {
            'notion': NotionConnector(),
            'airtable': AirtableConnector(),
            'database': DatabaseConnector()
        }
        self.sync_engine = SyncEngine()
```

---

## Design Principles Evolution

### Version 3.0 Principles
1. **Local-First**: No external dependencies
2. **Performance**: Sub-second response times
3. **Reliability**: Graceful degradation
4. **Simplicity**: Clear, focused functionality

### Version 4.0 Enhancements
1. **Optionality**: AI features completely optional
2. **Modularity**: Independent extension development
3. **Intelligence**: Proactive quality improvement
4. **Scalability**: Multi-model, multi-agent architecture
5. **Predictability**: Consistent interfaces and behavior

---

## Key Architectural Decisions

### Decision 1: Extension vs. Core Integration

**Options Considered**:
- A: Integrate AI directly into core system
- B: Create optional extension system

**Decision**: B - Extension system  
**Rationale**: Maintains zero-impact design, enables independent development

### Decision 2: Single vs. Multi-Model LLM

**Options Considered**:
- A: Single general-purpose model
- B: Task-specific model routing

**Decision**: B - Task-specific routing  
**Rationale**: Better performance through specialization, resource efficiency

### Decision 3: Agent Architecture

**Options Considered**:
- A: Monolithic quality analysis
- B: Specialized quality agents

**Decision**: B - Specialized agents  
**Rationale**: Modular development, focused functionality, easier testing

### Decision 4: GraphRAG Enhancement

**Options Considered**:
- A: Simple retrieval augmentation
- B: Multi-phase pipeline with quality analysis

**Decision**: B - Multi-phase pipeline  
**Rationale**: Higher quality results, comprehensive source attribution

---

## Lessons Learned

### Extension System Design

**Lesson**: Dynamic loading requires careful interface design  
**Application**: Comprehensive IExtension interface with lifecycle management

### Model Routing Complexity

**Lesson**: Task-specific routing adds configuration complexity  
**Application**: Provide sensible defaults, clear configuration documentation

### Agent Communication

**Lesson**: Agents need structured communication protocols  
**Application**: Event-driven architecture with typed messages

### Quality Automation

**Lesson**: Automated quality assessment requires human validation  
**Application**: Confidence scoring and user override capabilities