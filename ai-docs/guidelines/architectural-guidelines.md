# Python Architectural Guidelines

## Design Principles
- **Modular Architecture**: Clear package structure, single responsibility modules
- **SOLID Principles**: Interface segregation, dependency inversion, open/closed
- **Adaptive Design**: Protocol-based interfaces, composition over inheritance
- **Scalable Patterns**: Repository pattern, service layer, factory methods

## Python Technology Standards
- **Web Framework**: FastAPI (async APIs) or Django (full-featured)
- **Database**: SQLAlchemy 2.0+ with async support, PostgreSQL primary
- **API Design**: RESTful with Pydantic models, OpenAPI auto-generation
- **Authentication**: JWT with refresh tokens, OAuth2 integration
- **Caching**: Redis for sessions/cache, in-memory for temporary data

## Architecture Patterns

### Repository Pattern
```python
from abc import ABC, abstractmethod
from typing import Protocol, List, Optional

class UserRepository(Protocol):
    async def get_by_id(self, user_id: int) -> Optional[User]: ...
    async def create(self, user: UserCreate) -> User: ...
```

### Service Layer
```python
class UserService:
    def __init__(self, user_repo: UserRepository, email_service: EmailService):
        self._user_repo = user_repo
        self._email_service = email_service
    
    async def create_user(self, user_data: UserCreate) -> User:
        # Business logic here
```

### Dependency Injection
```python
# Use dependency-injector or simple constructor injection
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    user_repo = providers.Factory(SqlUserRepository)
    user_service = providers.Factory(UserService, user_repo=user_repo)
```

## Security Standards
- **Input Validation**: Pydantic models for all inputs
- **SQL Injection**: SQLAlchemy ORM, parameterized queries only
- **Authentication**: Secure JWT handling, password hashing with bcrypt
- **CORS**: Explicit origins, no wildcard in production
- **Rate Limiting**: slowapi for FastAPI, django-ratelimit for Django

## Performance Standards
- **Async**: Use async/await for I/O operations
- **Database**: Connection pooling, query optimization, lazy loading
- **Caching**: Redis for expensive operations, cache invalidation strategy
- **Monitoring**: Structured logging, health checks, metrics collection

## Integration Patterns
- **External APIs**: aiohttp client with circuit breaker, timeout handling
- **Message Queues**: Celery for background tasks, Redis as broker
- **File Storage**: Object storage (S3/MinIO) for files, local for temp
- **Environment Config**: Pydantic Settings for configuration management

---

# Jarvis‑Specific Architecture Guidelines

## Core Tenets
- **Local‑first**: All core functionality runs locally. External services are optional.
- **Structured by Default**: MCP tools should support `format: "json"` and return versioned payloads with a `schema_version` and `correlation_id`.
- **Thin MCP Layer**: The server delegates to a plugin registry. Plugins orchestrate; services compute.
- **Event‑Driven Analytics**: File changes publish events; analytics subscribe and invalidate caches.

## Layering Model
- **MCP (Protocol)**: `src/jarvis/mcp/*`
  - Server bootstraps DI and plugin registry
  - Tools expose contracts and map inputs → services → structured outputs
- **Plugins (Orchestration)**: `src/jarvis/mcp/plugins/tools/*`
  - Minimal logic: parameter validation, service calls, formatting selection
- **Services (Domain)**: `src/jarvis/services/*`
  - Deterministic logic, no protocol concerns
  - Testable in isolation; return domain objects or dicts
- **Infrastructure**: `src/jarvis/core/*` (DI, events, metrics, interfaces)

## Structured Output Conventions
- Include: `schema_version`, `correlation_id`, `timestamp`, `metrics`
- Shapes: semantic results, keyword results, graph neighborhoods, combined rankings, vault stats, health, performance
- Prefer returning JSON as text content for broad MCP compatibility; keep a single, shared serializer path in `mcp/structured`.

## Event & Cache Design
- Event types live in `EventTypes` (see `jarvis/core/event_integration.py`)
- Analytics caches must carry `cache_hit`, `content_hash`, and `generated_at`
- Invalidate on document add/update/delete and vault reindex

## Adaptive Context Loading (for AI Agents)
- Load only what’s needed:
  - Architecture questions → `docs/architecture/arc42.md` + architecture map
  - Tool behavior → `src/jarvis/mcp/plugins/tools/*` + `mcp/structured/*`
  - Analytics → `src/jarvis/services/analytics/*` + event integration
  - Configuration → `config/base.yaml`, `config/local.yaml`, `config/.env.example`
- Prefer links to files and anchors over large quote blocks
- Summarize objectives, decisions, and the exact code paths you will touch

## Adding a New MCP Tool (Checklist)
- Define input/outputs and pick a serializer model under `mcp/structured`
- Implement a plugin under `mcp/plugins/tools/`
- Wire to domain services; avoid doing domain work in the plugin
- Add `format` parameter; default markdown, support `json`
- Add unit tests for serializer + plugin happy path and errors
- Update docs: API reference, usage examples, and README quick hints

## Configuration Strategy
- YAML for structured defaults: `config/base.yaml`, overrides in `config/local.yaml`
- Secrets and environment overrides in `config/.env` (template provided)
- Pydantic Settings load from `config/.env` by default; CLI flags should override

## Observability
- Use structured logging including `correlation_id`
- Expose health/perf data via tools; keep metrics minimal but actionable
