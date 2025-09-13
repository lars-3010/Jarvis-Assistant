# Python Coding Standards

## Code Style
- **Formatting**: black (line length 100), isort for imports
- **Linting**: ruff for fast linting, mypy for type checking
- **Type Hints**: All functions, methods, class attributes (strict mode)
- **Docstrings**: Google style for public APIs, type information in hints

## File Organization
```
src/
├── domain/          # Business entities and interfaces
├── services/        # Business logic services
├── repositories/    # Data access layer
├── api/            # FastAPI routes or Django views
├── models/         # Database models (SQLAlchemy)
├── schemas/        # Pydantic models for API
└── core/           # Configuration, dependencies, utilities
```

## Testing Standards
- **Framework**: pytest with async support (pytest-asyncio)
- **Coverage**: >80% with pytest-cov, focus on business logic
- **Test Structure**: tests/ mirrors src/ structure
- **Fixtures**: Reusable test data with factory-boy
- **Mocking**: unittest.mock for external dependencies

## Type System
```python
# Good type hints
from typing import Optional, List, Dict, Protocol
from pydantic import BaseModel

async def get_users(
    limit: int = 10, 
    offset: int = 0
) -> List[UserResponse]:
    """Get paginated users."""
    pass

# Protocol for interfaces
class EmailService(Protocol):
    async def send_email(self, to: str, subject: str, body: str) -> bool: ...
```

## Error Handling
```python
# Custom exception hierarchy
class AppException(Exception):
    """Base application exception."""
    pass

class ValidationError(AppException):
    """Data validation error."""
    pass

class NotFoundError(AppException):
    """Resource not found."""
    pass

# FastAPI error handling
from fastapi import HTTPException, status

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )
```

## Database Standards
```python
# SQLAlchemy 2.0 style
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    
# Async queries
async def get_user(session: AsyncSession, user_id: int) -> Optional[User]:
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

## API Standards
```python
# FastAPI with Pydantic
from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, HTTPException

class UserCreate(BaseModel):
    email: str = Field(..., description="User email address")
    name: str = Field(..., min_length=1, max_length=100)

@app.post("/users/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """Create a new user."""
    return await user_service.create_user(user_data)
```

## Performance & Logging
```python
# Structured logging
import structlog

logger = structlog.get_logger()

async def process_user(user_id: int):
    logger.info("Processing user", user_id=user_id)
    try:
        # Process
        logger.info("User processed successfully", user_id=user_id)
    except Exception as e:
        logger.error("User processing failed", user_id=user_id, error=str(e))

# Async context managers for resources
async with get_db_session() as session:
    result = await session.execute(query)
```

---

# Jarvis‑Specific Coding Standards

## MCP Plugins
- Keep orchestration‑only: input validation, service calls, choose formatter
- Always accept `format` with `"markdown"|"json"` and default to markdown
- Return structured payloads via shared serializers in `mcp/structured`

## Structured Logging
- Include `correlation_id` and `schema_version` in logs for tool calls
- Avoid logging sensitive content (file contents, secrets)

## Async & Concurrency
- Prefer async service APIs; avoid blocking calls in MCP handlers
- Use bounded concurrency (e.g., asyncio.Semaphore) for fanout tasks

## Configuration & Secrets
- Load environment from `config/.env` (template in `config/.env.example`)
- Put structured defaults in `config/base.yaml`; override via `config/local.yaml`

## Testing
- Unit tests for: serializers, service behavior, plugin happy/edge paths
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.mcp`
- Use fakes/mocks for external systems (Neo4j, DuckDB) unless integration tests

## Error Handling
- Convert domain errors to MCP/tool‑friendly messages
- Prefer actionable messages (what failed, how to fix) over raw stack traces
