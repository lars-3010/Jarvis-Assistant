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