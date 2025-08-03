# [PROJECT_NAME] - System Design

## Architecture Overview
- **Pattern**: [Monolith/Microservices/Serverless/Layered]
- **Technology Stack**: [Primary frameworks, languages, databases]
- **Deployment**: [Cloud platform, containerization approach]
- **Communication**: [API style, messaging patterns]

## Architecture Visualization
```
[ASCII diagram showing component interactions]
User Request → API Gateway → Service Layer → Repository → Database
     ↓              ↓             ↓            ↓         ↓
  [Auth]      [Validation]  [Business Logic] [ORM]  [PostgreSQL]
```

## System Components

### Core Components
- **[Component 1]**: [Purpose and key responsibilities]
  ```python
  # Interface definition with type hints
  from abc import ABC, abstractmethod
  from typing import List, Optional

  class Component1Interface(ABC):
      @abstractmethod
      async def method_name(self, param: str) -> Optional[Result]:
          """Method description with clear contract."""
          pass
  ```

- **[Component 2]**: [Purpose and key responsibilities]
- **[Data Layer]**: [Database design and data management]
- **[API Layer]**: [External interfaces and communication]

### Data Models
```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserModel(BaseModel):
    id: int = Field(..., description="Unique user identifier")
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True
```

## Technical Decisions

### Decision 1: [Major Technical Choice]
- **Context**: [Problem requiring this decision]
- **Options Considered**: 
  - Option A: [Description] - Pros: [benefits] - Cons: [drawbacks]
  - Option B: [Description] - Pros: [benefits] - Cons: [drawbacks]
- **Decision**: [Chosen approach]
- **Rationale**: [Detailed reasoning with trade-off analysis]
- **Implementation Impact**: [How this affects development]

### Decision 2: [Database Choice]
- **Context**: [Data requirements and constraints]
- **Options Considered**:
  - PostgreSQL: Pros: [ACID, relations, JSON] - Cons: [complexity]
  - MongoDB: Pros: [flexibility, scalability] - Cons: [consistency]
- **Decision**: PostgreSQL with SQLAlchemy
- **Rationale**: [Structured data needs, ACID requirements, team expertise]

## Integration Architecture

### External System Integrations
```python
# External API interface example
class ExternalServiceAdapter(ABC):
    @abstractmethod
    async def authenticate(self, credentials: AuthRequest) -> AuthResponse:
        """Handle external authentication."""
        pass
    
    @abstractmethod
    async def fetch_data(self, query: DataQuery) -> DataResponse:
        """Retrieve data from external system."""
        pass
```

### Integration Points
- **Authentication Provider**: [OAuth2/SAML integration details]
- **External APIs**: [Third-party service connections]
- **Message Queues**: [Event-driven communication patterns]
- **File Storage**: [File handling and storage strategy]

## Security Architecture
- **Authentication Flow**: [Detailed auth implementation]
  ```python
  # JWT token validation example
  def validate_token(token: str) -> Optional[UserClaims]:
      try:
          payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
          return UserClaims(**payload)
      except jwt.InvalidTokenError:
          return None
  ```
- **Authorization Model**: [Role-based access control]
- **Data Protection**: [Encryption and sensitive data handling]
- **API Security**: [Rate limiting, CORS, input validation]

## Performance Architecture

### Scalability Strategy
- **Horizontal Scaling**: [Load balancing and service replication]
- **Database Optimization**: [Indexing strategy and query optimization]
- **Caching Strategy**: [Redis implementation and cache invalidation]
- **Performance Targets**: [Response time and throughput requirements]

### Optimization Considerations
```python
# Database optimization example
from sqlalchemy import Index
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True)  # Indexed for lookups
    created_at = Column(DateTime, index=True)  # Indexed for time-based queries
    
    # Efficient relationship loading
    orders = relationship("Order", lazy="select", back_populates="user")
```

## Error Handling Strategy
- **Exception Hierarchy**: [Custom exception design]
  ```python
  class AppException(Exception):
      """Base application exception."""
      def __init__(self, message: str, error_code: str = None):
          self.message = message
          self.error_code = error_code
          super().__init__(message)

  class ValidationError(AppException):
      """Data validation error."""
      pass
  ```
- **Error Response Format**: [Standardized API error responses]
- **Logging Strategy**: [Structured logging and error tracking]
- **Recovery Mechanisms**: [Retry logic and graceful degradation]

## Implementation Planning

### Migration Strategy (if applicable)
- **Current State**: [Existing system description]
- **Migration Phases**: 
  1. **Phase 1**: [Foundation setup and core components]
  2. **Phase 2**: [Data migration and service integration]
  3. **Phase 3**: [Feature migration and testing]
- **Rollback Plan**: [How to revert if migration fails]
- **Data Migration**: [Strategy for moving existing data]

### Development Roadmap
- **Foundation Setup**: [Environment, dependencies, core structure]
- **Core Implementation**: [Business logic and primary features]
- **Integration & Testing**: [External systems and comprehensive testing]
- **Performance Optimization**: [Scaling and performance tuning]

### Performance Considerations
- **Bottleneck Analysis**: [Potential performance issues and solutions]
- **Monitoring Strategy**: [Metrics collection and alerting]
- **Load Testing Plan**: [Performance validation approach]
- **Capacity Planning**: [Resource requirements and scaling triggers]

## Development Standards
- **Code Organization**: [Package structure and module design]
- **Testing Strategy**: [Unit, integration, and E2E testing approach]
- **Documentation**: [Code documentation and API docs]
- **Deployment Pipeline**: [CI/CD and environment management]