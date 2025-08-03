Implement functionality following design specifications: [description]

Execute developer-agent to write production-ready Python code following design.md and coding-standards.md.

**Agent**: developer-agent
**Guidelines**: `.claude/guidelines/coding-standards.md`
**Context**: Loads design.md and tasks.md from `.kiro/specs/` subfolder for implementation guidance
**Output**: Clean Python code with tests, type hints, and documentation
**Focus**: FastAPI/Django, SQLAlchemy, pytest, modular architecture

## Consultation Available:
Developer-agent can consult architecture-agent using: "@architecture-agent: [question]"

## Usage Examples:
```bash
/develop "implement user authentication with JWT tokens"
/develop "create REST API for task management with CRUD operations"
/develop "add Redis caching layer for expensive database queries"
```

Use when: Ready to implement specific functionality with existing specifications.