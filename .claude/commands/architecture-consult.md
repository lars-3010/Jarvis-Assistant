Get architectural guidance and consultation: [question]

Execute architecture-agent to provide architectural guidance and answer specific questions.

**Agent**: architecture-agent (guardian role) 
**Guidelines**: `ai-docs/guidelines/architectural-guidelines.md`
**Input**: Specific architectural question or design challenge from `.kiro/specs/` subfolders
**Output**: Expert guidance with rationale and implementation recommendations
**Focus**: Best practices, patterns, technology choices, design decisions

## Usage Examples:
```bash
/architecture-consult "How should I implement authentication with FastAPI?"
/architecture-consult "What's the best caching strategy for this use case?" 
/architecture-consult "Should I use Repository pattern for this simple CRUD API?"
```

Use when: Need architectural expertise for design decisions or implementation questions.
