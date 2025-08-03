Comprehensive code review with architectural validation: [target]

Execute both developer-agent and architecture-agent for complete code quality review.

**Agents**: developer-agent + architecture-agent coordination
**Guidelines**: coding-standards.md + architectural-guidelines.md
**Input**: Code files, modules, or implementation areas
**Output**: Quality assessment with coding standards and architectural compliance
**Focus**: Code quality, architectural compliance, best practices, security

## Review Process:
1. developer-agent reviews against coding-standards.md
2. architecture-agent validates against architectural-guidelines.md  
3. Combined recommendations for improvements

## Usage Examples:
```bash
/code-review src/services/
/code-review "review user authentication implementation" 
/code-review src/api/endpoints/users.py
```

Use when: Need comprehensive code quality and architectural compliance review.