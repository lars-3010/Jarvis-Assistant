Validate design or code against architectural guidelines: [target]

Execute architecture-agent to review specifications or code against architectural-guidelines.md.

**Agent**: architecture-agent (guardian role)
**Guidelines**: `ai-docs/guidelines/architectural-guidelines.md`
**Input**: design.md from `.kiro/specs/` subfolder, code files, or implementation for validation
**Output**: Compliance validation with specific recommendations
**Focus**: SOLID principles, Python patterns, security, scalability

## Usage Examples:
```bash
/architecture-review design.md
/architecture-review src/services/user_service.py
/architecture-review "validate current authentication implementation"
```

Use when: Need architectural compliance validation for design or code.
