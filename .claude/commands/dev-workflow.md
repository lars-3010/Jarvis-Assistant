Execute complete adaptive development workflow: $ARGUMENTS

Orchestrate full specification creation workflow using dev-workflow-orchestrator for Python projects.

**Agent**: dev-workflow-orchestrator
**Templates**: requirements-template.md, design-template.md, tasks-template.md
**Guidelines**: architectural-guidelines.md for design validation
**Output**: Complete project specifications ready for development

## Workflow Process:
1. **Requirements**: product-owner-agent creates requirements.md
2. **Design**: design-agent creates design.md with integrated architecture
3. **Architecture Validation**: architecture-agent validates design compliance
4. **Tasks**: tasks-agent creates implementation roadmap

## Expected Output:
```
.kiro/specs/$ARGUMENTS/
├── requirements.md    # Professional requirements with acceptance criteria
├── design.md         # System design with architectural decisions  
└── tasks.md          # Implementation roadmap with development tasks
```

## Quality Assurance:
- Architecture-agent validates design.md against architectural-guidelines.md
- All specifications follow template formats
- Context properly handed off between agents
- Professional quality suitable for development team

## Usage Examples:
```bash
/dev-workflow user-authentication-api
/dev-workflow task-management-system
/dev-workflow real-time-chat-service
```

Success: Complete specifications ready for development implementation.