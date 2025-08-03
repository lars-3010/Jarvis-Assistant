Comprehensive validation of project specifications and code: $ARGUMENTS

Validate complete project against all guidelines and templates for quality assurance.

## Validation Scope:
- **Requirements**: Completeness against requirements-template.md
- **Design**: Architecture compliance with architectural-guidelines.md
- **Tasks**: Implementation plan completeness and feasibility
- **Code**: Standards compliance with coding-standards.md (if implementation exists)

## Validation Process:
1. Check all specification files exist and follow templates
2. Architecture-agent validates design against guidelines
3. Verify context consistency between specifications
4. Code quality check against standards (if applicable)

## Output Format:
```
🔍 Validation Report: $ARGUMENTS

Specifications:
✅ requirements.md - Complete and well-structured
✅ design.md - Architecture compliant
⚠️  tasks.md - Missing performance testing tasks

Code Quality:
✅ Type hints and mypy compliance
✅ Test coverage >80%
❌ Missing docstrings in user_service.py

Recommendations:
1. Add performance testing tasks to tasks.md
2. Add docstrings to user_service.py per coding standards
```

Use when: Need comprehensive quality assurance before development or deployment.