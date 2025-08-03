# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# JARVIS ASSISTANT - MCP TOOLS FOR AI SYSTEMS

*project philosophy: "Excellent MCP tools for AI systems"*

## PROJECT MISSION

Production-ready MCP server providing semantic search and knowledge graph capabilities for AI systems. Built for Claude Desktop integration via Model Context Protocol (MCP) servers. Focus: Reliable, efficient tools for AI-powered knowledge discovery in Obsidian vaults.

## SPEC-DRIVEN DEVELOPMENT

For spec-driven development, follow `.kiro/specs/` specific folder for the feature.

## ESSENTIAL COMMANDS

**Setup & Quick Start:**
- `uv sync` - Install dependencies
- `uv run jarvis index --vault /path` - Index vault for search
- `uv run jarvis mcp --vault /path --watch` - Start MCP server

**Quality Assurance:**
- `uv run ruff check src/` - Lint code
- `uv run pytest resources/tests/` - Run tests
- `uv run pytest resources/tests/unit/test_interface_contracts.py -v` - Validate interface contracts

**Important**: Always use `python3` instead of `python` when running commands directly.

## ADAPTIVE CONTEXT SYSTEM

Load specific context based on task type using `.claude/` directory structure:

### Available Agents (`.claude/agents/`)
- **architecture-agent.md** - Senior System Architect for design enhancement and validation
- **design-agent.md** - Principal Architect for system design with visualization
- **dev-workflow-orchestrator.md** - Technical Lead for complete specification workflow
- **developer-agent.md** - Principal Python Developer for clean code with architectural compliance
- **product-owner-agent.md** - Senior Product Manager for professional requirements with business focus
- **tasks-agent.md** - Engineering Manager for implementation roadmap with development workflow

### Available Commands (`.claude/commands/`)
- **architecture-consult.md** - Architectural consultation and guidance
- **architecture-review.md** - Architecture review and validation
- **code-review.md** - Code review and quality assessment
- **design.md** - Create system design specification
- **dev-workflow.md** - Complete development workflow orchestration
- **develop.md** - Development task execution
- **requirements.md** - Create professional requirements specification
- **status.md** - Project status and progress tracking
- **tasks.md** - Create implementation task breakdown
- **validate.md** - Validation and testing workflows

### Guidelines (`.claude/guidelines/`)
- **architectural-guidelines.md** - SOLID principles, patterns, and architectural decisions
- **coding-standards.md** - Code quality standards and best practices

### Templates (`.claude/templates/`)
- **design-template.md** - System design specification template
- **requirements-template.md** - Requirements specification template
- **tasks-template.md** - Implementation tasks template

## CONTEXT LOADING STRATEGY

**For Development Tasks:**
- Agent: `developer-agent.md` - Clean code with architectural compliance
- Guidelines: `architectural-guidelines.md` + `coding-standards.md`

**For Architecture/Design:**
- Agent: `design-agent.md` or `architecture-agent.md` - System design and validation
- Guidelines: `architectural-guidelines.md`

**For Requirements/Planning:**
- Agent: `product-owner-agent.md` or `tasks-agent.md` - Business requirements and task planning
- Templates: Relevant specification templates

**For Workflow Orchestration:**
- Agent: `dev-workflow-orchestrator.md` - Complete specification workflow coordination

## DOCUMENTATION & SUPPORT

📚 **Complete Documentation**: [docs/README.md](docs/README.md) - 7-section comprehensive documentation system
🎯 **Troubleshooting**: [docs/07-maintenance/troubleshooting.md](docs/07-maintenance/troubleshooting.md)
⚡ **Performance**: [docs/07-maintenance/performance-tuning.md](docs/07-maintenance/performance-tuning.md)

## DEVELOPMENT REMINDERS

- Always use `python3` instead of `python` for direct command execution
- Follow `.kiro/specs/` for spec-driven development
- Use agents and guidelines for contextual development guidance
- Maintain interface contracts when making changes
- Update documentation when adding features or changing APIs