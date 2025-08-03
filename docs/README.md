# Jarvis Assistant Documentation

Welcome to the comprehensive documentation for Jarvis Assistant - an MCP server providing semantic search and graph analysis tools for AI systems to intelligently discover knowledge in Obsidian vaults.

## Documentation Map

### 📋 01-overview/ - Understanding the Project
Get oriented with what Jarvis Assistant is and how it works.

- **[project-overview.md](01-overview/project-overview.md)** - Mission, problem statement, solution approach, and current status
- **[architecture.md](01-overview/architecture.md)** - High-level system design, components, and technology choices
- **[key-concepts.md](01-overview/key-concepts.md)** - Core terminology and mental models for understanding the system
- **[architectural-roadmap.md](01-overview/architectural-roadmap.md)** - Technical infrastructure improvements and modularity roadmap
- **[ai-implementation-roadmap.md](01-overview/ai-implementation-roadmap.md)** - Comprehensive plan for AI agents and GraphRAG capabilities

### 🔧 02-system-design/ - How It Works
Deep dive into the technical implementation and system internals.

- **[system-overview.md](02-system-design/system-overview.md)** - High-level architecture and design philosophy
- **[architecture-decisions.md](02-system-design/architecture-decisions.md)** - Key technical decisions and their rationale (ADRs)
- **[extension-architecture.md](02-system-design/extension-architecture.md)** - ⭐ Plugin system design and implementation *(New in Phase 0)*
- **[deployment-patterns.md](02-system-design/deployment-patterns.md)** - Different deployment scenarios and configurations
- **[data-flow.md](02-system-design/data-flow.md)** - Request-to-response journey and processing pipeline
- **[component-interaction.md](02-system-design/component-interaction.md)** - How MCP server, databases, and services interact
- **[semantic-search-design.md](02-system-design/semantic-search-design.md)** - Vector search architecture and embeddings
- **[neo4j-schema.md](02-system-design/neo4j-schema.md)** - Graph database structure and relationships
- **[mcp-implementation-details.md](02-system-design/mcp-implementation-details.md)** - MCP protocol implementation details

### 🚀 03-getting-started/ - Getting Up and Running
Progressive setup guides from quickstart to advanced configuration.

- **[quick-start.md](03-getting-started/quick-start.md)** - 5-minute setup to get working system
- **[detailed-installation.md](03-getting-started/detailed-installation.md)** - Complete installation with all options
- **[configuration.md](03-getting-started/configuration.md)** - Understanding and customizing settings
- **[first-queries.md](03-getting-started/first-queries.md)** - Testing your setup and basic usage

### 📖 04-usage/ - Practical Application
Real-world usage patterns and examples.

- **[api-examples.md](04-usage/api-examples.md)** - Copy-paste code examples for each MCP tool
- **[common-workflows.md](04-usage/common-workflows.md)** - "I want to..." scenarios with step-by-step solutions
- **[advanced-queries.md](04-usage/advanced-queries.md)** - Power user examples and complex search patterns

### 💻 05-development/ - Contributing and Development
For developers who want to contribute or modify the system.

- **[developer-guide.md](05-development/developer-guide.md)** - Local development setup, workflow, and debugging
- **[testing-strategy.md](05-development/testing-strategy.md)** - Running tests and adding new test coverage
- **[code-standards.md](05-development/code-standards.md)** - Coding conventions, linting, and code quality
- **[contribution-guide.md](05-development/contribution-guide.md)** - How to contribute, PR process, and communication
- **[extension-development.md](05-development/extension-development.md)** - ⭐ Creating extensions for AI and custom functionality *(New in Phase 0)*

### 📚 06-reference/ - Quick Lookup
Comprehensive reference materials for specific information.

- **[api-reference.md](06-reference/api-reference.md)** - Complete MCP tool documentation with parameters and responses
- **[configuration-reference.md](06-reference/configuration-reference.md)** - All configuration options with examples
- **[error-codes.md](06-reference/error-codes.md)** - Error messages, causes, and solutions

### 🔧 07-maintenance/ - Operational Excellence
Keeping your system running smoothly and handling issues.

- **[troubleshooting.md](07-maintenance/troubleshooting.md)** - Common problems and step-by-step solutions
- **[performance-tuning.md](07-maintenance/performance-tuning.md)** - Optimization guides and monitoring
- **[backup-recovery.md](07-maintenance/backup-recovery.md)** - Data protection and recovery procedures
- **[updates-migration.md](07-maintenance/updates-migration.md)** - Handling system updates and migrations

### 📊 08-progression/ - Version Tracking & Evolution
System development progression and version history.

- **[version-history.md](08-progression/version-history.md)** - Complete chronological development history
- **[current-phase.md](08-progression/current-phase.md)** - Active development phase and milestone tracking
- **[milestone-log.md](08-progression/milestone-log.md)** - Detailed milestone completion records
- **[architectural-evolution.md](08-progression/architectural-evolution.md)** - How system architecture has evolved
- **[lessons-learned.md](08-progression/lessons-learned.md)** - Development insights and decision rationale

## Start Here If You're...

### 🆕 New User
Never used Jarvis Assistant before? Start with the basics:

1. **[Project Overview](01-overview/project-overview.md)** - Understand what this project does
2. **[Quick Start Guide](03-getting-started/quick-start.md)** - Get up and running in 5 minutes
3. **[Common Workflows](04-usage/common-workflows.md)** - Learn typical usage patterns

### 🔍 Curious About How It Works
Want to understand the technical implementation?

1. **[Key Concepts](01-overview/key-concepts.md)** - Learn the terminology and mental models
2. **[System Architecture](01-overview/architecture.md)** - Understand the overall design
3. **[Data Flow](02-system-design/data-flow.md)** - See how requests become responses

### 💻 Developer or Contributor
Want to modify, extend, or contribute to the project?

1. **[Developer Guide](05-development/developer-guide.md)** - Set up your development environment
2. **[System Design](02-system-design/component-interaction.md)** - Understand component interactions
3. **[Code Standards](05-development/code-standards.md)** - Follow project conventions
4. **[AI Implementation Roadmap](01-overview/ai-implementation-roadmap.md)** - Future AI capabilities and implementation plan

### 🤖 AI System or Tool
Need structured information for automated processing?

1. **[System Overview](02-system-design/system-overview.md)** - High-level architecture and capabilities
2. **[Component Interaction](02-system-design/component-interaction.md)** - Detailed system architecture
3. **[API Reference](06-reference/api-reference.md)** - Complete tool specifications
4. **[Configuration Reference](06-reference/configuration-reference.md)** - All configuration options

### 🚨 Problem Solver
Having issues or need specific information?

1. **[Troubleshooting Guide](07-maintenance/troubleshooting.md)** - Common problems and solutions
2. **[Error Codes Reference](06-reference/error-codes.md)** - Specific error messages and fixes
3. **[Performance Tuning](07-maintenance/performance-tuning.md)** - Optimization and monitoring

## Quick Links to Common Tasks

### Setup and Installation
- [5-minute quick start](03-getting-started/quick-start.md)
- [Complete installation guide](03-getting-started/detailed-installation.md)
- [Claude Desktop configuration](03-getting-started/configuration.md)

### Usage and Examples
- [Basic usage examples](04-usage/api-examples.md)
- [Semantic search patterns](04-usage/common-workflows.md)
- [Advanced search techniques](04-usage/advanced-queries.md)

### Development and Contributing
- [Local development setup](05-development/developer-guide.md)
- [Running tests](05-development/testing-strategy.md)
- [Code contribution process](05-development/contribution-guide.md)

### Maintenance and Support
- [Common troubleshooting](07-maintenance/troubleshooting.md)
- [Performance optimization](07-maintenance/performance-tuning.md)
- [System updates](07-maintenance/updates-migration.md)

## Documentation Philosophy

This documentation is designed with these principles:

- **Progressive Disclosure**: Start simple, get more detailed as needed
- **Multiple Entry Points**: Different paths for different audiences
- **Cross-Referenced**: Easy navigation between related topics
- **AI-Friendly**: Structured for both human and AI consumption
- **Practical Focus**: Real examples and working solutions
- **Maintained**: Updated with code changes and user feedback

## Need Help?

- **Quick Questions**: Check the [troubleshooting guide](07-maintenance/troubleshooting.md)
- **API Questions**: See the [API reference](06-reference/api-reference.md)
- **Setup Issues**: Follow the [installation guide](03-getting-started/detailed-installation.md)
- **Development Help**: Read the [developer guide](05-development/developer-guide.md)

---

*This documentation is maintained as part of the Jarvis Assistant project. For corrections or improvements, please see the [contribution guide](05-development/contribution-guide.md).*