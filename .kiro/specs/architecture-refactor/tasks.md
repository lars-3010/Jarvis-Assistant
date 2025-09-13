# Architecture Refactor & Codebase Clarity — Tasks

## Phase 1: Analysis & Planning
- [ ] 1. Codebase inventory
  - Map MCP tools, services, databases, extensions
  - Identify duplicates (e.g., tool registration paths) and large files
- [ ] 2. ADRs
  - DI usage in server context; structured response module; analytics invalidation via events

## Phase 2: Boundary & Composition
- [ ] 3. Integrate plugin registry with server listing/execution
- [ ] 4. Extract structured response schemas/formatters into shared module
- [ ] 5. Normalize naming and directory structure for MCP tools vs services vs extensions

## Phase 3: Docs Refresh
- [ ] 6. Update architecture docs
  - Extension system; analytics engine; DI path; event bus usage
- [ ] 7. Add quick reference for services and tool schemas

## Validation
- [ ] 8. Run tests; confirm no behavior regressions
- [ ] 9. Draft refactor changelog and migration notes (internal)

