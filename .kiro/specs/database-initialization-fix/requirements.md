# Requirements Document

## Introduction

The Jarvis MCP server is failing to start because it attempts to open a DuckDB database in read-only mode when the specific database file doesn't exist. The configuration points to `/Users/larsboes/Developer/Resources/data/Jarvis-Assistant-data/jarvis.duckdb`, but while the directory exists and contains other database files (like `jarvis-vector.duckdb`), the specific `jarvis.duckdb` file is missing. This creates a chicken-and-egg problem where the database needs to exist before the MCP server can start, but there's no clear initialization process for users to create the missing database file. This requirement addresses the need for robust database initialization and graceful handling of missing database files.

## Requirements

### Requirement 1

**User Story:** As a user setting up Jarvis for the first time, I want the system to automatically create and initialize the database if it doesn't exist, so that I don't encounter startup failures.

#### Acceptance Criteria

1. WHEN the MCP server starts AND the database file doesn't exist THEN the system SHALL create the database file and initialize the schema
2. WHEN the database directory doesn't exist THEN the system SHALL create the necessary directory structure
3. WHEN database initialization fails THEN the system SHALL provide clear error messages with actionable guidance
4. WHEN the database is successfully created THEN the system SHALL log the creation and proceed with normal startup

### Requirement 2

**User Story:** As a user, I want clear feedback about database status during startup, so that I can understand what's happening and troubleshoot issues if they occur.

#### Acceptance Criteria

1. WHEN the MCP server starts THEN the system SHALL log the database path being used
2. WHEN the database doesn't exist THEN the system SHALL log that it's creating a new database
3. WHEN database operations fail THEN the system SHALL provide specific error messages with the database path and failure reason
4. WHEN the database is successfully connected THEN the system SHALL log confirmation of the connection

### Requirement 3

**User Story:** As a developer, I want the database initialization to be robust and handle edge cases, so that the system works reliably across different environments and configurations.

#### Acceptance Criteria

1. WHEN the database path points to a directory instead of a file THEN the system SHALL detect this and provide a clear error message
2. WHEN the database file exists but is corrupted THEN the system SHALL attempt recovery or provide guidance for manual intervention
3. WHEN there are permission issues with the database path THEN the system SHALL provide clear error messages about file permissions
4. WHEN the database schema is outdated THEN the system SHALL handle migration or provide upgrade guidance

### Requirement 4

**User Story:** As a user, I want a command-line option to initialize or reset the database, so that I can manually manage the database state when needed.

#### Acceptance Criteria

1. WHEN I run a database initialization command THEN the system SHALL create a fresh database with proper schema
2. WHEN I run the initialization command on an existing database THEN the system SHALL ask for confirmation before overwriting
3. WHEN database initialization completes successfully THEN the system SHALL report the database location and basic statistics
4. WHEN initialization fails THEN the system SHALL provide specific error information and suggested remediation steps