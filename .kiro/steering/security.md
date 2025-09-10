# Security & Privacy Standards

## Local-First Security Philosophy

### Core Principles
- **Data Never Leaves Device**: All processing happens locally
- **No Network Dependencies**: Core functionality works offline
- **User Controls Data**: Users own and control their knowledge
- **Minimal Attack Surface**: Reduce external dependencies and network exposure

### Privacy by Design
- **No Telemetry**: No usage data collection or phone-home functionality
- **No Cloud Processing**: Embeddings and analysis happen locally
- **No External APIs**: No calls to external services for core functionality
- **Encrypted Storage**: Sensitive data encrypted at rest (when applicable)

## Data Handling Standards

### Vault Data Protection
```python
# Always validate vault paths to prevent directory traversal
import os
from pathlib import Path

def validate_vault_path(vault_path: str) -> str:
    """Validate and normalize vault path to prevent security issues"""
    # Resolve to absolute path
    abs_path = Path(vault_path).resolve()
    
    # Ensure path exists and is a directory
    if not abs_path.exists() or not abs_path.is_dir():
        raise ValueError(f"Invalid vault path: {vault_path}")
    
    # Prevent access to system directories
    forbidden_paths = ["/etc", "/usr", "/bin", "/sbin", "/root"]
    if any(str(abs_path).startswith(path) for path in forbidden_paths):
        raise ValueError(f"Access to system directory forbidden: {vault_path}")
    
    return str(abs_path)
```

### File Access Controls
```python
# Safe file reading with size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

def safe_read_file(file_path: str) -> str:
    """Safely read file with size and path validation"""
    # Validate path
    abs_path = Path(file_path).resolve()
    
    # Check file size before reading
    if abs_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {file_path}")
    
    # Check file extension (only allow markdown and text)
    allowed_extensions = {'.md', '.txt', '.markdown'}
    if abs_path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"File type not allowed: {abs_path.suffix}")
    
    # Read with encoding detection
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 for problematic files
        with open(abs_path, 'r', encoding='latin-1') as f:
            return f.read()
```

### Database Security
```python
# Secure database connections
import duckdb
from pathlib import Path

def create_secure_database_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """Create database connection with security considerations"""
    # Ensure database is in allowed directory
    db_path = Path(db_path).resolve()
    allowed_db_dir = Path("data").resolve()
    
    if not str(db_path).startswith(str(allowed_db_dir)):
        raise ValueError(f"Database path outside allowed directory: {db_path}")
    
    # Create connection with read-only mode for queries when possible
    conn = duckdb.connect(str(db_path))
    
    # Disable potentially dangerous functions
    conn.execute("SET enable_external_access=false")
    
    return conn
```

## Input Validation & Sanitization

### MCP Parameter Validation
```python
from pydantic import BaseModel, validator
import re

class SecureSearchRequest(BaseModel):
    query: str
    limit: int = 10
    vault: Optional[str] = None
    
    @validator('query')
    def sanitize_query(cls, v):
        # Remove potentially dangerous characters
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        
        # Limit query length
        if len(v) > 1000:
            raise ValueError('Query too long (max 1000 characters)')
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        return sanitized.strip()
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v
    
    @validator('vault')
    def validate_vault_name(cls, v):
        if v is None:
            return v
        
        # Only allow alphanumeric, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid vault name format')
        
        return v
```

### SQL Injection Prevention
```python
# Always use parameterized queries
def safe_database_query(conn: duckdb.DuckDBPyConnection, query: str, params: tuple):
    """Execute database query safely with parameters"""
    # Whitelist allowed SQL operations
    allowed_operations = ['SELECT', 'INSERT', 'UPDATE', 'CREATE INDEX']
    
    query_upper = query.strip().upper()
    if not any(query_upper.startswith(op) for op in allowed_operations):
        raise ValueError(f"SQL operation not allowed: {query}")
    
    # Execute with parameters to prevent injection
    return conn.execute(query, params)

# Example usage
results = safe_database_query(
    conn,
    "SELECT * FROM documents WHERE vault_id = ? AND content LIKE ?",
    (vault_id, f"%{search_term}%")
)
```

## Network Security

### MCP Server Security
```python
# Secure MCP server configuration
class SecureMCPServer:
    def __init__(self):
        # Disable network access by default
        self.network_enabled = False
        
        # Rate limiting for requests
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        
        # Request size limits
        self.max_request_size = 1024 * 1024  # 1MB
    
    def handle_request(self, request: dict) -> dict:
        # Rate limiting
        if not self.rate_limiter.allow_request():
            return self.error_response("RATE_LIMIT_EXCEEDED", "Too many requests")
        
        # Size validation
        request_size = len(str(request))
        if request_size > self.max_request_size:
            return self.error_response("REQUEST_TOO_LARGE", "Request exceeds size limit")
        
        # Process request
        return self.process_request(request)
```

### External Dependencies
```python
# Minimize and validate external dependencies
ALLOWED_DOMAINS = {
    "huggingface.co",  # For model downloads only
}

def validate_external_request(url: str):
    """Validate external requests (if any are needed)"""
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    
    if parsed.netloc not in ALLOWED_DOMAINS:
        raise ValueError(f"External request to unauthorized domain: {parsed.netloc}")
    
    # Only allow HTTPS
    if parsed.scheme != 'https':
        raise ValueError(f"Only HTTPS requests allowed: {url}")
```

## Error Handling Security

### Information Disclosure Prevention
```python
def safe_error_response(error: Exception, user_message: str) -> dict:
    """Return safe error response without exposing internal details"""
    # Log full error details internally
    logger.error(f"Internal error: {error}", exc_info=True)
    
    # Return sanitized error to user
    return {
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": user_message,  # Generic user-friendly message
            "suggestions": ["Check logs for details", "Verify input parameters"]
        }
    }

# Usage
try:
    result = risky_operation()
except DatabaseError as e:
    return safe_error_response(e, "Database operation failed")
except FileNotFoundError as e:
    return safe_error_response(e, "Requested file not found")
```

### Path Traversal Prevention
```python
def safe_file_path(base_path: str, relative_path: str) -> str:
    """Safely join paths to prevent directory traversal"""
    base = Path(base_path).resolve()
    target = (base / relative_path).resolve()
    
    # Ensure target is within base directory
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"Path traversal attempt detected: {relative_path}")
    
    return str(target)
```

## Logging Security

### Secure Logging Practices
```python
import logging
from pathlib import Path

def setup_secure_logging():
    """Configure logging with security considerations"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set restrictive permissions on log directory
    log_dir.chmod(0o750)  # Owner: rwx, Group: r-x, Other: none
    
    # Configure logger
    logger = logging.getLogger('jarvis')
    
    # File handler with rotation
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "jarvis.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Secure formatter (avoid logging sensitive data)
    formatter = SecureFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SecureFormatter(logging.Formatter):
    """Formatter that sanitizes sensitive data from logs"""
    
    SENSITIVE_PATTERNS = [
        r'password=\w+',
        r'token=\w+',
        r'api_key=\w+',
    ]
    
    def format(self, record):
        msg = super().format(record)
        
        # Redact sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            msg = re.sub(pattern, '[REDACTED]', msg, flags=re.IGNORECASE)
        
        return msg
```

## Configuration Security

### Secure Configuration Management
```python
from pydantic import BaseSettings, validator
import os

class SecureConfig(BaseSettings):
    # Database configuration
    database_path: str = "data/jarvis.db"
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/jarvis.log"
    
    # Security settings
    max_file_size_mb: int = 10
    max_vault_files: int = 50000
    enable_debug: bool = False
    
    @validator('database_path')
    def validate_db_path(cls, v):
        # Ensure database is in data directory
        if not v.startswith('data/'):
            raise ValueError('Database must be in data directory')
        return v
    
    @validator('enable_debug')
    def validate_debug(cls, v):
        # Warn about debug mode in production
        if v and os.getenv('JARVIS_ENV') == 'production':
            raise ValueError('Debug mode not allowed in production')
        return v
    
    class Config:
        env_prefix = 'JARVIS_'
        case_sensitive = False
```

## Security Checklist

### Development Security
- [ ] Validate all user inputs with Pydantic models
- [ ] Use parameterized database queries
- [ ] Implement file size and type restrictions
- [ ] Prevent path traversal attacks
- [ ] Sanitize error messages
- [ ] Set restrictive file permissions
- [ ] Disable debug mode in production

### Deployment Security
- [ ] Run with minimal privileges
- [ ] Restrict network access
- [ ] Enable logging with rotation
- [ ] Set up file system permissions
- [ ] Validate configuration
- [ ] Monitor resource usage
- [ ] Regular security updates

### Data Protection
- [ ] Keep all data local
- [ ] No external API calls for core functionality
- [ ] Encrypt sensitive data at rest (if applicable)
- [ ] Secure database connections
- [ ] Validate vault paths
- [ ] Limit file access scope

## Security Incident Response

### If Security Issue Detected
1. **Isolate**: Stop affected services immediately
2. **Assess**: Determine scope and impact
3. **Contain**: Prevent further damage
4. **Investigate**: Analyze logs and system state
5. **Remediate**: Fix vulnerability and restore service
6. **Document**: Record incident and lessons learned

### Emergency Commands
```bash
# Stop all Jarvis processes
pkill -f "jarvis"

# Check for suspicious files
find data/ -type f -newer /tmp/reference_time

# Review recent logs
tail -100 logs/jarvis.log | grep -i "error\|warning\|fail"

# Verify file permissions
ls -la data/ logs/
```

## Privacy Compliance

### Data Minimization
- Only process files explicitly in vault directories
- Don't store unnecessary metadata
- Implement data retention policies
- Provide data deletion capabilities

### User Control
- Users control what data is indexed
- Clear data export capabilities
- Easy data deletion and cleanup
- Transparent about what data is stored

### No Tracking
- No user behavior analytics
- No usage statistics collection
- No external service dependencies
- No network communication for core features