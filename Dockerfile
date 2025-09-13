# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (if needed)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml /app/
COPY src /app/src
COPY resources /app/resources
COPY config /app/config

# Install package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Volumes for data and logs
VOLUME ["/data", "/logs", "/vault"]

# Default environment
ENV JARVIS_VAULT_PATH=/vault \
    JARVIS_DATABASE_PATH=/data/jarvis.duckdb \
    JARVIS_VECTOR_DB_PATH=/data/jarvis-vector.duckdb \
    JARVIS_LOG_FILE=/logs/mcp_server.log

# Healthcheck placeholder (could be a small script/ping endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD ["python", "-c", "import sys; sys.exit(0)"]

# Default command: run MCP server (stdio for MCP clients)
CMD ["jarvis-mcp-stdio"]

