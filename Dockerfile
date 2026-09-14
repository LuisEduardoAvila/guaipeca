# ─── Stage 1: Builder ────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv for clean package isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install guaipeca with all needed extras from pyproject.toml.
# Letting pip resolve from pyproject.toml ensures deps stay in sync.
# We use specific markitdown extras (not [all]) to avoid heavy deps.
# fastembed uses onnxruntime (~46MB) instead of torch (~959MB).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[pdf,docx,pptx,xlsx,monitor]"

# ─── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    libffi8 \
    libjpeg62-turbo \
    zlib1g \
    poppler-utils \
    antiword \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder (has all deps + package installed)
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /app/guaipeca

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/guaipeca/src"
ENV GUAIPECA_CONFIG="/data/config/guaipeca.yaml"

# Create non-root user
RUN groupadd -r guaipeca && \
    useradd -r -g guaipeca -d /app/guaipeca -s /bin/bash guaipeca && \
    mkdir -p /data && \
    chown -R guaipeca:guaipeca /app/guaipeca

WORKDIR /app/guaipeca

# Expose MCP HTTP/SSE port
EXPOSE 8090

# Health check — /health endpoint already exists
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')" || exit 1

# Entrypoint: create dirs with correct perms, then run as non-root user
ENTRYPOINT ["docker-entrypoint.sh", "guaipeca", "--config", "/data/config/guaipeca.yaml", "serve", "--transport", "http", "--port", "8090"]