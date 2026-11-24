# Guaipeca Container Deployment Guide

This guide covers deploying Guaipeca in a Docker container on a VM (ARM64 or x86_64).

## Prerequisites

- Docker Engine 24+ with BuildX
- Docker Compose v2+
- A VM running Linux (Ubuntu 22.04+, Debian 12+, or equivalent)
- Network access to ghcr.io (to pull the image) or to build from source

## Option 1: Docker Compose (Recommended)

### Step 1: Get the Code

```bash
git clone https://github.com/LuisEduardoAvila/SmartSpud.git
cd SmartSpud/projects/guaipeca
```

### Step 2: Prepare the Data Directory

```bash
mkdir -p data/{config,corpora,index,models}
```

### Step 3: Configure Guaipeca

```bash
cp config/guaipeca.container.yaml data/config/guaipeca.yaml
```

Edit `data/config/guaipeca.yaml`:
- Update `corpora` sections with your corpus names and paths
- Set `auth_token` if you want authentication
- Adjust `search.hybrid` if you want BM25 hybrid search

### Step 4: Add Source Documents

```bash
# Create corpus directories matching your config
mkdir -p data/corpora/epm-docs

# Copy your documents
cp /path/to/your/documents/* data/corpora/epm-docs/
```

### Step 5: Start the Container

```bash
docker compose up -d
```

### Step 6: Verify

```bash
# Check container is running
docker compose ps

# Check health endpoint
curl http://localhost:8090/health
# Expected: {"status": "ok", "server": "guaipeca", "version": "0.1.0"}

# Check tools endpoint
curl http://localhost:8090/tools
```

#### Verify the MCP Streamable HTTP endpoint

The recommended endpoint for modern MCP clients is `POST /mcp` (Streamable HTTP transport, protocol version `2025-11-25`). Perform an `initialize` handshake to confirm it is working and to obtain a session ID:

```bash
curl -s -D - -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"curl-test","version":"1.0"}}}'
# Expected: HTTP/1.1 200, an Mcp-Session-Id header, and protocolVersion in the body
```

Then send a follow-up request using the session ID returned above:

```bash
curl -s -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: <session-id-from-initialize>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
# Expected: JSON response containing the tools array
```

Terminate the session when done:

```bash
curl -s -X DELETE http://localhost:8090/mcp \
  -H "Mcp-Session-Id: <session-id-from-initialize>"
# Expected: {"status":"session terminated"}
```

> **Session lifetime:** `Mcp-Session-Id` values are held in memory only. A server (container) restart invalidates all sessions — clients must re-run `initialize`.
>
> **Protocol versions:** the server advertises `2025-11-25` and negotiates down for known older versions. Note that `POST /mcp` requires at least `2025-03-26`; a client sending `2024-11-05` to `/mcp` is rejected with a JSON-RPC Invalid params error. `2024-11-05` remains valid on the legacy `GET /sse` transport.
>
> **Legacy clients:** `GET /sse` and `POST /messages?session_id=<id>` remain fully functional and can be used concurrently with `/mcp` sessions.

### Step 7: Index Your Corpora (First Time)

The embedding model downloads on first run (~90MB). Wait for the server to be healthy, then trigger indexing:

```bash
# Index all corpora
docker compose exec guaipeca guaipeca --config /data/config/guaipeca.yaml index

# Or index a specific corpus
docker compose exec guaipeca guaipeca --config /data/config/guaipeca.yaml index epm-docs

# Force re-index everything
docker compose exec guaipeca guaipeca --config /data/config/guaipeca.yaml index --force
```

### Step 8: Check Status

```bash
docker compose exec guaipeca guaipeca --config /data/config/guaipeca.yaml status
```

## Option 2: Pre-built Image from ghcr.io

### Step 1: Authenticate

```bash
# Create a Personal Access Token with read:packages scope
echo $GITHUB_TOKEN | docker login ghcr.io -u LuisEduardoAvila --password-stdin
```

### Step 2: Pull the Image

```bash
docker pull ghcr.io/luis-eduardo-avila/guaipeca:latest
# Or a specific version:
# docker pull ghcr.io/luis-eduardo-avila/guaipeca:abc1234
```

### Step 3: Prepare Data Directory

```bash
mkdir -p data/{config,corpora,index,models}

# Create a minimal docker-compose.yml or use the one from the repo
# See docker-compose.yml in the project root
```

### Step 4: Run

```bash
docker run -d \
  --name guaipeca \
  -p 8090:8090 \
  -v $(pwd)/data:/data \
  --restart unless-stopped \
  ghcr.io/luis-eduardo-avila/guaipeca:latest
```

## Option 3: Build from Source

```bash
git clone https://github.com/LuisEduardoAvila/SmartSpud.git
cd SmartSpud/projects/guaipeca

# Build the image
docker build -t guaipeca:local .

# Run
docker run -d \
  --name guaipeca \
  -p 8090:8090 \
  -v $(pwd)/data:/data \
  --restart unless-stopped \
  guaipeca:local
```

## Managing the Container

### View Logs

```bash
docker compose logs -f guaipeca
# Or: docker logs -f guaipeca
```

### Restart

```bash
docker compose restart guaipeca
```

### Update to New Version

```bash
# Pull latest image
docker compose pull

# Restart with new image
docker compose up -d
```

### Stop

```bash
docker compose down
# Data persists in the ./data directory
```

## Configuration Reference

### Container-Specific Settings

| Setting | Container Value | Notes |
|---------|----------------|-------|
| `server.host` | `0.0.0.0` | Bind to all interfaces for container access |
| `server.transport` | `http` | No stdio needed in container; serves Streamable HTTP (`POST /mcp`) and legacy SSE |
| `server.port` | `8090` | Exposed in Dockerfile |
| `embedding.cache_dir` | `/data/models` | Volume-mounted, persists across restarts |
| `indexing.data_dir` | `/data/index` | Volume-mounted, persists across restarts |
| `corpora[].path` | `/data/corpora/<name>` | Volume-mounted source documents |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GUAIPECA_CONFIG` | `/data/config/guaipeca.yaml` | Path to config file |

### Authentication

To enable Bearer token auth, add to your config:

```yaml
server:
  auth_token: "your-secret-token"
```

Clients must then send `Authorization: Bearer your-secret-token` headers.

### Reaching the Container From Another Machine

The container config binds `0.0.0.0`, which inside a container means "listen on the
container's interfaces" — the host reaches it through the published port, and the port
mapping decides who can reach it. Two ways to expose that port, and they are not
equivalent:

```yaml
# docker-compose.yml — prefer this
ports:
  - "127.0.0.1:8090:8090"     # host loopback only; reach it via tunnel
# vs
ports:
  - "8090:8090"               # every host interface, including the LAN
```

The second form publishes on your LAN in plaintext. If you want other machines to use
the service, publish on loopback and put an encrypted overlay (Tailscale/WireGuard) in
front of it, or publish bound to the overlay address:

```yaml
ports:
  - "100.x.y.z:8090:8090"     # this host's tailnet address only
```

In all cases keep `auth_token` set. A tailnet address authenticates *devices*, not
*service users*. Browser clients additionally need the exact origin in
`server.allowed_origins` — see
[docs/network-access-design.md](network-access-design.md).

### Health Check

The container includes a built-in health check hitting `GET /health`:

```json
{"status": "ok", "server": "guaipeca", "version": "0.1.0"}
```

Check health manually:

```bash
docker inspect --format='{{.State.Health.Status}}' guaipeca
```

## Model Caching

The embedding model (`all-MiniLM-L6-v2`, ~90MB) is NOT baked into the Docker image. On first start:

1. Container starts, server initializes
2. `_warmup_model()` triggers model download from HuggingFace Hub
3. Model is saved to `/data/models/` (the volume)
4. Subsequent starts load from cache — no network needed

**Pre-populate the model cache** (optional, avoids first-run download):

```bash
# Copy existing model cache into the volume
cp -r ~/.guaipeca/models/* data/models/

# Or run the check-model command
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/luis-eduardo-avila/guaipeca:latest \
  guaipeca --config /data/config/guaipeca.yaml --check-model
```

## Troubleshooting

### Container won't start

1. Check logs: `docker compose logs guaipeca`
2. Verify config exists: `ls data/config/guaipeca.yaml`
3. Verify config syntax: `docker compose exec guaipeca python -c "import yaml; yaml.safe_load(open('/data/config/guaipeca.yaml'))"`

### Health check fails

1. Wait 60+ seconds (start_period in healthcheck)
2. Check if model is downloading: `docker compose logs guaipeca | grep -i model`
3. Verify port mapping: `docker compose ps`

### Port 8090 already in use

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "9090:8090"  # Map host 9090 to container 8090
```

### Corpora not found

Ensure corpus paths in config match the volume structure:

```bash
# Config says: path: /data/corpora/epm-docs
# So on the host: data/corpora/epm-docs/ must exist
ls data/corpora/epm-docs/
```

### Index not working

```bash
# Check if corpus path exists inside container
docker compose exec guaipeca ls /data/corpora/

# Re-index
docker compose exec guaipeca guaipeca --config /data/config/guaipeca.yaml index --force
```

## Backup

The entire persistent state is in the `data/` directory:

```bash
# Backup
tar czf guaipeca-backup-$(date +%Y%m%d).tar.gz data/

# Restore
tar xzf guaipeca-backup-YYYYMMDD.tar.gz
docker compose up -d
```