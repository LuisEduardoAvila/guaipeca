# Deployment security posture

The short answer to "how should this be deployed, and what do I have to set."
Reasoning: [network-access-design.md](network-access-design.md); full container walkthrough: [container-deployment.md](container-deployment.md).

## The default

Binds `127.0.0.1:8090` with no token. Supported as-is: reachable only from this
host, and every start announces the tokenless loopback bind on stderr.

## The rule

The server refuses to start on any non-loopback bind without `auth_token` — that
is the only enforcement. Set it via `GUAIPECA_AUTH_TOKEN` (recommended; keeps the
secret out of the config) or `server.auth_token`.

## The three supported shapes

### 1. Same host / SSH tunnel (default)

Nothing to configure; other machines tunnel in: `ssh -L 8090:127.0.0.1:8090` (or `tailscale serve`).

### 2. Container behind a tunnel

The container binds `0.0.0.0` *inside* itself; the host port mapping decides
exposure. Publish on host loopback only, and set the token:

```yaml
ports:
  - "127.0.0.1:8090:8090"
environment:
  - GUAIPECA_AUTH_TOKEN=${GUAIPECA_AUTH_TOKEN:?set GUAIPECA_AUTH_TOKEN before starting}
```

### 3. Tailnet

Publish on this host's tailnet address only, token required:

```yaml
ports:
  - "100.x.y.z:8090:8090"
```

Bare metal / systemd binds the address directly instead:

```yaml
server:
  host: 100.x.y.z            # this host's tailscale0 address
  allowed_origins:
    - http://name.tailnet.ts.net:8090   # browser clients: exact origin, no wildcards
```

## Not supported

- **Plain LAN bind** — plaintext on the segment; queries, results and `/download`
  payloads are readable, not just the token. Use a tunnel.
- **Public bind** — a plaintext public address is not a supported configuration.

With a token set the code cannot tell either apart from a tailnet bind; the argument for why, and the trust-class table, live in the design note.
