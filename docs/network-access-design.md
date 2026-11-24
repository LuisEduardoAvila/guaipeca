# Network Access Design

Status: **proposal** — resolves [#69](https://github.com/LuisEduardoAvila/guaipeca/issues/69). Not yet implemented.

## The question

#69 says: *"Deployment: no TLS on a plaintext wildcard bind."* The concern is that guaipeca can be started listening on a non-loopback address, serving plaintext HTTP, and that anything on the network segment can read the traffic — including the bearer token, which is the only authentication there is.

There are three plausible answers and this document picks one:

1. **App-level TLS** — terminate HTTPS in guaipeca, manage certificates.
2. **Overlay network** — reach the service over an already-encrypted, already-authenticated network (Tailscale/WireGuard), and never bind it to a plaintext LAN.
3. **Loopback forever** — refuse every non-loopback bind, full stop.

## Recommendation

**Overlay network, with the conservative bind check left alone.**

Concretely: bind to the Tailscale interface, keep the auth token required, never bind to a plain LAN address, and do not teach `_check_bind_exposure` that Tailscale is special.

This satisfies the threat #69 names without adding certificate management to a service that currently has none.

## Why not app-level TLS

TLS in the application means: key material on disk, a renewal path, a hostname that matches the certificate, and a new failure mode where an expired cert takes the service down. On a Pi-class single-operator host, that is a permanent operational tax to solve a confidentiality problem that WireGuard already solves one layer down. If guaipeca is ever required to serve a public hostname, TLS becomes necessary — but that is a different deployment, and it should be a different decision rather than a default.

## Why not loopback forever

It is the most secure and the least useful. The read paths (`search`, `get_document`) are the point of the service, and they are wanted from a phone and from other machines. Loopback-only means every client needs its own tunnel, which is the same overlay network with more steps.

## The trust classes

The phrase "non-local" hides four situations with different requirements. This table is the normative part of the document.

| Class | Example | Encrypted? | Token required? | Can bind? |
|---|---|---|---|---|
| **Loopback** | `127.0.0.1`, `::1` | N/A (no wire) | No, but announced | Yes |
| **Tailnet** | `100.64.0.0/10`, `fd7a:115c:a1e0::/48` | Yes, WireGuard | **Yes** | Yes |
| **LAN** | `192.168.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12` | No | Yes, and plaintext is the residual risk | **No** — use a tunnel |
| **Public** | any routable address | No | Yes, and this is still wrong | **No** |

Two things follow from this table:

- **The token is required on every non-loopback bind, including the tailnet.** Tailscale authenticates *devices*, not *users of the service*. A compromised phone in your tailnet is an authenticated device. The token is what separates "on my network" from "allowed to read my documents". This is why the conservative choice costs nothing: the existing `_check_bind_exposure` already enforces exactly this rule, and it needs no change.
- **A Tailscale bind is not a public bind.** It is also not a loopback bind. The current check treats it as the former and warns/refuses accordingly, which is *correct but under-explained* — the error message tells the operator to set a token, which is the right instruction, but it does not say that a tailnet bind is a supported and expected configuration. That is a documentation fix, not a code fix.

## Why the bind check stays conservative

The temptation is to teach `_check_bind_exposure` about `100.64.0.0/10` and skip the warning, on the grounds that a tailnet address is not publicly reachable. The reason not to:

**The assumption decays.** Today the machine has `tailscale0`, `end0` and a pile of docker bridges. Tomorrow it has a subnet router, a second physical interface, or an exit node configured differently. The check's job is to be dumb enough that a change in network topology cannot silently turn a guarded configuration into an unguarded one. A check that reasons about address classes has to be correct about every interface, on every host, forever; a check that asks "is this loopback?" is correct by construction.

There is no override flag, and there should not be one. The existing docstring already says this: *"An operator who wants the service reachable sets an `auth_token`; one who does not want auth keeps the bind on loopback and reaches it through a tunnel."* That sentence is the whole policy, and Tailscale is the tunnel.

**Residual annoyance, accepted:** an operator binding to their tailnet address will get the same refusal as a LAN bind, and must set a token. That is one environment variable. It is not friction worth a special case.

## What actually needs to change

Almost nothing, which is the point.

### 1. Documentation (the real work)

- **`docs/container-deployment.md`** gains a "Reaching it from another machine" section: bind `server.host` to the tailnet address, set `GUAIPECA_AUTH_TOKEN`, add the MagicDNS origin to `server.allowed_origins`.
- **This document** records the trust classes so the next person does not have to re-derive them.

### 2. Bind-address strategy

Bind to the interface address, not `0.0.0.0`:

```yaml
server:
  host: 100.127.56.18      # this host's tailscale0 address
  port: 8090
  allowed_origins:
    - http://hspi.tailnet-name.ts.net:8090
```

`0.0.0.0` would also listen on `end0` (the LAN) and every docker bridge. The bind check cannot distinguish those from the tailnet one because it only sees what was asked for, so binding to the specific address is the control that matters here. It is also the honest statement of intent.

**Startup ordering:** `tailscale0` may not have an address when the service starts. Binding to an address that does not exist yet fails, and should fail loudly rather than fall back to `0.0.0.0`. If this becomes a problem in the container deployment, the fix is `ExecStartPre=` waiting for the interface, not a `0.0.0.0` fallback.

### 3. CORS origins

The UI from [#87](https://github.com/LuisEduardoAvila/guaipeca/issues/87) is the first browser client, so this matters more than it does for CLI/MCP clients. A browser at `http://hspi.tailnet-name.ts.net:8090` sends that Origin, and `server.allowed_origins` must contain it exactly (scheme, host, port). The `#37` rule applies: no wildcards, list each origin explicitly.

### 4. Tailnet ACLs (deployment, not code)

Encrypted transport says the packets are private. It does not say which devices may reach the port. That is Tailscale ACLs, and they live outside this repository:

```jsonc
{
  "grants": [
    { "src": ["autogroup:member"], "dst": ["hspi:8090"], "ip": ["tcp"] }
  ]
}
```

Worth writing down in the deployment doc that the default (before any ACL) is "every device in the tailnet can reach it". The token is the thing that actually stops a rogue device; ACLs narrow it further.

## What is explicitly not decided here

- **`tailscale cert` / HTTPS on the tailnet name.** Available and one command, but not required for a plain HTTP UI. If a future feature needs a secure browser context (clipboard, service workers, camera), this becomes required and is a small follow-up rather than a redesign.
- **Tailscale Funnel** (public exposure through Tailscale's infrastructure). Out of scope: that is a public bind by another name, and the trust-class table says no.
- **Whether the UI binds its own port** or reuses the existing one. That is #87's decision.

## Success criteria for closing #69

- [ ] `docs/container-deployment.md` documents the tailnet bind recipe, including the token requirement and the exact `allowed_origins` entry
- [ ] The doc states that `0.0.0.0` is not the recommended way to serve a tailnet client, and why
- [ ] A container deployment exists (or is documented) with `server.host` on the tailnet address and a token set
- [ ] `_check_bind_exposure` is **unchanged** — if a code change appears necessary, that is a signal the design drifted, not that the check is wrong
- [ ] The trust-class table above is referenced from `docs/ARCHITECTURE.md`

## Non-goals

- Changing the auth model. One shared token, as in #53.
- Supporting multiple users, roles or sessions.
- Adding a `--insecure` style override to the bind check.
- Making TLS mandatory for loopback.

## Context

Written after confirming that hspi (`100.127.56.18`) and evo (`100.120.69.29`) are already communicating over Tailscale, with six peers in the tailnet. The service is therefore already on an overlay network; this document formalises that as the supported deployment shape rather than inventing a new capability.
