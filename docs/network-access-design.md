# Network Access Design

Status: **adopted** — addresses [#69](https://github.com/LuisEduardoAvila/guaipeca/issues/69). The design decision is settled; the operator-side container step is tracked separately.

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

**TLS is not a no-op, though, and the earlier framing of "adds nothing" was too strong.** Where WireGuard is in use, TLS still buys four things that have nothing to do with wire confidentiality:

1. **The client can verify the server's identity by name.** `tailscale cert` issues a publicly trusted cert for the `*.ts.net` name. WireGuard authenticates the *channel endpoints*; it does not tell a browser that the HTTP service behind those endpoints is the one it expected.
2. **It defeats an on-path attacker *inside* the tailnet.** A compromised co-member device or a misconfigured subnet router can sit between two peers it is allowed to talk to. WireGuard protects the tunnel from outside the peers; TLS protects the request from anything between the service and the client.
3. **Secure-context browser APIs.** Service workers, clipboard, camera and geolocation all require a secure context. A future browser UI ([#87](https://github.com/LuisEduardoAvila/guaipeca/issues/87)) that wants any of them will need `https://` regardless of the network underneath.
4. **Compliance and expectation.** "Encrypted in transit" is usually read as TLS. Explaining WireGuard to an auditor is a conversation; a cert is not.

None of these is load-bearing today, which is why the recommendation stands. But they are the reasons `tailscale cert` is listed as a one-command follow-up rather than something to argue against.

## Why not loopback forever

It is the most secure and the least useful. The read paths (`search`, `get_document`) are the point of the service, and they are wanted from a phone and from other machines. Loopback-only means every client needs its own tunnel, which is the same overlay network with more steps.

## The trust classes

The phrase "non-local" hides four situations with different requirements. This table is the normative part of the document.

**Read the two middle columns carefully.** They describe what the *code enforces*; only the right-hand column describes *policy*. `_check_bind_exposure` distinguishes exactly two states: loopback (bindable, announced if tokenless) and everything else (refused unless a token is set). It does not classify anything in between — a `192.168.x` address, a `100.64.0.0/10` address, `0.0.0.0`, and a public IP are all identical to it.

| Class | Example | Encrypted in transit? | What the code enforces | Recommended posture (policy, not enforced) |
|---|---|---|---|---|
| **Loopback** | `127.0.0.1`, `::1` | N/A (no wire) | Binds freely; announcement when tokenless | Fine. Also the target when a proxy terminates TLS in front. |
| **Tailnet** | `100.64.0.0/10`, `fd7a:115c:a1e0::/48` | Yes, WireGuard | **Token required** | Recommended for reaching other machines. Bind the specific address. |
| **LAN** | `192.168.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12` | **No** | **Token required** — same as any other non-loopback bind | Discouraged. Use a tunnel instead. If you must, know that queries, results, document text and `/download` payloads are all readable on the segment — not just the token. |
| **Wildcard / public** | `0.0.0.0`, `::`, any routable IP | No | **Token required** — same as any other non-loopback bind | Discouraged. Reaching a public address without TLS is not a supported configuration. |

Two things follow from this table:

- **The token is required on every non-loopback bind, including the tailnet, and that is the only thing the code enforces.** Tailscale authenticates *devices*, not *users of the service*. A compromised phone in your tailnet is an authenticated device. The token is what separates "on my network" from "allowed to read my documents".
- **The policy column is advice, not an interlock.** Nothing stops an operator from binding a public address with a token; the code cannot tell that bind apart from a tailnet one. The controls that actually bound exposure live outside this process: what you publish in `docker-compose.yml`, your firewall, and your tailnet ACLs. This document's recommendation is therefore a recommendation, and the failure mode it does not prevent is deliberate misconfiguration.

## Why the bind check stays conservative

The temptation is to teach `_check_bind_exposure` about `100.64.0.0/10` and skip the warning, on the grounds that a tailnet address is not publicly reachable. The reason not to:

**The real reason is that reachability is unknowable, not that addresses are unclassifiable.** A process *can* trivially classify an address: `ipaddress` alone separates RFC1918 from `100.64.0.0/10` from a globally-routable one, and `getifaddrs` reveals interface flags. What it cannot determine is **who can route to that interface** — whether a subnet router exposes the tailnet to a LAN, whether a port-forward maps this bind to a WAN address, whether a bridge network is published on the host's uplink. That is the question the policy is actually about, and it is not answerable from inside the process.

An earlier draft of this section argued that the process "cannot classify addresses" and that the loopback check is "correct by construction". Both claims overstate the case. The check is not correct by construction — once a token is set it exercises no control at all, and any public bind passes. And whitelisting `100.64.0.0/10` would not in itself be fragile: it compares a string, not live interface state.

The argument that actually holds is narrower: **a whitelist of "trusted" address classes is a permanent exemption for a property the process cannot verify.** "This bind is on the tailnet" is a claim about deployment, and deployments drift. Today `100.64.0.0/10` means a WireGuard tunnel; tomorrow the same host may route that range differently, or the operator may bind a subnet router's address. The check cannot tell, so it should not be asked to.

**The consequence is honest and worth stating plainly:** the check enforces one thing — no token, no non-loopback bind — and it does not enforce the policy column of the table above. The policy is enforced by what the operator publishes and routes, not by this process.

There is no override flag, and there should not be one. The existing docstring already says this: *"An operator who wants the service reachable sets an `auth_token`; one who does not want auth keeps the bind on loopback and reaches it through a tunnel."* That sentence is the whole policy, and Tailscale is the tunnel.

**Residual annoyance, accepted:** an operator binding to their tailnet address will get the same refusal as a LAN bind, and must set a token. That is one environment variable. It is not friction worth a special case.

## What actually needs to change

Almost nothing, which is the point.

### 1. Documentation (the real work)

- **`docs/container-deployment.md`** describes the container-specific shape, which differs from the bare-metal one below — see §2. Container exposure is controlled by **host-side port publishing**, not by `server.host`, because a bridge-mode container cannot bind the host's tailnet address. Its `auth_token` instructions now state that the token is required, not optional.
- **This document** records the trust classes so the next person does not have to re-derive them.

### 2. Bind-address strategy

Two shapes, and they are not interchangeable.

**Bare metal / systemd — bind the interface address:**

```yaml
server:
  host: 100.x.y.z          # this host's tailscale0 address
  port: 8090
  allowed_origins:
    - http://host.tailnet-name.ts.net:8090
```

`0.0.0.0` would also listen on `end0` (the LAN) and every docker bridge. The bind check cannot distinguish those from the tailnet one because it only sees what was asked for, so binding to the specific address is the control that matters here. It is also the honest statement of intent.

**Container (bridge networking) — keep `0.0.0.0` inside, publish narrowly on the host:**

```yaml
# docker-compose.yml
ports:
  - "127.0.0.1:8090:8090"   # host loopback only; reach it via a tunnel
# or
  - "100.x.y.z:8090:8090"   # publish on this host's tailnet address only
```

Inside a bridge-mode container, `server.host: 0.0.0.0` means "the container's own interfaces" — it is not host exposure, and binding it to the host's tailnet address does not work without host networking. The exposure decision is therefore entirely in the port mapping, and the token is still required because the container sees a non-loopback bind.

**Startup ordering (bare metal):** `tailscale0` may not have an address when the service starts. Binding to an address that does not exist yet fails, and should fail loudly rather than fall back to `0.0.0.0`. The systemd fix is `ExecStartPre=` waiting for the interface; in a container, the port binding simply fails and the container restarts, which is the equivalent loud failure.

### 3. CORS origins

The UI from [#87](https://github.com/LuisEduardoAvila/guaipeca/issues/87) is the first browser client, so this matters more than it does for CLI/MCP clients. A browser at `http://host.tailnet-name.ts.net:8090` sends that Origin, and `server.allowed_origins` must contain it exactly (scheme, host, port). The `#37` rule applies: no wildcards, list each origin explicitly.

### 4. Tailnet ACLs (deployment, not code)

Encrypted transport says the packets are private. It does not say which devices may reach the port. That is Tailscale ACLs, and they live outside this repository:

```jsonc
{
  "grants": [
    { "src": ["autogroup:member"], "dst": ["host:8090"], "ip": ["tcp"] }
  ]
}
```

Worth writing down in the deployment doc that the default (before any ACL) is "every device in the tailnet can reach it". The token is the thing that actually stops a rogue device; ACLs narrow it further.

## What is explicitly not decided here

- **`tailscale cert` / HTTPS on the tailnet name.** Available and one command, but not required for a plain HTTP UI. If a future feature needs a secure browser context (clipboard, service workers, camera), this becomes required and is a small follow-up rather than a redesign.
- **Tailscale Funnel** (public exposure through Tailscale's infrastructure). Out of scope: that is a public bind by another name, and the trust-class table says no.
- **Whether the UI binds its own port** or reuses the existing one. That is #87's decision.

## Success criteria for closing #69

**Done by this document:**

- [x] The chosen posture is stated explicitly: where the service listens, what encrypts the traffic, and what the operator must configure
- [x] The trust-class table (§"The trust classes") is recorded and referenced from `docs/ARCHITECTURE.md`'s "Network exposure" section
- [x] Both deployment shapes are described — bare-metal bind-to-address, and container publish-narrowly — with the reason they differ

**Operator tasks, not code — tracked in #69 until done:**

- [ ] A deployment is actually running with the narrow publish (or the specific bind) and a token set
- [ ] `docs/container-deployment.md` has its `auth_token` language corrected so `host: 0.0.0.0` is never presented as optional-token
- [ ] Issue #69's acceptance criterion "`docs/` gains a short *Deployment security posture* section" — either add one that points here, or amend the criterion to say this document is it

**Guard for the next person:**

- [ ] `_check_bind_exposure` is **unchanged**. If a code change appears necessary, that is a signal the design drifted, not that the check is wrong. The check enforces "token or loopback" and nothing else — do not add address-class exemptions to it.

## Non-goals

- Changing the auth model. One shared token, as in #53.
- Supporting multiple users, roles or sessions.
- Adding a `--insecure` style override to the bind check.
- Making TLS mandatory for loopback.

## Context

Written after confirming that the host and a second machine on the same network are already communicating over Tailscale. The service is therefore already on an overlay network; this document formalises that as the supported deployment shape rather than inventing a new capability.
