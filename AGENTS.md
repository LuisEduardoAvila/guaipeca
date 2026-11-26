# AGENTS.md

Instructions for AI agents working in this repository.

## Sanitize private information before committing documents or architecture notes

`docs/` is **tracked and public**. Design notes and architecture documents in this
repository are written during real deployments, so drafts naturally contain the
author's own infrastructure. None of that may reach a commit.

**Remove or generalize, in any document under `docs/` or in architecture prose:**

- Real IP addresses of home/lab machines (LAN `192.168.x`, tailnet `100.x`, public).
  Write `100.x.y.z`, `192.168.x`, or a named placeholder instead.
- Real hostnames, machine names, and tailnet DNS names. Use
  `host.tailnet-name.ts.net` and `host`/`second machine` rather than the actual
  identifiers.
- Peer counts, ACL snippets naming real devices, and topology sentences such as
  "machine A and machine B are already communicating over Tailscale".
- Personal filesystem paths (`/home/<name>/...`), usernames, and service file paths.
- Tokens, keys, session identifiers, and anything copied out of a credential store.

**Keep, they are not private:** RFC1918 and CGNAT ranges discussed as classes
(`192.168.0.0/16`, `100.64.0.0/10`), loopback (`127.0.0.1`), example ports, and
generic placeholders.

**Why this is a rule and not a preference:** the previous `.gitignore` entry for
`docs/` existed to stop exactly this leak. It is now removed because `docs/` is
meant to be public, so the sanitisation step is the control that replaced it.
Sanitise before staging, not after review — a commit is already public once pushed.

**Check before committing:**

```bash
git diff --cached -- docs/ | grep -nE "100\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+|/home/[a-z]+|\.ts\.net|hspi|evo"
```

An empty result is the expected one. If it matches, fix the document rather than
the check.

## Scope

- Work only inside the repository. Do not add agent state directories
  (`.openclaw/`, `.claude/`, `.codex/`) to the tree; they are ignored.
- Commit locally and let the operator push unless the task explicitly says otherwise.
- Do not modify `docs/` policy, `.gitignore` visibility rules, or this file as part of
  a feature task; raise it instead.
