# Security Policy

## Supported Versions

smoltorrent does not have versioned releases. Security fixes are applied to the latest commit on `master` only.

| Version | Supported |
|---|---|
| Latest (`master`) | Yes |
| Older commits | No |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's [Private Vulnerability Reporting](https://github.com/YuvrajSingh-mist/smoltorrent/security/advisories/new) to report issues confidentially. Only you and the maintainer can see the report until it is resolved.

Include as much of the following as you can:

- A description of the vulnerability and where it exists in the codebase
- Steps to reproduce the issue
- The potential impact (what an attacker could achieve)
- Any suggested fix or mitigation, if you have one

You will receive an acknowledgement within **72 hours**. If the issue is confirmed, a fix will be prioritised and you will be kept in the loop throughout. You will be credited in the release notes unless you prefer otherwise.

## Scope

**In scope:**

- The FastAPI HTTP API (`backend/api.py`) — endpoint security, input validation, path traversal
- The TCP shard protocol on workers (`algorithms/SyncPS/worker.py`) — protocol abuse, malformed payloads
- SHA-256 integrity verification — bypass or collision attacks
- The mDNS/discovery layer (`discovery/`) — spoofing or poisoning
- The watcher daemon (`watcher/watch.py`) — arbitrary file access via crafted checkpoint paths

**Out of scope:**

- The Raspberry Pi hardware and OS-level security
- Tailscale VPN itself (report to [Tailscale](https://tailscale.com/security))
- Third-party Python packages (report to the respective maintainers)
- Attacks that require physical access to the cluster

## Security Design Decisions

smoltorrent is designed for a **trusted private network** (Tailscale VPN). As a result:

- The HTTP API and TCP worker protocol have **no authentication by default** — this is intentional; the Tailscale network is the trust boundary
- If you expose the API or worker ports outside of a trusted VPN, you are responsible for adding your own authentication layer
- Every shard is SHA-256 verified on receipt; corrupted or tampered shards are rejected and deleted

## Disclosure Policy

This project follows **coordinated disclosure**. Please keep the vulnerability private until a fix has been released. Once patched, you are welcome to publish your findings — the fix commit will reference the report.
