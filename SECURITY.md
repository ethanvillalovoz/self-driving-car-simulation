# Security Policy

## Supported Version

Security fixes target the latest commit on `main`.

## Reporting

Please do not open a public issue for a suspected vulnerability. Email `ethan.villalovoz@gmail.com` with the affected path, reproduction steps, and impact. You should receive an acknowledgment within seven days.

## Scope Notes

The inference server binds to `127.0.0.1` by default. Exposing the legacy simulator protocol to an untrusted network is unsupported. Release artifacts are checksum-verified before link-safe extraction; report any manifest or downloader bypass privately.

## Legacy Simulator Boundary

The Udacity simulator client requires the pinned Engine.IO 3 / Socket.IO 2-compatible Python packages. Their maintained successors no longer speak that protocol. The adapter therefore has a deliberately narrow support boundary:

- loopback connections only;
- one in-process server with no message queue or multi-server client manager;
- the default JSON serializer, never pickle;
- simulator telemetry only, with bounded image payloads and neutral control on invalid input.

Dependabot advisories against the legacy protocol packages are treated as a documented tolerable risk for this local compatibility path. Do not bind the server to `0.0.0.0`, place it behind a public endpoint, or accept connections from an untrusted client. Use the offline replay when simulator connectivity is not required.
