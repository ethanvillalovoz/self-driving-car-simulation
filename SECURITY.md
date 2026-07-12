# Security Policy

## Supported Version

Security fixes target the latest commit on `main`.

## Reporting

Please do not open a public issue for a suspected vulnerability. Email `ethan.villalovoz@gmail.com` with the affected path, reproduction steps, and impact. You should receive an acknowledgment within seven days.

## Scope Notes

The inference server binds to `127.0.0.1` by default. Exposing the legacy simulator protocol to an untrusted network is unsupported. Release artifacts are checksum-verified before link-safe extraction; report any manifest or downloader bypass privately.
