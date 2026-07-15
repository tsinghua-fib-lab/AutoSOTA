# Security Policy

This repository is a research artifact and simulator. It should not be exposed
as a network service without additional review.

## Reporting

Please report security issues privately to the maintainers before opening a
public issue. Include:

- A minimal reproduction.
- The affected commit or release.
- Whether the issue can expose credentials, local files, or remote execution.

## Sensitive Data

Do not commit:

- SSH credentials, tunnel URLs, Tailnet auth links, or IP-specific lab details.
- Dataset credentials or private model weights.
- Raw logs that include hostnames, usernames, tokens, or internal network
  addresses.
