# SSL/TLS Credentials

This directory used to ship a `localhost.crt` / `localhost.key` / `root.crt`
trio borrowed from the gRPC examples, intended for local demos. Those
files have been removed: shipping a private key in a PyPI wheel is a
supply-chain hazard (every `pip install appfl` user gets the same key on
disk, and any on-path attacker can MITM `localhost` deployments using the
matching public certificate).

## Generating credentials

Use the bundled console script to generate a fresh local CA and a
server certificate signed by it:

```bash
appfl-setup-ssl
```

The script writes (by default) to `~/.appfl/ssl/`:

- `ca.key` and `ca.crt` — the CA private key and self-signed certificate.
  Keep `ca.key` private; ship `ca.crt` to every client that should trust
  this federation.
- `server.key` and `server.crt` — the server's private key and the
  CA-signed certificate to present on the wire.

Then point your APPFL config at those paths:

```yaml
server:
  server_certificate_key: /home/<you>/.appfl/ssl/server.key
  server_certificate:     /home/<you>/.appfl/ssl/server.crt
client:
  root_certificates:      /home/<you>/.appfl/ssl/ca.crt
```

In code, paths are loaded via `appfl.comm.grpc.load_credential_from_file`.

## Production deployments

`appfl-setup-ssl` is a convenience for self-signed local CAs. For
production, mint server certificates from a real CA the participating
clients already trust (your institutional PKI, Let's Encrypt for an
internet-facing endpoint, an AWS PCA, etc.) and configure the same fields
to point at those PEM files.
