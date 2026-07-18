# Security Model

FuseFSS follows the paper's two-server, semi-honest setting.

## Parties and Assets

- Two non-colluding servers execute the protocol.
- The client's input is secret shared or masked across the servers.
- Model parameters are public in the artifact's main evaluation setting.
- Servers see public tensor shapes and the public operator specification.

In the Sigma integration, the two servers correspond to Sigma party 0 and party
1. The artifact validation runs those parties as separate processes, normally
bound to two different GPUs.

## Masking Semantics

The Sigma bridge uses a fresh input mask per scalar wire. Keygen and eval paths
receive vectors of masked inputs and masks rather than one shared tensor mask.
The public value is:

```text
x_hat = x + r_in mod 2^n
```

The operator compiler and runtime keep public shapes independent of concrete
mask values. Shape leakage records public quantities such as interval count,
comparison count, payload width, output count, and post-processing primitive
counts.

## Post-Processing

The strict generic executor supports share-based post-processing with:

- Beaver-style multiplication
- Boolean AND and OR
- B2A
- A2B bit extraction for explicit bit-indexed expressions
- arithmetic and Boolean kappa inputs

Unsupported strict features fail before execution. The production path does not
fall back to host-side secret reconstruction.

## Private Model Boundary

This artifact does not implement full private model-weight serving. Matmul,
MHA, and layernorm weight interfaces in the current Sigma stack still use public
CPU-side model parameters.

If `FUSEFSS_MODEL_PRIVATE=1` or `SIGMA_MODEL_WEIGHTS_PRIVATE=1` is set, the
integrated backend rejects public-weight production paths. This fail-fast guard
prevents the artifact from silently claiming private-model protection.

## What Is Not Claimed

- Malicious-security guarantees.
- Protection of model weights in the current production path.
- Hiding public shapes, model architecture, sequence length, or selected
  operator specifications.
- A complete replacement of Sigma tensor-level protocols.
