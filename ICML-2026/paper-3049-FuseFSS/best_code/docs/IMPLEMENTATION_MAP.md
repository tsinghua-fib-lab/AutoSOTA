# Implementation Map

This note maps the paper-level FuseFSS design to the artifact files.

## Paper Abstractions

| Paper concept | Artifact implementation |
|---|---|
| Operator specification | `include/suf/operator_spec.hpp` |
| Shape leakage | `ShapeLeakage` in `include/suf/operator_spec.hpp` |
| Packed comparison plan | `PackedComparisonInstance` in `include/suf/operator_spec.hpp` |
| Vector interval lookup payload | `IntervalLookupInstance` and `VectorLutPlan` |
| Share-based post-processing `Phi` | `include/suf/postprocess.hpp`, `src/secure_program.cpp` |
| Mask-aware lowering | `src/masked_compile.cpp` |
| GPU secure program | `include/suf/secure_program.hpp`, `src/secure_program.cpp` |
| Fused vector LUT backend | `third_party/EzPC_vendor/GPU-MPC/fss/gpu_lut.*` |
| Sigma production bridge | `src/fusefss_sigma_bridge.cu` |

## Runtime Paths

FuseFSS has three runtime paths:

1. Optimized Sigma path. This is the default benchmark path. It registers
   built-in operator specifications for GELU, SiLU, nExp, reciprocal, and
   rsqrt, then uses optimized DPF/LUT lowerings to preserve the paper's
   performance profile.

2. Generic canary path. `FUSEFSS_SIGMA_GENERIC=1` routes built-in operators
   through the compiled-operator ABI for single-output dense-table checks.

3. Strict generic path. `FUSEFSS_SIGMA_GENERIC_STRICT=1` exercises the
   production-style Sigma integration for vector payloads, multi-output
   arithmetic results, Boolean outputs, kappa inputs, fused vector LUT keys, and
   share-based post-processing primitives.

The default optimized path is the artifact's performance path. The strict path
is a semantic and integration audit path, not the measured performance path and
not a blanket claim that every generic production lowering is exactly the
two-backend-call theorem instance.
Reference canaries that use `PaperStrictSharedX` are not performance claims;
they check the typed generic semantics against a deterministic runtime.

## Compatibility Names

The code still exposes these internal compatibility names:

- `namespace suf`
- `include/suf/...`
- `SUFDescriptor`
- `suf_sigma_*` C ABI
- `SUF_HAVE_CUDA`

These names predate the paper title. They are not separate systems. User-facing
scripts and documentation use FuseFSS naming, and runtime configuration accepts
both `FUSEFSS_*` and legacy `SUF_*` environment variables.

## Nonlinear and Helper Operators

The Sigma integration replaces scalar nonlinear/helper gates:

- GELU and SiLU activations
- nExp inside softmax
- reciprocal inside softmax normalization
- rsqrt inside layer normalization

Reductions, matrix multiplication, MHA, and other tensor-level operations remain
Sigma GPU-MPC components. This matches the paper scope: FuseFSS targets scalar
fixed-point gates and helper operations, not the entire transformer layer stack.

## Model Evaluation Matrix

The maintained reproduction script covers the paper-main BERT/GPT points at
sequence length 128 and the Llama extension points:

- `LLaMA-7B` at sequence lengths 16, 32, and 64.
- `LLaMA-3.1-8B` at sequence lengths 16, 32, and 64.

Internally, Sigma names the latter model `llama3-8b`; artifact scripts expose
aliases such as `llama8b` and display it as `LLaMA-3.1-8B`.
