# Progressive Cramming

The research domain and public artifacts for compressing a span of text into one
(or a few) learnable input embeddings of a frozen language model, and the language
used across the repo, paper, poster, slides, and project page.

## Method language

**Cramming**:
Compressing a span of text into one or a few *learnable memory embeddings* by
optimising those input embeddings (via gradient descent) until a **frozen** LM
reconstructs the original tokens. The weights never change; all information is
squeezed into the embedding.
_Avoid_: encoding, prompt tuning, soft prompting.

**Full cramming**:
Cram a fixed-length span all at once — fix the number of tokens, optimise the
embedding to reconstruct them.
_Avoid_: one-shot cramming, batch cramming.

**Progressive cramming**:
Grow the target span one token (or a small step) at a time, advancing to the next
length only after the current span reconstructs **exactly**. Halting at the first
failure yields a sharp per-sample capacity boundary.
_Avoid_: incremental cramming, curriculum cramming.

**Low-dim projection**:
Optimise the embedding inside a learned rank-*k* subspace ($\mathbf{e}=W\mathbf{z}+\mathbf{b}$,
$k\ll d$) instead of the full hidden size.
_Avoid_: low-rank cramming, PCA cramming, bottleneck.

## Measurement language

**Compression horizon**:
The longest prefix (in tokens) that a single embedding reconstructs **exactly** —
the per-sample answer that progressive cramming finds.
_Avoid_: capacity, context length, max tokens.

**Reconstruction**:
The fraction of tokens the frozen model decodes correctly from the embedding,
teacher-forced (the `final_convergence` field). `1.0` = exact.
_Avoid_: accuracy, fidelity, recall.

**Steps-to-converge**:
The number of optimiser steps a stage needs to reach exact reconstruction; the
*marginal* (differenced) value is the cost of absorbing one newly added token.
_Avoid_: convergence time, iterations, training steps.

**Per-token surprisal**:
The frozen base model's next-token surprisal $s(L)=-\log_2 p(x_L\mid x_{<L})$ in
bits — the per-token difficulty signal that predicts steps-to-converge.
_Avoid_: perplexity, loss, entropy.

**Reconstruction ≠ understanding**:
The headline finding that perfect reconstruction is *brittle steering*, not
transferable semantics: prepending a crammed embedding drops downstream benchmark
accuracy and collapses generative MMLU to ~0%.
_Avoid_: hallucination, forgetting, capability loss.

## Project-page language

**Project page**:
The static academic landing page for the public repo, served at the GitHub Pages
**site root**. Distinct from the **video deck** (the reveal.js slides at `/slides`).
_Avoid_: homepage, website, docs site, microsite.

**Per-token explorer**:
The interactive, pre-computed widget on the project page that renders a real
crammed sample's tokens, colours each by reconstructed-up-to-horizon vs not, shows
per-token surprisal and steps-to-converge on hover, and grows the horizon with a
slider.
_Avoid_: playground, live demo, sandbox.
