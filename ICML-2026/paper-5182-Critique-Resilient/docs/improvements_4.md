# Code Review: Outstanding Issues and Follow-ups (Round 4)

Review Date: 2025-12-30
Scope: Cross-cutting issues from docs/issues.md and code inspection

## Findings

### 1) OpenRouter payload includes `temperature: None`
- If `temperature` is unset, the OpenRouter payload still includes `"temperature": None`, which some APIs reject or interpret unexpectedly.
- Affected: `model_api.py:190`, `model_api.py:204`.
- Decision: omit the field entirely when the value is None.

Human decision: approve

### 2) Metadata/version tracking helpers are unused
- `save_json_with_metadata()` and `get_model_metadata()` exist but are not wired into writers, so experiment/model version tracking remains unresolved.
- Affected: `utils.py:52`, `model_api.py:39`, `docs/issues.md:137`, `docs/issues.md:154`.
- Decision: integrate metadata capture into all output writers or document why it is intentionally skipped.

Human decision: Decide whether worth using. If yes use them, if not remove them

### 3) Benchmark answer fallback semantics remain unclear
- Multiple entry points use `benchmark_answers()`/`benchmark_answer_for_index()` without a documented policy for when fallback is correct or acceptable.
- Affected: `docs/issues.md:9`, `generate_critiques.py:73`, `debate.py:66`, `automated_judge.py:65`.
- Decision: document expected behavior and apply consistently.

Human decision: unify in a single method to be called in utils.py, allow fallback with a boolean argument


### 4) Debate early-stop asymmetry still undocumented
- The intentional asymmetry (defender concedes: immediate stop; claimant concedes: both messages collected) is not documented in code.
- Affected: `debate.py:160`, `docs/issues.md:83`.
- Decision: add a short code comment explaining the rationale.

Human decision: Document

### 5) Redaction still over-matches via substring replacement
- `redact_text()` uses case-insensitive substring replacement and can redact unintended tokens (e.g., surnames or theorem names).
- Affected: `automated_judge.py:96`, `docs/issues.md:96`.
- Decision: monitor; if problematic, consider boundaries or whitelists.

Human decision: switch to a safer redaction system or drop altogether

## Deferred Analysis/Repro

### 6) Judge coverage analysis not run
- Need minimum judges per claim, zero-judge cases, and statistical power impact.
- Affected: `docs/issues.md:23`.

Human decision: ignore

### 7) Unknown verdict rate analysis not run
- Need frequency, model breakdown, and retry policy for `verdict="unknown"`.
- Affected: `docs/issues.md:38`.

Human decision: ignore

### 8) Stats presentation enhancements pending
- Per-topic breakdowns and alternative cross-model presentation options are still untested.
- Affected: `docs/issues.md:55`, `docs/issues.md:67`.

Human decision: ignore

### 9) Reproducibility policy unresolved
- Thread nondeterminism/run IDs are not resolved at the project level.
- Affected: `docs/issues.md:115`, `docs/issues.md:137`.

Human decision: ignore

## Deferred Hardening

### 10) Reliability hardening still deferred
- No API-key validation, no retries/backoff, no request timeouts, hardcoded polling sleeps, and hardcoded `.json` extension.
- Affected: `docs/improvements.md:340`, `docs/improvements.md:555`, `docs/improvements.md:570`, `docs/improvements.md:586`, `model_api.py:204`.

Human decision: put some basic retries, ignore the rest

### 11) Security/IO hardening still deferred
- Path traversal guard for slugs and repeated full-file writes in judging are still deferred.
- Affected: `docs/improvements.md:649`, `docs/improvements.md:928`.

Human decision: fix repeated full-writes, ignore slugs

### 12) Performance/memory concerns still deferred
- Load-entire-file patterns and JSON parsing hot paths remain as-is.
- Affected: `docs/improvements.md:472`, `docs/improvements.md:958`.

Human decision: ignore

### 13) Data quality choices still deferred
- LaTeX cleaning limitations, skipped "correct answer" critiques, and lack of self-answering baseline remain open.
- Affected: `docs/improvements.md:782`, `docs/improvements.md:796`, `docs/improvements.md:807`.

Human decision: improve LaTeX cleaning, add a flag to force the evaluation of critiques with decision "correct" (not "correct answer": that's a legacy value), add --allow-self-answering. Basically handle all three

### 14) Reproducibility/traceability hardening deferred
- Timezone handling for run_id, no seeds, no integrity hashes, guidance drift risk.
- Affected: `docs/improvements.md:819`, `docs/improvements.md:849`, `docs/improvements.md:882`, `docs/improvements.md:895`.

Human decision: ignore

### 15) Role naming consistency remains unresolved
- Inconsistent naming across pipeline roles still needs standardization.
- Affected: `docs/improvements.md:449`.

Human decision: address
