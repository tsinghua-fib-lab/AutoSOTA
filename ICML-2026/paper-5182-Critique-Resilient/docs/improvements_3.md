# Code Review: Issues and Inconsistencies (Round 3)

Review Date: 2025-12-29
Scope: Core pipeline scripts, schema models, and tooling helpers

## Findings

### 1) Schema/data mismatches
- Benchmark statuses can be `"ill-posed"` in the pipeline, but the schema only allows `"succeeded"`/`"failed"` for `BenchmarkEntry` and `GenerationRound`. See `data_models.py#L44`, `data_models.py#L61`, `self_improvement.py#L106`, `generate_benchmark.py#L214`.
- Debate records use `"Alice"`/`"Bob"` speaker labels and 1-indexed rounds, while the schema expects `"defender"`/`"claimant"` and 0-indexed rounds. Critique debate outputs also omit the required `claimant` field. See `data_models.py#L183`, `data_models.py#L204`, `debate.py#L150`, `debate.py#L456`.
- Automated evaluation schema expects `type="critique_debate"` and required fields like `claim`, `claimant`, `defender`, `debate_history`, but the judge output uses `type="critique"` and omits those fields. See `data_models.py#L236`, `automated_judge.py#L230`, `automated_judge.py#L370`.

Decision: fix the schema. Additionally, use the schema validation throughout the codebase

### 2) Inconsistent model identity formats
- `benchmark_answers` writes `answer_model` as a slug in `debate.py` but as a full model name in `generate_critiques.py`, which can lead to mixed identifier formats in outputs. See `debate.py#L60`, `generate_critiques.py#L44`.

Decision: always use slug

### 3) Tooling checks do not match produced data
- `check_issues.py` looks for `final_question`, `final_answer`, and `final_critique`, but those fields are never written by the generators, so the checker will flag false "empty" issues. See `check_issues.py#L21`.

Decision: have check_issues look at the correct fields

### 4) Critique selection logic and unused flag
- `critic_specs` is computed from `--models` but never used; instead, critic selection is driven by `answer_models`, so critique-only models are skipped and `--models` does not actually constrain critics. See `generate_critiques.py#L280`, `generate_critiques.py#L296`.

Decision: use critic_specs instead


### 5) API parameter edge cases
- OpenAI requests always include `"temperature": None` when configured that way, which can be rejected by the API (it expects the field to be omitted). See `model_api.py#L122`, `model_api.py#L350`.
- Gemini batch always builds `ThinkingConfig` even when `reasoning=None`, which may not be valid for the SDK. See `model_api.py#L583`.

Decisions:
- If temperature is None, omit the field entirely in the request
- Don't put ThinkingConfig when reasoning is None

### 6) Dead/unused parameter
- `illposed_debate` accepts `guidance_a` but never uses it. See `debate.py#L120`.

Decision: If useless, remove the argument. If necessary, use it