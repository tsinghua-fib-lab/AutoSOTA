# ANCHOR Paper 4218 — SOTA Preparation Repair & Code Analysis

## Original Preparation Failure

The normal SOTA preparation failed with two issues:

### 1. vLLM Server Not Running
The evaluation produced 30/30 "Connection error" errors because the vLLM server serving Qwen3-8B on port 8000 was not running. The reproduction manifest's setup_notes clearly states this prerequisite but the orchestrator didn't start it.

**Fix**: Started vLLM inside the container using GPU 0:
```
CUDA_VISIBLE_DEVICES=0 vllm serve /models/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 8192
```
GPU 0 (A100-SXM4-80GB) used, model loads in ~30s.

### 2. Python String Quoting Bug in run_eval.sh
The Step 3 Python one-liner used unquoted strings causing NameError:
```python
# BROKEN:
with open(evaluation_results.json) as f:  # NameError!
    m = data[metadata]                     # NameError!
```
**Fix**: Added proper string quoting:
```python
# FIXED:
with open('evaluation_results.json') as f:
    m = data['metadata']
```

## Corrected In-Container Evaluation Command
```bash
cd /repo/local_eval && bash run_eval.sh
```
This runs: trajectory_gen.py -> evaluate_judges.py -> metrics summary.

## Baseline Verification

| Metric | Manifest | Repaired Baseline | Match |
|--------|----------|-------------------|-------|
| Refusal Rate | 0.0% | 0.0% | Exact |
| Harm Score | 47.33 | 48.93 | Within noise |
| Catastrophic Risk | 34.5 | 34.48 | Exact |
| Harm & Risk Score | 40.92 | 41.71 | Within noise |

The H&R difference (~0.8) is within expected stochastic variance (temperature=0.7).

## Key Optimization Target Files

### /repo/local_eval/trajectory_gen.py
- Lines 23-25: MAX_TURNS=8, MAX_TOKENS=2048, TEMPERATURE=0.7
- Lines 29-57: INSTRUCTION_TEMPLATE
- Lines 60-90: TOOL_RESPONSES (placeholder responses)
- Lines 158-168: System prompt in run_single_task()
- Around line 220: Tool response follow-up logic

### /repo/local_eval/evaluate_judges.py
- Judge evaluation using Qwen3-8B via vLLM
- Should NOT be modified

## Safe Optimization Targets
1. System prompt (line 168) - persona conditioning
2. Instruction template (lines 29-57) - linguistic style, reframing
3. Tool responses (lines 60-90) - realism enhancement
4. Sampling params (lines 23-25) - temperature, max_tokens sweep
5. Multi-turn logic (around line 220) - extended turns, follow-up prompts

## Red Lines
- evaluate_judges.py
- Sample data / test tasks
- Metric calculation
- The fundamental evaluation protocol

## Container Info
- Container: autosota_sota_paper_4218
- GPUs: 2x A100-SXM4-80GB (indices 0,1)
- vLLM: port 8000, GPU 0
- Model: /models/Qwen3-8B
- Repo: /repo (git managed)
