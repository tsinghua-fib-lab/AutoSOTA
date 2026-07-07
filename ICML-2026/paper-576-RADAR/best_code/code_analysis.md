# SOTA Preparation Repair: Paper 576 (RADAR)

## Preparation Failure

**Root cause**: The container `autosota_sota_paper_576` was started but `DEEPSEEK_API_KEY` was not set in the container environment. The `SandboxGrader` constructor in `Static/src/dataset_utils.py:133` raises `RuntimeError("DEEPSEEK_API_KEY 未设置...")` when the env var is missing.

**Fix**: Exported `DEEPSEEK_API_KEY="[REDACTED]"` (from config.yaml) in the container before running the evaluation. Also needs to be set for every `docker exec` command since env vars do not persist between exec sessions.

## Corrected In-Container Evaluation Command

```bash
export DEEPSEEK_API_KEY="[REDACTED]"
cd /repo
cd Static && python3 main.py --model_name deepseek-chat --dataset_name realtimeqa \
  --top_k 10 --defense_method mincut --attack_method PIA --attackpos 0 \
  --use_open_model_api
```

## Baseline Verification

| Metric | Manifest | Reproduced | Match |
|--------|----------|------------|-------|
| Acc    | 70.0     | 70.0       | ✓     |
| ASR    | 17.0     | 17.0       | ✓     |

100 samples of RealTimeQA with PIA attack at Pos1, MinCut defense. ~15.5 minutes runtime.

## Container State

- Container: `autosota_sota_paper_576` (running)
- Image: `autosota/paper-576:reproduced`
- GPU: NVIDIA A100-SXM4-80GB
- Python: 3.10 (conda)
- Torch: 2.5.1+cu121
- Repo: `/repo` at commit `0e2da54` (tagged `_baseline`)
- NLI model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- Embed model: `all-MiniLM-L6-v2`
- LLM: `deepseek-chat` via API

## Safe Optimization Targets

1. `Static/src/defense.py` — MinCut defense implementation (NLI matrices, Min-Cut selection, prompt construction)
2. `Static/main.py` — CLI argument parser and main orchestration
3. `Static/src/dataset_utils.py` — Data loading and grading (DO NOT MODIFY benchmark logic)

## Known Issues to Fix (Idea-08)
- C-matrix pollution: Both branches of NLI if/else set `C[i,j]`, polluting conflict matrix
- Hardcoded `batch_size=16` vs declared `self.nli_batch_size=32`

## Optimization Levers
- `--top_k`: retrieval depth
- `--isolation_threshold` (hardcoded 0.3): post-Min-Cut filtering
- NLI model choice: `--nli_model_path`
- Embed model: `--embed_model_path`
- M/C matrix construction in `_build_sim_and_conflict_matrices()`
- Prompt construction in `MinCutRRAG.query()`
- Self-consistency voting, answer verification, document deduplication
