# Code Analysis - AutoVSR Paper 6315 SOTA

## Evaluation Path
- **Script**: `/repo/eval_v5.py`
- **Command**: `python3 /repo/eval_v5.py`
- **Timeout**: 60 minutes (manifest), ~27 min in practice
- **Data**: `/datasets/CircuitSense/Analysis/synthetic/level1/`
- **Output**: `/repo/output/type2_tf_accuracy.json` + `/repo/output/eval_log.txt`

## Core Pipeline
The eval script implements deterministic transfer function generation:
1. Load Type 2 (RLC) circuit TF questions from CircuitSense dataset (336 samples)
2. For each sample: fix netlist (step/dc -> s), compute TF via Lcapy subprocess
3. Check symbolic equivalence with SymPy (symbolic -> rational -> numerical fallback)
4. Report accuracy (excluding timeouts)

## Key Bottleneck
- **17/336 samples (5%) timeout** at 30s Lcapy `cct.transfer()` call
- Root cause: `cct.transfer()` hangs when source and output element share a node (especially ground)
- All non-timeout samples compute correctly (100% accuracy excluding timeouts)

## Optimization Strategy (ITER-1)
- Replace single `cct.transfer()` call with multi-strategy approach:
  1. Primary: `cct.transfer()` with 15s timeout (fast for ~95% of circuits)
  2. Fallback: `V_elem.s / V_src.s` voltage ratio method with 120s timeout
- The voltage ratio method avoids the pathological case in `cct.transfer()`
- Validated: 16/17 timeout samples compute correctly via voltage ratio (all symbolically equivalent)
- Remaining issue: q1808 (14-element circuit) needs ~85s for V.s extraction

## Config/Levers
- `TIMEOUT_PRIMARY = 15s` (transfer method)
- `TIMEOUT_FALLBACK = 120s` (vratio method)
- `fix_netlist()`: converts step/dc sources to s-domain
- `check_eq()`: 3-tier equivalence checking (symbolic -> rational -> numerical)

## Safe Modification Targets
- `compute_transfer_subprocess()`: change computation method (done in ITER-1)
- `fix_netlist()`: netlist preprocessing (no need to append source name)
- Timeout values: tune for performance vs coverage trade-off
- `check_eq()`: add more sophisticated equivalence strategies

## Risky Files (DO NOT MODIFY)
- Dataset files in `/datasets/CircuitSense/`
- Expected answers (`_ta.txt` files)
- `record_score.sh` at `/tools/record_score.sh`
- Metric computation logic in `check_eq()` (must preserve equivalence protocol)

## Full AutoVSR Pipeline (not usable)
- Located in `src/` (nodes, graph, tools, ir)
- Requires GLM-4.6V-Flash API (ZhipuAI) - API key invalid (403)
- `main.py`: entry point for full pipeline
- `src/nodes/netlist/solve.py`: LLM-based solver agent
- `src/tools/netlist_tools.py`: Lcapy tools with 50s default timeout
