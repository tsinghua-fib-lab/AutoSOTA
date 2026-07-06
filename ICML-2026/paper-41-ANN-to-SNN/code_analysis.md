# Code Analysis for Paper 41: Error Amplification Limits ANN-to-SNN Conversion

## Evaluation Path
- `eval.sh` runs `cd /repo/MuJoCo && python3 convert.py --policy_name TD3 --env <env> --SNN_ts 8 --eval_seed <seed>` for 4 envs x 5 seeds
- APR computed as mean across seeds of `(max_alpha SNN_return / ANN_baseline_return) * 100`
- ANN baselines hardcoded in eval.sh: Ant=6505, HalfCheetah=13193, Hopper=3594, Walker2d=4582
- Grid-search optimal alpha per environment (0.0 to 1.0 in 0.1 steps)

## Key Files
- `MuJoCo/convert.py` -- Main conversion script (SpikingNeuron, Actor, eval_policy, alpha sweep)
- `MuJoCo/TD3.py` -- TD3 training (pretrained models used, not retrained)
- `MuJoCo/SAC.py` -- SAC Actor variant
- `MuJoCo/DDPG.py` -- DDPG Actor variant
- `MuJoCo/eval.sh` -- Batch evaluation driver
- `MuJoCo/models/` -- 12 pretrained actor checkpoints (TD3x4, DDPGx3, SACx4)

## Metric Parser
- eval.sh prints "Overall APR (grid-search optimal): XX.XX%" -- this is the key metric
- Also prints per-environment APRs

## Safe Modification Targets
1. `convert.py:60` -- Change `torch.max()` to `torch.quantile()` for percentile threshold (ALGO-01)
2. `convert.py:49-66` -- `SpikingNeuron.finalize()` -- strip zero row (CODE-01), add percentile logic
3. `convert.py:71` -- Membrane init bias (ALGO-03)
4. `convert.py:257-262` -- Alpha sweep loop -- layer-wise alpha (ALGO-02)
5. `convert.py:115-123` -- SNN forward weighted averaging (ALGO-05)
6. `convert.py:219` -- Add deterministic seeding (CODE-02)
7. `convert.py:27,45` -- Pre-allocate act_buffer (CODE-04)
8. `convert.py:189` -- SNN_ts parameter (PARAM-01)
9. `convert.py:188` -- TIME parameter (PARAM-02)

## Risky Files (do not modify)
- `eval.sh` -- Evaluation protocol and metric computation
- ANN baselines in eval.sh
- Model files in `models/`
- Environment seeds and evaluation protocol

## /paper_data Resources
- Contains DDPG&TD3&SAC_MuJoCo and DrQ-v2_DMC directories
- Models already available in `/repo/MuJoCo/models/` -- paper_data not needed

## Known Gotchas
- Line 27: act_buffer initialized with zero row -- biases percentile (CODE-01)
- Line 189: SNN_ts default is 32 but eval.sh uses 8
- Line 45: torch.cat O(n^2) memory pattern (CODE-04)
- No torch/numpy seeding before calibration loop (CODE-02)
- reset() called between alphas may carry over state (CODE-03)
- EGL rendering not available (DMC tasks won't work)
