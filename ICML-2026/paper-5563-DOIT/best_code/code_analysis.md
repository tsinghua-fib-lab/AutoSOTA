## Code Analysis - Paper 5563: Doob h-Transform

### Evaluation Path
- `run_eval.sh` → `run_repro.py` → `Pipeline(args).eval()` → loads DoobHGuidanceSampler → runs 5 seeds
- Each seed: sample actions for evaluation episodes → return mean/std of episode returns
- Final output: per-seed [DONE] lines + overall mean (paper scale × 100)

### Config Path
- `search/configs.py`: Arguments dataclass with all hyperparameters
- `exp_doob.py`: DATASET_CONFIGS dict maps dataset→(eta, gamma, tau, t_star_idx, particles)
- `run_repro.py`: Hard-codes halfcheetah-medium-v2 config inline

### Metric Parser
- Output: `overall_mean * 100` = "Normalized Score" (D4RL paper scale 0-100)
- Per-seed means printed as `mean=0.XXXX` with `[DONE]` prefix
- Final result printed with "FINAL RESULT" and "Overall mean: X.XXXX (paper scale: XX.X)"
- CSV output at `results/halfcheetah-medium-v2/doob.csv`

### Key Files
1. `search/doob_search.py` - Main optimization target: DoobHGuidanceSampler
   - `estimate_doob_grad()`: MC estimation of Doob correction gradient
   - `guide_step()`: Per-step denoising with guidance
   - `__call__()`: Full sampling loop
2. `search/ddim.py` - Base DDIMSampler with tensor_to_obj() and get_reward()
3. `search/configs.py` - Arguments dataclass (add new params here)
4. `exp_doob.py` - Multi-dataset experiment runner
5. `run_repro.py` - Single-dataset repro runner (5 seeds, cuda:0)
6. `pipeline.py` - Pipeline wrapper that connects sampler to env eval
7. `search/utils.py` - rescale_grad utility for gradient clipping

### Safe Modification Targets
- `search/doob_search.py`: estimate_doob_grad, guide_step, __call__
  - All changes are inference-time only
  - No model weights modified
  - No training data touched
- `search/configs.py`: Add new hyperparameters as dataclass fields
- `search/ddim.py`: tensor_to_obj (action selection), get_reward (Q computation)
- `run_repro.py`: Can modify hyperparameters for testing

### Risky Files (DO NOT MODIFY)
- Dataset splits, labels, env definitions (d4rl/gym)
- Model weights in `models_rl/`
- Evaluation metric computation in `pipeline.py` (eval function)
- Scoring scripts

### Red-Line Confirmations
All proposed changes are:
- ✅ Inference-time sampler modifications only
- ✅ No evaluation protocol change
- ✅ No dataset/label modification
- ✅ No hard-coded metric values
- ✅ Full rollback via git

### Baseline Configuration
- eta=0.2, doob_gamma=0.25, doob_tau=0.5, doob_t_star_idx=10
- particles=4, doob_M=32, inference_steps=15
- doob_antithetic_sampling=True, doob_store_best=False
- doob_clip_scale=None (disabled)
- Baseline Normalized Score: 55.25 (paper reports 55.3 ± 0.3)

### Manifest Recovery Notes
- eval_command `bash run_eval.sh` runs correctly inside container at /repo
- No ambiguous host commands to repair
- GPU devices: 2,3 available (but run_repro.py uses cuda:0, single-GPU sequential)
