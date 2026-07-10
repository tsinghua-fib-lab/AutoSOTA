# Code Analysis — Paper 1780 (ProtoMech)

## Evaluation Path
1. `run_eval.py` orchestrates:
   - Step 1: `01_extract_embeddings.py` — extracts ESM2 layer 6 MLP output → `all_acts_8M.npz`
   - Step 2: `02_discover_circuits_clt.py` — per family: load data → train probe → compute attribution → greedy circuit search → evaluate on test set
   - Metrics: reads all `families_8M/CLT_sequential/*.json` → average `max_f1` → prints `METRIC:F1=<value>`
2. `--skip-step1` available for probe-only changes
3. `--metric-only` computes metrics from cached JSON results

## Key Files (Optimization Targets)

| File | Function | Change Risk |
|------|----------|-------------|
| `family_circuit/family_utils.py:train_probe()` | LogisticRegression probe | LOW — probe only |
| `family_circuit/family_utils.py:get_data()` | Data sampling (4x neg ratio) | LOW — data only |
| `family_circuit/family_utils.py:evaluate_circuit()` | Probe inference | LOW |
| `family_circuit/family_utils.py:split_data()` | Train/val/test split | LOW |
| `circuit_utils/clt_circuit.py:CircuitDiscovererCLT` | CLT reconstruction, TopK | MED — inference code |
| `training/clt_model.py:CrossLayerTranscoder` | CLT model, LN, decoder init | HIGH — requires retraining |
| `training/run_clt.py` | CLT training config | HIGH — requires retraining |

## Known Code Issues
- `clt_model.py:41`: `skip_ln = (num_layers != 6)` — LN is DISABLED for ESM2-8M (6 layers). Unusual.
- `family_utils.py:20`: Fixed 4x negative ratio regardless of family size
- `family_utils.py:36`: Default C=1.0 without tuning
- `clt_circuit.py:133`: TopK k is hardcoded at `self.clt.k` (k=16)

## Baseline Resources
- ESM2-8M weights: `/repo/models/esm2_t6_8M_UR50D.pt`
- CLT checkpoint: `/repo/models/CLT_L6_D3200/checkpoints/last.ckpt` (346M)
- Data: `/repo/data/swissprot_seqid30_75k_all_info_with_3di.parquet`
- Cached embeddings: `/repo/family_circuit/families_8M/all_acts_8M.npz` (55M)
- 558 cached family result JSONs in `families_8M/CLT_sequential/`

## Safe Modification Pattern
For probe-only changes: modify `family_utils.py` → `python3 run_eval.py --skip-step1 --overwrite`
For inference-only changes: modify `clt_circuit.py` → `python3 run_eval.py --skip-step1 --overwrite`
For CLT retraining: modify `training/` → retrain CLT → `python3 run_eval.py --overwrite`

## Corrected Eval Command
`cd /repo/family_circuit && python3 run_eval.py --overwrite`
(Matches manifest. Inside container.)

## Metric Parsing
Parse stdout for line: `METRIC:F1=<float>`
