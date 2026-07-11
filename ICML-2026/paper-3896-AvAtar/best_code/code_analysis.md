# Code Analysis — AvAtar Paper 3896

## Evaluation Path
- Entry: /repo/evaluate.py — runs 5 seeds, parses MRR via regex, takes round 10 value
- Metric parser: regex in stdout capturing per-round MRR values
- Output: stdout + printed average

## Train/Inference Path
- Main loop: /repo/source/active_na.py
- Query strategies: /repo/source/utils/active_utils.py
- PARROT model: /autosota_cache/PlanetAlign/PlanetAlign/algorithms/parrot.py
- Dataset: PlanetAlign.datasets.PhoneEmail loads from /datasets/phone-email.pt

## Config Path
- Settings: /repo/settings/PARROT/phone-email.json
- Settings loading: /repo/source/utils/utils.py -> read_settings()
- Parameter dicts: /repo/source/utils/dicts.py

## Safe Modification Targets
1. active_utils.py: get_adjoint_grad_scores_sparse — fix T_norm double-multiplication (CODE-01)
2. active_utils.py: query_anchors — add entropy weighting (ALGO-01), weights passthrough
3. active_utils.py: query_anchor_offline — add k-hop diversity penalty (ALGO-03)
4. active_na.py: main loop — pass weights to query_anchors, schedule outer_iters, schedule lambda_n
5. parrot.py: con_prox_pt_opt — accept optional S_init for warm-start (CODE-02)

## Risky Files (DO NOT MODIFY)
- evaluate.py — evaluation protocol
- PlanetAlign/metrics.py — metric computation
- PlanetAlign/datasets/ — dataset loading
- settings/PARROT/phone-email.json — can modify values only, not structure

## Key Bug Found
get_adjoint_grad_scores_sparse has extra * T_norm vs dense version.
node_inf_scores = (pair_inf_scores * T_norm).sum(dim=1)  # sparse
node_inf_scores = torch.sum(pair_inf_scores, dim=1)       # dense (correct)

## Missing Functionality
--weights CLI arg accepted but never passed to query_anchors().
