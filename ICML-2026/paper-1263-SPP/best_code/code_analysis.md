# Code Analysis — Paper 1263 SOTA

## Evaluation path
- `scripts/reproduce_arc.py` — main eval script
- Loads model, probes, head importance, whitelist from `outputs/Qwen2.5-7B-Instruct/ppd_pipeline/`
- Runs dense and pruned evaluation on ARC scenarios
- Metrics: Token-level Recall, Speedup, Memory, Retention

## Key source files
- `src/probe/session_pruning.py` — Core pruning logic. SessionPruner class.
- `src/probe/domain_inference.py` — Domain inference from probe predictions
- `src/preorientation/linear_probe.py` — Probe training and inference (LinearProbe, MultiLayerProbe)
- `src/model/base_model.py` — Model wrapper with mask hooks
- `src/probe/head_importance.py` — Head importance computation
- `scripts/train_offline.py` — Offline training pipeline (probes, importance, whitelist)

## Config
- `src/utils/config.py` — Central config
- Pruning strength: 0.5 (eta)
- Domain inference: min_probability_threshold=0.05, cross_domain_threshold=0.15
- Session breadth: computed from probe predictions, controls keep_ratio

## Bottlenecks identified
1. **Short context**: Tokenization uses max_length=512 everywhere (lines 212 of session_pruning.py)
2. **First-turn only**: extract_session_description() only uses first turn (line 134-146)
3. **Uniform pruning**: All layers have same min retention ratio (0.1)
4. **Greedy head selection**: No redundancy penalty — may pick functionally identical heads
5. **Binary masks**: All-or-nothing head masking loses partial contributions

## Safe modification targets
- session_pruning.py: extract_session_description(), _select_heads_for_domain(), _prune_based_on_domains()
- domain_inference.py: thresholds (min_probability_threshold, cross_domain_threshold)
- reproduce_arc.py: only to add CLI parameters
- base_model.py: NO changes needed for current ideas

## Risky files (avoid modifying)
- answer_evaluator.py — metric computation
- test_result_logger.py — evaluation protocol
- Any test data files
- record_score.sh
