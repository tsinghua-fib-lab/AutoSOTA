# Paper 29 — CDQGeoEmbed

**Full title:** *Conditional Dichotomy Quantification via Geometric Embedding*

**Registered metric movement (internal ledger):** +14.0%(0.4907→0.5594 avg_dcf)

## Summary

**avg_dcf** improved from **0.4907** to **0.5594** without changing DCF definitions or datasets. A new checkpoint **`ckpts/cross_scenario/defeasible_bert_v1d_nli`** comes from **sequential** fine-tuning rounds across debate, defeasible NLI, and causal scenarios, starting from the public defeasible-bert base. At inference, **`eval_all_scenarios.py`** blends embeddings **0.7** (fine-tuned) / **0.3** (original baseline) so cross-scenario gains and strong NLI geometry are combined.

## Key ideas

- **Sequential multi-domain fine-tuning** beats a single joint mix for this setup.
- **Weighted embedding ensemble** at eval time is cheap vs. retraining the metric stack.

## Where to look

- **`eval_all_scenarios.py`**, **`ckpts/cross_scenario/defeasible_bert_v1d_nli/`**.
- Pipeline run **`run_20260324_211015`** under `optimizer/papers/paper-367/runs/` on the source machine.
