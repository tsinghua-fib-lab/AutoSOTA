# §3 — Cost model

The paper's linear iteration-time model (§3) is validated empirically here.
Two subdirectories cover the two figures:

- [`linear_model/`](linear_model/) — `T_iter = α + β_p · L_p + β_d · N_d`
  (Figs. `execution_time.pdf`, `execution_time_gpus.pdf`,
  `prefill_linearity_all_models.png`).
- [`kernel_breakdown/`](kernel_breakdown/) — attention kernel time vs token
  count for `pure_prefill`, `pure_decode`, and `mixed` workloads
  (Figs. `kernel_breakdown*.pdf`).
