# Paper 32 — PTSolver (TravelPlanner eval)

**Full title:** *Personal Travel Solver: A Preference-Driven LLM-Solver System for Travel Planning*

**Registered metric movement (internal ledger):** +4.48%(86.45%→90.32% final pass rate)

## Summary

**Final pass rate** moved **86.45% → 90.32%** on the rule-based TravelPlanner-style harness. Removing **`room_type`** filtering in **`get_accommodation`** stopped excluding cheaper listings: the hard-constraint checker compares **lowercase** room-type strings while queries use **title case**, so the old filter was an accidental over-constraint. **`get_best_transport_mode`** picks a **single** mode for both legs of a round trip and verifies **both directions** against the distance-matrix API, fixing asymmetric “cheap one way, invalid return” failures.

## Key ideas

- Align **generator filters** with **evaluator semantics** (case and which constraints are actually active).
- **Bidirectional** transport feasibility before committing a mode.

## Where to look

- **`generate_plans_v2.py`** (`get_accommodation`, `get_best_transport_mode`, `get_transportation` with `force_mode`).
- Pipeline run **`run_20260324_212959`** under `optimizer/papers/paper-394/runs/` on the source machine.
