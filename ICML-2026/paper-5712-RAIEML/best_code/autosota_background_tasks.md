# Background Tasks Ledger

## Task 1: exp1_fine_rho_sweep (IDEA-5712-11)
- **Started:** 2026-07-15 12:21 UTC
- **Command:** `cd /repo && python3 exp1_fine_rho_sweep.py > /repo/exp1_output.log 2>&1`
- **PID pattern:** exp1_fine_rho_sweep
- **Log path:** /repo/exp1_output.log
- **Expected output:** JSON results with rho sweep values
- **Deadline:** ~25 min from start
- **Status:** running

## Task 2: exp6_mild_sig_filter (IDEA-5712-01/02 variant)
- **Started:** 2026-07-15 ~12:30 UTC
- **Command:** `cd /repo && python3 exp6_mild_sig_filter.py > /repo/exp6_output.log 2>&1`
- **Log path:** /repo/exp6_output.log
- **Status:** running
- **Description:** Milder sig_level=0.01 + adaptive smoothing

## Task 3: exp3_per_group_rho (IDEA-5712-03)
- **Started:** 2026-07-15 ~12:30 UTC
- **Command:** `cd /repo && python3 exp3_per_group_rho.py > /repo/exp3_output.log 2>&1`
- **Log path:** /repo/exp3_output.log
- **Status:** running
- **Description:** Per-group rho with gamma=0.5
