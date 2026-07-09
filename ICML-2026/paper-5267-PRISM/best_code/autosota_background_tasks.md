# SOTA Optimization Background Tasks — Paper 5267

## Active Tasks

### Task 1: Iter 1 — IDEA-11 Wider Beam Search (eval-only)
- **Started:** 2026-07-08 15:55 UTC
- **GPUs:** 0,1
- **Command:** train_eval.py --run_train false --run_eval true --force_eval --num_beams 8 --max_new_tokens 384
- **Adapter:** Existing baseline adapter (seed 42)
- **Log:** /tmp/iter1_idea11.log
- **Expected duration:** ~80 min (slower due to num_beams=8)
- **Status:** RUNNING

### Task 2: Iter 2 — IDEA-08 + IDEA-09 Combined Training
- **Started:** 2026-07-08 16:07 UTC
- **GPUs:** 2,3
- **Command:** train_eval.py --force_train --force_eval --no_resume (with code defaults: DataLoader seed fix + dp_debias_second_moment=True)
- **Adapter:** New training from scratch
- **Log:** /tmp/iter2_idea08_09.log
- **Expected duration:** ~90 min training + ~10 min eval = ~100 min
- **Status:** RUNNING

## Planned Tasks

### Task 3: Iter 3 — IDEA-05 geometry floor mode
- **Command:** train_eval.py --force_train --force_eval --no_resume --prism_floor_mode geometry
- **Expected duration:** ~100 min

### Task 4: Iter 4 — IDEA-02 noise decay
- **Command:** train_eval.py --force_train --force_eval --no_resume --noise_decay_enabled true
- **Expected duration:** ~100 min

### Task 5: Iter 5 — IDEA-05 floor_factor=0.25
- **Command:** train_eval.py --force_train --force_eval --no_resume --prism_floor_factor 0.25 --prism_floor_mode geometry
- **Expected duration:** ~100 min

### Task 6: Iter 6 — Combined best settings
- **Command:** TBD based on results
- **Expected duration:** ~100 min

## Completed Tasks
- Baseline eval verified (2026-07-08 15:55 UTC)
