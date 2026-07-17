# Background Tasks Ledger

## Task 1: Iteration 1 (CODE-01) — Activation Checkpointing + level=1
- **Iteration**: 1
- **Idea ID**: CODE-01
- **Title**: Activation Checkpointing + fold_level=1
- **Command**: `bash /repo/run_iter1.sh`
- **Working Directory**: /repo
- **Log Path**: /repo/iter1_run.log
- **PID**: TBD
- **Start Time**: TBD
- **Deadline**: 2026-07-17 (start + 150 min)
- **Expected Output**: final_eval_loss in log, checkpoint at checkpoints/foam2_llama60m_iter1/
- **Score Row**: iter=1, idea_id=CODE-01, status=TBD
- **Status**: pending

## Task 2: Iteration 2 (CODE-01+02+03) — Compound
- **Iteration**: 2
- **Idea ID**: CODE-01+CODE-02+CODE-03
- **Title**: Compound: Activation Checkpointing + level=1 + WSD + Grad Clipping 1.0
- **Command**: `bash /repo/run_iter2.sh`
- **Working Directory**: /repo
- **Log Path**: /repo/iter2_run.log
- **PID**: TBD
- **Start Time**: 
- **Deadline**: 2026-07-17T01:17:32Z
- **Expected Output**: final_eval_loss in log, checkpoint at checkpoints/foam2_llama60m_iter2/
- **Score Row**: iter=2, idea_id=CODE-01+CODE-02+CODE-03, status=TBD
- **Status**: running

## Task 3: Iteration 3 (CODE-01 + ALGO-04)
- **Iteration**: 3
- **Idea ID**: CODE-01+ALGO-04
- **Title**: Activation Checkpointing + level=1 + Per-layer Adaptive res_scale
- **Command**: `bash /repo/run_iter3.sh`
- **Working Directory**: /repo
- **Log Path**: /repo/iter3_run.log
- **PID**: TBD
- **Start Time**: 
- **Deadline**: 2026-07-17T02:24:13Z
- **Expected Output**: final_eval_loss in log
- **Score Row**: iter=3, idea_id=CODE-01+ALGO-04, status=TBD
- **Status**: running

## Task 4: Iteration 4 (CODE-01 + WSD only, no grad clipping)
- **Iteration**: 4
- **Idea ID**: CODE-01+WSD-only
- **Title**: Activation Checkpointing + level=1 + WSD scheduler (stable_ratio=0.8)
- **Command**: `bash /repo/run_iter6_no_clip.sh`
- **Working Directory**: /repo
- **Log Path**: /repo/iter4_run.log
- **PID**: TBD
- **Start Time**: 
- **Deadline**: 2026-07-17T03:33:42Z
- **Score Row**: iter=4, idea_id=CODE-01+WSD-only
- **Status**: running

## Task 5: Iteration 5 (CODE-01 + WSD only — retry)
- **Iteration**: 5
- **Idea ID**: CODE-01+WSD-only-retry
- **Title**: Activation Checkpointing + level=1 + WSD scheduler (no grad clipping) — RETRY
- **Command**: `bash /repo/run_iter6_no_clip.sh`
- **Working Directory**: /repo
- **Log Path**: /repo/iter5_run.log
- **PID**: TBD
- **Start Time**: 
- **Deadline**: 2026-07-17T04:27:02Z
- **Score Row**: iter=5
- **Status**: running
