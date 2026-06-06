# Paper Optimization Mission

You are an autonomous AI research engineer. Your mission: optimize the performance of **Mean Flows for One-step Generative Modeling** beyond its paper-reported metrics, by iteratively modifying the code inside a Docker container and running experiments.

## Environment

| Parameter | Value |
|-----------|-------|
| Docker container | `paper_opt_paper-84` |
| Repo path (in container) | `/py-meanflow` |
| Evaluation command | `cd / && python eval_cifar10.py` (run from `/py-meanflow`) |
| Eval timeout | `2700` seconds |
| GPU devices | `2,3` |
| Docker image | `docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |

## Baseline Metrics (Paper-Reported)

| Metric | Baseline | Direction |
|--------|----------|-----------|
| Fid | **2.8883** | ↓ lower is better |

**Optimization goal**: Decrease `fid` to <= **2.8305** (↓ 2.0% vs baseline of 2.8883). Lower is better for this metric.

## Your Output Directory

All memory, results, and logs go to: **`/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200`**

```
/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/
├── memory/
│   ├── code_analysis.md      ← your deep understanding of the paper's code
│   ├── research_report.md    ← research insights (pre-populated if available)
│   └── idea_library.md       ← running list of optimization ideas + iteration log
└── results/
    ├── scores.jsonl           ← one JSON line per iteration (append only)
    └── final_report.md        ← written at the very end
```

## scores.jsonl Format

**Never write to scores.jsonl directly.** Always use `record_score.sh` (copied to `/tools/record_score.sh` inside the container). It performs `git commit`, captures the **real** commit hash, updates the `_best` tag when appropriate, and appends the JSON line atomically.

```bash
# Determine IS_NEW_BEST = true if new_primary_metric < BEST_SCORE, else false
docker exec paper_opt_paper-84 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/scores.jsonl' \
    --iter     ITERNUM \
    --idea-id  'IDEA-XXX' \
    --title    'Your idea title' \
    --status   success \
    --primary  58.5 \
    --metrics  '{\"overall\": 58.5, \"egocentric_dir\": 85.1}' \
    --notes    'Overall improved +0.5%' \
    --is-best  IS_NEW_BEST
"
```

For failures use `--status failed --primary <last_known_value> --metrics '{}' --is-best false`.

---

## PHASE 0 — Setup & Baseline

### 0.1 Start Docker Container

```bash
# Check if running
docker ps --filter "name=paper_opt_paper-84" --format "{{.Names}}"

# Start if not running
docker run -d \
  --name paper_opt_paper-84 \
  --gpus '"device=2,3"' \
  --shm-size=16g \
  \
  -v /home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results:/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results  docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime sleep infinity
```

### 0.2 Ensure Git is Available (inside container)

Git is required for version control and rollback. Check and install if missing:
```bash
docker exec paper_opt_paper-84 bash -c "
  if which git > /dev/null 2>&1; then
    echo '[Git] already installed:' \$(git --version)
  else
    echo '[Git] not found, installing...'
    # Try apt-get first (Debian/Ubuntu), then conda, then yum
    (apt-get update -qq && apt-get install -y -qq git 2>/dev/null && echo '[Git] installed via apt') ||
    (conda install -y git -c conda-forge -q 2>/dev/null && echo '[Git] installed via conda') ||
    (yum install -y git -q 2>/dev/null && echo '[Git] installed via yum') ||
    echo '[Git] WARNING: could not install git automatically'
    which git && git --version || echo '[Git] FATAL: git still unavailable after install attempts'
  fi
"
```

### 0.3 Initialize Git Snapshot & Copy Tools (run once at the very beginning)

```bash
# 1. Initialize git inside container and create baseline snapshot
docker exec paper_opt_paper-84 bash -c "
  cd /py-meanflow
  git rev-parse --git-dir 2>/dev/null && echo 'git already init' || git init -q
  git config user.name optimizer && git config user.email opt@local
  git add -A
  git commit -q -m 'baseline' --allow-empty
  git tag -f _baseline
  echo '[Git] Baseline snapshot created. HEAD:' && git rev-parse HEAD
"

# 2. Copy the record_score.sh helper into the container
docker exec paper_opt_paper-84 mkdir -p /tools
docker cp /home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/scripts/record_score.sh paper_opt_paper-84:/tools/record_score.sh
docker exec paper_opt_paper-84 chmod +x /tools/record_score.sh
echo '[Tools] record_score.sh installed at /tools/record_score.sh'
```

### 0.4 Run Baseline Evaluation

Run the evaluation to confirm it works and record the true baseline:
```bash
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && timeout 2700 cd / && python eval_cifar10.py 2>&1"
```

Parse the output to extract all metric values.

**Eval output format hint** (from onboarding):
```
The eval script prints two sections at the end:
  === FID score (net_ema1): 2.8883 ===
  === FID RESULT ===
  FID (1-NFE, CIFAR-10, net_ema1): 2.8883
Parse the float after "FID (1-NFE, CIFAR-10, net_ema1): " as the metric value.
Lower FID is better.
```

Record this as iteration 0 (baseline) using `record_score.sh`:
```bash
docker exec paper_opt_paper-84 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/scores.jsonl' \
    --iter     0 \
    --idea-id  'baseline' \
    --title    'Paper baseline' \
    --status   success \
    --primary  <actual_primary_value> \
    --metrics  '{<actual_metrics_json>}' \
    --notes    'Paper-reported baseline' \
    --is-best  true
"
```

**If the baseline evaluation fails**: read the error, investigate the repo, fix any environment issues. This is critical — you cannot optimize if you can't even run the eval.

---

## PHASE 1 — Code & Paper Understanding

Deeply explore the repository. Your goal: understand everything needed to optimize it.

```bash
# Repo structure
docker exec paper_opt_paper-84 bash -c "find /py-meanflow -name '*.py' | grep -v __pycache__ | head -50"
docker exec paper_opt_paper-84 bash -c "find /py-meanflow -name '*.yaml' -o -name '*.json' -o -name '*.cfg' -o -name '*.ini' | head -30"

# README
docker exec paper_opt_paper-84 bash -c "cat /py-meanflow/README.md 2>/dev/null | head -150"

# Evaluation script — understand output format thoroughly
docker exec paper_opt_paper-84 bash -c "cat /py-meanflow/eval_cifar10.py"

# Find all configurable parameters
docker exec paper_opt_paper-84 bash -c "grep -rn 'threshold\|n_frame\|weight\|alpha\|beta\|lr\|epoch\|batch\|topk\|sigma\|margin' /py-meanflow --include='*.py' | grep -v '#' | head -40"

# Find argparse / config loading
docker exec paper_opt_paper-84 bash -c "grep -rn 'argparse\|add_argument\|yaml.load\|json.load\|config\[' /py-meanflow --include='*.py' -l | head -20"
```

Read the key source files. Understand:
1. **Pipeline flow**: Data → Processing → Prediction → Metric computation
2. **Evaluation script**: How does `cd / && python eval_cifar10.py` work? What exact output does it produce? How are metrics calculated?
3. **Optimization levers**: Every parameter, threshold, hyperparameter, algorithm branch
4. **Hard constraints** (DO NOT CHANGE): pretrained model weights, dataset, evaluation protocol, metric definitions, metric computation logic (e.g. do not change how scores are aggregated, which columns/outputs are used, or how results are reduced to a single number). Also read the paper's experimental setup carefully — any settings the paper explicitly aligns to (e.g. decoding strategy, sampling parameters, number of samples, ensemble method) must be preserved as-is.
5. **Run procedure**: How long does a full eval take? Are there faster partial runs?

**Save your analysis to `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/memory/code_analysis.md`**:

```markdown
# Code Analysis: Mean Flows for One-step Generative Modeling

## Pipeline Summary
<describe the high-level flow>

## Key Source Files
| File | Purpose |
|------|---------|
| ... | ... |

## Evaluation Procedure
- Command: `cd / && python eval_cifar10.py`
- Output format: <describe exactly how to parse the metrics from stdout>
- Estimated runtime: <X minutes>

## Optimization Levers
| Parameter | Current Value | File:Line | Type | Notes |
|-----------|---------------|-----------|------|-------|
| ...       | ...           | ...       | threshold/hyperparam/algorithm | ... |

## Hard Constraints (DO NOT CHANGE)
- [ ] ...

## Initial Hypotheses
- ...
```

---

## PHASE 2 — Research Insights & Idea Library

A deep research model has pre-gathered related literature and optimization ideas for this paper. The full research report is appended at the end of this prompt under **## Appended Research Report**. Read it carefully and incorporate its insights into your idea library.

**Known optimization levers** (from onboarding):
- net_ema selector: eval uses net_ema1 (ema_decay=0.99995); alternatives are
  model.net_ema (net_ema0, ema_decay ~0.9999) or net_ema2 (ema_decay ~0.9999)
- batch_size: controls inference batch size (default 128); larger may be faster
- fid_samples: number of generated samples for FID (default 50000); lower is faster
  but noisier — keep at 50000 for accurate metric
- model.sample() integration steps: MeanFlow uses 1-NFE (single function evaluation)
  via the mean flow ODE; the number of ODE steps is fixed at 1 for 1-NFE metric
- use_edm_aug: True (EDM augmentation was used during training; must match checkpoint)
- args in train_arg_parser.py: ema_decays, model architecture hyperparameters
  (hidden_size, depth, num_heads in DiT-S/2 config)
- model_configs.py: instantiate_model builds a DiT-S/2 model (32x32 CIFAR-10)
- Sampling temperature / noise scaling in model.sample() in meanflow.py
- Quantization in synthetic samples: synthetic_samples = floor(x*255)/255
  (this quantization step in eval_cifar10.py may affect FID)

Combine your code analysis, the research insights, and any known levers above to generate a comprehensive idea library.

**Save to `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/memory/idea_library.md`**:

```markdown
# Optimization Idea Library: Mean Flows for One-step Generative Modeling

Last updated: <date>

## Ideas

### IDEA-001: <title>
- **Type**: PARAM / CODE / ALGO
- **Priority**: HIGH / MEDIUM / LOW
- **Risk**: LOW / MEDIUM / HIGH  (risk of breaking the pipeline)
- **Description**: What exactly to change, and how
- **Hypothesis**: Expected effect on `fid` and other metrics
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-002: ...

(generate at least 12-15 ideas)

## Iteration Log

| Iter | Idea | Type | Before | After | Delta | Status | Key Takeaway |
|------|------|------|--------|-------|-------|--------|--------------|
```

**Idea generation guidelines:**
- **HIGH priority, LOW risk** (do first): simple parameter tuning — thresholds, counts, distances (Type: PARAM)
- **MEDIUM priority, MEDIUM risk**: algorithm logic improvements, better post-processing, smarter fusion (Type: CODE)
- **LOW priority, HIGH risk**: pipeline structural changes, new modules, loss function changes (Type: ALGO)
- **LEAP** ideas (auto-generated during Leap iterations): cross-pollination "A+B" integrations — importing a proven technique from another paper/domain into this codebase
- Generate ideas at multiple granularities: micro (single value), meso (function-level logic), macro (pipeline stage)

---

## PHASE 3 — Optimization Loop

Run up to **12 iterations**. Stop early if `fid` <= 2.8305.

> ⛔ **HARD ITERATION LIMIT: 12 iterations.**
> After completing iteration 12, you MUST immediately proceed to **PHASE 4 (Finalize)** — do NOT start iteration 12+1 under any circumstances, even if you believe further improvement is possible. Running more iterations than allowed is a critical violation of the task constraints.

> **Metric direction for `fid`**: lower is better (↑ = maximize / ↓ = minimize accordingly).

Maintain these variables in your working memory:
```
BEST_SCORE   = <value from Phase 0 baseline evaluation>
CURRENT_ITER = 0

# Leap honeymoon tracking (set automatically — do not manually initialize)
IN_LEAP_HONEYMOON   = false   # whether we are inside a post-leap honeymoon period
HONEYMOON_REMAINING = 0       # how many honeymoon iterations are left
```
> Note: do NOT track commit hashes manually — `record_score.sh` captures the real hash automatically from `git rev-parse HEAD`.

### ─── For each iteration ───────────────────────────────────────────

#### ⓪ PRE-ITERATION REFLECTION — decide Normal vs. Leap

Before selecting an idea, reflect on **the last 3 completed iterations** (check the Iteration Log table in `idea_library.md`).

Classify each past iteration's type:
- **PARAM** = changed a single threshold, hyperparameter, or numeric value
- **CODE** = modified algorithmic logic, post-processing, or code flow
- **ALGO** = introduced a new module, algorithm, or external technique

**Trigger rule**: If the last 3 iterations are **all PARAM** (simple parameter tuning), then this iteration MUST use the **Leap Path** below. Otherwise, use the **Normal Path** (① SELECT IDEA).

> The first 3 iterations (iter 1-3) always use the Normal Path since there isn't enough history yet.

---

#### NORMAL PATH → ① SELECT IDEA

Read `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/memory/idea_library.md`.
- Pick the highest-priority **PENDING** idea
- Prefer **LOW risk** ideas in early iterations (iter ≤ 3)
- After exhausting low-risk ideas, move to MEDIUM then HIGH
- If all ideas are used up, generate 5 new ones based on what you've learned

Mark the selected idea as `IN_PROGRESS` in idea_library.md.

Then skip to **② SNAPSHOT** below.

---

#### LEAP PATH → ①' CROSS-POLLINATION IDEATION

When triggered, **do NOT pick from idea_library.md**. Instead, conduct a dedicated "A + B" ideation session.

**What is A+B optimization?**

This paper is **A** — its core method, pipeline, and framework. **B** is a technique, module, or algorithm from *a different paper or domain* that can be imported into A's framework to create a meaningfully improved system. This is how many breakthroughs happen in ML research:

| A (Base System) | B (Imported Module) | Result |
|---|---|---|
| GAN (adversarial generative model) | Wasserstein distance (optimal transport theory) | **Wasserstein GAN** — eliminated mode collapse, stabilized training |
| Transformer (self-attention on sequences) | Graph structure (adjacency-aware message passing) | **Graph Transformer** — captured relational topology that flat sequences miss |
| RNN encoder-decoder (seq-to-seq translation) | Attention mechanism (learnable alignment weights) | **Seq2Seq + Attention** — broke the information bottleneck of fixed-size context vectors |
| ResNet (deep residual image classifier) | Squeeze-and-Excitation (channel-wise recalibration) | **SE-ResNet** — adaptive channel weighting boosted accuracy with minimal cost |
| U-Net (encoder-decoder segmentation) | Attention gates (region-focused skip connections) | **Attention U-Net** — suppressed irrelevant features in skip connections |

Notice the pattern: **B is not a random addition** — it addresses a specific weakness or bottleneck in A. The key is diagnosing *what A lacks* and finding *a proven B that fills that gap*.

**Your ideation task:**

1. **Diagnose A's bottleneck**: Re-read `code_analysis.md` and the Iteration Log. What has consistently limited improvement? Is it noisy features? Lack of global context? Suboptimal fusion? Fragile calibration? Write down the top 1-2 structural weaknesses.

2. **Brainstorm 3 candidate B modules**: For each, specify:
   - **B name**: e.g., "Test-Time Augmentation (TTA)", "Attention-based Feature Fusion", "Contrastive Loss Regularization"
   - **Source**: where this technique comes from (cite a paper/method name if possible)
   - **How it integrates**: exactly which file/function would be modified, and what the new code flow would look like
   - **Expected effect**: which specific metric(s) should improve and why
   - **Implementation complexity**: LOW (< 30 lines), MEDIUM (30-100 lines), HIGH (> 100 lines)
   - **Risk assessment**: what could go wrong

3. **Self-evaluate (Reflection)**: For each of the 3 candidates, critically assess:
   - Is this actually feasible given the codebase constraints?
   - Does it conflict with the hard constraints (no changing pretrained weights, dataset, eval protocol)?
   - Has something similar already been tried in the iteration log?
   - What's the realistic probability of improvement vs. regression?

4. **Select the best one**: Pick the candidate with the highest [expected gain × feasibility] and lowest risk of catastrophic regression. Clearly state your reasoning for why this B is chosen.

5. **Log it**: Append this as a new `### IDEA-0XX` in `idea_library.md` with:
   - `**Type**: LEAP`
   - A detailed description of the A+B integration
   - Mark as `IN_PROGRESS`

> **Important**: A Leap idea is inherently riskier, but that's intentional. The goal is to escape the plateau of incremental parameter tuning. Even if it fails, the diagnostic insights are valuable — record them thoroughly.

> ⚠️ **MANDATORY EXECUTION RULE**: Once the LEAP PATH is triggered and you have selected the best B, you **MUST** implement and evaluate it in this iteration. You are **NOT allowed** to abandon this B in favor of any other idea from `idea_library.md`. The whole point of the Leap Path is to force a bold, structural change — bypassing it defeats the purpose. Proceed directly to **② SNAPSHOT** now.

Then proceed to **② SNAPSHOT** below.

---

#### ② SNAPSHOT (MANDATORY — do this BEFORE any code modification)

```bash
# Save current state as a rollback point (pre-modification snapshot)
docker exec paper_opt_paper-84 bash -c "
  cd /py-meanflow
  git config user.name optimizer && git config user.email opt@local
  git add -A
  git commit -q -m 'pre-iter-ITERNUM: IDEA_TITLE' --allow-empty
  PRE_COMMIT=\$(git rev-parse HEAD)
  git tag -f _pre_iter
  echo \"PRE_COMMIT=\$PRE_COMMIT\"
"
```

Note the printed `PRE_COMMIT` hash. Use it to rollback if things go wrong.

---

#### ③ IMPLEMENT

Make focused, targeted changes in the container. **Change only ONE logical thing per iteration** to clearly attribute improvements.

```bash
# Option A: Edit file directly via Python one-liner
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && python3 -c \"
import re
content = open('path/to/file.py').read()
content = re.sub(r'PATTERN', 'REPLACEMENT', content)
open('path/to/file.py', 'w').write(content)
print('Done')
\""

# Option B: Write a patch script to host, then copy and execute
cat > /tmp/patch_iter.py << 'PATCH'
# your Python patch code here
PATCH
docker cp /tmp/patch_iter.py paper_opt_paper-84:/tmp/patch_iter.py
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && python3 /tmp/patch_iter.py"

# Option C: For multiline edits, write the full file to host and copy in
docker cp /path/to/edited_file.py paper_opt_paper-84:/py-meanflow/path/to/file.py

# Verify the change looks correct
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && grep -n 'CHANGED_LINE' path/to/file.py | head -5"
```

---

#### ④ EVALUATE

```bash
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && timeout 2700 cd / && python eval_cifar10.py 2>&1"
```

Parse the stdout to extract all metric values as numbers.

---

#### ⑤ DEBUG (if run crashes or produces errors)

If the evaluation fails, errors out, or produces clearly wrong output:

**Timing**: Note the current time when you start debugging. You have **15 minutes** total.
**Attempts**: You have at most **3** fix-and-retry attempts.

For each debug attempt:
1. Read the error message carefully
2. Identify the root cause (syntax error, wrong variable name, shape mismatch, etc.)
3. Apply a targeted fix
4. Re-run the evaluation

**If debugging fails** (exceeded attempts OR exceeded time):
```bash
# Rollback to the state before this iteration started
docker exec paper_opt_paper-84 bash -c "
  cd /py-meanflow
  git checkout -- .
  git clean -fd
  echo '[Rollback] Restored to PRE_COMMIT state'
  git status
"
```
Mark idea as `FAILED (could not debug)` in idea_library.md. Proceed to next iteration.

---

#### ⑥ RECORD RESULT

**Use `record_score.sh` — do NOT write to scores.jsonl directly.** The script performs the git commit, records the real hash, and updates the `_best` tag when `--is-best true` is passed.

First, determine whether this result is a new best:
```
IS_NEW_BEST = new_primary_metric < BEST_SCORE   # e.g. new_primary_metric < BEST_SCORE (lower-is-better)
              OR (this is the baseline record)
```

Then call the script, passing `--is-best` accordingly:
```bash
# Success (replace IS_NEW_BEST with true or false):
docker exec paper_opt_paper-84 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/scores.jsonl' \
    --iter     ITERNUM \
    --idea-id  'IDEA-XXX' \
    --title    'Your idea title' \
    --status   success \
    --primary  NEW_PRIMARY_VALUE \
    --metrics  '{\"metric_name\": VALUE}' \
    --notes    'Change description' \
    --is-best  IS_NEW_BEST
"

# Failed iteration (--is-best false, no git tag change):
docker exec paper_opt_paper-84 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/scores.jsonl' \
    --iter     ITERNUM \
    --idea-id  'IDEA-XXX' \
    --title    'Your idea title' \
    --status   failed \
    --primary  LAST_KNOWN_PRIMARY \
    --metrics  '{}' \
    --notes    'Error description' \
    --is-best  false
"
```

Then update working-memory state and handle rollback / honeymoon logic:

```
IF new_primary_metric < BEST_SCORE:
    # ─── New best result! ───────────────────────────────────────
    # (_best tag was already moved by record_score.sh --is-best true)
    BEST_SCORE = new_primary_metric
    # If we were in honeymoon, the leap ultimately paid off — exit honeymoon
    IN_LEAP_HONEYMOON   = false
    HONEYMOON_REMAINING = 0
    Print: "✓ New best! fid: PREV → NEW"

ELIF this_iteration_was_LEAP AND NOT new_primary_metric < BEST_SCORE:
    # ─── Leap did not immediately improve — start Honeymoon Period ──
    # Do NOT roll back. Keep the leap changes as the new working base.
    # Allow up to 3 more normal iterations to evolve on top of this leap.
    IN_LEAP_HONEYMOON   = true
    HONEYMOON_REMAINING = 3
    docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && git tag -f _leap_entry"
    Print: "⚡ Leap result did not improve best score (fid: PREV → NEW)."
    Print: "   Starting 3-iteration Honeymoon Period — continuing to evolve from leap state."
    Print: "   Pre-leap best ({BEST_SCORE}) is preserved at tag _best."

ELIF IN_LEAP_HONEYMOON:
    # ─── Inside honeymoon — monitor progress ────────────────────
    HONEYMOON_REMAINING = HONEYMOON_REMAINING - 1
    IF HONEYMOON_REMAINING == 0:
        # Honeymoon expired — 3 iters passed with no new best
        docker exec paper_opt_paper-84 bash -c "
          cd /py-meanflow
          git checkout _best
          git checkout -- .
          git clean -fd
          echo '[Rollback] Honeymoon expired. Restored to pre-leap best.'
        "
        IN_LEAP_HONEYMOON = false
        Print: "↩ Honeymoon expired (3 iterations, no improvement). Rolled back to pre-leap best ({BEST_SCORE})."
    ELSE:
        # Still in honeymoon — keep current state, keep iterating
        Print: "⏳ Honeymoon period: {HONEYMOON_REMAINING} iteration(s) remaining. Continuing from leap state."
        Print: "   fid: PREV → NEW  (best so far: {BEST_SCORE})"

ELIF new_primary_metric > BEST_SCORE * 1.03:
    # ─── Normal significant regression (outside honeymoon) ──────
    docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && git checkout _best && git checkout -- . && git clean -fd"
    Print: "↩ Regression detected. Rolled back to _best ({BEST_SCORE})"

ELSE:
    # ─── Minor change — keep as is and continue ─────────────────
    Print: "→ No improvement. fid: PREV → NEW"
```

> **Honeymoon summary**: After a LEAP iteration that doesn't immediately improve the score, the framework enters a 3-iteration "Honeymoon Period". During this period, normal optimizations continue on top of the leap changes (treating the leap state as the new working base). If any of those 3 iterations sets a new best, the leap is considered successful and the honeymoon ends. If all 3 pass without a new best, the code rolls back to the pre-leap best (`_best` tag), discarding the leap changes entirely.

---

#### ⑦ REFLECT & UPDATE IDEAS  ← **MANDATORY — do not skip**

You MUST update `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/memory/idea_library.md` after every iteration. This file is your persistent memory across iterations; skipping this step means future iterations lose context.

Specifically:
1. **Mark the executed idea's Status** — change from `PENDING`/`IN_PROGRESS` to `SUCCESS` or `FAILED`:
   - Success: `**Status**: SUCCESS — overall X→Y (+Z%), metric_a A→B, metric_b C→D`
   - Failed: `**Status**: FAILED — reason: <what went wrong>`
2. **Add 2-3 new ideas** suggested by this iteration's observations (append new `### IDEA-0XX` blocks).
   - After a **LEAP** iteration (whether it succeeded or failed), pay special attention: write down what you learned about the codebase's structural constraints, and if the leap partially worked, propose follow-up ideas that refine the approach.
3. **Append one row** to the `## Iteration Log` table at the bottom. **Include the Type column** — this is critical for the pre-iteration reflection in step ⓪:

```
| {ITER} | IDEA-0XX | Type | {PREV_SCORE} | {NEW_SCORE} | {DELTA} | SUCCESS/FAILED | <one sentence takeaway> |
```

The `Type` column must be one of: `PARAM`, `CODE`, `ALGO`, `LEAP`.

> If this iteration ran during a **Honeymoon Period**, append `[HP-N]` to the Notes column (e.g., `[HP-2]` means 2 honeymoon iters remaining after this one). This makes it easy to trace which iterations were leap-influenced.

To verify the update happened, read back the last 10 lines of the file after writing:
```bash
tail -10 /home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/memory/idea_library.md
```

---

### ─── End of iteration loop ─────────────────────────────────────────

**Stop condition check (MANDATORY at the start of every iteration):**
```
CURRENT_ITER += 1

IF CURRENT_ITER > 12:
    # Hard limit reached — this line should never be hit if you checked at end of prev iter
    STOP. Proceed to PHASE 4 immediately.

IF BEST_SCORE <= 2.8305:
    Print: "🎯 Target reached! Proceeding to finalize."
    STOP. Proceed to PHASE 4 immediately.

IF CURRENT_ITER == 12:
    # This is the LAST allowed iteration — run it, then go to PHASE 4. No exceptions.
    Print: "⚠️  Final iteration (12/12). Will finalize after this."
```

---

## PHASE 4 — Finalize

### 4.1 Restore Best State
```bash
docker exec paper_opt_paper-84 bash -c "
  cd /py-meanflow
  git checkout _best 2>/dev/null || git checkout _baseline
  echo '[Final] Container restored to best known state.'
  git log --oneline -3
"
```

### 4.2 Final Evaluation
Run one last full evaluation and record in scores.jsonl as iter=`final` using `record_score.sh`:

```bash
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && timeout 2700 cd / && python eval_cifar10.py 2>&1"
```

Parse the output, then:
```bash
docker exec paper_opt_paper-84 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/scores.jsonl' \
    --iter     final \
    --idea-id  'final' \
    --title    'Final best state' \
    --status   success \
    --primary  FINAL_PRIMARY_VALUE \
    --metrics  '{<final_metrics_json>}' \
    --notes    'Final evaluation after restoring _best' \
    --is-best  false
"
```

### 4.3 List Changes from Baseline
```bash
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && git diff _baseline HEAD --stat"
docker exec paper_opt_paper-84 bash -c "cd /py-meanflow && git diff _baseline HEAD"
```

### 4.4 Write Final Report
Save to `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/final_report.md`:

```markdown
# Optimization Results: Mean Flows for One-step Generative Modeling

## Summary
- Total iterations: N
- Best `fid`: X% (baseline: 2.8883%, improvement: +Y%)
- Best commit: <hash>

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| ...    | ...      | ...  | ...   |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|

## What Worked
- ...

## What Didn't Work
- ...

## Top Remaining Ideas (for future runs)
- ...
```

### 4.5 Print scores.jsonl Summary
Output the full contents of `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260322_183200/results/scores.jsonl` so the user can see the complete optimization trajectory.

---

## General Rules

1. **One change per iteration** — isolate variables to understand causality
2. **Always snapshot before modifying** — never lose a working state
3. **Record everything** — even failures teach us something; log them all
4. **Never modify**: evaluation script logic, dataset files, pretrained model weights, metric definitions, or metric computation logic (e.g. do not change score aggregation, column selection, or result reduction). Respect the paper's experimental setup: any settings the paper explicitly aligns to (decoding strategy, sampling parameters, number of samples, ensemble configuration, etc.) are off-limits.
5. **Container crashes**: restart with `docker start paper_opt_paper-84` then re-check git status
6. **Long evals**: if full eval takes too long, first confirm with a quick sanity-check run (subset of data), then do full eval only if sanity check looks promising
7. **Be scientific**: form hypotheses, test them, update beliefs accordingly

---



---

## Appended Research Report

# Deep Research Report: Mean Flows for One-step Generative Modeling

Generated by: openai/o4-mini-deep-research
Date: 2026-03-22 18:46:35

---

# Related Follow-up Works  
- **α-Flow (Zhang et al., ICLR 2026)** – analyzes and generalizes MeanFlow into a unified “α-Flow” family. By smoothly interpolating between flow-matching and MeanFlow objectives via a curriculum, α-Flow achieves significantly better sampling quality. In class-conditional ImageNet-256 experiments (vanilla DiT backbones), their largest model achieves **FID 2.58 (1-NFE)** and **2.15 (2-NFE)** ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released)), far surpassing MeanFlow’s reported FID 3.43 (1-NFE) on the same task. This shows that adopting a multi-objective loss and even doing a *two*-step sampling (2-NFE) can yield large gains in sample fidelity.  
- **Flow Generator Matching (Huang et al., ICLR 2025)** – proposes **Flow Matching Generators (FGM)** to distill multi-step flow models into a one-step model. On CIFAR-10, their one-step FGM model achieves **FID 3.08** (unconditional) ([openreview.net](https://openreview.net/forum?id=B5IuILRdAX#:~:text=innovative%20approach%20designed%20to%20accelerate,When%20evaluated%20on%20GenEval)), setting a new record among one-step flow-based models at that time. (For reference, the MeanFlow PyTorch baseline already beats this, but FGM is the closest prior one-step flow result.) FGM is notable for combining implicit ODE knowledge into the loss to match teacher and student.  
- **Second-Order MeanFlow (Cao et al., ICLR 2026)** – extends MeanFlow by incorporating **average acceleration fields** (“second-order” velocities). This theoretical work proves a consistency condition for accelerating trajectories in one step ([openreview.net](https://openreview.net/forum?id=rj7PF436kF#:~:text=Abstract%3A%20Generative%20modelling%20has%20seen,We%20then)). It does not report sample metrics, but it lays groundwork for more expressive one-step flows.  
- **Flow Uniqueness Model (Zhang et al., ICLR 2026)** – introduces **Flow Uniqueness Models (FUM)**, enforcing a one-to-one mapping by constructing unique image pairs. Empirically they report “remarkable one-step generative performance” across standard image datasets ([openreview.net](https://openreview.net/forum?id=ZMqIgONdJZ#:~:text=generative%20modeling,benchmark%20datasets%20comprehensively%20validate%20the)). (Exact FID numbers aren’t given in the abstract, but the framework is explicitly designed to balance quality vs. # of steps.)  
- **Modular MeanFlow (You et al., preprint 2025)** – a unified framework that mixes flow-matching and consistency ideas. Using a curriculum on the gradient term (similar to α-Flow’s weighted stop-gradient), they report competitive single-step image quality. For instance, on CIFAR-10 their curriculum-trained model attains **FID ≈3.41** (1-NFE) ([www.emergentmind.com](https://www.emergentmind.com/topics/modular-meanflow#:~:text=%2A%20For%20image%20synthesis%2C%20curriculum,03%20s%2Fimage%29%20%28%206)). This demonstrates MeanFlow-style one-step sampling can be made more stable by module design and gradient scheduling. (Related papers like *MeanFlowSE* for speech and video–audio *MeanFlow* variants also show one-step results in other modalities, indicating wide interest in this one-step paradigm.)  

# State-of-the-Art Techniques (2023–2025)  
- **Classifier-Free Guidance / Scaling** – While originally a conditional diffusion trick, one can analogously “sharpen” unconditional flows by **adjusting the initial noise scale** (effectively a temperature). Reducing the sampling “temperature” (e.g. scaling the latent Gaussian by <1) can concentrate outputs and often improves FID at the cost of diversity (the classic BigGAN truncation trick). Many recent works tune sampling temperature or truncation (scaling ℕ(0, I) to ℕ(0, σ²I) with σ<1) to trade fidelity/diversity.  
- **Test-Time Augmentation** – In generative modeling, one can apply minor augmentations at inference and aggregate outputs. For example, sample an image, apply a random horizontal flip or small crop, and then “unflip” or pad back. Averaging results from a few such transforms can smooth artifacts. This mirrors techniques in classification (like ensembling flips) but is rarely used because it can hurt diversity; however, it is a possible SOTA trick to slightly boost FID.  
- **Multi-Stage Sampling / Iteration** – Even for one-step models, using *more than one evaluation pass* can help. For instance, α-Flow and other work show **2-step sampling** dramatically improves FID (e.g. from 2.58 to 2.15 on ImageNet ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released))). Although MeanFlow is trained for 1-NFE, one can try splitting the step (e.g. integrate from t=1→0.5 and 0.5→0 separately) or using a simple solver like two Euler steps. This can reduce discretization error. (The risk is that the model wasn’t trained for two steps, so it may deviate; but follow-up work suggests it’s beneficial.)  
- **Post-Processing / Dithering** – SOTA diffusion and GAN papers sometimes apply post-processing to optimize metrics. For example, **dithering** the 8-bit quantization or adding sub-pixel jitter (tiny uniform noise before quantization) can reduce bias introduced by flooring pixel values. Instead of `floor(x*255)/255`, one might round or add uniform [0,1)/255 noise. This more closely matches how real images are distributed and often slightly improves FID. Conversely, one can apply a mild Gaussian blur (σ≪1) to outputs to suppress high-frequency noise. Such smoothing has been used in GAN literature to improve visual quality, though it can reduce sharpness.  
- **Ensembles / Mixtures** – Modern generative works sometimes mix multiple models or latent samples. A primitive approach: generate, say, two samples per latent (with different fixed seeds or latent splits) and take a pixel-wise average or select the sharper image. More fittingly, one can run the network twice on the same noise (e.g. one pass as-is, one pass with a slightly perturbed seed) and then combine. This can be seen as a crude ensemble that may boost FID. Another idea is **latent ensembling**: take two random noise vectors \(z_1,z_2\), map each to images, then output their average image; this tends to produce more “average” images (lower diversity) but sometimes higher fidelity. Similarly, trimming output extremes (“top-k” sampling of outputs accord. to some quality measure) is possible. These ensemble tricks can lower FID at the cost of recall.  
- **Quantization Strategy** – The exact way we quantize to 8-bit can matter. As noted, rounding with dither is one approach. Another is to convert images to a color space (e.g. YCbCr) and quantize only luminance (a trick in JPEG) before converting back. Or simply skip quantization altogether (if the FID implementation supports float images) to avoid information loss. These tweaks are inspired by SOTA GANs where evaluation details (clipping, rounding) are carefully matched to the training domain.  
- **Noise Calibration / Filtering** – Some flows add random noise at the last step (like a tiny Gaussian). Tuning the scale of this “scheduler noise” (analogous to DDIM noise) or even setting it to 0 (pure deterministic push) can trade off diversity vs. fidelity. Likewise, one can filter out outputs with extremely low “confidence” (for some proxy); for example, pass generated images through a pretrained classifier or discriminator and discard or tweak those with very low likelihood. This amounts to calibrating the final distribution to look more like the training set.  
- **Data Augmentation Consistency** – Since the model was trained with EDM-style augmentations, ensuring any test-time augmentation strategy matches the training (e.g. random flips, small croppings, noise) is crucial. Some SOTA diffusion workflows explicitly “augment” generated samples (add tiny noise or slight offsets) to mimic training noise. Although largely a training technique, at evaluation one can still enforce that any random seed for augmentation matches the checkpoint’s augmentation setting (which here is held fixed). In practice, this means setting `use_edm_aug=True` (matching training) as a simple best practice.  
- **Consistency Models / Distillation** – Recent SOTA includes techniques like **Consistency Models** (Song et al. 2023) and distillation methods that precompute teacher outputs. Inference-wise, one might precompute a small lookup (e.g. running a fast first-stage network on the noise) to “guess” a refined initial image, then apply MeanFlow as a second step. While MeanFlow is already one-step, one could imagine a two-network pipeline (flow-distilled refinement) in post-hoc fashion. Though not directly published for flows, analogous ideas from diffusion (DDIM/Consistent models) suggest any shortcut that predicts an intermediate output can help.  
- **Regularized Outputs** – Ensuring generation doesn’t produce out-of-distribution artifacts is critical. Some SOTA works use *soft clipping*: instead of hard clipping to [−1,1], they blend values outside that range back into it with a smooth function, preserving color integrity. Similarly, one can apply a small amount of 2D spectral normalization (filter the Fourier spectrum to match the training set’s frequency profile). These are advanced image processing steps adopted in top-tier GAN papers to boost FID.  
- **Ensemble of Checkpoints** – If multiple checkpointed models are available (net_ema0/1/2), one can *interpolate* between them or ensemble their outputs. For example, take the output image from net_ema1 and clip it by combining (pixelwise average or weighted sum) with the output from net_ema0. This smooths out model-specific biases. More simply, generate half of the samples with one EMA and half with another, then pool them all for FID. Mixing EMA levels can sometimes improve overall distribution coverage.  

# Parameter Optimization Insights  
- **EMA Decay Selection** – The MeanFlow repo gives three EMAs (net_ema0,1,2). In practice, the **higher-decay EMA** (net_ema1 with 0.99995 decay) typically yields slightly better sample quality. Similar works often find that an EMA of ~0.9999–0.99999 is best ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released)) ([openreview.net](https://openreview.net/forum?id=B5IuILRdAX#:~:text=innovative%20approach%20designed%20to%20accelerate,When%20evaluated%20on%20GenEval)). One should test all provided EMA checkpoints and pick the lowest FID. (In DiT flows, authors often find the newest EMA is optimal for FID.)  
- **Sampling Temperature / Noise Scale** – Diffusion literature commonly uses a “temperature” factor τ on the initial noise, with τ<1 giving sharper images and τ>1 more variation. In practice, values around 0.7–1.0 are typical: for example, many GANs use truncation at ~0.7. For flows one can analogously try τ = 0.8, 0.9, 1.0, 1.1. We expect gains near τ≈0.8–0.9 for higher fidelity (lower FID), with the risk of mode collapse if τ≪1. Aside from Gaussian scale, one can also try scaling the model’s output vector (clamp strength), but usually the latent.scale is the most effective.  
- **Batch Size (Inference)** – Most papers use large batch sizes at test time for efficiency. Here, the default is 128. It’s safe to increase this to the GPU max (e.g. 256 or 512) to speed up generation. Batch size does *not* affect FID quality (aside from numerical noise), but larger batches usually improve GPU utilization. No downside except memory limits.  
- **Number of Samples for FID** – We keep 50k samples for accuracy. (Reducing to 10k might save time but makes FID noisy.) SOTA evaluations always use ≥50k for reliable FID ([news.mit.edu](https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321#:~:text=against%20a%20starry%20background)), so no savings to quality if lowered.  
- **Image Quantization** – As a parameter, one can toggle how output floats are quantized. The default used floor(×255)/255, but one can experiment with round(×255)/255 or dithering. GAN papers often note that precise bit-depth treatment can shift FID by ~0.1–0.2. If running for final results, test both rounding and dithering – the best practice (minimal bias) is often using round-to-nearest integer plus a 0.5 shift. This is a small constant tweak but can tip FID by a few tenths.  
- **Latent Truncation Radius** – Some flows allow clamping the random vector’s norm. For instance, limit \|z\|≤r for some threshold (like r=2.0 or 2.5, instead of unbounded normal). Truncation to ~2σ is common in BigGAN. Values between 1.5–2.5 are worth trying: lower values (≤2) reduce extreme latents. Works have shown that reducing the allowed latent norm (even just scaling z to have magnitude ≤c) improves sample quality at modest coverage loss. The ideal radius depends on training (often around the 98th percentile of radius). Testing r=2.0 vs r=3.0 is a simple hyper-parameter sweep.  
- **Integration Steps** – By default MeanFlow uses exactly 1 ODE step. To generalize, one can set the solver to use 2 or more steps (if the code supports it). For example, a simple Euler split (t=1→0.5, then 0.5→0) effectively attempts a two-stage flow. In some MeanFlow variants (e.g. α-Flow), 2-NFE dramatically lowers FID ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released)). We can try increasing “steps” to 2 (if implemented) or manually iterate twice. Expected gain: potentially large (several tenths of FID) if the model tolerates it; risk is that the model was not trained for multi-step, so results could degrade or even fail. Nonetheless, it’s a strong lever (moderate gain, medium–high risk).  
- **Net-EMA Ensemble/Interpolation** – One can linearly interpolate between EMA weights (net_ema0 & net_ema1) before sampling. For example, average their parameters or outputs. In practice, a simple trick is to generate half the batch with EMA1 and half with EMA0 and mix these images. Such ensembling rarely hurts fidelity and can boost diversity slightly. Estimated gain is small (maybe 0.05–0.1 FID) but risk is minimal.  
- **Post-Sample Filtering** – Use a pretrained image discriminator or classifier to *filter* outputs: e.g. discard images with low “realism scores” (like a low Softmax probability under an Inception classifier). This isn’t commonly published, but as a last resort, selecting only the top 90% of samples (by some quality metric) can reduce FID slightly. This is essentially mode-pruning (hurts recall). Gain is uncertain (~0.1–0.3), risk is diversity loss.  
- **Output Clipping and Normalization** – After generation, one can enforce pixel range clipping (e.g. clamp output to [0,1] or [−1,1]) and optionally normalize the color histogram to match the training set (via a simple affine transform on RGB channels). Clipping is often done in diffusion outputs. If the model occasionally overshoots range, clamping can prevent extremely bad samples. The gain from clipping is minor (it mostly fixes outliers) and the risk is zero. Histogram matching is more aggressive: it can reduce slight color biases but may distort images if not done carefully (risk moderate).  

# Parameter Ranges and Hyperparameters in Similar Work  
- **Network size**: Transformers like DiT or U-ViT are common backbones. Typical DiT variants use hidden dimensions from 512 to 1024 and depths of 6–12. For example, Zhang *et al.* report using DiT-XL/2 (~6 layers) with ~675M parameters ([www.researchgate.net](https://www.researchgate.net/publication/397934965_Understanding_Accelerating_and_Improving_MeanFlow_Training#:~:text=Flow%E2%80%99s%201,NFE%20FID)). Smaller “S/2” models are ∼100–150M parameters. In practice, more capacity usually improves FID (e.g. DiT-S/2 yields 11.6, B/2 yields 7.85 FID on CIFAR-10 in one survey ([www.researchgate.net](https://www.researchgate.net/publication/397934965_Understanding_Accelerating_and_Improving_MeanFlow_Training#:~:text=Method%20FID%20%281))). We are fixed by the checkpoint, but this suggests that any larger available model may give lower FID.  
- **EMA decay**: Values tested are ~0.9999 (net_ema0/2) and 0.99995 (net_ema1). Most papers favor the slightly higher decay (0.99995) for best FID ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released)). If working with other checkpoints, common EMA decays range 0.999–0.99995 (too low decays hurt convergence, too high decays borderline model).  
- **Sampling step size in equation**: Although MeanFlow is “one-step,” analogous works try a small number of solver steps (NFE=2–5) when allowed. Their gains plateau quickly – e.g. 2 steps often suffices. We’d limit to 2 steps if possible.  
- **Noise scale / Temperature**: In diffusion models, temperatures ~0.7–1.0 are common; flows similarly can use a scaling factor k in [0.5,1.5]. Past work (e.g. α-Flow) even experimented with injecting noise of varying magnitude (they cite a 3×3 grid of scale factors) and found an optimum around the default noise level. We would likely try scaling factors 0.7, 1.0, 1.3 as representative.  
- **Truncation radius**: StyleGAN-style truncation uses radii from 1.0 (none) down to 0.5 (very tight). For flows, applying r ∈ [1.5, 2.5] (in standard deviation units) is reasonable. Some GAN papers choose r=0.7–1.0 in their latent space; for Gaussian noise a radius ~2 (covering 95% of mass) is a balance of fidelity/diversity.  
- **Quantization**: The two main choices are floor(x*255) vs. round(x*255). Modern GAN evaluation typically uses **round**. Optionally adding uniform dither ∼U(0,1) before rounding is also common to reduce banding. No published flow/diffusion paper benchmarks this explicitly, but it’s a known GAN trick.  
- **Precision**: Many SOTA generative pipelines run in float16 or bfloat16 at inference. If numeric issues arise, try float32 to ensure highest fidelity. (Rarely does precision alone change FID, but it prevents underflow errors when clipping very bright/dark pixels.)  

# Concrete Optimization Ideas (Inference-only)  
1. **Switch EMA checkpoint**: Evaluate all offered EMA weights (net_ema0, net_ema1, net_ema2). In MeanFlow’s ablations, EMA1 (decay=0.99995) was found best. Still, try each – especially net_ema0 or 2 (decay=0.9999) – on your 50k samples. *Expected gain:* Small (tenths of an FID point); different EMA may marginally reduce noise. *Risk:* Very low (just inference variants).  
2. **Increase batch size**: Raise the inference batch size as high as GPU memory allows (e.g. 256 or 512). This doesn’t change FID per se, but speeds up generation volume. *Gain:* Orders-of-magnitude time speed-up. *Risk:* None to FID (only if out-of-memory).  
3. **Adjust sampling temperature**: Try scaling the latent noise by τ<1 (e.g. 0.8, 0.9). This often sharpens images. For example, in DiffusionGAN-style models, a truncation/temperature of ≈0.7–0.9 can cut FID a few points. *Expected gain:* 0–0.3 FID improvement (sharpness boost); *Risk:* High τ reduction (≪1) collapses diversity, so stay moderate. Also try τ>1 (e.g. 1.1) to check if slight “oversampling” helps (usually it doesn’t).  
4. **Two-stage sampling (2-NFE)**: If the code allows, perform two ODE steps instead of one. Concretely, integrate from t=1→0.5 and then 0.5→0 with the learned mean-velocity field. α-Flow’s results ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released))and others show 2-step greatly lowers FID. *Expected gain:* Possibly large (≈0.5–1.0 FID drop if model tolerates it); *Risk:* Moderate–high. The model was trained for 1 step, so results are unpredictable: it may overshoot or distort. Always inspect outputs. This is a complex tweak but a prime candidate for big gain.  
5. **Latent truncation**: Clamp or scale the initial noise. For example, sample z∼N(0,I) and if ‖z‖>r (e.g. r=2.0), resample or project it to the shell of radius 2. Or simply multiply all latents by 0.8. *Expected gain:* Small to moderate (≈0.1–0.5 FID) – reduces output extremity. *Risk:* Lower diversity (some modes collapse). This is similar to BigGAN truncation. Even if only applied to the tails (e.g. ignore latents beyond 2σ), it can help fidelity.  
6. **Quantization finesse**: Change the rounding scheme for pixel values. Instead of `floor(x*255)/255`, try `round(x*255)/255` **with optional dithering** (add a tiny uniform U[0,1)/255). Dithering is known in GANs to slightly boost scores. *Gain:* ~0.05–0.2 FID by reducing quantization error. *Risk:* Very low. (Alternatively, skip quantization entirely if the FID implementation allows float inputs; though real images are discrete, so rounding is safer.)  
7. **Output clipping/normalization**: After generation, clamp all pixels strictly to [0,1] (or [−1,1]) to eliminate out-of-range artifacts. Then optionally apply a global color histogram normalization (match mean/variance per channel to the train data). Gan literature allows slight color adjustments to reduce dataset bias. ([www.emergentmind.com](https://www.emergentmind.com/topics/modular-meanflow#:~:text=%283.41%29%20and%201,24%20Aug%202025)) *Gain:* Minor (fixes unnatural outliers); *Risk:* Low. Histogram matching can distort palette if misused (risk moderate). Clipping alone is safe and recommended.  
8. **Add tiny Gaussian blur**: Apply a 1–2 pixel Gaussian blur (σ≈0.5) to each image before FID. This is akin to “antialiasing” and is used occasionally in evaluation to suppress high-frequency noise. *Gain:* Potentially small (≲0.2 FID) if the model left artifacts. *Risk:* Loss of detail (could slightly hurt crispness and Precision, even if improving FID). Use with caution.  
9. **Post-sample filtering with discriminator/classifier**: If you have a pretrained Inception (or discriminator) model, score each generated image and remove the worst 5–10% before computing FID. This “prunes” blatantly bad images. *Gain:* Could drop FID a few tenths. *Risk:* Loss of diversity/modes. This is essentially tuning to the metric – it’s a delicate cheat. Only mention for completeness, as it’s “post hoc.”  
10. **Ensemble EMAs**: Generate half the samples with net_ema1 and half with net_ema0 (or average their weights). Pool all images for FID. Blending two models can sometimes approximate an interpolation that neither model alone achieves. *Gain:* Very small (few hundredths FID) but can fill tiny gaps in the distribution. *Risk:* None significant. It’s an easy low-risk test.  
11. **Multiple passes with noise**: Feed each latent twice through the model with tiny added random perturbations (e.g. add ε∼𝒩(0,0.01^2) to the input halfway) and average the outputs. This simulates a rudimentary “denoising” step. *Gain:* Uncertain (this is a guess from diffusion practice). *Risk:* Generated image could become blurry or otherwise corrupted, so test visually.  
12. **Auxiliary calibration step**: Run a short optimization on each sample (e.g. one gradient descent step) against a frozen “real image” discriminator, to fine-tune it. This is a heavy post-process (often called “discriminator guidance” in diffusion ([news.mit.edu](https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321#:~:text=DMD%20cleverly%20has%20two%20components,image%20with%20the%20student%20model))). *Gain:* If perfectly done, could raise every sample’s realism. *Risk:* Extremely high complexity; likely only a portrait’s local optimum. We list it more as a theoretical possibility from the diffusion literature than a practical first step.  

# Common Failure Modes and Pitfalls  
- **Overfitting to FID (Mode Collapse)** – Many of the above tricks (truncation, filtering, tuning temperature) will artificially **reduce diversity** while improving Perceptual Quality. A low FID can mask mode-dropping. Beware: if FID drops drastically but samples look “too similar,” you may have collapsed modes. Always inspect recall/diversity metrics (e.g. PR curves) when tuning.  
- **Mismatch of data augmentation** – If the model was trained with a certain augmentation (EDM augmentation, random flips, etc.), failing to match that at inference can hurt quality. For example, if you mistakenly disable `use_edm_aug` or apply the wrong flipping, the sampled images may systematically differ from training data. This can spike FID even if individual images look plausible. Always ensure augmentation usage is consistent with training.  
- **Quantization bias** – Using an incorrect quantization (e.g. floor instead of round) can systematically shift generated image colors toward one corner of pixel space, raising FID. Conversely, unbounded outputs can produce a few extreme pixels (values <0 or >1) that severely skew Inception statistics. It’s easy to forget to properly clamp or quantize, so carefully handle this step.  
- **Arithmetic under/overflow** – Pushing the model output beyond [−1,1] without clipping can produce NaNs or infinities when quantizing. This breaks FID entirely. Similarly, using float16 inference might cause tiny bright/dark values to flush to 0. Double-check numeric ranges. Typically, one should clip to [−1,1] before any conversion.  
- **Batch-statistics mismatch** – Generative FID implementations usually run on the CPU with float32. If you collect images in GPU tensor with normalization, forgetting to convert or normalize may yield garbage statistics. Ensure images fed to FID are proper uint8 arrays. Also, do not “augment” generated images (like full resizing) unless you apply the same to real images.  
- **Too few samples** – Reducing `fid_samples` to speed up evaluation will introduce high variance. People sometimes see seeming improvements with 10k samples that vanish at 50k. For reproducible optimization, always use ≥50k as in the baseline. (Smaller FID on fewer samples when the model is unchanged is just noise, not a real gain.)  
- **Neglecting random seed coverage** – Because generative outputs can vary widely with the seed, an optimizer might cherry-pick seeds that look good. In practice, always compute FID on a fixed random seed or average over several seeds. Tuning tricks on a single 5k batch can mislead you.  
- **Improper detector usage** – If you use a classifier/discriminator for filtering or guidance, poor choice of network or miscalibration (threshold too strict) can discard valid samples. For instance, a pretrained Inception might have blind spots (e.g. on certain CIFAR classes). Relying on such a “critic” can introduce its own bias.  
- **Dimensionality mismatches** – MeanFlow is class-conditional by default. If your pipeline accidentally feeds in mismatched labels (e.g. zeros for all images in an unconditional run), sample quality can degrade significantly. Ensure that if using a class-conditional checkpoint, you supply the correct class vector or use “unconditional” mode. Mismatched inputs are a common bug.  
- **Mis-specifying time indexing** – This one-step model expects the interval [t₀=0, t₁=1]. Calling the sampler with wrong time endpoints (even something like 0 to 0.99) will produce wrong outputs. It’s a subtle bug if you try to adjust the diffusion noise schedule “by hand.” Always confirm that the ODE integrator spans the full unit interval as intended.  

Each of the above leverages only inference-time changes (no retraining) and has documented use in recent generative modeling work ([openreview.net](https://openreview.net/forum?id=adacb4JTIv#:~:text=objectives%2C%20and%20achieves%20better%20convergence,checkpoints%20will%20be%20publicly%20released)) ([openreview.net](https://openreview.net/forum?id=B5IuILRdAX#:~:text=innovative%20approach%20designed%20to%20accelerate,When%20evaluated%20on%20GenEval)) ([www.emergentmind.com](https://www.emergentmind.com/topics/modular-meanflow#:~:text=%2A%20For%20image%20synthesis%2C%20curriculum,03%20s%2Fimage%29%20%28%206)). They range from very low risk (batch size, EMA choice) to higher risk (multi-step sampling, discriminator guidance) but can offer incremental FID improvements if tuned carefully.

