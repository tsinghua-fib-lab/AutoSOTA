# Paper Optimization Mission

You are an autonomous AI research engineer. Your mission: optimize the performance of **Mean Flows for One-step Generative Modeling** beyond its paper-reported metrics, by iteratively modifying the code inside a Docker container and running experiments.

## Environment

| Parameter | Value |
|-----------|-------|
| Docker container | `paper_opt_paper-84` |
| Repo path (in container) | `/py-meanflow` |
| Evaluation command | `cd / && python eval_cifar10.py` (run from `/py-meanflow`) |
| Eval timeout | `2700` seconds |
| GPU devices | `6,7` |
| Docker image | `docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |

## Baseline Metrics (Paper-Reported)

| Metric | Baseline | Direction |
|--------|----------|-----------|
| Fid | **2.8883** | ↓ lower is better |

**Optimization goal**: Decrease `fid` to <= **2.8305** (↓ 2.0% vs baseline of 2.8883). Lower is better for this metric.

## Your Output Directory

All memory, results, and logs go to: **`/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650`**

```
/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/
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
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/scores.jsonl' \
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
  --gpus '"device=6,7"' \
  --shm-size=16g \
  \
  -v /home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results:/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results  docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime sleep infinity
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
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/scores.jsonl' \
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

**Save your analysis to `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/memory/code_analysis.md`**:

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

No deep research was conducted for this run. Generate your idea library based purely on your code analysis and domain knowledge.

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

**Save to `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/memory/idea_library.md`**:

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

Read `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/memory/idea_library.md`.
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
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/scores.jsonl' \
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
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/scores.jsonl' \
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

You MUST update `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/memory/idea_library.md` after every iteration. This file is your persistent memory across iterations; skipping this step means future iterations lose context.

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
tail -10 /home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/memory/idea_library.md
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
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/scores.jsonl' \
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
Save to `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/final_report.md`:

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
Output the full contents of `/home/dataset-assist-0/AUTOSOTA/sota-6/auto-pipeline/optimizer/papers/paper-84/runs/run_20260319_131650/results/scores.jsonl` so the user can see the complete optimization trajectory.

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


