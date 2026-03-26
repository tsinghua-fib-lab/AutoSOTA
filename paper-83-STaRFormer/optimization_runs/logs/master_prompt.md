# Paper Optimization Mission

You are an autonomous AI research engineer. Your mission: optimize the performance of **STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data** beyond its paper-reported metrics, by iteratively modifying the code inside a Docker container and running experiments.

## Environment

| Parameter | Value |
|-----------|-------|
| Docker container | `paper_opt_paper-3938` |
| Repo path (in container) | `/repo` |
| Evaluation command | `CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard` (run from `/repo`) |
| Eval timeout | `3600` seconds |
| GPU devices | `2,3` |
| Docker image | `autosota/paper-3938:reproduced` |

## Baseline Metrics (Paper-Reported)

| Metric | Baseline | Direction |
|--------|----------|-----------|
| Test Acc | **0.9738** | ↑ higher is better |
| Test F1 | **0.9755** | ↑ higher is better |
| Test Precision | **0.98** | ↑ higher is better |
| Test Recall | **0.9716** | ↑ higher is better |

**Optimization goal**: Improve `test_f1` to >= **0.995** (↑ 2.0% vs baseline of 0.9755). Higher is better for this metric.

## Your Output Directory

All memory, results, and logs go to: **`/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148`**

```
/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/
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
# Determine IS_NEW_BEST = true if new_primary_metric > BEST_SCORE, else false
docker exec paper_opt_paper-3938 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/scores.jsonl' \
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
docker ps --filter "name=paper_opt_paper-3938" --format "{{.Names}}"

# Start if not running
docker run -d \
  --name paper_opt_paper-3938 \
  --gpus '"device=2,3"' \
  --shm-size=16g \
  \
  -v /home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results:/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results  autosota/paper-3938:reproduced sleep infinity
```

### 0.2 Ensure Git is Available (inside container)

Git is required for version control and rollback. Check and install if missing:
```bash
docker exec paper_opt_paper-3938 bash -c "
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
docker exec paper_opt_paper-3938 bash -c "
  cd /repo
  git rev-parse --git-dir 2>/dev/null && echo 'git already init' || git init -q
  git config user.name optimizer && git config user.email opt@local
  git add -A
  git commit -q -m 'baseline' --allow-empty
  git tag -f _baseline
  echo '[Git] Baseline snapshot created. HEAD:' && git rev-parse HEAD
"

# 2. Copy the record_score.sh helper into the container
docker exec paper_opt_paper-3938 mkdir -p /tools
docker cp /home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/scripts/record_score.sh paper_opt_paper-3938:/tools/record_score.sh
docker exec paper_opt_paper-3938 chmod +x /tools/record_score.sh
echo '[Tools] record_score.sh installed at /tools/record_score.sh'
```

### 0.4 Run Baseline Evaluation

Run the evaluation to confirm it works and record the true baseline:
```bash
docker exec paper_opt_paper-3938 bash -c "cd /repo && timeout 3600 CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard 2>&1"
```

Parse the output to extract all metric values.

**Eval output format hint** (from onboarding):
```
The script prints a Lightning-style table at the end, like:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/acc          │    0.9738482236862183     │
│          test/f1          │    0.9755123853683472     │
│       test/loss_ce        │    0.14982345700263977    │
│      test/precision       │    0.9800432324409485     │
│        test/recall        │    0.9716342091560364     │
└───────────────────────────┴───────────────────────────┘
Metric keys in output use slash notation (test/acc, test/f1, etc.).
Map to snake_case: test/acc -> test_acc, test/f1 -> test_f1, etc.
```

Record this as iteration 0 (baseline) using `record_score.sh`:
```bash
docker exec paper_opt_paper-3938 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/scores.jsonl' \
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
docker exec paper_opt_paper-3938 bash -c "find /repo -name '*.py' | grep -v __pycache__ | head -50"
docker exec paper_opt_paper-3938 bash -c "find /repo -name '*.yaml' -o -name '*.json' -o -name '*.cfg' -o -name '*.ini' | head -30"

# README
docker exec paper_opt_paper-3938 bash -c "cat /repo/README.md 2>/dev/null | head -150"

# Evaluation script — understand output format thoroughly
docker exec paper_opt_paper-3938 bash -c "cat /repo/train.py"

# Find all configurable parameters
docker exec paper_opt_paper-3938 bash -c "grep -rn 'threshold\|n_frame\|weight\|alpha\|beta\|lr\|epoch\|batch\|topk\|sigma\|margin' /repo --include='*.py' | grep -v '#' | head -40"

# Find argparse / config loading
docker exec paper_opt_paper-3938 bash -c "grep -rn 'argparse\|add_argument\|yaml.load\|json.load\|config\[' /repo --include='*.py' -l | head -20"
```

Read the key source files. Understand:
1. **Pipeline flow**: Data → Processing → Prediction → Metric computation
2. **Evaluation script**: How does `CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard` work? What exact output does it produce? How are metrics calculated?
3. **Optimization levers**: Every parameter, threshold, hyperparameter, algorithm branch
4. **Hard constraints / Red Lines** (ABSOLUTE DO NOT CHANGE — these are non-negotiable):
   - **Pretrained model weights**: never replace or fine-tune the paper's pretrained weights
   - **Dataset**: never modify, filter, augment, or mix train/test splits; test data must never be mixed into training
   - **Evaluation protocol**: the eval command, eval script logic, and how metrics are computed must not change
   - **Metric definitions and computation**: do not alter score aggregation, column/output selection, or result reduction logic. Specific examples:
     - `recall@k` — the value of `k` must not be changed
     - Text task context window size used during evaluation must not be changed
     - Sequence history window length used during evaluation (prediction tasks) must not be changed
     - If the paper averages results, do not switch to reporting the best/maximum instead
   - **Algorithm output integrity**: never directly overwrite or hard-code algorithm outputs/predictions; results must come from genuine model inference
   - **Metric-dimension trade-off**: do not sacrifice other important metric dimensions to inflate only the primary target metric
   - **Paper's core method**: optimizations must build on top of the paper's proposed algorithm/architecture — do not replace the paper's core contribution with a fundamentally different approach
   - Also read the paper's experimental setup carefully — any settings the paper explicitly aligns to (e.g. decoding strategy, sampling parameters, number of samples, ensemble method) must be preserved as-is.
5. **Run procedure**: How long does a full eval take? Are there faster partial runs?

**Save your analysis to `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/memory/code_analysis.md`**:

```markdown
# Code Analysis: STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data

## Pipeline Summary
<describe the high-level flow>

## Key Source Files
| File | Purpose |
|------|---------|
| ... | ... |

## Evaluation Procedure
- Command: `CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard`
- Output format: <describe exactly how to parse the metrics from stdout>
- Estimated runtime: <X minutes>

## Optimization Levers
| Parameter | Current Value | File:Line | Type | Notes |
|-----------|---------------|-----------|------|-------|
| ...       | ...           | ...       | threshold/hyperparam/algorithm | ... |

## Hard Constraints / Red Lines (DO NOT CHANGE — non-negotiable)

The following are absolute red lines that must never be crossed by any optimization idea:

- [ ] **Eval metric parameters**: recall@k → k value must not be changed; context window size for text tasks must not be changed; history window length for prediction tasks must not be changed; averaging must not be replaced with best/max reporting
- [ ] **Evaluation script logic**: eval command, eval script code, and metric computation logic must not be modified
- [ ] **Algorithm output integrity**: never hard-code, overwrite, or fabricate model predictions/outputs
- [ ] **Metric-dimension trade-off**: do not sacrifice other metric dimensions to inflate only the primary metric
- [ ] **Dataset integrity**: train/test split must be preserved; test data must not be mixed into training
- [ ] **Pretrained weights**: original pretrained model weights must not be replaced or fine-tuned
- [ ] **Core method**: the paper's proposed algorithm/architecture must be preserved — optimizations build on top of it, not replace it
- [ ] <paper-specific constraint 1: fill in>
- [ ] <paper-specific constraint 2: fill in>

## Initial Hypotheses
- ...
```

---

## PHASE 2 — Research Insights & Idea Library

No deep research was conducted for this run. Generate your idea library based purely on your code analysis and domain knowledge.

**Known optimization levers** (from onboarding):
# Loss / contrastive learning
- loss.lambda_cl: float (currently 0.6001) — weight on contrastive loss term
- loss.temp: float (currently 0.5) — temperature for contrastive loss
- loss.lambda_fuse_cl: float (currently 0.5) — fused contrastive loss weight

# Transformer model hyperparameters
- model.sequence_model.n_head: int (currently 8) — number of attention heads
- model.sequence_model.d_model: int (currently 32) — model hidden dimension
- model.sequence_model.dropout: float (currently 0.1017) — dropout rate
- model.sequence_model.dim_feedforward: int (currently 128) — FFN hidden dim
- model.sequence_model.num_encoder_layers: int (currently 3) — transformer depth

# Dynamic attention-based regional masking (DAREM)
- model.sequence_model.mask_threshold: float (currently 0.2078) — attention threshold for masking
- model.sequence_model.mask_region_bound: float (currently 0.1) — boundary for masked regions
- model.sequence_model.ratio_highest_attention: float (currently 0.3) — fraction of tokens to keep

# Training hyperparameters
- training.learning_rate: float (currently 0.00760) — initial learning rate
- training.batch_size: int (currently 256) — training batch size
- training.epochs: int (currently 300) — max epochs (early stop at patience=30)

# Output head
- model.output_head.activation: str (currently "selu") — activation in classifier head

# Optimizer
- optimizer.beta1: float (currently 0.857) — Adam beta1
- optimizer.beta2: float (currently 0.939) — Adam beta2
- optimizer.weight_decay: float (currently 0.000410) — L2 regularization

# LR scheduler
- callbacks.lr_scheduler.factor: float (currently 0.8) — LR reduction factor
- callbacks.lr_scheduler.patience: int (currently 8) — epochs before LR reduction
- callbacks.early_stop.patience: int (currently 30) — early stopping patience

Combine your code analysis, the research insights, and any known levers above to generate a comprehensive idea library.

**Save to `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/memory/idea_library.md`**:

```markdown
# Optimization Idea Library: STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data

Last updated: <date>

## Ideas

### IDEA-001: <title>
- **Type**: PARAM / CODE / ALGO
- **Priority**: HIGH / MEDIUM / LOW
- **Risk**: LOW / MEDIUM / HIGH  (risk of breaking the pipeline)
- **Description**: What exactly to change, and how
- **Hypothesis**: Expected effect on `test_f1` and other metrics
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-002: ...

(generate at least 12-15 ideas — quality over quantity; the majority must be Tier 1 ALGO and Tier 2 CODE, not parameter tuning)

## Iteration Log

| Iter | Idea | Type | Before | After | Delta | Status | Key Takeaway |
|------|------|------|--------|-------|-------|--------|--------------|
```

**Idea generation guidelines — CREATIVE-FIRST mandate:**

> ⚠️ **You are expected to produce bold, innovative ideas.** Experience shows that pure hyperparameter tuning (Type: PARAM) almost never achieves significant improvements on its own. The high-value breakthroughs come from architectural changes, algorithmic rewrites, and cross-paper technique integrations. PARAM ideas are strictly last-resort fine-tuning — do not lead with them.

Generate ideas in **three tiers**, in this order of importance:

**Tier 1 — Architecture & Cross-paper Integration (Type: ALGO) — generate at least 6 ideas**
Architectural changes, new modules, novel training objectives, loss function redesign, or "A+B" integrations that import a proven technique from another paper/domain into this codebase. These are the highest-potential ideas.
- For each Tier 1 idea, **explicitly name the source paper or technique** that inspires it (e.g., "inspired by SE-Net channel attention [Hu et al. 2018]").
- These ideas should come directly from synthesizing `research_report.md` with `code_analysis.md`.
- *Examples*: replacing a pooling layer with a cross-attention module from Paper X; adding contrastive regularization from Paper Y; introducing a test-time ensemble strategy from Paper Z.

**Tier 2 — Algorithmic Logic Rewrite (Type: CODE) — generate at least 6 ideas**
Non-trivial changes to how the algorithm works — smarter feature fusion, novel post-processing pipelines, better aggregation or decoding strategies, inference-time computation flow changes. These are *not* "change one number" — they change *how* the code operates.
- *Examples*: rewriting a greedy decoder as a beam search; replacing a fixed-window aggregation with an adaptive one; redesigning the scoring function.

**Tier 3 — Parameter Tuning (Type: PARAM) — generate at most 4 ideas**
Simple threshold, count, distance, or numeric value adjustments. These are **strictly last resort** — to be executed only after all Tier 1 and Tier 2 ideas have been explored. Mark every PARAM idea with `**Priority**: LOW`.

**Mandatory research integration (non-negotiable)**:
After reading `research_report.md`, extract **at least 3 specific techniques** from cited papers that are plausibly applicable to this codebase. Each must appear as a dedicated Tier 1 (ALGO) idea with the source paper/method name explicitly cited in the description. If the research report is unavailable, use your domain knowledge to identify 3 recent techniques from related literature.

Generate ideas at multiple granularities: micro (single formula/function tweak), meso (module-level rewrite), macro (new pipeline stage or module addition).

---

### ⛔ MANDATORY RED LINE SELF-CHECK — run this BEFORE finalising the idea library

After generating all ideas, go through **every single idea** and verify it does NOT violate any of the following red lines. If an idea violates even one, mark it `**Status**: REJECTED (red line violation)` and briefly note which rule it broke. Do NOT keep a red-line-violating idea as PENDING.

**Red Line Checklist** (apply to each idea):

| # | Red Line | Question to ask |
|---|----------|-----------------|
| R1 | Eval metric parameters must not change | Does this idea change `k` in recall@k, the context window size for text tasks, the history window length for prediction tasks, or switch averaging to best/max reporting? |
| R2 | Evaluation script and metric computation logic must not be modified | Does this idea touch the eval script or how metrics are calculated/aggregated? |
| R3 | No fabricating or hard-coding algorithm outputs | Does this idea directly overwrite model predictions or hard-code output values? |
| R4 | No sacrificing other metric dimensions | Does this idea gain on the primary metric by severely degrading other reported metrics? |
| R5 | No train/test data contamination | Does this idea mix test data into training, or modify the train/test split? |
| R6 | No modifying the dataset | Does this idea alter, filter, or re-label training or test data? |

After completing the self-check, append a `## Red Line Audit` section to `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/memory/idea_library.md`:

```markdown
## Red Line Audit

| Idea | R1 | R2 | R3 | R4 | R5 | R6 | Decision |
|------|----|----|----|----|----|----|----|
| IDEA-001 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-002 | ✗ R1 | ... | | | | | REJECTED |
```

Only ideas marked `CLEARED` are eligible for execution.

---

## PHASE 3 — Optimization Loop

Run up to **24 iterations**. Stop early if `test_f1` >= 0.995.

> ⛔ **HARD ITERATION LIMIT: 24 iterations.**
> After completing iteration 24, you MUST immediately proceed to **PHASE 4 (Finalize)** — do NOT start iteration 24+1 under any circumstances, even if you believe further improvement is possible. Running more iterations than allowed is a critical violation of the task constraints.

> **Metric direction for `test_f1`**: higher is better (↑ = maximize / ↓ = minimize accordingly).

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

#### NORMAL PATH → ① CONTINUOUS IDEATION + SELECT IDEA

##### 1a. Continuous Ideation (mandatory before every selection)

Before picking an idea, spend a moment generating **2-3 new ideas** informed by everything you know so far. This keeps the idea pool fresh and grounded in actual observations rather than upfront speculation.

Read the **Iteration Log** in `idea_library.md` and ask yourself:
- What patterns have emerged from the iterations so far? (what types of changes help vs. hurt?)
- What parts of the pipeline have NOT been explored yet?
- Are there techniques from `research_report.md` that past iterations make more or less promising now?
- Did any recent failure reveal a structural weakness that suggests a new approach?

Based on this reflection, write **2-3 new ideas** (Tier 1 or Tier 2 only — no PARAM ideas here) and append them to `idea_library.md` as new `### IDEA-0XX` blocks. Then run them through the Red Line checklist (R1–R6) and mark them CLEARED or REJECTED before proceeding.

> This step takes only a few minutes but ensures the idea pool continuously improves with accumulated knowledge. Do not skip it, even if the pool already has many PENDING ideas.

##### 1b. Select Idea

Read the updated `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/memory/idea_library.md`.

**Selection rules — strict priority order:**

1. **Pick a Tier 1 (ALGO) CLEARED PENDING idea first.** These have the highest potential. If multiple are available, prefer the one with the strongest hypothesis and the most direct research backing.
2. **Then Tier 2 (CODE) CLEARED PENDING ideas.** After all Tier 1 ideas are exhausted (all executed or REJECTED).
3. **PARAM ideas are only permitted in exactly two situations:**
   - ✅ You are currently inside a **Leap Honeymoon Period** — using these iterations to do focused parameter fine-tuning *on top of the leap's structural change* is appropriate and encouraged. It helps fully exploit the leap's potential.
   - ✅ ALL Tier 1 and Tier 2 CLEARED ideas are genuinely exhausted (all executed, failed, or rejected).
   - ❌ In any other situation, selecting a PARAM idea is **not allowed**.

> ⚠️ If you are about to pick a PARAM idea and neither of the two allowed conditions is met — **stop**. Instead, generate 3 new Tier 1 or Tier 2 ideas based on what you've observed so far, pass them through the Red Line checklist, and pick one.

- If all Tier 1 + Tier 2 ideas are genuinely exhausted, generate 5 new non-PARAM (CODE or ALGO) ideas before falling back to any remaining PARAM ideas.

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
   - **Red Line Check** — explicitly verify against all 8 red lines:
     - R1: Does it change eval metric parameters (recall@k's k, context window, history window, avg→best)?
     - R2: Does it modify eval script logic or metric computation?
     - R3: Does it hard-code or fabricate model outputs?
     - R4: Does it sacrifice other metric dimensions to inflate the primary metric?
     - R5: Does it mix test data into training or alter the train/test split?
     - R6: Does it modify or filter the dataset?
   - Any candidate that fails even one red line check must be **discarded entirely** — do not proceed with it.

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
docker exec paper_opt_paper-3938 bash -c "
  cd /repo
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
docker exec paper_opt_paper-3938 bash -c "cd /repo && python3 -c \"
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
docker cp /tmp/patch_iter.py paper_opt_paper-3938:/tmp/patch_iter.py
docker exec paper_opt_paper-3938 bash -c "cd /repo && python3 /tmp/patch_iter.py"

# Option C: For multiline edits, write the full file to host and copy in
docker cp /path/to/edited_file.py paper_opt_paper-3938:/repo/path/to/file.py

# Verify the change looks correct
docker exec paper_opt_paper-3938 bash -c "cd /repo && grep -n 'CHANGED_LINE' path/to/file.py | head -5"
```

---

#### ④ EVALUATE

```bash
docker exec paper_opt_paper-3938 bash -c "cd /repo && timeout 3600 CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard 2>&1"
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
docker exec paper_opt_paper-3938 bash -c "
  cd /repo
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
IS_NEW_BEST = new_primary_metric > BEST_SCORE   # e.g. new_primary_metric < BEST_SCORE (lower-is-better)
              OR (this is the baseline record)
```

Then call the script, passing `--is-best` accordingly:
```bash
# Success (replace IS_NEW_BEST with true or false):
docker exec paper_opt_paper-3938 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/scores.jsonl' \
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
docker exec paper_opt_paper-3938 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/scores.jsonl' \
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
IF new_primary_metric > BEST_SCORE:
    # ─── New best result! ───────────────────────────────────────
    # (_best tag was already moved by record_score.sh --is-best true)
    BEST_SCORE = new_primary_metric
    # If we were in honeymoon, the leap ultimately paid off — exit honeymoon
    IN_LEAP_HONEYMOON   = false
    HONEYMOON_REMAINING = 0
    Print: "✓ New best! test_f1: PREV → NEW"

ELIF this_iteration_was_LEAP AND NOT new_primary_metric > BEST_SCORE:
    # ─── Leap did not immediately improve — start Honeymoon Period ──
    # Do NOT roll back. Keep the leap changes as the new working base.
    # Allow up to 5 more normal iterations to evolve on top of this leap.
    IN_LEAP_HONEYMOON   = true
    HONEYMOON_REMAINING = 5
    docker exec paper_opt_paper-3938 bash -c "cd /repo && git tag -f _leap_entry"
    Print: "⚡ Leap result did not improve best score (test_f1: PREV → NEW)."
    Print: "   Starting 5-iteration Honeymoon Period — continuing to evolve from leap state."
    Print: "   Pre-leap best ({BEST_SCORE}) is preserved at tag _best."

ELIF IN_LEAP_HONEYMOON:
    # ─── Inside honeymoon — monitor progress ────────────────────
    HONEYMOON_REMAINING = HONEYMOON_REMAINING - 1
    IF HONEYMOON_REMAINING == 0:
        # Honeymoon expired — 5 iters passed with no new best
        docker exec paper_opt_paper-3938 bash -c "
          cd /repo
          git checkout _best
          git checkout -- .
          git clean -fd
          echo '[Rollback] Honeymoon expired. Restored to pre-leap best.'
        "
        IN_LEAP_HONEYMOON = false
        Print: "↩ Honeymoon expired (5 iterations, no improvement). Rolled back to pre-leap best ({BEST_SCORE})."
    ELSE:
        # Still in honeymoon — keep current state, keep iterating
        Print: "⏳ Honeymoon period: {HONEYMOON_REMAINING} iteration(s) remaining. Continuing from leap state."
        Print: "   test_f1: PREV → NEW  (best so far: {BEST_SCORE})"

ELIF new_primary_metric < BEST_SCORE * 0.97:
    # ─── Normal significant regression (outside honeymoon) ──────
    docker exec paper_opt_paper-3938 bash -c "cd /repo && git checkout _best && git checkout -- . && git clean -fd"
    Print: "↩ Regression detected. Rolled back to _best ({BEST_SCORE})"

ELSE:
    # ─── Minor change — keep as is and continue ─────────────────
    Print: "→ No improvement. test_f1: PREV → NEW"
```

> **Honeymoon summary**: After a LEAP iteration that doesn't immediately improve the score, the framework enters a 5-iteration "Honeymoon Period". During this period, normal optimizations continue on top of the leap changes (treating the leap state as the new working base). PARAM fine-tuning is explicitly allowed and encouraged during honeymoon — it helps explore the full potential of the leap. If any of those 5 iterations sets a new best, the leap is considered successful and the honeymoon ends. If all 5 pass without a new best, the code rolls back to the pre-leap best (`_best` tag), discarding the leap changes entirely.

---

#### ⑦ REFLECT & UPDATE IDEAS  ← **MANDATORY — do not skip**

You MUST update `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/memory/idea_library.md` after every iteration. This file is your persistent memory across iterations; skipping this step means future iterations lose context.

Specifically:
1. **Mark the executed idea's Status** — change from `PENDING`/`IN_PROGRESS` to `SUCCESS` or `FAILED`:
   - Success: `**Status**: SUCCESS — overall X→Y (+Z%), metric_a A→B, metric_b C→D`
   - Failed: `**Status**: FAILED — reason: <what went wrong>`
2. **Add 2-3 new ideas** suggested by this iteration's observations (append new `### IDEA-0XX` blocks).
   - New ideas must be **Tier 1 (ALGO) or Tier 2 (CODE)** whenever possible — do not add more PARAM ideas unless you have a very specific hypothesis about an untested parameter range.
   - After a **LEAP** iteration (whether it succeeded or failed), pay special attention: write down what you learned about the codebase's structural constraints, and propose follow-up ideas that refine or build further on the leap's approach (these should also be Tier 1/2).
3. **Append one row** to the `## Iteration Log` table at the bottom. **Include the Type column** — this is critical for the pre-iteration reflection in step ⓪:

```
| {ITER} | IDEA-0XX | Type | {PREV_SCORE} | {NEW_SCORE} | {DELTA} | SUCCESS/FAILED | <one sentence takeaway> |
```

The `Type` column must be one of: `PARAM`, `CODE`, `ALGO`, `LEAP`.

> If this iteration ran during a **Honeymoon Period**, append `[HP-N]` to the Notes column (e.g., `[HP-2]` means 2 honeymoon iters remaining after this one). This makes it easy to trace which iterations were leap-influenced.

To verify the update happened, read back the last 10 lines of the file after writing:
```bash
tail -10 /home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/memory/idea_library.md
```

---

### ─── End of iteration loop ─────────────────────────────────────────

**Stop condition check (MANDATORY at the start of every iteration):**
```
CURRENT_ITER += 1

IF CURRENT_ITER > 24:
    # Hard limit reached — this line should never be hit if you checked at end of prev iter
    STOP. Proceed to PHASE 4 immediately.

IF BEST_SCORE >= 0.995:
    Print: "🎯 Target reached! Proceeding to finalize."
    STOP. Proceed to PHASE 4 immediately.

IF CURRENT_ITER == 24:
    # This is the LAST allowed iteration — run it, then go to PHASE 4. No exceptions.
    Print: "⚠️  Final iteration (24/24). Will finalize after this."
```

---

## PHASE 4 — Finalize

### 4.1 Restore Best State
```bash
docker exec paper_opt_paper-3938 bash -c "
  cd /repo
  git checkout _best 2>/dev/null || git checkout _baseline
  echo '[Final] Container restored to best known state.'
  git log --oneline -3
"
```

### 4.2 Final Evaluation
Run one last full evaluation and record in scores.jsonl as iter=`final` using `record_score.sh`:

```bash
docker exec paper_opt_paper-3938 bash -c "cd /repo && timeout 3600 CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard 2>&1"
```

Parse the output, then:
```bash
docker exec paper_opt_paper-3938 bash -c "
  bash /tools/record_score.sh \
    --scores   '/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/scores.jsonl' \
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
docker exec paper_opt_paper-3938 bash -c "cd /repo && git diff _baseline HEAD --stat"
docker exec paper_opt_paper-3938 bash -c "cd /repo && git diff _baseline HEAD"
```

### 4.4 Write Final Report
Save to `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/final_report.md`:

```markdown
# Optimization Results: STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data

## Summary
- Total iterations: N
- Best `test_f1`: X% (baseline: 0.9755%, improvement: +Y%)
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
Output the full contents of `/home/dataset-assist-0/AUTOSOTA/sota-11/auto-pipeline-ab/optimizer/papers/paper-3938/runs/run_20260324_234148/results/scores.jsonl` so the user can see the complete optimization trajectory.

---

## General Rules

1. **One change per iteration** — isolate variables to understand causality
2. **Always snapshot before modifying** — never lose a working state
3. **Record everything** — even failures teach us something; log them all
4. **Never cross these red lines** (absolute prohibitions — no exceptions under any circumstances):
   - **Eval metric parameters**: do not change `k` in recall@k; do not change context window size for text evaluation tasks; do not change history window length for prediction evaluation tasks; do not replace mean/average aggregation with best/max reporting
   - **Evaluation script and metric computation**: do not modify the eval script, score aggregation logic, column selection, or how results are reduced to a single number
   - **Algorithm output integrity**: do not hard-code, overwrite, or fabricate model predictions or inference outputs — all results must come from genuine model inference
   - **Metric-dimension trade-off**: do not sacrifice other important metric dimensions to inflate only the primary target metric
   - **Dataset contamination**: do not mix test data into training sets; do not alter, filter, re-label, or re-split the training or test data
   - **Pretrained weights**: do not replace or fine-tune the paper's pretrained model weights
   - **Core method preservation**: do not replace the paper's fundamental algorithm or architecture with a different approach — optimizations must build ON TOP OF the paper's contribution
5. **Container crashes**: restart with `docker start paper_opt_paper-3938` then re-check git status
6. **Long evals**: if full eval takes too long, first confirm with a quick sanity-check run (subset of data), then do full eval only if sanity check looks promising
7. **Be scientific**: form hypotheses, test them, update beliefs accordingly

---


