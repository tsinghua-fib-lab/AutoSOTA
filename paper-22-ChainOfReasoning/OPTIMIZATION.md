# Paper 22 — ChainOfReasoning

**Full title:** *Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective*

**Registered metric movement (internal ledger, ASCII only):** +3.38%(85.29->88.17)

# Optimization Results: Chain-of-Reasoning (GSM8K)

## Summary
- **Total iterations**: 3
- **Best `accuracy`**: 88.17% (baseline: 85.29%, improvement: +2.88%)
- **Target**: ≥87.2304% — **ACHIEVED ✓**
- **Best commit**: 9946ab8e66

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| Accuracy | 85.29% | 88.17% | +2.88% |
| Correct | 1125/1319 | 1163/1319 | +38 |
| Total | 1319 | 1319 | — |
| Sampling strategy | Greedy (T=0.0, N=1) | Self-consistency (T=0.6, N=5) | — |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Self-consistency: N=5 samples, T=0.6, majority vote | +2.88% accuracy | Main driver of improvement |
| Improved answer extraction regex | +0.0% accuracy | Model already uses \\boxed{} correctly |

## What Worked

- **Self-Consistency (IDEA-004)**: Sampling 5 diverse reasoning paths at T=0.6 then taking majority vote dramatically improved accuracy. This is the key finding — the single most impactful change. The model's reasoning is stochastic at non-zero temperature, and different paths sometimes make different errors. Majority voting aggregates these to find the most commonly-arrived-at answer.

## What Didn't Work

- **Plan-and-Solve Prompting (IDEA-011)**: Changing the system prompt to ask the model to plan before solving actually hurt performance (-0.53%). This indicates Qwen2.5-Math-1.5B-Instruct is optimized for the standard step-by-step CoT format.
- **Improved Answer Extraction (IDEA-002)**: Adding more regex patterns for answer extraction had no effect. The model already produces clean \\boxed{} answers in nearly all cases.

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter Optimization Insights**

Insights from recent results include:

- **Decoding temperature and sampling:** A modest temperature (T≈0.5–0.7) typically balances creativity and reliability. Wang et al. found T=0.5–0.7 improved consistency on CoT tasks (your-ai-staff.com). Qwen2.5 guidelines specifically use *T=0.7*, top-p=0.8 for majority-vote sampling (github.com). Too high T (1.0) adds diversity but more errors; too low (0.0) yields deterministic chains that may trap into a single reasoning path. Nucleus (top-p≈0.8–0.9) or top-k (k≈40) sampling is common; in one study GPT-3 used T=0.7 without top-k, PaLM T=0.7 with k=40 (your-ai-staff.com).

- **Best-of-N size (N):** Typical choices are N=5–20. Self-consistency papers used around 40 samples for their big models (your-ai-staff.com), but smaller N may still help. The Qwen example uses “maj@8” (8 shots) and “RM@8” for a reward model ensemble (github.com). In practice, even N=5–10 yields gains, but more runs cost more time. A practical tip is to try increasing N until gains saturate (often diminishing returns after ~10–20).

- **Answer extraction patterns:** Chains usually end with phrases like “The answer is X.” or a boxed number. Common regex patterns include `\\boxed{([^}]*)}`, `The answer is ([\d\.\-+()/]+)`, or `#### (\d+)` (if model outputs Markdown). Standardizing on one format (e.g. always “\boxed{}”) and using a robust parser (e.g. removing LaTeX symbols) improves extraction. The existing CoR code looks for `\boxed{}`, `####`, or *"answer is"* (github.com); one can add more patterns (e.g. “= X”, or final sentence contexts) to catch edge cases.

- **Context length (max_model_len):** For very long problems or multi-step solutions, more context (4096+ tokens) may be needed. Extending beyond 4096 (if model supports it) could allow including examples or extra scratch space. Qwen2.5 uses 4096; if practical, raising to 8192 can accommodate extra few-shot examples or fuller chains. Watch GPU utilization and batch size when increasing window.

- **Memory & parallel settings:** The code has `tensor_parallel_size=2` and `gpu_memory_utilization=0.90` (github.com). Tuning these can allow larger batch or model size usage, but are mostly engineering parameters. Ensuring high GPU utilization (e.g. `gpu_memory_utilization` ~0.95) maximizes throughput, but may require reducing batch size.

**4. Concrete Optimization Ideas**

Below are *10+ actionable ideas*, roughly ordered from low to high complexity. For each, we estimate likely accuracy gain and risk/drawback:

1. **Tune the System Prompt:** Adjust wording (“step by step,” “show your work,” etc.) or add emphases (e.g. “ensure all steps are written out” or “double-check your answer”). Small wording tweaks (0.5–2% gain) can clarify the task. *Risk:* Very low (just prompt text).

2. **Few-Shot Examples:** Add 1–2 solved examples in the prompt (chain-of-thought style). May gain a few percent if examples are well-chosen. *Risk:* Low to moderate—longer prompt uses more tokens, and if examples mismatch the question style it can confuse the model.

3. **Temperature Sampling + Majority Vote:** Set temperature to ~0.5–0.7 and generate multiple answers (e.g. N=5–10). Take the most frequent final answer (your-ai-staff.com). This “self-consistency” often produces substantial gains (on similar tasks, >10% lift (your-ai-staff.com)). *Risk:* Medium (increased latency and GPU cost; if N is too small may not cover errors).

4. **Beam Search Re-ranking:** Instead of multiple random samples, use beam search (beam≥5) to generate top-K chains and pick the best. Could give a mild boost (1–3%) by exploring near-optimal alternatives. *Risk:* Low computation-wise, but beams tend to be shallow; may not improve much on already high-accuracy model.

5. **Top-k/Nucleus Tuning:** Try different top-p or top-k values. E.g. if current run is greedy (T=0, no top-k), allow T=0.3 or top-p=0.9 to introduce variation. This can uncover alternative reasoning paths. Expected gain small (~1%), but combined with voting might compound. *Risk:* Low.

6. **Answer Extraction Improvements:** Enhance the regex or parsing logic to catch answers. For instance, after decoding, strip LaTeX (`$`,`\\`) and match numerals at end of text. Add patterns like `= X` or look for any standalone number in final line. Could recover a few percent of correct answers that were previously missed by strict patterns. *Risk:* Low (mostly implementation overhead).

7. **Numerical Normalization:** Post-process model answers to canonical form (e.g. convert fractions to decimals, remove trailing `.0`, unify “−0.5” vs “0.5”). Likewise, ensure close numeric answers (within 1e-3) count as equal. This avoids counting correct answers as wrong due to formatting. Possible gain ~1–2% (fixing false negatives). *Risk:* Low.

8. **Chain Verification Pass:** After an answer is produced, prompt the model (or a smaller model) to double-check: e.g. “Is the above reasoning correct? If not, correct the steps.” This meta-query can catch simple arithmetic mistakes. In some studies, self-critique yields an extra 2–3% on reasoning tasks. *Risk:* Moderate (requires an extra inference; model may just re-affirm its own answer without change).

9. **Tool-Assisted Checking:** Use a mathematics library (e.g. Sympy) to verify the result. For each solution, extract the final numeric answer and plug into the original formula, or parse the chain’s math to check each step. If a discrepancy is found, perhaps try the next-best candidate answer. This can significantly reduce obvious mistakes (gain maybe up to +5%). *Risk:* High complexity (parsing is error-prone) and overhead; if parsing fails, may lose answers.

10. **Programmatic (TIR) Prompting:** Instead of only natural language, instruct the model to produce Python-style calculations (as Qwen supports). E.g. “solve using Python code and answer in a box.” This leverages the model’s learned coding ability and may yield more precise arithmetic. Potential gain might be +5% if done right. *Risk:* High – the environment must support code syntax (and possibly execution), and the model may output malformed code or move away from plain reasoning.

11. **Ensemble of Prompts:** Test multiple prompt variants (e.g. phrasing, examples, or method prompts) and ensemble their outputs. For example, run one prompt that emphasizes CoT and another that emphasizes TIR, then vote on answers. This could yield a few percent boost by combining different “reasoning styles.” *Risk:* Medium (multiple calls needed, risk in how to fairly combine answers).

12. **Increase Model Context:** If GPU allows, raise `max_model_len` beyond 4096 to include more context/examples. Longer context might let the model keep more intermediate work or extra hints. Gains are uncertain (maybe small), but could fix failures where the chain was truncated. *Risk:* Moderate hardware cost (memory, speed).

13. **Explicit “Final Answer” Prompt:** End every prompt with a clear cue, e.g. “Answer in the format: \boxed{<number>}.” This anchors the output format. Qwen already does this, but experimenting with different cues (like “####” prefix) might recover outputs that were previously parsed incorrectly. *Risk:* Low.

14. **Rollback Strategies for Failed Chains:** If the model’s output is clearly wrong (e.g. violates basic arithmetic), attempt a second inference with a stronger hint (“the previous reasoning had an error, try again more carefully”). This is akin to a “retry” or minor prompt-engineering. Gains likely small (~1-2%), as it catches only obvious blunders. *Risk:* Low to moderate.

_(Research digest truncated.)_

## Idea library snapshot (`idea_library.md`)

### IDEA-001: Improve System Prompt Wording
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Modify the SYSTEM_PROMPT to be more specific about step-by-step math reasoning and final answer format. Possible options:
- **Hypothesis**: Better prompt guidance reduces reasoning errors; expected +0.5-2% accuracy
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-002: Improve Answer Extraction Regex
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: The current `extract_answer()` function uses \\boxed{}, #### prefix, and "answer is X" patterns. Improvements:
- **Hypothesis**: Could recover 5-20 missed correct answers (+0.4-1.5% accuracy)
- **Status**: SUCCESS — 85.29%→85.29% (+0.0%). Model already uses \\boxed{} correctly in nearly all cases; extraction wasn't the bottleneck.
- **Result**: No improvement. Extended patterns added (=X, total is X) but no accuracy gain.

### IDEA-003: Better Numeric Normalization
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Improve `normalize()` to handle edge cases:
- **Hypothesis**: Fixes false negatives where model gives correct answer in different format; +0.3-1% accuracy
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-004: Self-Consistency (Temperature Sampling + Majority Voting)
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Use temperature=0.6-0.7 + N=5-8 samples per question, take majority vote on final answer.
- **Hypothesis**: Self-consistency paper shows +10-17% improvement on math tasks; expected +2-5% here
- **Status**: SUCCESS — 85.29%→88.17% (+2.88%). TARGET EXCEEDED! N=5, T=0.6 with majority vote delivers strong gains.
- **Result**: +2.88% accuracy. Majority voting across 5 diverse samples dramatically reduces error rate.

### IDEA-005: Extend max_tokens to 4096
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Increase `max_tokens` from 2048 to 4096 in SamplingParams to allow longer chain-of-thought reasoning for complex problems.
- **Hypothesis**: Some problems may be truncated with 2048 tokens; extending may recover +0.3-1%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-006: Few-Shot Examples in Prompt
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Add 2-4 solved GSM8K examples in the system prompt to demonstrate the reasoning format. Use examples that cover:
- **Hypothesis**: Few-shot examples clarify expected reasoning style; expected +1-3% accuracy
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-007: Explicit Final Answer Instruction
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Add explicit final answer formatting instruction at the END of each user question: "Please provide your final numerical answer in \\boxed{...}."
- **Hypothesis**: Ensures consistent \\boxed{} format for answer extraction; +0.3-1%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-008: Increase gpu_memory_utilization to 0.95
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Increase `gpu_memory_utilization` from 0.90 to 0.95 to allow vLLM to use more GPU memory, potentially enabling larger batch sizes.
- **Hypothesis**: Better throughput; minimal accuracy impact, but faster eval means we can try more ideas
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-009: Qwen Math-Specific System Prompt
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Use Qwen2.5-Math's recommended system prompt and CoT format from their documentation. The Qwen math models have specific prompt templates optimized for their training. Their recommended prompt: "Please reason step by step, and put your final answer within \\boxed{}." is what we already use, but we can try:
- **Hypothesis**: Model is trained for specific prompt format; matching exactly may help +0.5-1%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-010: Add Verification Pass to Prompt
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: After primary reasoning chain, add "Please verify your answer by reviewing the key steps" instruction. Two approaches:
- **Hypothesis**: Self-verification can catch simple arithmetic errors; +0.5-2%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-011: Plan-and-Solve Prompting
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Replace simple CoT prompt with Plan-and-Solve: "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve it step by step."
- **Hypothesis**: Better problem decomposition leads to fewer errors; expected +0.5-2%
- **Status**: FAILED — 85.29%→84.76% (-0.53%). This prompt format conflicts with Qwen2.5-Math training. Model performs worse with plan-and-solve wording.
- **Result**: Regression. Rolled back to baseline.

### IDEA-012: Best-of-N with Multiple System Prompts
- **Type**: ALGO
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Run multiple different system prompts (each as a "voter"), take majority vote:
- **Hypothesis**: Diversity in reasoning paths improve accuracy; +1-3%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-013: Fraction and Mixed Number Parsing in Extractor
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Specific enhancement to `extract_answer()`:
- **Hypothesis**: LLMs sometimes output fractions in LaTeX form; recovering these improves accuracy +0.3-0.8%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-014: Temperature Sweep (Find Optimal T)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Test T=0.0 (baseline), T=0.3, T=0.5, T=0.7 with N=1 (single sample) to find the optimal temperature for this model. Even single-sample with non-zero T can sometimes be better than greedy.
- **Hypothesis**: Model may get stuck in suboptimal reasoning chains with greedy; slight randomness helps +0.5-1%
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-015: Chain-of-Thought with Code Verification (LEAP)
- **Type**: LEAP
- **Priority**: MEDIUM
- **Risk**: HIGH
- **Description**: A+B integration: A = NLR chain-of-thought answers, B = Python code execution (AR from CoR framework). For each question:
- **Hypothesis**: Python code execution produces exact arithmetic; recovers errors in NLR; expected +2-5%
- **Status**: PENDING
- **Result**: (fill in after execution)
