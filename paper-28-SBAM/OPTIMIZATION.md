# Paper 28 — SBAM

**Full title:** *Segment-Based Attention Masking for GPTs*

**Registered metric movement (internal ledger, ASCII only):** +0.22%(68.0->68.15)[ ]

## Iteration trace (`scores.jsonl`)

Structured log of each optimizer attempt (including failures and rollbacks):

- **Iter 0** — *Paper baseline* — `success` — primary `68` — Baseline evaluation with MAS model (MasLlamaForCausalLM + LoRA). Dataset reconstructed from HuggingFace parquet/BEIR. My baseline avg=68.0 vs paper=68.67.
- **Iter 1** — *Case-insensitive + last match extraction* — `success` — primary `68` — Changed extract_answer to lowercase and return last match. No improvement - model already outputs lowercase consistently.
- **Iter 2** — *Temperature=0.0 greedy decoding* — `success` — primary `68` — Temperature=0.0 with num_beams=4 gives identical results. Beam search dominates temperature in this regime.
- **Iter 3** — *batch_size=1 (no padding)* — `success` — primary `68.02` — batch_size=1 matches original eval protocol. Tiny +0.02% improvement, not significant. Results are nearly identical - MAS padding handling works correctly.
- **Iter 4** — *Training-format prompt (2 spaces + indented blank lines)* — `success` — primary `67.89` — Matching training prompt format hurt boolq (-1.71%) while helping arc_c (+0.34%) and obqa (+0.80%). Net -0.13%. Reverted.
- **Iter 5** — *Dataset-specific prompt format: training format for ARC+obqa* — `success` — primary `68.15` — NEW BEST 68.15%. Apply training format (2 spaces + indented blank lines) for ARC-Challenge, ARC-Easy, openbookqa. Keep original format for boolq/piqa/siqa/hellaswag/winogrande. arc_c +0.59%, obqa +0.60%.

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter-Optimization Insights**
Typical inference settings from recent studies suggest: 
- **Beam Size**: State-of-the-art often use greedy or small beam (beam=1–5). Wu et al. (2023) and others found that small beam search yields modest gains. For example, Wang et al. note sampling (beam=1, multiple draws) with self-consistency outperforms single-beam greedy (openreview.net). Try expanding `num_beams` to 3–5 (with appropriate num_return_sequences) to allow diversity. 
- **Temperature**: Many analyses use low temperature (0–0.5) for accuracy. Wei et al. (2022) found CoT performance peaked around T≈0.0–0.5; very high T (≈1.0) often degrades correctness. So sweep T ∈ [0.0,0.2,0.5]: lower T makes answers more deterministic (lower variance), higher T yields more exploration (useful for self-consistency). 
- **Top-p / Top-k**: Common choices are top_p ≈0.8–0.95 and top_k ≈20–50. For example, the MAS baseline uses top_p=0.75, top_k=40. In SOTA CoT work, top_p around 0.9 was often used to allow flexibility while avoiding extremely low-prob events. Try ranges top_p∈[0.7,0.9,1.0], top_k∈[10,40,100] to balance diversity vs focus. 
- **Max Length**: The default max_new_tokens=32 may cut off explanations. Tasks like HellaSwag or ARC sometimes benefit from a longer rationale. Try increasing to 64 or 96 tokens when using CoT or when answers are sentences. (Longer output has minimal cost if beam=1, but ensure stopping rules prevent run-on.) 
- **Batch Size**: Larger batches speed throughput but do not affect accuracy. Keep batch_size as large as GPU memory allows (e.g. 8–16) for speed, but be wary that batch decoding might change answer ordering or race conditions in some decoding implementations (usually low risk). 

*(No direct citations for these hyperparameter ranges; they are drawn from typical values reported in LLM benchmarks.)* 

**4. Concrete Optimization Ideas**
1. **Tune Segmentation N-gram (###) Threshold**: Experiment with the MAS template function (`calculate_attention_mask_for_commonsenseDS_template`) to detect segment boundaries differently. Instead of only `' ###'`, try other delimiters or remove the n-gram (e.g. always split at question/answer boundary). Gains: If segmentation currently misses logical breaks, proper partitioning could allow better contextual integration (estimated +1–3%). Risk: High – improper segmentation can actually scramble context (if, e.g., the separator occurs inside text, or if no separator exists and MAS does nothing). Reward only if you validate it improves individual examples. 
2. **Beam Search + Self-Consistency Voting**: Increase `num_beams` (e.g. from 4 to 8) and generate multiple answer hypotheses. Then use self-consistency by majority-voting the final answers or aggregating softmax scores. Expected gain: **~2–5%** on reasoning tasks (Wang et al. achieved +3.9% on ARC-Challenge with a similar idea (openreview.net)). Risk: Medium – more beams increases runtime and may produce very similar/high-prob answers, so diminishing returns. Also risk of beam degeneracy (repeated text) if beams are too many. 
3. **Adjust Temperature and Sampling**: Sweep `temperature` higher (e.g. 0.2, 0.5, 0.8) to induce variability, or lower it to 0.0 for determinism. For multiple-choice, low-T (0.0) often gives stable top answers. If using self-consistency, a higher T (0.5–0.8) can explore different reasoning paths. Expected gain: **~1–3%**, since even minor randomness can reveal alternatives. Risk: Low – wrong T may slightly reduce accuracy (e.g. T=1.0 often yields nonsense), but testing on dev can filter it. 
4. **Top-p/Top-k Tuning**: Try `top_p=0.85–0.95` and vary `top_k`. Larger top_k (100) with top_p high allows more answer diversity; smaller top_p (0.5–0.7) restricts to high-prob tokens. Gains: **~1–3%** by avoiding overly random or overly rigid decoding. Risk: Low – only output style changes, but too-low top_p risks missing correct low-prob words. 
5. **Answer Option Scoring (Discriminative Decoding)**: For each question, feed each multiple-choice option as the (masked) answer and compute its log-likelihood given the prompt and MAS context. Select the highest-likelihood option. This often outperforms free text generation for MCQs. Expected gain: **~3–7%** (common in benchmarks; it bypasses answer-extraction errors). Risk: Medium – if model embeddings for choices are not well-aligned with MAS context, or if choices are long sentences, scoring can be expensive and could bias toward longer answers (length normalization may be needed). 
6. **Prompt Engineering / Few-Shot CoT**: Insert a few hand-picked QA examples with step-by-step reasoning in the prompt before evaluation (few-shot), or prepend instructions like “Let’s think step by step.” This can guide the model to produce explicit rationales. Gains: **~5–10%** on tasks requiring multi-hop reasoning (per Wang et al. in related settings (openreview.net), Wei et al. 2022 also demonstrate large leaps with CoT). Risk: High – constructing good examples is time-consuming, and irrelevant or poor exemplars can confuse the model. Also increases prompt length (but MAS can still segment by question). 
7. **Increase Output Length for Rationale**: Allow more tokens (e.g. max_new_tokens=64) so the model can generate a brief explanation before the final answer token. Then use regex to extract the final answer letter/or word. Gains: **~1–3%**, as sometimes the model knows the answer but only emits it after reasoning. Risk: Low – just decoding more; ensure the regex still correctly finds the final answer. (Also check that increased length doesn’t cause out-of-memory.) 
8. **Refine Answer-Extraction Regex**: Improve the `extract_answer` logic by allowing synonyms, ignoring case, and catching answer letters in context. For instance, match “The answer is (A)” or just “A.” at end of response. Also post-normalize (“A.”→“A”). Gains: **~1–2%**, by recovering answers lost to parsing errors. Risk: Low – incorrect regex could mis-extract wrong text, so test carefully on dev. 
9. **Mixed Precision (bf16)**: Switch model inputs/weights to bfloat16 instead of float16. On modern GPUs, bf16 often preserves accuracy (due to larger dynamic range) without performance loss. Expected gain: **~0–1%** (usually negligible difference), but risk is minimal. It may slightly change the model’s internal numerics and sometimes yield small improvements or more stable outputs on certain hardware. 
10. **Ensemble Multiple Decoding Strategies**: Combine outputs from diverse settings. For instance, run once with greedy decoding and once with sampling, or use both the LoRA-MAS model and the original fine-tuned model (without MAS), then vote. Gains: **~1–3%** aggregate improvement by covering different failure modes. Risk: Medium – requires combining outputs robustly; if one strategy dominates, ensemble may skew or add confusion. Inference cost is higher too. Each new strategy should ideally give a complementary perspective. 

Each of these ideas should be validated on a held-out subset. For example, self-consistency (Idea 2) has been shown to give ~3.9% on ARC (openreview.net); option scoring (Idea 5) is a standard trick in multi-choice QA and often yields single-digit improvements; prompt engineering (Idea 6) can occasionally give very large gains (in arithmetic reasoning benchmarks, CoT gave +10–20%). The actual gains will vary by task, so estimate conservatively (~1–5% per idea). 

**5. Common Failure Modes and Pitfalls**

_(Research digest truncated.)_

## Idea library snapshot (`idea_library.md`)

### IDEA-001: Improve case-insensitive answer extraction
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: The current `extract_answer` regex is case-sensitive. Added lowercase conversion and case-insensitive matching.
- **Hypothesis**: Expected +0.5-2% across datasets.
- **Status**: SUCCESS — BUT no improvement observed. Model already outputs lowercase consistently.
- **Result**: average 68.0% → 68.0% (+0%). Model always generates lowercase answer labels.

### IDEA-002: Improve answer extraction to search full generated text
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Changed extract_answer to return last match instead of first match.
- **Hypothesis**: Expected +0.5-2% improvement.
- **Status**: SUCCESS — BUT no improvement. With max_new_tokens=32, outputs are too short for multiple matches.
- **Result**: average 68.0% → 68.0% (+0%). Short outputs mean first=last match always.

### IDEA-003: Tune temperature to 0.0 (greedy) for deterministic outputs
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change temperature from 0.1 to 0.0. At temperature=0.0, the model uses greedy decoding in combination with num_beams=4, which is fully deterministic. Literature suggests lower temperature is better for accuracy on MCQ tasks.
- **Hypothesis**: Expected ±1% change. Lower variance, might improve or slightly decrease accuracy.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-004: Increase num_beams from 4 to 8
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Increase beam search size from 4 to 8. More beams explore more of the output distribution. Research shows larger beams often (but not always) improve accuracy, especially for MCQ tasks.
- **Hypothesis**: Expected +0.5-2% improvement, possibly better coverage of correct answer patterns.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-005: Increase max_new_tokens from 32 to 64
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Some datasets might need more tokens for the model to generate the correct answer label. Currently max_new_tokens=32. Increasing to 64 gives the model more room to reason.
- **Hypothesis**: Expected small improvement (+0.5-1%) for datasets with longer expected outputs.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-006: Add letter-based answer extraction fallback
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: For ARC, OpenBookQA datasets where answer is labeled A/B/C/D in original data, the model might output "A" or "(A)" instead of "answer1". Add a secondary extraction that maps "A"→"answer1", "B"→"answer2", etc. for multi-choice datasets.
- **Hypothesis**: Expected +1-3% improvement on ARC-Challenge and ARC-Easy. These datasets originally use A/B/C/D labels.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-007: Use batch_size=1 with better padding for evaluation accuracy
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Padding in batched evaluation can sometimes cause MAS to compute incorrect attention masks, because the padding tokens may interfere with the `#` separator detection. Running with batch_size=1 eliminates padding.
- **Hypothesis**: Expected +0.5-2% improvement, particularly for datasets with variable-length inputs.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-008: Adjust top_p from 0.75 to 0.9
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Higher top_p=0.9 includes more probability mass. When combined with low temperature and beam search, this might allow the model to find better answers in some cases.
- **Hypothesis**: Expected ±0.5% effect.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-009: Adjust top_k from 40 to 20
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Lower top_k=20 focuses on more probable tokens. Combined with beam search, this may improve answer generation quality.
- **Hypothesis**: Expected ±0.5% effect.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-010: Improve BoolQ instruction format to explicitly ask for yes/no
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Current format just appends "Based on the passage above, is the following statement true or false?" which expects "true" or "false". Adding explicit instructions like "Answer only with 'true' or 'false'" may reduce non-answer responses.
- **Hypothesis**: Expected +0.5-2% on boolq specifically.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-011: Try non-MAS adapter (Llama-3.2-1B_epoch3) for comparison
- **Type**: CODE
- **Priority**: LOW
- **Risk**: MEDIUM
- **Description**: Temporarily switch to the non-MAS fine-tuned model (Llama-3.2-1B_epoch3) to understand the MAS contribution. This is for analysis, not optimization - but if non-MAS performs better on some datasets, we could use it.
- **Hypothesis**: Understanding MAS contribution helps prioritize ideas.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-012: Answer with explicit labels format in instruction
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Modify the instruction format to explicitly tell the model to respond with "answer1", "answer2", etc. For example, add "Please respond with exactly one of: answer1, answer2, answer3" at the end of the instruction.
- **Hypothesis**: Expected +1-3% by reducing answer extraction failures.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-013: BFloat16 precision instead of float16
- **Type**: CODE
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Try bf16 instead of fp16 for model precision. BF16 has better dynamic range and may produce slightly different (potentially better) numerical results.
- **Hypothesis**: Expected ±0.5% effect on accuracy.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-014: Add explicit answer format instructions to prompt
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Append to the prompt: "Answer with the exact label (e.g., 'answer1', 'true/false', 'option1')." This tells the model exactly what format to output.
- **Hypothesis**: Expected +1-3% improvement in answer extraction accuracy.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-015: LEAP - Self-consistency voting across multiple beam outputs
- **Type**: ALGO
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Instead of using just the top-1 beam output, generate multiple diverse outputs (via sampling with different temperatures/seeds) and take majority vote.
- **Hypothesis**: Expected +1-3% improvement following Wang et al. (2023) self-consistency results.
- **Status**: PENDING
- **Result**: (fill in after execution)
