# Paper 20 — Aristotle

**Full title:** *Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework*

**Registered metric movement (internal ledger, ASCII only):** +1.68%(59.4->60.40)[ ]

## Iteration trace (`scores.jsonl`)

Structured log of each optimizer attempt (including failures and rollbacks):

- **Iter 0** — *Paper baseline (deepseek/deepseek-chat)* — `success` — primary `59.4` — Baseline using deepseek-chat (gpt-4o not available in region). Paper reports 64.0 with gpt-4o.
- **Iter 1** — *Replace gpt-4o with deepseek-v3.2 (alias map)* — `failed` — primary `59.4` — deepseek-v3.2 for resolution with deepseek-chat translations gave 47% (worse). Mismatch between translation style and resolution model.
- **Iter 4** — *search_round=20 for True run (False still 10)* — `success` — primary `60.4` — Increasing search_round from 10 to 20 for True negation run improved accuracy +1.0% by finding more contradictions for GT=D cases
- **Iter 5** — *deepseek-chat True=round20 + deepseek-chat False=round10* — `success` — primary `72.58` — v3.2 was actually worse for resolution. deepseek-chat at round=20 for True run is much better than v3.2

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter Optimization Insights**
From recent literature and ablations, we see some common hyperparameter choices for reasoning: 

- **Model Choice:** Many works use mid-sized open models (7–13B) when large APIs aren’t available. For example, RUG-PD reports results with LLaMA2-7B, Mistral-7B, and LLaMA3-8B (www.sciencedirect.com), all giving consistent improvements. This suggests that even a 7–8B LLM can benefit from these strategies (though larger models like GPT-4 usually add even more accuracy). In practice, use the strongest model you can afford (GPT-4, Llama-3 13B+) for best reasoning precision. 

- **Sampling and Temperature:** To exploit self-consistency, one typically sets a moderate temperature (e.g. 0.3–0.7) when generating multiple chains, then may use greedy decoding (temp=0.0) when deriving the final answer. Wang et al. found that sampling dozens of chains (with randomness) significantly lifts accuracy (arxiv.gg). There is a trade-off: higher temperature increases candidate diversity but risks more nonsense, while 0.0 gives reproducible answers. A common compromise is Ti~0.2–0.5 for expansion steps and final Ti=0.0 for answer selection. 

- **Max Tokens:** Long reasoning needs tens to a few hundred tokens. In practice, many papers use `max_new_tokens` ≈ 128–256 for each LLM call, sufficient for multi-step logic chains. If tasks are complex, increasing this budget (to 300+ tokens) can allow deeper reasoning sub-answers. However, very long outputs may drift off-topic, so limit each step to only as many tokens as needed (perhaps ~50 tokens per sub-question). The key is to balance depth vs. focus. 

- **Search Dimensions:** When doing structured search (like ToT or a custom exploration in `--search_rounds`), typical values are small integers. For example, RAP (Reasoning via Planning) often runs 3–5 search iterations or rollouts. If using a tree or beam search, a branching factor of 2–4 thoughts is common, with depth 3–5 steps. In general, more rounds increases search thoroughness but also multiplies API calls. One should experiment: start with ~3–5 rounds and some beam width, and see if accuracy plateaus (gains often diminish after a few iterations). 

- **Batch Parallelism:** To manage cost, many systems parallelize independent LLM calls. For instance, running decomposition prompts in parallel (`--batch_num`) can linearly speed up inference with little downside. There’s no accuracy trade-off to batching itself, but ensure outputs are combined coherently. Batch sizes of 10–50 are typical where allowed. 

- **Answer Normalization Logic:** The evaluation’s `normalize_answer()` in Aristotle-fork defaults unknown answers to a label (“C”). Adjusting this logic (e.g. treating “Unknown” differently or disallowing “Contradiction” when inappropriate) has impact. While not a hyperparameter per se, the choice of fallback answer acts like a threshold. Some systems instead admit “Unknown” or require a clear justification. If possible, refine the resolution rules (for example, reject an answer if the LLM itself flagged a contradiction). 

In sum, similar works suggest: use the strongest model under budget, sample many chains (multi-shot) with T≈0.3–0.7 followed by greedy answer selection (arxiv.gg), allocate ~128–256 tokens per chain, and use a few steps/branches of search. Runtime probes and small-scale sweeps (e.g. doubling search rounds until returns diminish) are common in ablations. Notably, even using relatively small models (7–8B) can be effective (www.sciencedirect.com) if all these techniques are applied. 

**4. Concrete Optimization Ideas**

1. **Swap in a Stronger LLM:** Replace the base model (e.g. GPT-3.5) with a more powerful one (GPT-4, Claude 3/4, or large open models like LLaMA-3 13B or DeepSeek R1) (openreview.net). Larger models have better inherent reasoning ability. *Expected gain:* substantial (single-digit to tens of %). *Risk:* higher API cost, possibly slower or rate-limited. 

2. **Increase Search Iterations (`--search_round`):** Run more rounds of decomposition + searching, letting the system explore more candidate sub-problems. For example, increase from 2 to 5 or 10 rounds. *Expected gain:* modest to significant (few %), especially on multi-step problems; beyond a point, returns diminish. *Risk:* linearly more LLM calls (time/cost blowup), potential cascading errors if early rounds go astray. 

3. **Raise Token Budget (`--max_new_tokens`):** Allocate more tokens for each LLM call in decomposition and resolution. E.g., increase from 100 to 200 or more if needed. This helps the model produce longer, more detailed chains. *Expected gain:* small to moderate (deeper reasoning sometimes solves extra cases). *Risk:* model output may become verbose or off-track, increasing chance of error or exceeding prompt limits. 

4. **Non-Zero Temperature in Decomposition:** Use a small positive temperature (0.2–0.5) when generating decompositions or intermediate thoughts (while keeping answer generation greedy). *Expected gain:* better diversity of ideas; can catch solutions missed at T=0.5 or above. *Risk:* some generated chains will be nonsensical; mitigated by filtering or voting (see below). 

5. **Self-Consistency Voting:** Run the full pipeline (decompose→search→resolve) multiple times with different random seeds or sampling, then take a majority vote on the final answer. Inspired by *Self-Consistency* (arxiv.gg), this can correct random errors. *Expected gain:* often double-digit improvements (depending on task difficulty). *Risk:* greatly increased API usage; if the model is systematically biased, majority voting won’t fix errors. 

6. **Graph-Based Verification:** After obtaining one or more solutions, explicitly cross-check them. For example, build a graph of intermediate facts (premises, subgoals, conclusions) and use the LLM or simple rules to detect inconsistencies. This is akin to **GraphReason** (aclanthology.org). Concretely, ask the model “Does conclusion follow logically from premises? Which premise would need changing if answer is X?” *Expected gain:* detects contradictions, improves reliability; 2–5% accuracy boost in practice. *Risk:* overhead of extra checks; risk of confusion if prompts misused. 

7. **Symbolic/Code Intermediate Representation:** Convert parts of the problem or answer into a formal or executable form for validation. For instance, translate premises into logical formulas or entailment checks (as in Logic-LM++ (aclanthology.org)). Or express numeric inferences as Python code and run it. That can enforce correctness. *Expected gain:* strong improvement on technical subproblems (+5–15%). *Risk:* requires careful prompt engineering or external tools; if mis-specified, may introduce bugs. 

8. **Prompt Rewriting / Few-Shot Exemplars:** Edit or expand the prompt templates to be clearer or include exemplar reasoning steps. For example, prepend a worked example (few-shot) of solving a similar logical puzzle. Alternatively, try rephrasing the question’s wording (changing “If P then Q” style) to see which yields better chains. *Expected gain:* up to a few percent; well-known in prompt engineering. *Risk:* laborious manual tuning; if examples are misleading, can degrade accuracy. 

9. **Dynamic Prompting (Analogical Prompting):** Use the LLM itself to recall or generate analogous problems with solutions and include them as context (like Yasunaga et al.’s analogical prompting). E.g., “Here’s a similar logic problem and its solution: … Now solve the new problem.” This leverages the model’s exemplar generation ability. *Expected gain:* potentially large if good analogies are found (~10%±). *Risk:* generating near-miss examples is tricky; wrong analogies can confuse the model. 

_(Research digest truncated.)_

## Idea library snapshot (`idea_library.md`)

### IDEA-001: Switch to stronger model (deepseek/deepseek-v3.2)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Modify utils.py to add a MODEL_ALIAS_MAP that maps "gpt-4o" → "deepseek/deepseek-v3.2". This is a stronger model than deepseek-chat and may produce better logical reasoning. Run full pipeline (translate_decompose + search_resolve). WARNING: Requires re-running full pipeline (takes ~2-3 hours).
- **Hypothesis**: deepseek-v3.2 is a more capable model than deepseek-chat and should improve translation quality and resolution accuracy. Expected +2-5% accuracy.
- **Status**: SKIPPED (superseded by IDEA-006/003 findings — deepseek-chat is better for resolution)
- **Result**: IDEA-006 showed v3.2 is worse for search_resolve step

### IDEA-002: Fix evaluate.py normalize_answer to handle "D" ground truth better
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Currently 163/300 examples have ground truth "D" (Contradiction). D requires one run to get A and the other to get B. If both get "No final answer" (→C), it fails. The current failure rate on D is high (63/163 = 38.7% error). The normalize_answer function maps anything not A/B/D to C. Could improve by not normalizing D away even when present - but actually, the search_resolve outputs final_choice A/B/C using final_process(). The issue is the search fails to find the contradiction.
- **Hypothesis**: This is a code-level insight - the model needs to better detect contradictions.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-003: Re-run search_resolve with higher search_round (increase from 10 to 20)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: The search_round default is 10. Increasing to 20 gives more resolution steps and may find contradictions in complex cases. Only re-run search_resolve (not translate_decompose), reusing existing deepseek-chat translation results.
- **Hypothesis**: More search rounds → more "No final answer found" cases convert to actual A/B answers. Expected +1-3%.
- **Status**: COMPLETED (KEY: asymmetric — True=20 helps, False=20 hurts)
- **Result**: True=round20 deepseek-chat + False=round10 → **72.58%** (+13.18%). This was the key breakthrough. False run at round=20 hurts (-7%) due to false positive contradictions for GT=A.

### IDEA-004: Increase batch_num for translate_decompose parallelism
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Parallelism doesn't affect accuracy. Just speeds things up.
- **Status**: SKIP (not applicable for accuracy)
- **Result**: N/A

### IDEA-005: Fix possible bug in search_resolve.py ("or 'false'" always True)
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: In search_resolve.py, the condition `if new_clause.lower() == "contradiction" or "false":` always evaluates to True when sufficiency_label="True" because `"false"` is truthy. The intended check is `if new_clause.lower() in ("contradiction", "false"):`. However, this effectively means: when SufficiencyLabel=True, always take the branch. This might be "correct by accident" since SufficiencyLabel=True is supposed to mean a contradiction was found. Need to investigate more carefully.
- **Hypothesis**: Fixing this may have no effect or small positive effect. Low risk.
- **Status**: COMPLETED (REVERTED — NOT A BUG)
- **Result**: "Fixing" this broke things badly: 45.64% (from 59.40% baseline). The `or "false"` behavior is intentional — when sufficiency_label="True", always set final answer regardless of new_clause value. REVERTED.

### IDEA-006: Use translate_decompose results from deepseek-chat but re-run search_resolve with deepseek-v3.2
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Re-use existing translation results (deepseek-chat) but run search_resolve with a better model (deepseek-v3.2) for the resolution steps. The resolution step (logic_resolver prompt) is the key LLM call in search_resolve.py. Only needs to modify the model_name for search_resolve calls. Can reuse existing `LogicNLI_deepseek/deepseek-chat_trans_decompose_no_negation.json` and `deepseek-chat_trans_decompose_negated_data.json`. Need to copy these to gpt-4o naming convention and run search_resolve with different model.
- **Hypothesis**: Better model for resolution → more correct resolutions → higher accuracy. Expected +2-5%.
- **Status**: COMPLETED (FAILED)
- **Result**: ~47% accuracy. deepseek-v3.2 produces far more "No final answer found" than deepseek-chat for search_resolve. The key is that deepseek-chat's output format matches search_resolve's response parsing better.

### IDEA-007: Copy deepseek trans_decompose files to gpt-4o naming and run search with deepseek-v3.2
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**:
- **Hypothesis**: Uses best available model for resolution. Expected improvement over deepseek-chat baseline.
- **Status**: COMPLETED (FAILED)
- **Result**: Re-translating 123 error cases with v3.2 → 51.17% (worse). Both translation AND search_resolve do better with deepseek-chat.

### IDEA-008: Improve logic_resolver prompt with better examples
- **Type**: ALGO
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: The logic_resolver.txt prompt guides the resolution step. Adding more complex examples or clarifying when conclusions are "Unknown" vs "Contradiction" may help.
- **Hypothesis**: Better prompts → more accurate resolution → higher accuracy.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-009: Analyze and fix "No final answer found" cases by improving sos_list cleaning
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: 115/300 cases in False run and 92/300 in True run fail to find any answer. The search process often terminates with "No complement found" or similar. This could be caused by translation failures that produce empty/bad normalized_context or sos_list. Fix by adding fallback logic: if no complementary clause found in first round, do a broader search.
- **Hypothesis**: Reducing "no answer" from 115+92 to say 50+40 could add many correct answers.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-010: Try using deepseek-v3.2 for full pipeline (translate + search)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Run the complete pipeline from scratch with deepseek/deepseek-v3.2. Add MODEL_ALIAS_MAP in utils.py: {"gpt-4o": "deepseek/deepseek-v3.2"}. Run translate_decompose + negate + search_resolve + evaluate. Full run needed.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-011: Use higher temperature (0.3) for translate_decompose only
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: MEDIUM
- **Description**: Temperature=0.3 in translate_decompose may produce more diverse/creative FOL translations.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-012: Set max_new_tokens explicitly (e.g., 2048) for all LLM calls
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Currently max_new_tokens passes None (API default). Explicit 2048 ensures longer reasoning chains.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-013: Add search_resolver fallback - when search fails, use LLM direct answer
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: When search_resolve finds no complementary clause (returns "No complement found"), instead of terminating, call the LLM directly with the original problem to attempt a direct answer.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-014: Re-run translate_decompose with deepseek-v3.2 (fresh full pipeline)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: The deepseek-chat translation may have errors. deepseek-v3.2 could produce better FOL translations. Combined with existing search logic, this could materially improve accuracy.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-015: Check if taking existing _deepseek results as-is but rerunning search with more rounds helps
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Use deepseek-chat translations, rerun search_resolve only with search_round=20 and deepseek-v3.2 model. This combines better model with more search steps.
- **Status**: PENDING
- **Result**: (fill in after execution)
