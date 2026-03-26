# Paper 16 — MathFusion

**Full title:** *MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion*

**Registered metric movement (internal ledger, ASCII only):** +20.97%(0.2420->0.2928)

# Final Report: MathFusion Paper-229 Optimization

## Summary

- **Baseline accuracy**: 0.2420
- **Final accuracy**: 0.2928
- **Absolute improvement**: +0.0508 (+21.0% relative)
- **Target**: 0.2468 (EXCEEDED by +0.0460)

## Best Configuration

- **Prompt template**: `cot-qa-boxed2` — adds inline `\boxed{}` instruction in query suffix
- **Few-shot examples**: `n_shots=4` (4 ICL examples for college-math)
- **Max new tokens**: 4096 (vs 2048 baseline)
- **ICL example format**: Examples end with "Therefore, the final answer is \boxed{...}."
- **Temperature**: 0 (greedy decoding)

## Key Changes from Baseline

1. **Prompt template change** (`/repo/evaluation/dart_math/utils.py`): Added `cot-qa-boxed2` template that instructs the model to put the final answer in `\boxed{}` via a suffix in the query: `\nPlease reason step by step, and put your final answer within \\boxed{}.\n`

2. **Few-shot prompting**: Changed `n_shots=0` to `n_shots=4` to provide 4 in-context learning examples.

3. **Increased max tokens**: Changed `max_new_toks=2048` to `max_new_toks=4096` to allow deeper reasoning chains.

4. **Modified ICL examples** (`/repo/evaluation/dart_math/data.py`): Updated the 4 ICL examples for `mwpbench/college-math/test` to end with "Therefore, the final answer is \boxed{...}." format, consistent with the boxed instruction in the prompt.

## Iteration History

| Iter | Configuration | Accuracy | Delta vs Baseline |
|------|--------------|----------|-------------------|
| 0 | Baseline (cot-qa, n_shots=0, max_new_toks=2048) | 0.2420 | - |
| 1 | n_shots=4 | 0.2594 | +0.0174 |
| 2 | cot-qa-boxed + n_shots=4 | 0.2708 | +0.0288 |
| 3 | Boxed ICL examples + cot-qa-boxed + n_shots=4 | 0.2828 | +0.0408 |
| 4 | cot-qa-boxed2 + n_shots=4 | 0.2864 | +0.0444 |
| 5 | cot-qa-boxed2 + n_shots=4 + max_new_toks=4096 | 0.2889 | +0.0469 |
| 6 | n_shots=3 (ablation) | 0.2708 | +0.0288 |
| 7 | cot-qa-boxed3 (alternative phrasing) | 0.2871 | +0.0451 |
| 8 | cot-qa-boxed4 (CoT prefix + boxed) | 0.2697 | +0.0277 |
| 9 | cot-qa-boxed2 + n_shots=4 + max_new_toks=4096 + adjusted ICL | **0.2928** | **+0.0508** |
| final | (same as iter 9) | **0.2928** | **+0.0508** |

## Analysis

### What Worked
- **Few-shot prompting** (iter 1): +1.74pp gain from providing 4 college-math examples
- **Boxed instruction in prompt** (iter 2): +1.14pp additional from instructing model to use `\boxed{}`
- **Consistent ICL format** (iter 3): +1.20pp additional from updating ICL examples to also use `\boxed{}`
- **Better prompt phrasing** (iter 4): cot-qa-boxed2 with inline instruction was slightly better than cot-qa-boxed
- **Longer context window** (iter 5): +0.25pp from allowing 4096 tokens instead of 2048
- **Refined ICL phrasing** (iter 9): +0.39pp from using "Therefore, the final answer is \boxed{...}." in ICL examples

### What Did Not Work
- **n_shots=3** (iter 6): Fewer examples hurt performance; 4 is the right number
- **CoT prefix + boxed** (iter 8): Adding "Let's think step by step." as prompt_before_resp alongside boxed instruction caused format inconsistency

### Key Insight
The model (Mistral-7B-MathFusion) responds well to explicit formatting instructions. The combination of (a) telling the model to use `\boxed{}` in the query prompt and (b) providing ICL examples that demonstrate this exact format creates a strong signal for the model to both express its reasoning clearly and format the answer for reliable extraction.

Analysis of model outputs showed 98.8% of responses used `\boxed{}` format in the final configuration, so the primary remaining gap to perfect accuracy is mathematical correctness rather than formatting/extraction issues.

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter Optimization Insights**
- **Temperature (T):** Studies show CoT reasoning often uses moderate T (0.6–0.8) for diverse samples (arxiv.gg), while greedy (T=0) yields deterministic chains. For Mistral, try a range (0, 0.3, 0.7) and compare consistency. 
- **Number of Samples (n_trials):** Self-consistency works with ~10–40 samples in literature (arxiv.gg). With limited GPUs, even 5–10 runs can help. 
- **Few-Shot Count (n_shots):** Many math prompting papers use 4–8 examples. Too many shots can exceed context. If problems vary widely, keep n_shots ≤5. 
- **Max Tokens (`max_new_toks`):** Set high enough to accommodate long CoT (e.g. >500 tokens per chain). 2048 default is safe for multi-step reasoning. For very complex problems, one might increase it if outputs get truncated. 
- **Stop Tokens:** If using GPT-like stop sequences, ensure the chosen stop (e.g. `\n\nAnswer:` or similar) doesn’t appear in reasoning. Some templates use “\|” or “```” as guards. 
- **Answer Parsing:** The sympy equivalence check requires valid LaTeX or parseable math. Commonly, models output a final answer like “\boxed{42}” or “x = 42”. Ensure the answer is in a format that `latex2sympy` can parse (e.g. use “=”, avoid text like “forty-two”). 
- **Top-k / Top-p:** If using probabilistic sampling, tune top-k (e.g. 40–50) or top-p (0.9–0.95) to control diversity. Self-consistency often uses high diversity (top-p≈0.9). 
- **Prompts (Templates):** Past studies use “Alpaca” or “GPT-3 chat” templating. Some report differences: an instruction-based template (like Alpaca’s Q/A format) vs a plain Q followed by “Let’s think”. Test multiple known math CoT templates. 
- **Error Thresholds:** For ensemble voting, decide how to handle ties or disagreeing answers. One can fall back to a second-round verification if no majority. 
- **Sympy Equivalence Tolerance:** If numeric, allow numeric tolerance for floats. Ensure the evaluation script is not too strict (e.g. 1.000 vs 1 might flag incorrectly). 
- **Output Length:** Some problems need only short answers; forcing too many tokens can confuse. Conversely, too-short chains lack justification. Observe distribution of solution lengths on MWPBench and adjust. 

**4. Concrete Optimization Ideas**

1. **Self-Consistency Voting:** Run the model *N* times per problem with *T≈0.6–0.8* and *N=5–10*. Take the most common numeric answer (using sympy to normalize). 
 - *Expected gain:* ~5–10% absolute (Wang *et al.* report big CoT gains (arxiv.gg)). 
 - *Risk:* **Low** – simply more inference; risk of increased compute only. If model frequently ties or is confused, gain may be smaller. 
2. **Few-Shot CoT Prompting:** Add a few solved example math problems with chain-of-thought to the prompt (e.g. 3–5 shots). Use problems of similar types if possible (e.g. same category or complexity). 
 - *Expected gain:* ~5–15%. Related works often see multi-point boosts with few-shot CoT. 
 - *Risk:* **Medium** – Requires careful example selection. Bad examples or mismatched format might mislead the model. Watch context length. 
3. **Prompt Engineering – “Think Step by Step”:** Use the “chain-of-thought” style prompt (e.g. suffix questions with “Let’s think step by step.”) or an explicit instruction like “Explain your reasoning before answering.” (arxiv.gg). 
 - *Expected gain:* ~3–7%. Often CoT triggers more accurate reasoning. 
 - *Risk:* **Low** – Little downside. If too wordy, might exceed token limit on lengthy problems; should manage lengths. 
4. **Multiple Prompt Templates Ensemble:** For each problem, generate answers using 2–3 different prompt formats (e.g. ALPACA style, GPT-3 style, or different CoT phrasing) and vote on final answer. 
 - *Expected gain:* ~2–5%. Diversity in prompting often yields complementary answers. 
 - *Risk:* **Low–Medium** – More inference runs. Prompts may conflict if too different, but voting mitigates. Complexity adds overhead. 
5. **Answer Verification Query:** After obtaining an answer, append a follow-up (in the same session) asking, “Is this step-by-step solution correct? The final answer was X. If not, correct it.” This invites the model to self-criticize. Possibly run a second query to revise. 
 - *Expected gain:* ~3–8%. Inspired by PRP and LEMMA, catching mistakes can improve final accuracy. 
 - *Risk:* **Medium–High** – Extra round of inference doubles cost. The model may agree with itself and not change. Must carefully parse any new solution. 
6. **Chain Subdivision (Multi-step Prompting):** Break each problem into parts. For example, first prompt “What equations can you set up?” then use those in a second prompt to solve. You can use the LLM itself to generate and solve sub-problems sequentially (like a simple debate). 
 - *Expected gain:* ~5–10% for complex multi-step problems (if sub-questions align to hidden steps). 
 - *Risk:* **High** – Crafting sub-questions may be tricky. Over-length prompts. Inference chaining requires careful engineering. Not always easy to automate reliably. 
7. **Mathematical Checks in Prompt:** Encourage the model to check its arithmetic. For example, add to the prompt: “After computing, verify each arithmetic step yourself.” 或者 “Double-check the algebra.” 
 - *Expected gain:* ~2–5%. Could reduce simple calculation mistakes. 
 - *Risk:* **Low** – Minimal change to prompt. The model may still ignore or pretend the check. 
8. **Optimize Temperature vs Strategy:** Experiment with T=0 (greedy) vs T=0.7. For single-shot CoT, T=0 (deterministic) might yield more coherent proofs, whereas for sampling-vote strategies use T>0 to diversify. 
 - *Expected gain:* *Tuning benefit.* Up to ~3% improvement by finding sweet spot. 
 - *Risk:* **Low** – Only inference settings adjust; no real danger. 
9. **Interactive Few-Shot Retrieval:** Use a small MWP retrieval: for each test problem, find a near-similar problem (from training or synthetic set) and its solution, include it as an example. Methods like computing TF-IDF or embedding similarity. 
 - *Expected gain:* ~5–10% (Learning by Analogy saw ~6.7% on average (openreview.net)). 
 - *Risk:* **Medium** – Requires building a retrieval index and selecting “good” analogies. Might introduce mislabeled focus if retrieval fails. 
10. **Post-hoc Answer Filtering:** If multiple answers/solutions are generated, use Sympy to quickly verify feasibility. For instance, if model outputs two candidate solutions (e.g. via different prompts), pick the one that actually satisfies the problem’s equations (using a quick solver). 
 - *Expected gain:* ~5%. Ensures the chosen answer is mathematically valid. 
 - *Risk:* **Medium** – Implementation complexity (automatically extracting equations). Sympy might fail to parse messy answers. 
11. **Enforce Answer Format:** Modify prompt to produce easily parsed output (e.g. “Final answer: [latex math]”). Reduces loss from mis-parsed answers. 
 - *Expected gain:* *Indirect* – fewer evaluation errors. Might be a couple percent of false negatives fixed. 
 - *Risk:* **Low** – Mostly beneficial. If model ignores the rule, no harm done. 
12. **Use Sympy for Reranking:** Generate several chain-of-thoughts via sampling; for each, extract the final computed result. Then programmatically check consistency: if multiple chains yield the same final number via valid equations, trust those. Discard or rerun ones that yield inconsistencies. 
 - *Expected gain:* ~5–8%. Re-runs invalid/junk answers. 
 - *Risk:* **High** – Complex implementation, plus if the model’s reasoning is wrong syntactically, extraction fails. 

Each idea ranges from *low-risk* prompt tweaks (few-shot, instructive cues) to *moderate-risk* ensemble or self-verification loops (more compute and complexity). Gains are estimates; actual improvement depends on problem distribution and careful implementation.

**5. Common Failure Modes & Pitfalls**

_(Research digest truncated.)_
