# Paper 13 — APPL

**Full title:** *APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts*

**Registered metric movement (internal ledger, ASCII only):** -14.29%(35->30)[AST ]

# Optimization Results: APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts

## Summary
- Total iterations: 1
- Best `ast_size`: 30 (baseline: 35, improvement: -14.3%)
- Target: ≤ 34.3 — **ACHIEVED in iteration 1**
- Best commit: b3bae979c2

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| ast_size | 35 | 30 | -5 (-14.3%) |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| Inlined `marginalize(results)` into return: `results = [gen() for _ in range(num_trials)]` + `return marginalize(results)` → `return marginalize([gen() for _ in range(num_trials)])` | -5 AST nodes (35→30) | Eliminates: Assign node, Name('results') with Store ctx, Name('results') with Load ctx in return |

## What Worked
- **Eliminating intermediate variables**: The `results = ...` assignment created 5 extra AST nodes: `Assign`, `Name('results', Store)`, and in the return statement `Name('results', Load)` plus the `Return` node overhead. Inlining the list comprehension directly into `marginalize()` removes the assignment entirely.
- The change is semantically equivalent — Python evaluates the list comprehension identically whether stored first or passed directly.

## What Didn't Work
- N/A — target achieved in first iteration

## Code Diff
```
- results = [gen() for _ in range(num_trials)]
- return marginalize(results)
+ return marginalize([gen() for _ in range(num_trials)])
```

## AST Node Count Breakdown
**Before (35 nodes):**
- Module, FunctionDef, arguments: 3
- 3 x arg: 3
- 2 x Expr + 2 x Name (cot_examples, question stmts): 4
- Assign + Name('results', Store) + Store: 3
- ListComp + Call(gen) + comprehension + Name('_', Store) + Call(range) + Name('num_trials') + ...: 10
- Return + Call(marginalize) + Name(results) + Load: 6
- Various Load ctx nodes: 6

**After (30 nodes):**
- Eliminated: Assign, Name('results', Store), Name('results', Load) = -5 nodes
- Remaining structure unchanged

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter Optimization Insights**
From related work and practice, typical effective choices include: 
- **Temperature (T):** Often set in [0.5–0.8] for reasoning tasks. For diversity (self-consistency), T≈0.7 (medium.com). Low T (≤0.3) makes outputs more deterministic but sometimes misses creative solutions. 
- **Top-\(k\)/Top-\(p\):** Many use top_p=1.0 (or 0.9–1.0) to allow variability. Limiting top_k or lowering top_p (e.g. top_p=0.9) can sometimes improve coherence for factual answers. Empirically, open-ended tasks do well with high “mass” (near 1.0), while simple tasks may use a small top_k to prevent random rare tokens. 
- **Number of Samples:** Self-consistency studies generally sample \(N=5\)–\(10\) outputs per query (sometimes up to 20) and vote. In our context, increasing sampling (and parallel gen calls) usually improves accuracy roughly logarithmically (each doubling of samples yields diminishing returns). 
- **Few-shot Examples:** If the APPL snippet allows, adding a few examples to the model prompt can help (e.g. 3–5 examples of the chain-of-thought). Many systems find ~5 examples is a sweet spot. (Too many examples can exceed model context limits.) 
- **Model Choice:** Use the largest/best model affordably available. For example, switching from GPT-3.5 to GPT-4 can often yield ~10–20% absolute accuracy gains on reasoning benchmarks (www.nature.com). 
- **Concurrency:** APPL’s async runtime can launch many LLM calls at once. Setting the parallelism (number of concurrent threads or API calls) up to the rate-limit can speed up wall time nearly linearly (APPL claims near-ideal speedups (aclanthology.org)). If the bottleneck is latency, increasing concurrency is safe; if the bottleneck is throughput (API rate-limits), tune it just below the limit. 
- **Stop Criteria:** Configure sensible stopping sequences (e.g. newline or final answer markers) to truncate generation once the answer is complete. This avoids wasteful tokens and reduces chances of runaway text. 
- **Pruning and Caching:** If some LLM calls are repeated (in loops), caching results can halve calls. In our multiplication example, one could cache 3*4=12 since it’s constant. Even without entwining the LLM, reusing identical calls or splitting tasks to half-size (and doubling results) can cut cost.

**4. Concrete Optimization Ideas**
Below are ten strategies (with rough expected impact and risks) for improving the APPL code/metrics **without retraining**:

1. **Code Simplification:** *Combining expressions or using concise constructs.* For example, replace separate nested loops and joins with a single nested comprehension. E.g.: 
 ```python
 @ppl
 def table(n: int):
 f"Generate an {n}×{n} multiplication table (format: a*b=c with rows separated by newlines and columns by spaces)."
 return gen()
 ``` 
 This uses one `gen()` instead of N×N gens. *Expected gain:* **Very high** (dramatically shrink AST and number of calls). *Risk:* **High** – model must correctly output the entire table format; if it hallucinate or misformats, answers become unusable. 

2. **Flatten Loops via Prompts:** Instead of iterating in Python, ask the LLM to enumerate sub-answers. E.g. prompt “What is 3×1, 3×2, …, 3×n?” to generate a batch of values. *Gain:* **High** (fewer AST nodes, fewer gen calls). *Risk:* **High** – LLM might list results in an unpredictable format or make an error in arithmetic. 

3. **Inline Generation Calls:** Replace intermediate variables and string futures with inlined f-strings. For example, instead of building partial strings, directly do: 
 ```python
 f"{x}*{y}="; return gen()
 ``` 
 every time. *Gain:* AST nodes reduced (dropping temporary placeholders). *Risk:* **Low** – mainly stylistic, should preserve semantics. 

4. **Remove Redundant Decorators or Returns:** If any `@ppl`-decorated helper function is trivial, merge it. For example, if a helper just returns `gen()`, call `gen()` directly. *Gain:* Small AST reduction. *Risk:* **Low**. 

5. **Temperature Tuning:** Lower or raise `temperature` to improve answer quality. For instance, if currently T=1.0, try T=0.7 (or vice versa). *Gain:* **Moderate** – may reduce obvious errors. *Risk:* **Low** – might slightly underperform if lowered too much (model sticks to top token) or produce gibberish if raised too high. 

6. **Increase Parallel Samples (Self-Consistency):** If the snippet currently generates one answer per branch, modify it to sample multiple times (e.g. loop performing gen() 5 times and vote). *Gain:* **Moderate** (accuracy improves with more samples). *Risk:* **Medium** – increases AST/node count if not careful, and doubles/triples latency. However, APPL can parallelize these calls. 

7. **Model Upgrade:** Switch to a stronger LLM backend (e.g. GPT-4 vs GPT-3.5). *Gain:* **High** – as reported, GPT-4 can be 10–20% more accurate on complex tasks (www.nature.com). *Risk:* **Medium/High** – higher cost, possible rate limits, and output style differences may require minor prompt tweaks. 

8. **Answer Verification:** After generation, re-prompt the LLM (or a chain) to verify or correct the answer. For example, append each candidate answer with “Is this correct? Answer Yes or No.” *Gain:* **Small to moderate** (catches obvious nonsense). *Risk:* **Medium** – uses extra calls (AST increases), and model might over-reject correct answers, requiring fallback logic. 

9. **Prompts with Assertions:** Use APPL’s DSL to embed simple code checks. E.g., convert LLM strings to integers and assert `a*b == c`. If the assertion fails, retry that call. *Gain:* **Moderate** (ensures arithmetic correctness). *Risk:* **High** – mixing programmatic checks may complicate flow and error handling. If a check fails repeatedly, logic must handle it (risk of infinite retry loop). 

10. **Trim or Post-Process Output:** After collecting results, post-process the combined string to remove extraneous characters (e.g. trailing newline, quotes). This can be simple Python regex or string operations outside APPL. *Gain:* **Low** – cleans output format for evaluation. *Risk:* **Low** – straightforward filtering, but ensure not to accidentally remove valid content. 

Each idea should be tested incrementally. For example, (1) or (2) could reduce AST dramatically but must be validated carefully. Parameter tweaks like (5) or (6) are safer “knobs” with incremental gains. Model change (7) often yields the largest improvement in accuracy, at the cost of utility (risk of hitting rate limits or budget). 

**5. Common Failure Modes and Pitfalls**
- **Broken Semantics:** Aggressive code rewriting can inadvertently change program logic. For instance, merging loops may alter execution order or synchronization in APPL. Always verify that the new code produces identical outputs on a few examples. 
- **Generation Hallucinations:** LLMs sometimes “hallucinate” (produce incorrect but plausible-looking answers). Techniques like self-consistency, verification prompts, or embedding arithmetic checks can mitigate this. Without them, errors may go undetected. 
- **Overfitting Prompts:** Too much prompt-engineering (e.g. adding irrelevant qualifiers or over-specific examples) can make the prompt brittle—working only on narrow cases. Avoid hardcoding example outputs unless they generalize. 
- **Rate Limits and Latency:** Parallel or repeated calls (ensembles, self-consistency) may hit API rate limits or greatly increase response time. It’s easy to over-parallelize; monitor throughput and back off if needed. 
- **Excessive Complexity:** Adding many new tricks (ensembles, post-processing) can make the pipeline fragile and hard to debug. Each added component (e.g. a checker or vote logic) introduces potential new bugs. Test each in isolation. 

_(Research digest truncated.)_

## Idea library snapshot (`idea_library.md`)

### IDEA-001: Inline marginalize into return statement
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Instead of `results = [gen() for _ in range(num_trials)]` + `return marginalize(results)`, use `return marginalize([gen() for _ in range(num_trials)])`. This eliminates the Assign node, the intermediate Name(results) Store, and the separate Name(results) Load in the return.
- **Hypothesis**: Removes the Assign statement and intermediate variable: saves ~5 nodes (35→30)
- **Status**: SUCCESS — ast_size 35→30 (-14.3%)
- **Result**: Confirmed 30 nodes. Target achieved.

### IDEA-002: Remove @ppl decorator from snippet
- **Type**: CODE
- **Priority**: LOW
- **Risk**: HIGH (changes semantics)
- **Description**: Remove the `@ppl` decorator from the function. Saves the decorator node.
- **Hypothesis**: Saves 2 nodes (Name('ppl') + Load ctx) but fundamentally changes what the function does
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-003: Use lambda instead of def
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Replace `def cot_consistency(...)` with a lambda expression. Lambdas have fewer AST components.
- **Hypothesis**: Lambda has fewer argument nodes, could reduce AST
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-004: Rename parameter to shorter name
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Rename `num_trials` to `n`. Variable names don't affect node count (Name nodes are still Name nodes regardless of the string id). So this won't help.
- **Hypothesis**: No change to node count - Name node count stays same
- **Status**: PENDING (likely ineffective)
- **Result**: (fill in after execution)

### IDEA-005: Use *args or **kwargs
- **Type**: CODE
- **Priority**: LOW
- **Risk**: MEDIUM
- **Description**: Replace explicit parameters with *args to reduce arg nodes.
- **Hypothesis**: Might reduce argument node count
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-006: Combine both Expr statements in a tuple
- **Type**: CODE
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Replace `cot_examples\nquestion` with `cot_examples, question` (single tuple expression). But this adds a Tuple node: 2 Expr + 2 Name → 1 Expr + 1 Tuple + 2 Name = +1 node.
- **Hypothesis**: Actually increases by 1 (verified: 36 nodes)
- **Status**: PENDING (ineffective)
- **Result**: (fill in after execution)

### IDEA-007: Reduce to minimal single-argument function
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Create the most minimal valid CoT-SC snippet possible - single parameter, no intermediate vars
- **Hypothesis**: Major reduction possible
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-008: Use walrus operator for gen() results
- **Type**: CODE
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Explore if walrus operator `:=` can reduce node count vs separate assign
- **Hypothesis**: Walrus might save a node or two
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-009: Remove one parameter (question or cot_examples)
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: If the snippet is just measuring AST size and doesn't need to be runnable, compress to fewer parameters. E.g. combine cot_examples+question into a single `context` param.
- **Hypothesis**: Each param = 1 arg node; removing one saves 1 node
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-010: Remove type annotations from snippet
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: The original cot_sc.py has type annotations (list[str], str, int). Without annotations: 35 nodes. With annotations: 45 nodes. Our current snippet has no annotations (35 nodes). Confirmed already done.
- **Hypothesis**: Already applied (35 nodes without annotations vs 45 with)
- **Status**: ALREADY APPLIED
- **Result**: Current baseline is already without annotations

### IDEA-011: Use direct expression for results without variable
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: The main IDEA-001: inline marginalize. Already computed: gives 30 nodes. This is the primary optimization to try first.
- **Status**: PENDING (same as IDEA-001)

### IDEA-012: Minimal 2-param snippet
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Use only 2 params (examples, n) and inline everything. Saves 1 arg node + reduced body nodes.
- **Hypothesis**: ~28-29 nodes possible
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-013: Single-line body function
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Compress to a function with just a single return statement: `@ppl\ndef f(e,q,n):\n e\n q\n return marginalize([gen() for _ in range(n)])`
- **Hypothesis**: Short param names + inline = ~29 nodes
- **Status**: PENDING
- **Result**: (fill in after execution)
