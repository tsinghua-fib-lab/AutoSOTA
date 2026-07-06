# Questions for Author

These questions arose during code review. Answers will help determine the correct fixes for identified issues.

---

## 1. Statistical Methodology

### Q1.1: Self-Answer Correctness for "Correct" Verdicts
**Context:** [stats.py:203-213](stats.py#L203-L213)

When a critic declares a self-answer "correct" (no debate needed), this answer is currently excluded from all statistics. However, self-answers that trigger debates are tracked in `model_self_answers`.

**Question:** Should self-answers declared "correct" by critics be:
- A) Added to `model_self_answers` with 100% judge agreement (treating "declared correct" as unanimous judge agreement)?
- B) Tracked in a separate "self_answers_no_debate" category?
- C) Excluded entirely (current behavior) - and if so, why?

**Impact on results:** This affects how we report model performance on self-generated questions.

Answer: B

---

### Q1.2: Majority Vote with Even Number of Judges
**Context:** [stats.py:184](stats.py#L184)

The code uses `correct_count > len(verdicts) / 2` to determine majority, which means:
- With 2 judges, 1-1 split counts as "wrong" (1 > 1.0 = False)
- With 4 judges, 2-2 split counts as "wrong"

**Question:** Is this intentional behavior?
- A) Yes, ties should favor the critic (claimant)
- B) No, ties should favor the defender (answer author)
- C) No, ties should be excluded from analysis as inconclusive
- D) We should ensure an odd number of judges per claim

**Current impact:** Affects which answers are counted as correct in cross-model statistics.

Answer: C

---

### Q1.3: Statistical Independence of Judges
**Context:** Multiple judge models evaluate the same debates

**Question:** Are the judge models statistically independent?
- Do you report per-judge agreement rates, or aggregate across all judges?
- Should inter-rater reliability (Cohen's kappa, Fleiss' kappa) be computed?
- Should we report confidence intervals on the percentages?

**Relevance:** Determines if we can treat judge verdicts as independent samples for statistical tests.

Answers:
- Per-judge
- Yes
- Yes

---

## 2. Data Semantics

### Q2.1: Temperature with Reasoning in Anthropic Models
**Context:** [model_api.py:165-166](model_api.py#L165-L166)

The code checks:
```python
if 'anthropic' in model and (temperature is not None and temperature != 1) and reasoning is not None:
    raise ValueError("Cannot set both temperature and reasoning")
```

**Question:** Why is `temperature=1` specifically allowed with reasoning?
- A) Temperature=1 is the default and equivalent to None, so it's okay
- B) Anthropic API specifically allows temperature=1 with reasoning
- C) This is a bug - ANY temperature should be rejected with reasoning
- D) The check should be `temperature != 1.0` to handle floating point comparison

**Impact:** Affects valid parameter combinations for API calls.

Answer: B
---

### Q2.2: Debate Early Stop Asymmetry
**Context:** [debate.py:152-154](debate.py#L152-L154), [debate.py:160-162](debate.py#L160-L162)

When the defender (Bob) concedes, the loop breaks immediately before the claimant (Alice) can respond in that round. But when the claimant concedes, both messages for that round are already collected.

**Question:** Is this intentional?
- A) Yes, we want debates to end immediately when defender concedes
- B) No, both parties should get to speak in the same round before stopping
- C) Doesn't matter for your analysis

**Impact:** Affects debate length statistics and potentially judge evaluations.

Answer: A

---

### Q2.3: Critique Verdict Semantics
**Context:** Multiple files use critique verdicts

The prompt (prompt_library.py:178) defines these critique verdicts:
- `"correct"` - no issues
- `"incorrect"` - has errors
- `"insufficient"` - incomplete
- `"obscure"` - unclear

**Question:** How should these map to statistics?
- Should "insufficient" and "obscure" be treated as "wrong" or as a separate category?
- Do you want to report them separately in results?
- Should critiques with "obscure" verdict be excluded from analysis?

**Impact:** Affects how answers are categorized in statistical analysis.

Answers:
- As "wrong"
- Yes
- No

---

## 3. Pipeline Behavior

### Q3.1: Concurrent File Updates in Critique Generation
**Context:** [generate_critiques.py:363-380](generate_critiques.py#L363-L380)

Multiple jobs can update different indices in the same output file concurrently. This creates a race condition where updates can be lost.

**Question:** Is this a known issue?
- A) Yes, we're aware and it's rare enough not to matter
- B) No, this is a bug that needs fixing
- C) We thought each job wrote to a different file

**Observed behavior:** Have you seen any missing critiques or corrupted output files?

Answer: introduce file locks, but only if you're confident that there can be concurrent access

---

### Q3.2: Ill-Posed Override Semantics
**Context:** [generate_answers.py:40-45](generate_answers.py#L40-L45)

The code adds an override note to the self-check prompt to prevent marking questions as ill-posed.

**Question:** What is the intended use case?
- A) Human reviewer identified false positives and wants to force re-evaluation
- B) Specific question-answer pairs known to be valid
- C) Override model tendency to claim ill-posedness too often

**Impact:** Affects interpretation of ill-posed statistics.

Answer: A

---

### Q3.3: Benchmark Answer Fallback
**Context:** Multiple files use `benchmark_answers()` to extract answers from benchmark files when answer files don't exist

**Question:** When should this happen?
- A) For self-answers (question_author == answer_author), the answer is in the benchmark file
- B) This is a fallback for missing data
- C) This handles a specific edge case

**Context:** Appears in [generate_critiques.py:56-76](generate_critiques.py#L56-L76), [debate.py:58-78](debate.py#L58-L78), [automated_judge.py:61-74](automated_judge.py#L61-L74)

Answer: I don't know what happened, document in docs/issues.md for now

---

## 4. Research Methodology

### Q4.1: Debate Rounds and Convergence
**Context:** Debates run for fixed number of rounds (default 5)

**Question:** How did you choose the number of rounds?
- Is 5 rounds sufficient for convergence?
- Do most debates end early due to concession?
- Should the number of rounds vary by claim type?

**Impact:** Affects whether debates provide sufficient information for judge models.

Answer: 5 is fine

---

### Q4.2: Self-Improvement Rounds
**Context:** All generation tasks use up to 5 self-improvement rounds

**Question:**
- What percentage of generations pass on first attempt?
- What percentage reach max rounds without passing?
- Should max_rounds be tuned per model or per task type?

**Relevance:** Helps understand if 5 rounds is appropriate across all models and tasks.

Answer: 5 is fine

---

### Q4.3: Judge Model Selection
**Context:** [automated_judge.py:459-467](automated_judge.py#L459-L467) - judges cannot evaluate debates they participated in

**Question:** How do you ensure sufficient judge coverage?
- With N models, each debate has (N-2) eligible judges
- What's the minimum number of judges per claim?
- Do some claims have zero eligible judges?

**Impact:** Affects statistical power and validity of majority voting.

Answer: will be studied later, write in issues.md

---

## 5. Data Quality

### Q5.1: Unknown Verdicts in Production
**Context:** [check_issues.py](check_issues.py), [automated_judge.py:378](automated_judge.py#L378)

Automated evaluations can have `verdict="unknown"` when parsing fails or model output is invalid.

**Question:** What percentage of judgments end up as "unknown"?
- Should these be retried with different models?
- Should they be excluded from statistics?
- Is this a quality issue that needs addressing?

**Impact:** Affects sample size and validity of results.

Answer: will be studied later, write in issues.md

---

### Q5.2: Failed Status Handling
**Context:** Multiple status values: "succeeded", "failed", "ill-posed"

**Question:** What should be done with "failed" items?
- A) Retry with different parameters
- B) Exclude from analysis
- C) Manual review
- D) They represent genuine model limitations

**Observed:** `--rerun-failures` flag exists for some scripts but not others.

Answer: C & D

---

### Q5.3: Math LaTeX Cleaning
**Context:** [utils.py:89-94](utils.py#L89-L94) - converts `\(` to `$` and `\[` to `$$`

**Question:**
- Are all models expected to output LaTeX in `\( \)` format?
- Do any models already output in `$ $` format (making double conversion a risk)?
- Should cleaning be model-specific?

**Impact:** Potential corruption of mathematical notation if applied incorrectly.

Answer: should be fine as-is

---

## 6. Configuration

### Q6.1: Model Display Names
**Context:** [configs/models.json](configs/models.json) - some models have display_name, others don't

**Question:**
- Should all models have display names for paper publication?
- Are the full model names (e.g., "openai/gpt-5.2-2025-12-11") okay to publish?
- Should display names be anonymized for blind review?

**Impact:** Affects what appears in published results.

Answers:
- Yes
- Yes
- No
---

### Q6.2: Default Reasoning Level
**Context:** [configs/models.json:3](configs/models.json#L3) - `"default_reasoning": null`

Some models specify `"reasoning": "high"`, others use default (null).

**Question:**
- Is "high" reasoning always better for this task?
- Should all models use the same reasoning level for fair comparison?
- Or is this testing different configurations of each model?

**Impact:** Affects fairness of model comparisons.

Answer: We use high only where it's supported

---

## 7. Output and Reporting

### Q7.1: Statistics Aggregation
**Context:** [stats.py](stats.py) prints averages across critiques

**Question:**
- Should results be aggregated by topic?
- Should confidence intervals be reported?
- Should per-question results be exported for further analysis?

**Impact:** Affects what statistics appear in the paper.

Answer: not for now, write in issues.md

---

### Q7.2: Cross-Model Statistics Presentation
**Context:** [stats.py:233-255](stats.py#L233-L255) - detailed breakdown by question maker

**Question:**
- Is the three-way split (declared correct / critiqued but correct / critiqued and wrong) the desired presentation?
- Should this be a confusion matrix instead?
- Should we report percentages or raw counts?

**Impact:** Affects clarity of results presentation.

Answer: write in issues.md

---

## 8. Edge Cases

### Q8.1: Empty Debates
**Context:** [automated_judge.py:144-146](automated_judge.py#L144-L146), [automated_judge.py:216-218](automated_judge.py#L216-L218)

With `--allow-no-debate`, judgments can be made without debate transcripts.

**Question:**
- When would debates be empty?
- Should judgments without debates be treated differently in statistics?
- Is this for handling incomplete data or a valid use case?

**Impact:** Affects validity of judgments.

Answer: keep for now

---

### Q8.2: Parsing with Model Unavailability
**Context:** [utils.py:122-156](utils.py#L122-L156) - JSON repair uses an LLM

**Question:**
- What happens if the parsing model is unavailable or rate-limited?
- Should there be a fallback or retry mechanism?
- How often does parsing fail in practice?

**Impact:** Pipeline robustness to API failures.

Answer: fail loudly (but let other jobs proceed)

---

## 9. Reproducibility

### Q9.1: Run IDs and Versioning
**Context:** [generate_benchmark.py:38-47](generate_benchmark.py#L38-L47) - uses run IDs from config

**Question:**
- How are run IDs generated?
- Should there be a global experiment ID for tracking which outputs go together?
- How do you ensure reproducibility when rerunning parts of the pipeline?

**Impact:** Ability to reproduce results.

Answers:
- Hand-made
- Later, add in issues.md
- Hard to do so, we work with what we have

---

### Q9.2: API Model Versions
**Context:** [model_api.py:25-50](model_api.py#L25-L50) - captures model metadata

**Question:**
- Are you tracking and reporting the exact model versions used?
- Some APIs (like OpenAI) serve different versions over time - how do you handle this?
- Should model versions be included in result files?

**Impact:** Long-term reproducibility and transparency.

Answer: ignore for now, write in issues.md

---

## Priority Questions

If time is limited, these are the most critical to answer:

1. **Q1.2** (Majority vote handling) - Affects correctness of statistics
2. **Q3.1** (Race conditions) - Potential data corruption
3. **Q4.3** (Judge coverage) - Affects validity of results
4. **Q1.1** (Self-answer stats) - Affects completeness of results
5. **Q2.3** (Critique verdicts) - Affects statistical categories

---

**Total questions:** 25 across 9 categories
