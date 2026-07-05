# Outstanding Issues for Future Work

This document tracks issues identified during code review that are deferred for later analysis or implementation.

---

## Data Quality and Analysis

### Benchmark Answer Fallback Behavior
**Context:** Q3.3 from questions.md

Multiple files use `benchmark_answers()` to extract answers from benchmark files when answer files don't exist. The exact semantics of when this should happen is unclear and needs investigation.

**Files affected:**
- [generate_critiques.py:56-76](generate_critiques.py#L56-L76)
- [debate.py:58-78](debate.py#L58-L78)
- [automated_judge.py:61-74](automated_judge.py#L61-L74)

**Action needed:** Document the expected behavior and ensure consistency across all uses.

---

### Judge Model Coverage Analysis
**Context:** Q4.3 from questions.md, [automated_judge.py:459-467](automated_judge.py#L459-L467)

With N models in the system, each debate has at most (N-2) eligible judges (excluding the two debate participants).

**Questions to investigate:**
- What's the minimum number of judges per claim in practice?
- Do some claims have zero eligible judges?
- How does this affect statistical power?
- Should we require a minimum number of judges per claim?

**Action needed:** Analyze actual judge coverage in generated data and document findings.

---

### Unknown Verdict Rate Analysis
**Context:** Q5.1 from questions.md

Automated evaluations can have `verdict="unknown"` when parsing fails or model output is invalid.

**Questions to investigate:**
- What percentage of judgments end up as "unknown" in practice?
- Which models produce more unknown verdicts?
- Should these be retried automatically?
- Should they be excluded from statistics or treated as a separate category?

**Action needed:** Run statistics on current data to quantify unknown verdict rate and analyze patterns.

---

## Statistical Analysis Enhancements

### Results Aggregation by Topic
**Context:** Q7.1 from questions.md

Current statistics aggregate across all topics. Consider adding:
- Per-topic breakdown of results
- Confidence intervals on percentages
- Per-question result exports for deeper analysis

**Action needed:** Determine if topic-level analysis would provide meaningful insights for the paper.

---

### Cross-Model Statistics Presentation
**Context:** Q7.2 from questions.md, [stats.py:233-255](stats.py#L233-L255)

Current presentation uses three-way split (declared correct / critiqued but correct / critiqued and wrong).

**Alternatives to consider:**
- Confusion matrix format
- Raw counts vs percentages
- Different aggregation strategies

**Action needed:** Experiment with different presentations and choose the clearest for the paper.

---

## Code Quality Improvements

### Debate Early Stop Asymmetry
**Context:** §4.2 from improvements_2.md, [debate.py:152-154](debate.py#L152-L154)

When defender concedes, loop breaks immediately. When claimant concedes, both messages for that round are collected. This asymmetry is intentional but should be documented.

**Behavior:**
- Defender concession: immediate break, claimant doesn't respond in that round
- Claimant concession: both parties have spoken in that round

**Action needed:** Add comment in code explaining this intentional asymmetry and why it's designed this way.

---

### Redaction Over-Matching
**Context:** §10.2 from improvements_2.md, [automated_judge.py:90-98](automated_judge.py#L90-L98)

Redaction uses case-insensitive substring matching, which may over-redact (e.g., "claude" in "Claude's theorem").

**Current approach:**
```python
redacted = re.sub(re.escape(needle), replacement, redacted, flags=re.IGNORECASE)
```

**Potential issue:** May redact unintended matches where model names appear as parts of mathematical terms or person names.

**Action needed:** Monitor for over-redaction in actual judge prompts. If it becomes a problem, consider:
- Word boundary matching (but may break with special characters in model names)
- Exact case-sensitive matching
- Whitelist of terms to never redact

---

### Thread Execution Non-Determinism
**Context:** §11.1 from improvements_2.md

Thread execution order is non-deterministic, which could affect reproducibility.

**Current state:**
- No random seeds set
- Thread scheduling affects API call order
- Different runs may produce slightly different results

**Mitigation strategies:**
- Set random seeds at program start
- Add run IDs and timestamps to all output files (partially done)
- Log execution order for debugging
- Accept minor non-determinism as acceptable for research

**Action needed:** Decide if perfect reproducibility is required, or if current approach is acceptable for research purposes.

---

## Versioning and Reproducibility

### Experiment Tracking and Global IDs
**Context:** Q9.1 from questions.md

**Current state:**
- Run IDs are hand-made and stored in configs
- No global experiment ID to track which outputs belong together
- Rerunning parts of pipeline may create inconsistencies

**Desired improvements:**
- Global experiment ID that ties together all outputs from a single run
- Timestamp-based versioning
- Better tracking of which data files were generated together

**Action needed:** Design and implement an experiment tracking system if needed for reproducibility.

---

### API Model Version Tracking
**Context:** Q9.2 from questions.md, [model_api.py:25-50](model_api.py#L25-L50)

**Current state:**
- Code exists to capture model metadata including versions
- Not clear if this is being saved to output files consistently

**Questions:**
- Are exact model versions being tracked in output files?
- How to handle APIs that serve different versions over time?
- Should model versions be reported in the paper?

**Action needed:** Verify model version metadata is being captured and decide on reporting strategy for paper.

---

## Notes

All items in this document are deferred for future work. They represent:
- Analysis tasks that require running statistics on actual data
- Design decisions that need more information
- Nice-to-have improvements that aren't critical for current publication
- Documentation tasks to make code more maintainable

Priority should be given to items that affect:
1. Statistical validity of results
2. Reproducibility for paper reviewers
3. Code maintainability for future researchers
