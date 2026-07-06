## Critique Guidance (Verification Role)

This rubric guides the critique/verification process.

### Primary Goal

Determine whether an answer to a given question is correct and complete according to the answer quality rubric. Your role is to **verify**, not to regenerate the solution from scratch.

### Verification Process

**1. Check Correctness**
- Verify all mathematical steps for logical validity
- Identify any errors, such as computational errors, sign mistakes, or algebraic slip-ups
- Check that all theorem applications are valid (hypotheses satisfied, conclusions properly drawn)
- Ensure claimed results actually follow from stated premises

**2. Check Completeness**
- Verify that all parts of the question are addressed
- Check that edge cases, boundary conditions, and special values are handled
- For "find all" questions: assess whether the solution space is fully characterized
- For existence/uniqueness claims: verify both directions are proven if required

**3. Check Rigor**
- Identify gaps in reasoning (unjustified steps, missing lemmas, unproven claims)
- Flag handwaving or appeals to intuition where rigorous argument is needed
- Verify that validity conditions (domains, convergence, etc.) are stated where necessary
- Check that assumptions are explicit and reasonable

**4. Check Well-Posedness Recognition**
- If the question itself is ill-posed, verify that the answer correctly identifies this
- Check that the answer explains which well-posedness criterion is violated
- If an answer proceeds despite an ill-posed question, this is an error

### Evidence-Based Assessment

- **Quote specific passages** from the answer when identifying errors or gaps
- Provide concrete examples of what is wrong (e.g., "Line 3 claims $f$ is continuous, but this doesn't follow from the given hypotheses")
- For incomplete answers, specify what is missing
- For incorrect answers, explain where and why the reasoning fails

### Verdict Guidelines

Use these verdicts (must choose exactly one):

- **"correct"**: The answer is mathematically sound, complete, and adequately rigorous. Minor stylistic issues are acceptable if the mathematics is right.

- **"incorrect"**: The answer contains mathematical errors, unjustified claims, or fundamental logical flaws that invalidate the conclusion. This includes:
  - Incorrect final answer
  - Invalid reasoning steps
  - Misapplication of theorems
  - Significant gaps that undermine the solution's validity

- **"insufficient"**: The answer is on the right track and contains no outright errors, but is meaningfully incomplete:
  - Missing required cases or edge conditions
  - Partial proof that doesn't establish the full claim
  - Correct approach but stops short of the final result
  - Incomplete verification of all question components

- **"obscure"**: The answer is too unclear to verify:
  - Critical steps are missing or unexplained
  - Notation is undefined or inconsistent
  - Reasoning is so compressed that validity cannot be assessed
  - Structure makes it impossible to follow the logical flow

### Response Format

Provide:
1. **Verdict**: One of {correct, incorrect, insufficient, obscure}
2. **Notes**: Concise, evidence-based explanation citing specific issues or confirming correctness
3. **Suggestions** (optional): For non-correct verdicts, indicate what would need to be fixed

### Important Constraints

**Stay Concise**
- Focus on substantive issues, not minor style preferences
- Quote only the relevant portions needed to identify problems
- Prioritize mathematical correctness over expositional elegance

**Be Objective**
- Base judgments on the rubric, not on whether you would have solved it differently
- If the answer is correct, say so, even if an alternative approach exists
- Don't penalize correct answers for using advanced techniques or taking non-standard paths

**Verification vs. Generation**
- You are checking an existing solution, not solving from scratch
- If you cannot verify the answer's correctness within reasonable effort, mark it as "obscure" rather than guess

**Borderline Cases**
There might be some borderline cases. In such cases, the litmus test should always be: "Would a mathematician, given the information at hand, reasonably conclude the answer to satisfy the above points?"

### Allowed Conclusions

It is perfectly acceptable to conclude that an answer is fully correct and needs no changes. Do not feel obligated to find problems where none exist.
Also, if the only issues are very minor, there is no need to consider the answer incorrect/incomplete/obscure.
Note: what counts as a very minor flaw? A flaw is very minor if it's more stylistic than substantive, and if the flaw doesn't meaningfully undermine the correctness of the answer. When in doubt, consider what a mathematician would do.

**Examples of very minor flaws:**
- Notational inconsistency (e.g., switching between $f(x)$ and $f$ when context is clear)
- Missing explicit statement of a standard assumption (e.g., not stating "for $n \geq 1$" when solving a problem clearly about positive integers)
- Slightly informal language in an otherwise rigorous proof (e.g., "we can see that" instead of "it follows that")
- Omitting a trivial verification step that any mathematician would immediately recognize (e.g., not explicitly checking $0 < 1$ in an inequality chain)
- Minor notational ambiguity that doesn't affect understanding (e.g., using $\sin^2 x$ without clarifying it means $(\sin x)^2$, not $\sin(\sin x)$, when context makes it obvious)

**Examples of substantive (NOT minor) flaws:**
- Using a theorem without verifying its hypotheses are satisfied
- Missing a case in a case analysis (e.g., not considering $x = 0$ separately when dividing by $x$)
- Claiming uniqueness without proof when multiple solutions might exist
- Computational error that propagates to the final answer
- Unjustified step in the logical chain (e.g., "clearly $f$ is continuous" when this requires proof)
- Incomplete proof that establishes only partial results

### Examples

**Good Critique (JSON format):**
```json
{
  "verdict": "incorrect",
  "notes": "The answer claims 'by L'Hôpital's rule, the limit equals 2' but fails to verify that the limit has indeterminate form $0/0$ or $\\infty/\\infty$. At $x=0$, the numerator equals 1 (not 0), so L'Hôpital's rule does not apply.",
  "suggestions": "Verify the conditions for L'Hôpital's rule before applying it. In this case, direct substitution works."
}
```

*Why this is good:* Identifies a specific error (misapplication of theorem), quotes the problematic claim, explains why it's wrong (hypotheses not satisfied). Note that providing the correct answer is NOT required (though appreciated, if possible).

**Poor Critique (JSON format):**
```json
{
  "verdict": "incorrect",
  "notes": "The solution doesn't look right and seems to have mistakes.",
  "suggestions": "Redo the problem more carefully."
}
```

*Why this is poor:* Vague ("doesn't look right"), no specific errors identified, no evidence or quotes, unhelpful suggestions.