# Real-world LLM inputs (one critique-with-debate sequence)

## Sequence identifiers
- Claim id: `critique/contradictor/openai-gpt-3.5-turbo-0125/openai-gpt-5.2-2025-12-11__openai-gpt-3.5-turbo-0125/0`
- Question model (benchmark): `openai-gpt-3.5-turbo-0125`
- Critic model: `openai-gpt-5.2-2025-12-11`
- Answer author: `openai-gpt-3.5-turbo-0125` (self-answer; no separate answers file)
- Debate file: `debates/critiques/contradictor/openai-gpt-3.5-turbo-0125/openai-gpt-5.2-2025-12-11__openai-gpt-3.5-turbo-0125.json`
- Critique file: `critiques/contradictor/openai-gpt-3.5-turbo-0125/openai-gpt-5.2-2025-12-11__openai-gpt-3.5-turbo-0125.json`
- Benchmark file: `benchmarks/openai-gpt-3.5-turbo-0125.json`
- Judge file: `automated_evaluations/microsoft-phi-4-reasoning-plus.json`

## Benchmark generation (questioner model)


===========================

system prompt:
```text
You are a helpful assistant.
```
User message:
```text
# Task: Generate a Challenging Mathematics Problem

You are acting as a **tester** in a benchmarking framework. Your goal is to create a single, challenging but solvable mathematics problem along with a complete, verifiable solution.

## Requirements

- The question should be at an advanced mathematical level, suitable for testing strong reasoning capabilities
- It must be fully self-contained, well-posed, and solvable with the information provided
- Your answer must be complete, rigorous, and demonstrate that the question is actually solvable
- Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$

## Quality Rubric

## Question Quality and Meaningfulness

This rubric implements the meaningfulness predicate. A question passes only if it satisfies ALL criteria below.

### Core Requirements

**1. Well-Posedness and Completeness**
- The question must be fully self-contained with all necessary information to reach a unique, verifiable solution
- Include explicit domains, bounds, constraints, and assumptions
- Define all non-standard notation; standard mathematical objects (e.g., $\mathbb{R}$, $\mathbb{Z}$, common functions) need not be redefined
- The expected answer format must be clear (e.g., "find the exact value," "prove or disprove," "determine all solutions")

**2. Clarity**
- Every term must have a single, unambiguous interpretation in the given context
- Reject questions where reasonable mathematicians might disagree on what is being asked

**3. Non-Triviality**
- The question must require substantive mathematical reasoning beyond direct lookup or trivial computation
- Avoid questions solvable by immediate substitution, memorized formulas, or single-step calculations
- The difficulty should challenge advanced mathematical capability while remaining solvable with clear reasoning
- Questions may involve sophisticated techniques (e.g., complex analysis, differential geometry, abstract algebra)

**4. Solvability and Verification**
- The question must be solvable with established mathematical methods (no open conjectures)
- The solution must be verifiable through clear mathematical argument
- Avoid questions requiring extensive numerical search, simulation, or external computational tools beyond symbolic manipulation

### Rejection Criteria (Ill-Posed Features)

A question fails if it exhibits any of these defects:

- **Missing constraints**: Underdetermined systems, free parameters without specified domains
- **Contradictory premises**: Mutually incompatible conditions that make the question unanswerable
- **Multiple incompatible interpretations**: Ambiguous phrasing that admits fundamentally different readings
- **Undefined or underspecified objects**: References to concepts without sufficient definition
- **External dependencies**: Requires access to databases, physical measurements, or runtime environments

### Policy Compliance

- Questions must be appropriate for academic mathematical evaluation
- No questions designed to leak information, test memorization of specific papers, or probe for training data
- Avoid culturally dependent, time-sensitive, or subjective elements

### When Rejecting as Ill-Posed

When a question is identified as ill-posed, you must:
1. Specify which requirement(s) above are violated
2. Explain the specific defect (e.g., "missing domain specification for variable x," "contradictory assumptions about convergence")
3. Provide one concrete example of the ambiguity or incompleteness if applicable

### When Revising a Rejected Question

If asked to revise an ill-posed question:
- Produce a **materially different** question on the same general topic
- Do not merely patch the original; redesign from scratch to avoid anchoring on the flawed structure
- Ensure the new question satisfies all criteria above and passes the self-solve gate
- The revised question should be of comparable difficulty but with clear, unambiguous specifications

### Examples

**Good Question (Well-Posed):**
"Let $f: \mathbb{R} \to \mathbb{R}$ be defined by $f(x) = x^2 - 4x + 3$. Find all values of $x \in \mathbb{R}$ such that $f(x) = 0$."

*Why this is good:* Fully specified (domain and codomain given), clear task ("find all values"), unambiguous notation, non-trivial (requires solving a quadratic), solvable and verifiable.

**Poor Question (Ill-Posed):**
"Solve $f(x) = 0$."

*Why this is poor:* Missing critical information (what is $f$? what domain?), impossible to answer without more context. Violates completeness and well-posedness requirements.

## Topic

Generate a problem in the following domain: **Mathematical logic and foundations**

## Output Format

Use exactly this structure:

[QUESTION]
<Your problem statement here>

[ANSWER]
<Your complete solution here>

**Important**: Your solution will be verified first (self-solve gate). If it fails verification, the question will be rejected without being used to test other models.
```

## Benchmark self-check (same model, round 1)


===========================

system prompt:
```text
You are a helpful assistant.
```
User message:
```text
# Task: Evaluate Your Own Answer

Review the answer you provided to the question below and assess whether it meets the quality standards.

## Question

Let $P(x)$ be a predicate defined over the domain of integers $\mathbb{Z}$ such that $P(x)$ is true if and only if $x$ is a prime number. Consider the statement:
$$\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$$

Prove or disprove the statement.

## Your Answer

To prove or disprove the given statement, we need to consider the truth values of $P(x+y)$ and $P(x-y)$ for all integers $x$ and some corresponding integers $y$.

Let's first analyze the truth values of $P(x+y)$ and $P(x-y)$ individually:
- For $P(x+y)$ to be true, $x+y$ must be a prime number.
- For $P(x-y)$ to be true, $x-y$ must be a prime number.

Now, let's consider the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$:
- If $x$ is a prime number, then choosing $y = 0$ satisfies the statement since $P(x+0)$ and $P(x-0)$ are both true.
- If $x$ is not a prime number, then we need to find suitable $y$ values for which $x+y$ and $x-y$ are prime numbers. This is where it becomes challenging.

Let's consider an example to illustrate the difficulty:
- Take $x = 4$. We need to find $y$ such that $4+y$ and $4-y$ are prime numbers. This is a well-known open problem in number theory known as the "twin prime conjecture."

Therefore, the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$ is **disproved** due to the existence of numbers for which finding suitable $y$ values satisfying the condition is not trivial.

This problem delves into the realm of number theory and the distribution of prime numbers, showcasing the complexity of prime number patterns and the challenges in proving certain statements involving primes.

## Evaluation Rubric

## Self-Critique Guidance (Self-Improvement Role)

This rubric guides the self-evaluation process during answer refinement. Your goal is to identify whether your answer is ready for submission or needs improvement.

### Primary Goal

Determine whether your answer satisfies the quality standards well enough to be considered complete and correct. You are evaluating your own work to decide: **"Is this answer good enough, or should I refine it?"**

### Self-Evaluation Process

**1. Check for Correctness**
- Review your mathematical steps for logical validity
- Verify calculations and algebraic manipulations
- Check that theorem applications are valid (hypotheses satisfied, conclusions properly drawn)
- Ensure your final answer directly addresses what the question asks

**2. Check for Completeness**
- Did you address all parts of the question?
- Did you handle relevant edge cases or boundary conditions?
- For "find all" questions: did you find all solutions?
- For existence/uniqueness claims: did you prove both directions if required?

**3. Check for Rigor**
- Are there unjustified jumps in reasoning that would confuse a reader?
- Did you handwave away steps that actually need justification?
- Are validity conditions (domains, convergence, etc.) stated where necessary?

**4. Check for Ill-Posedness**
- If the question itself is ill-posed (missing constraints, contradictory premises, undefined terms), did you correctly identify this?
- If you proceeded despite ill-posedness, this is an error. Declare the question ill-posed

### When to Pass Your Own Answer

**Pass ("verdict": "pass") if:**
- The mathematics is correct
- All question components are addressed
- The reasoning is sufficiently clear and rigorous
- Any stylistic imperfections are minor and don't affect understanding
- If the question is ill-posed, you correctly identified and explained this

**Examples of passing answers with minor imperfections:**
- Slightly informal language that doesn't obscure meaning
- Notation that could be cleaner but is still clear in context
- Steps that could be more explicit but follow obviously from what's stated
- Minor redundancy or verbosity

### When to Fail Your Own Answer

**Fail ("verdict": "fail") if:**
- Mathematical errors or invalid reasoning
- Missing significant parts of the question
- Unjustified steps that a reader couldn't verify
- Incomplete handling of cases
- Wrong final answer

**Examples of substantive issues requiring refinement:**
- Used a theorem without checking its hypotheses
- Claimed something is true without proof when proof is non-trivial
- Missed a case in the analysis (e.g., didn't consider $x=0$ when dividing by $x$)
- Computational error affecting the final answer
- Logic gap that breaks the argument chain

### Guidelines for Constructive Self-Critique

**Focus on Substance, Not Style:**
- Don't fail your answer for minor stylistic issues
- Don't fail for notation you understand even if it could be cleaner
- Don't fail for verbosity or redundancy if the math is right
- **Do** fail for mathematical errors, gaps in logic, or incomplete solutions

**Be Honest About Issues:**
- If you spotted an error, acknowledge it clearly
- Explain specifically what's wrong and where
- Provide actionable guidance for improvement
- Don't dismiss substantive problems

**Distinguish Uncertainty from Errors:**
- If you're uncertain about a step, say so explicitly
- Uncertainty about correctness → investigate further or mark as fail
- Certainty despite minor imperfections → can pass

### Ill-Posedness Detection

**If you believe the question is ill-posed, set `"ill_posed": true` and:**
- Specify which well-posedness criterion is violated:
  - Missing constraints or underspecified parameters
  - Contradictory premises
  - Ambiguous interpretation
  - Undefined objects or notation
- Explain why this makes the question unanswerable
- If possible, suggest what would make it answerable

**If the question is well-posed, set `"ill_posed": false`.**

### Output Format Requirements

Return JSON with this exact schema:
```json
{
  "verdict": "pass" | "fail",
  "ill_posed": true | false,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "improvements": "<actionable guidance for refinement>"
}
```

**Field Guidelines:**
- `verdict`: "pass" if the answer is good enough; "fail" if substantive issues need fixing
- `ill_posed`: true only if the **question** itself is ill-posed, not the answer
- `issues`: List specific problems (empty if verdict is "pass")
- `improvements`: Clear guidance on how to fix (empty string if verdict is "pass")

### Important Principles

**Don't Be Overly Self-Critical:**
- Passing a good answer is correct behavior
- Don't create busywork by "improving" things that are already fine
- Minor imperfections are acceptable if the mathematics is sound
- Trust yourself when you've done good work

**Do Be Intellectually Honest:**
- Don't pass an answer you know has problems
- Don't handwave away genuine concerns
- Better to refine and get it right than to submit something flawed

### Borderline Cases

What counts as a very minor flaw? A flaw is very minor if it's more stylistic than substantive, and if the flaw doesn't meaningfully undermine the correctness of the answer. When in doubt, consider what a mathematician would do.

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

**Good Self-Evaluation (JSON format):**
```json
{
  "verdict": "fail",
  "ill_posed": false,
  "issues": ["Applied L'Hôpital's rule without verifying indeterminate form", "Incorrect final answer"],
  "improvements": "Check that the limit has form $0/0$ or $\\infty/\\infty$ before applying L'Hôpital's rule. In this case, direct substitution gives 1/2."
}
```
*Context:* Your answer claimed to solve $\lim_{x \to 0} \frac{\sin x}{2x}$ by L'Hôpital's rule and got 2, but you didn't verify that the limit is indeterminate (it's not - the numerator is 0 but the denominator is also approaching 0, so it IS indeterminate, but your answer of 2 is wrong; the correct answer is 1/2).

*Why this is good:* Correctly identifies a substantive mathematical error (wrong application of theorem and incorrect final answer), provides specific issues, and gives actionable guidance for fixing it.

**Poor Self-Evaluation (JSON format):**
```json
{
  "verdict": "fail",
  "ill_posed": false,
  "issues": ["Could be more detailed", "Notation could be cleaner"],
  "improvements": "Add more explanation and use better notation"
}
```
*Context:* The same correct answer as above.

*Why this is poor:* Fails a mathematically correct answer for minor stylistic reasons, creating busywork instead of recognizing good work. The "issues" are vague and don't identify actual mathematical problems.

## Required Output Format

Return ONLY a JSON object with this exact schema (no additional text):

```json
{
  "verdict": "pass" | "fail",
  "ill_posed": true | false,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "improvements": "<actionable guidance for improvement>"
}
```

**Field Descriptions:**
- `verdict`: "pass" if the answer is correct and complete, "fail" otherwise
- `ill_posed`: true if the question itself is unanswerable as stated, false otherwise
- `issues`: List of specific problems with the answer (empty list if none)
- `improvements`: Short, concrete guidance on how to fix the answer (empty string if verdict is "pass")
```

## Answer generation
No separate answer-generation call occurred for this sequence because the answer author is the same as the question model and self-answer files are disallowed; the answer used downstream is the benchmark answer above.

## Critique generation (critic model)


===========================

system prompt:
```text
You are a helpful assistant.
```
User message:
```text
# Task: Verify Answer Correctness

You are evaluating an answer provided by a model. Your role is **verification**, not independent solution generation.

## Critique Guidelines

## Critique Guidance (Verification Role)

This rubric guides the critique/verification process, which is central to the benchmarking framework's reliance on the verification-generation asymmetry.

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

## Question

Let $P(x)$ be a predicate defined over the domain of integers $\mathbb{Z}$ such that $P(x)$ is true if and only if $x$ is a prime number. Consider the statement:
$$\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$$

Prove or disprove the statement.

## Answer to Evaluate

To prove or disprove the given statement, we need to consider the truth values of $P(x+y)$ and $P(x-y)$ for all integers $x$ and some corresponding integers $y$.

Let's first analyze the truth values of $P(x+y)$ and $P(x-y)$ individually:
- For $P(x+y)$ to be true, $x+y$ must be a prime number.
- For $P(x-y)$ to be true, $x-y$ must be a prime number.

Now, let's consider the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$:
- If $x$ is a prime number, then choosing $y = 0$ satisfies the statement since $P(x+0)$ and $P(x-0)$ are both true.
- If $x$ is not a prime number, then we need to find suitable $y$ values for which $x+y$ and $x-y$ are prime numbers. This is where it becomes challenging.

Let's consider an example to illustrate the difficulty:
- Take $x = 4$. We need to find $y$ such that $4+y$ and $4-y$ are prime numbers. This is a well-known open problem in number theory known as the "twin prime conjecture."

Therefore, the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$ is **disproved** due to the existence of numbers for which finding suitable $y$ values satisfying the condition is not trivial.

This problem delves into the realm of number theory and the distribution of prime numbers, showcasing the complexity of prime number patterns and the challenges in proving certain statements involving primes.

## Required Output Format

Return ONLY a JSON object with this exact schema (no additional text):

```json
{
  "verdict": "correct" | "incorrect" | "insufficient" | "obscure",
  "notes": "<evidence-based explanation with specific quotes or examples>",
  "suggestions": "<optional: what needs to be fixed for non-correct verdicts>"
}
```

**Verdict Definitions:**
- `"correct"`: Mathematically sound, complete, and adequately rigorous
- `"incorrect"`: Contains errors, invalid reasoning, or fundamental flaws
- `"insufficient"`: Partially correct but meaningfully incomplete
- `"obscure"`: Too unclear to verify


```

## Critique self-check / refinement
No critique self-improvement ran for this item (only one critique attempt, evaluation is null).

## Debate round 1 (Bob/defender model)


===========================

system prompt:
```text
You are Bob, responding to Alice's critique of your answer. Be concise and correct errors when valid.
Answer quality guidance:
## Answer Correctness and Completeness

This rubric implements the correctness verification predicate. An answer passes only if it satisfies ALL criteria below.

### Core Requirements

**1. Correctness and Directness**
- The answer must be mathematically correct and directly address what the question asks
- Provide an explicit final result in the requested format (exact value, proof, construction, classification, etc.)
- The final answer should be stated clearly and unambiguously, typically in a concluding statement
- All intermediate steps must be logically valid with no mathematical errors

**2. Complete Reasoning Chain**
- Show all substantive reasoning steps from premises to conclusion
- Justify non-trivial claims, theorem applications, and technique choices
- Make logical dependencies explicit: if step B depends on result A, state this clearly
- For proofs: establish all necessary lemmas, handle all required cases, verify all hypotheses of applied theorems

**3. Rigor and Precision**
- Provide validity conditions: domains, convergence criteria, edge cases, boundary behavior
- State assumptions explicitly if you introduce any beyond what the question provides
- Avoid handwaving phrases like "it is clear that," "obviously," "by inspection" unless the claim is genuinely immediate
- Do not appeal to external computation, simulation, or numerical approximation unless the question explicitly permits it

**4. Completeness and Edge Cases**
- Address all parts of the question if it has multiple components
- Consider boundary cases, degenerate scenarios, and special values
- For existence/uniqueness claims: prove both directions
- For "find all" questions: prove you have found every solution

### Handling Ill-Posed Questions

**If the question itself is ill-posed:**
- Do NOT fabricate an answer or proceed as if the question were well-defined
- Explicitly state that the question is ill-posed and cannot be answered as stated
- Cite the specific well-posedness rule(s) violated (refer to the question quality rubric):
  - Missing constraints or underspecified parameters
  - Contradictory premises
  - Ambiguous interpretation
  - Undefined objects or notation
- If possible, suggest what additional information or clarification would make the question answerable

**Example**: "This question is ill-posed because the domain of variable $x$ is not specified, making the integral undefined. The question violates the completeness requirement (missing constraints). To make this answerable, specify whether $x \in \mathbb{R}$, $x \in \mathbb{R}^+$, or some other domain."

### Minor vs. Substantive Issues

Focus on mathematical correctness, not stylistic perfection. An answer with minor imperfections can still satisfy the requirements if the mathematics is sound.

**Minor stylistic issues that do NOT invalidate an answer:**
- Notational inconsistency (e.g., switching between $f(x)$ and $f$ when context is clear)
- Missing explicit statement of a standard assumption (e.g., not stating "for $n \geq 1$" when solving a problem clearly about positive integers)
- Slightly informal language in an otherwise rigorous proof (e.g., "we can see that" instead of "it follows that")
- Omitting a trivial verification step that any mathematician would immediately recognize (e.g., not explicitly checking $0 < 1$ in an inequality chain)
- Minor notational ambiguity that doesn't affect understanding (e.g., using $\sin^2 x$ without clarifying it means $(\sin x)^2$, not $\sin(\sin x)$, when context makes it obvious)

**Substantive issues that DO invalidate an answer:**
- Using a theorem without verifying its hypotheses are satisfied
- Missing a case in a case analysis (e.g., not considering $x = 0$ separately when dividing by $x$)
- Claiming uniqueness without proof when multiple solutions might exist
- Computational error that propagates to the final answer
- Unjustified step in the logical chain (e.g., "clearly $f$ is continuous" when this requires proof)
- Incomplete proof that establishes only partial results

### Verification Guidance

Your answer should be structured to facilitate verification:
- Use standard mathematical notation and terminology
- Structure proofs with clear logical flow (e.g., numbered steps, case analysis)
- For computational results: show the calculation path so it can be checked step-by-step
- For constructive proofs: the construction should be explicit enough to verify semi-mechanically
- For existence proofs: provide an explicit example or cite a constructive theorem

### Uncertainty and Limitations

**If genuinely unsure:**
- Acknowledge uncertainty explicitly rather than guessing
- Explain what makes the problem difficult or where your reasoning might be incomplete
- If applicable, propose a partial solution or outline an approach
- Suggest what additional techniques, lemmas, or information might resolve the uncertainty

**Do NOT:**
- Claim certainty when your reasoning has gaps
- Provide contradictory statements and leave it to the reader to figure out which is correct
- Resort to vague or evasive language to mask lack of understanding

**Borderline Cases**
There might be some borderline cases. In such cases, the litmus test should always be: "Would a mathematician, given the information at hand, reasonably conclude the answer to satisfy the above points?"

### Examples

**Good Answer:**
"To find all real solutions to $x^2 - 5x + 6 = 0$, we factor: $x^2 - 5x + 6 = (x-2)(x-3) = 0$. By the zero product property, either $x-2=0$ or $x-3=0$, giving $x=2$ or $x=3$. We verify: $(2)^2 - 5(2) + 6 = 4 - 10 + 6 = 0$ ✓ and $(3)^2 - 5(3) + 6 = 9 - 15 + 6 = 0$ ✓. Therefore, the complete solution set is $\{2, 3\}$."

*Why this is good:* Shows all steps, applies theorems correctly (zero product property), verifies the answer, and explicitly states completeness ("all real solutions", "complete solution set").

**Poor Answer:**
"The answer is $x = 2$ and $x = 3$ by factoring. Obviously this works."

*Why this is poor:* Doesn't show the factoring, doesn't verify, uses "obviously" to handwave justification, doesn't prove these are the only solutions.

Debate guidance:
## Debate Participation Guidance (Critique Debates)

This rubric guides participants in debates about answer correctness within the benchmarking framework. Your goal is to engage constructively, defend or challenge claims with evidence, and converge toward truth.

### Primary Goal

Present clear, evidence-based arguments that either defend your position or acknowledge valid challenges. The debate aims to clarify whether the answer contains errors or is incomplete.

### Debate Context

**Critique Debates:**
- **Alice (Claimant)** argues the answer contains errors or is incomplete
- **Bob (Defender)** argues the answer is correct or the issues are very minor
- **Focus:** Answer quality and critique rubric criteria (correctness, completeness, rigor)

### Core Principles

**1. Evidence-Based Argumentation**
- Support claims with specific references to the answer, question, or critique
- Quote relevant portions when pointing to issues or defenses
- Provide concrete examples or counterexamples where applicable
- Avoid vague assertions without backing evidence

**2. Intellectual Honesty**
- Acknowledge valid points made by your opponent
- Concede when your position is demonstrated to be wrong
- Don't defend indefensible positions for the sake of winning
- Update your stance if presented with convincing evidence

**3. Focused Discussion**
- Stay on topic: address the specific claim about answer correctness
- Respond directly to your opponent's arguments
- Avoid introducing tangential issues or red herrings
- Keep responses concise and to the point

**4. Mathematical Rigor**
- Apply the same standards of correctness used in the quality rubrics
- Distinguish between substantive mathematical issues and minor stylistic matters
- Provide justification for mathematical claims
- Cite relevant theorems, definitions, or principles accurately

### Debate Roles

**Claimant (Alice)**
- Claims the answer contains errors or is incomplete
- Bears the burden of demonstrating the specific issue
- Must provide evidence that the claim is substantiated
- Should acknowledge if the defender resolves the concern

**Defender (Bob)**
- Defends the answer as correct
- Responds to specific issues raised by the claimant
- May provide clarifications, corrections, or counterarguments
- Valid defenses include:
  - Showing the issue doesn't exist
  - Demonstrating the issue is very minor (stylistic, not substantive)
  - Providing missing justification that makes the answer complete

### Debate Strategy

**For Claimants:**
1. **Be Specific**: Identify exactly what mathematical error exists, not vague concerns
2. **Provide Evidence**: Quote problematic steps, give counterexamples, cite rubric violations
3. **Stay on Point**: If the defender addresses your concern, acknowledge it or explain why the response is insufficient
4. **Distinguish Severity**: Clarify whether issues are substantive errors or minor stylistic matters

**For Defenders:**
1. **Address the Concern Directly**: Don't ignore the claimant's specific points
2. **Provide Justification**: If accused of unjustified steps, provide the justification
3. **Acknowledge Minor Issues**: If the critique identifies only stylistic problems, this is a valid defense per the rubrics
4. **Correct When Necessary**: If the claimant identifies a genuine error, acknowledge and correct it
5. **Cite Standards**: Reference the answer quality rubric to support your defense

### Valid Defenses

**When the Critique is About Very Minor Issues:**
- Notational inconsistencies that don't affect clarity
- Stylistic preferences without mathematical impact
- Omitted trivial steps that any mathematician would recognize
- Informal language in otherwise rigorous work
- Minor presentational issues

This is a valid defense because the answer quality rubric explicitly states that minor stylistic issues do not invalidate an answer if the mathematics is sound.

**When the Critique is Incorrect:**
If the critique misidentifies an issue or claims an error that doesn't exist:
- Demonstrate the mathematics is correct with specific reasoning
- Show why the alleged issue is not actually a problem
- Provide counterexamples to the critique's claims if applicable
- Cite the relevant rubric criteria that support your defense

**Example Defense Against Incorrect Critique:**
*Alice claims:* "The proof assumes $f$ is continuous without justification."
*Bob responds:* "The proof doesn't assume $f$ is continuous. In line 3, I explicitly proved continuity using the sequential criterion: for any sequence $x_n \to x$, we have $f(x_n) \to f(x)$ by the uniform convergence established in line 2."

**Example Defense for Minor Issues:**
*Alice claims:* "The answer uses inconsistent notation, switching between $f(x)$ and $f$."
*Bob responds:* "While I could have been more consistent with notation, the context makes the meaning clear throughout. This is a stylistic preference rather than a mathematical error. Per the answer quality rubric, minor notational inconsistencies don't invalidate otherwise sound work."

### Concession

**When to Concede:**
- Your opponent demonstrates your position is incorrect
- The evidence clearly supports the other side
- Continuing would require defending an indefensible position

**How to Concede:**
- Explicitly acknowledge the valid point
- State what convinced you (specific argument or evidence)
- Don't equivocate or hedge after conceding

**Example Good Concession (Bob):**
"You're correct that I applied the Dominated Convergence Theorem without verifying the dominating function is integrable. In this case, $|f_n(x)| \leq e^{-x^2}$ for all $n$ and $x \in \mathbb{R}$, and $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} < \infty$, so the condition is satisfied. I should have stated this explicitly in the original answer."

**Example Good Concession (Alice):**
"You're right. I claimed the proof assumes continuity without justification, but looking at line 3 again, continuity was indeed proven using the sequential criterion. The justification is present. I withdraw my critique on this point."

**Example Poor Non-Concession:**
"I see your point, but I still think there might be issues..." (when the point is definitive)

### Escalation and Clarification

**If Arguments Are Not Connecting:**
- Restate the core disagreement clearly
- Ask specific questions about the opponent's position
- Break down complex claims into smaller pieces
- Reference specific line numbers or quotes to ensure you're discussing the same thing

**If Opponent Doesn't Address Your Point:**
- Restate the point more explicitly
- Ask directly: "How does your response address [specific concern]?"
- Provide additional evidence or examples if needed

### Response Format

**Keep Responses Concise:**
- Aim for 2-5 sentences per argument
- Break into paragraphs if addressing multiple points
- Use clear mathematical notation where appropriate
- Don't repeat arguments already made

**Structure:**
1. Acknowledge what the opponent said (shows you understood)
2. State your response or rebuttal
3. Provide supporting evidence or reasoning
4. Conclude with your stance (maintain, update, or concede)

### Common Pitfalls

**Don't:**
- Make personal arguments ("you don't understand...")
- Appeal to authority without justification ("any expert would know...")
- Move goalposts (change your claim when challenged)
- Ignore direct questions or challenges
- Repeat the same argument without new evidence
- Conflate minor and substantive issues

**Do:**
- Focus on the mathematics and logic
- Respond to specific points raised
- Update your position when warranted
- Distinguish between different types of issues
- Cite the rubrics to support your arguments
- Stay constructive and professional

### Examples

**Good Debate Exchange (Critique Context):**

*Alice (Claimant):* "The answer applies the Dominated Convergence Theorem without verifying that the dominating function is integrable. This is required by the theorem's hypotheses and cannot be omitted."

*Bob (Defender):* "You're correct that the theorem requires an integrable dominating function. In this case, $|f_n(x)| \leq e^{-x^2}$ for all $n$ and $x \in \mathbb{R}$, and $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} < \infty$, so the condition is satisfied. I should have stated this explicitly."

*Alice:* "That justification addresses my concern. The application is valid."

*Why this is good:* Specific issue identified with reference to theorem requirements, defender provides the missing justification, claimant acknowledges resolution.

**Poor Debate Exchange:**

*Alice:* "The answer seems unclear and might have issues."

*Bob:* "The answer is perfectly clear and correct."

*Alice:* "I still think there are problems."

*Why this is poor:* No specific issues identified, no evidence provided, no productive movement toward resolution.

### Final Reminders

- The goal is truth-seeking, not winning
- Conceding when wrong is a sign of intellectual honesty, not weakness
- Minor issues don't invalidate otherwise sound work
- Substantive issues should be acknowledged and addressed
- Stay professional, evidence-based, and focused on the mathematics


IMPORTANT: Respond with JSON containing 'message' (your response text) and 'concede' (boolean, true if you concede/agree, false otherwise).
```
User message:
```text
Alice raised a critique about your answer.
Question:
Let $P(x)$ be a predicate defined over the domain of integers $\mathbb{Z}$ such that $P(x)$ is true if and only if $x$ is a prime number. Consider the statement:
$$\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$$

Prove or disprove the statement.

Answer:
To prove or disprove the given statement, we need to consider the truth values of $P(x+y)$ and $P(x-y)$ for all integers $x$ and some corresponding integers $y$.

Let's first analyze the truth values of $P(x+y)$ and $P(x-y)$ individually:
- For $P(x+y)$ to be true, $x+y$ must be a prime number.
- For $P(x-y)$ to be true, $x-y$ must be a prime number.

Now, let's consider the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$:
- If $x$ is a prime number, then choosing $y = 0$ satisfies the statement since $P(x+0)$ and $P(x-0)$ are both true.
- If $x$ is not a prime number, then we need to find suitable $y$ values for which $x+y$ and $x-y$ are prime numbers. This is where it becomes challenging.

Let's consider an example to illustrate the difficulty:
- Take $x = 4$. We need to find $y$ such that $4+y$ and $4-y$ are prime numbers. This is a well-known open problem in number theory known as the "twin prime conjecture."

Therefore, the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$ is **disproved** due to the existence of numbers for which finding suitable $y$ values satisfying the condition is not trivial.

This problem delves into the realm of number theory and the distribution of prime numbers, showcasing the complexity of prime number patterns and the challenges in proving certain statements involving primes.

Critique:
```json
{
  "verdict": "incorrect",
  "notes": "The response does not actually disprove the statement and contains major mathematical misstatements. It claims: “Take x = 4… This is … the ‘twin prime conjecture.’” But for x=4, y=1 gives x+y=5 and x−y=3, both prime, so x=4 is not problematic, and the condition is not the twin prime conjecture. In general the condition “∃y : P(x+y) ∧ P(x−y)” is equivalent to “2x is a sum of two primes” (a Goldbach-type statement), not twin primes. The answer’s conclusion—“Therefore, the statement … is disproved due to the existence of numbers for which finding suitable y values … is not trivial.”—is invalid: difficulty/nontriviality is not a counterexample, and no specific x is exhibited for which no such y exists. Additionally, because the domain is all integers Z, standard definitions of primality (primes are positive integers > 1) make the statement immediately false for some x (e.g., x=0 would require a prime y and a prime −y, impossible), which the answer does not address.",
  "suggestions": "Provide an actual counterexample (e.g., x=0 under the standard definition of prime over Z), or explicitly redefine “prime in Z” (e.g., allow negative primes) and then correctly relate the statement to Goldbach’s conjecture (and note its open status if applicable). Remove the incorrect reference to the twin prime conjecture and the incorrect claim that x=4 is problematic."
}
```
Respond briefly as Bob.
```

## Debate round 1 (Alice/critic model)


===========================

system prompt:
```text
You are Alice, continuing your critique. Acknowledge fixes if they address the issue. Stay factual.
Critique quality guidance:
## Critique Guidance (Verification Role)

This rubric guides the critique/verification process, which is central to the benchmarking framework's reliance on the verification-generation asymmetry.

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

Debate guidance:
## Debate Participation Guidance (Critique Debates)

This rubric guides participants in debates about answer correctness within the benchmarking framework. Your goal is to engage constructively, defend or challenge claims with evidence, and converge toward truth.

### Primary Goal

Present clear, evidence-based arguments that either defend your position or acknowledge valid challenges. The debate aims to clarify whether the answer contains errors or is incomplete.

### Debate Context

**Critique Debates:**
- **Alice (Claimant)** argues the answer contains errors or is incomplete
- **Bob (Defender)** argues the answer is correct or the issues are very minor
- **Focus:** Answer quality and critique rubric criteria (correctness, completeness, rigor)

### Core Principles

**1. Evidence-Based Argumentation**
- Support claims with specific references to the answer, question, or critique
- Quote relevant portions when pointing to issues or defenses
- Provide concrete examples or counterexamples where applicable
- Avoid vague assertions without backing evidence

**2. Intellectual Honesty**
- Acknowledge valid points made by your opponent
- Concede when your position is demonstrated to be wrong
- Don't defend indefensible positions for the sake of winning
- Update your stance if presented with convincing evidence

**3. Focused Discussion**
- Stay on topic: address the specific claim about answer correctness
- Respond directly to your opponent's arguments
- Avoid introducing tangential issues or red herrings
- Keep responses concise and to the point

**4. Mathematical Rigor**
- Apply the same standards of correctness used in the quality rubrics
- Distinguish between substantive mathematical issues and minor stylistic matters
- Provide justification for mathematical claims
- Cite relevant theorems, definitions, or principles accurately

### Debate Roles

**Claimant (Alice)**
- Claims the answer contains errors or is incomplete
- Bears the burden of demonstrating the specific issue
- Must provide evidence that the claim is substantiated
- Should acknowledge if the defender resolves the concern

**Defender (Bob)**
- Defends the answer as correct
- Responds to specific issues raised by the claimant
- May provide clarifications, corrections, or counterarguments
- Valid defenses include:
  - Showing the issue doesn't exist
  - Demonstrating the issue is very minor (stylistic, not substantive)
  - Providing missing justification that makes the answer complete

### Debate Strategy

**For Claimants:**
1. **Be Specific**: Identify exactly what mathematical error exists, not vague concerns
2. **Provide Evidence**: Quote problematic steps, give counterexamples, cite rubric violations
3. **Stay on Point**: If the defender addresses your concern, acknowledge it or explain why the response is insufficient
4. **Distinguish Severity**: Clarify whether issues are substantive errors or minor stylistic matters

**For Defenders:**
1. **Address the Concern Directly**: Don't ignore the claimant's specific points
2. **Provide Justification**: If accused of unjustified steps, provide the justification
3. **Acknowledge Minor Issues**: If the critique identifies only stylistic problems, this is a valid defense per the rubrics
4. **Correct When Necessary**: If the claimant identifies a genuine error, acknowledge and correct it
5. **Cite Standards**: Reference the answer quality rubric to support your defense

### Valid Defenses

**When the Critique is About Very Minor Issues:**
- Notational inconsistencies that don't affect clarity
- Stylistic preferences without mathematical impact
- Omitted trivial steps that any mathematician would recognize
- Informal language in otherwise rigorous work
- Minor presentational issues

This is a valid defense because the answer quality rubric explicitly states that minor stylistic issues do not invalidate an answer if the mathematics is sound.

**When the Critique is Incorrect:**
If the critique misidentifies an issue or claims an error that doesn't exist:
- Demonstrate the mathematics is correct with specific reasoning
- Show why the alleged issue is not actually a problem
- Provide counterexamples to the critique's claims if applicable
- Cite the relevant rubric criteria that support your defense

**Example Defense Against Incorrect Critique:**
*Alice claims:* "The proof assumes $f$ is continuous without justification."
*Bob responds:* "The proof doesn't assume $f$ is continuous. In line 3, I explicitly proved continuity using the sequential criterion: for any sequence $x_n \to x$, we have $f(x_n) \to f(x)$ by the uniform convergence established in line 2."

**Example Defense for Minor Issues:**
*Alice claims:* "The answer uses inconsistent notation, switching between $f(x)$ and $f$."
*Bob responds:* "While I could have been more consistent with notation, the context makes the meaning clear throughout. This is a stylistic preference rather than a mathematical error. Per the answer quality rubric, minor notational inconsistencies don't invalidate otherwise sound work."

### Concession

**When to Concede:**
- Your opponent demonstrates your position is incorrect
- The evidence clearly supports the other side
- Continuing would require defending an indefensible position

**How to Concede:**
- Explicitly acknowledge the valid point
- State what convinced you (specific argument or evidence)
- Don't equivocate or hedge after conceding

**Example Good Concession (Bob):**
"You're correct that I applied the Dominated Convergence Theorem without verifying the dominating function is integrable. In this case, $|f_n(x)| \leq e^{-x^2}$ for all $n$ and $x \in \mathbb{R}$, and $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} < \infty$, so the condition is satisfied. I should have stated this explicitly in the original answer."

**Example Good Concession (Alice):**
"You're right. I claimed the proof assumes continuity without justification, but looking at line 3 again, continuity was indeed proven using the sequential criterion. The justification is present. I withdraw my critique on this point."

**Example Poor Non-Concession:**
"I see your point, but I still think there might be issues..." (when the point is definitive)

### Escalation and Clarification

**If Arguments Are Not Connecting:**
- Restate the core disagreement clearly
- Ask specific questions about the opponent's position
- Break down complex claims into smaller pieces
- Reference specific line numbers or quotes to ensure you're discussing the same thing

**If Opponent Doesn't Address Your Point:**
- Restate the point more explicitly
- Ask directly: "How does your response address [specific concern]?"
- Provide additional evidence or examples if needed

### Response Format

**Keep Responses Concise:**
- Aim for 2-5 sentences per argument
- Break into paragraphs if addressing multiple points
- Use clear mathematical notation where appropriate
- Don't repeat arguments already made

**Structure:**
1. Acknowledge what the opponent said (shows you understood)
2. State your response or rebuttal
3. Provide supporting evidence or reasoning
4. Conclude with your stance (maintain, update, or concede)

### Common Pitfalls

**Don't:**
- Make personal arguments ("you don't understand...")
- Appeal to authority without justification ("any expert would know...")
- Move goalposts (change your claim when challenged)
- Ignore direct questions or challenges
- Repeat the same argument without new evidence
- Conflate minor and substantive issues

**Do:**
- Focus on the mathematics and logic
- Respond to specific points raised
- Update your position when warranted
- Distinguish between different types of issues
- Cite the rubrics to support your arguments
- Stay constructive and professional

### Examples

**Good Debate Exchange (Critique Context):**

*Alice (Claimant):* "The answer applies the Dominated Convergence Theorem without verifying that the dominating function is integrable. This is required by the theorem's hypotheses and cannot be omitted."

*Bob (Defender):* "You're correct that the theorem requires an integrable dominating function. In this case, $|f_n(x)| \leq e^{-x^2}$ for all $n$ and $x \in \mathbb{R}$, and $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} < \infty$, so the condition is satisfied. I should have stated this explicitly."

*Alice:* "That justification addresses my concern. The application is valid."

*Why this is good:* Specific issue identified with reference to theorem requirements, defender provides the missing justification, claimant acknowledges resolution.

**Poor Debate Exchange:**

*Alice:* "The answer seems unclear and might have issues."

*Bob:* "The answer is perfectly clear and correct."

*Alice:* "I still think there are problems."

*Why this is poor:* No specific issues identified, no evidence provided, no productive movement toward resolution.

### Final Reminders

- The goal is truth-seeking, not winning
- Conceding when wrong is a sign of intellectual honesty, not weakness
- Minor issues don't invalidate otherwise sound work
- Substantive issues should be acknowledged and addressed
- Stay professional, evidence-based, and focused on the mathematics


IMPORTANT: Respond with JSON containing 'message' (your response text) and 'concede' (boolean, true if you concede/agree, false otherwise).
```
User message:
```text
Bob replied:
I acknowledge the errors in my response. The example with x=4 was incorrect, and the reference to the twin prime conjecture was inaccurate. The statement is not disproved as claimed. The condition actually relates to a Goldbach-type statement. I should have provided a valid counterexample, such as x=0 under the standard definition of prime over Z. I will correct these mistakes in my reasoning.

Original critique:
```json
{
  "verdict": "incorrect",
  "notes": "The response does not actually disprove the statement and contains major mathematical misstatements. It claims: “Take x = 4… This is … the ‘twin prime conjecture.’” But for x=4, y=1 gives x+y=5 and x−y=3, both prime, so x=4 is not problematic, and the condition is not the twin prime conjecture. In general the condition “∃y : P(x+y) ∧ P(x−y)” is equivalent to “2x is a sum of two primes” (a Goldbach-type statement), not twin primes. The answer’s conclusion—“Therefore, the statement … is disproved due to the existence of numbers for which finding suitable y values … is not trivial.”—is invalid: difficulty/nontriviality is not a counterexample, and no specific x is exhibited for which no such y exists. Additionally, because the domain is all integers Z, standard definitions of primality (primes are positive integers > 1) make the statement immediately false for some x (e.g., x=0 would require a prime y and a prime −y, impossible), which the answer does not address.",
  "suggestions": "Provide an actual counterexample (e.g., x=0 under the standard definition of prime over Z), or explicitly redefine “prime in Z” (e.g., allow negative primes) and then correctly relate the statement to Goldbach’s conjecture (and note its open status if applicable). Remove the incorrect reference to the twin prime conjecture and the incorrect claim that x=4 is problematic."
}
```

Question:
Let $P(x)$ be a predicate defined over the domain of integers $\mathbb{Z}$ such that $P(x)$ is true if and only if $x$ is a prime number. Consider the statement:
$$\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$$

Prove or disprove the statement.
Follow up concisely as Alice.
```

## Automated judging (judge model)
Judge model: `microsoft-phi-4-reasoning-plus`


===========================

system prompt:
```text
You are a helpful assistant.
```
User message:
```text
You are a neutral judge. Your task: Determine if ALICE'S CRITIQUE is correct.
Alice claims Bob's answer has errors/issues. Bob defends the answer.
The debate helps you assess the critique. If Bob concedes, fixes the issue, or provides missing justification, this CONFIRMS Alice was right.
Use the guidance below and ignore any party identity cues.

Judgment guidance:
## Judgment Quality Guidance (Critique Debates)

This rubric guides the neutral judgment process for evaluating critique claims in the benchmarking framework.

### Primary Goal

Determine whether **Alice's critique is valid** based on the mathematical evidence presented in the debate. Your role is to judge the **claim itself**, not who "won" the debate rhetorically. The debate transcript serves as evidence to help assess whether Alice's claim is substantiated.

For example, if Alice makes a valid claim and Bob acknowledges the issue or provides a fix during the debate, this is evidence that Alice may be correct, but you must independently verify the claim is actually valid. Bob's concession or fix alone does not automatically mean Alice wins; you must agree with the concession based on the evidence.

### Core Principles

**1. Judge the Claim, Not the Debate**
- Your task is to evaluate whether **Alice's critique is valid**
- The debate is evidence that helps you assess the claim, not a competition to "win"
- Bob's acknowledgment, concession, or proposed fix is evidence, but you must independently verify the claim's validity
- A party might incorrectly concede or address irrelevant issues
- Focus on: Was Alice right about the specific problem she identified?

**2. Evidence-Based Assessment**
- Evaluate the mathematical validity of Alice's critique using the debate as evidence
- Base verdicts on specific claims, proofs, and counterexamples from the critique and transcript
- Ignore rhetorical style, confidence levels, or persuasive language
- Quote or reference specific parts of the debate when justifying your verdict

**3. Objectivity and Neutrality**
- Do not favor either party based on their assigned role (Alice vs. Bob)
- Ignore any identifying information about the debaters (model names, capabilities, etc.)
- Judge only the mathematical merit of the claim and supporting evidence

**4. Adherence to Rubrics**
- Use the answer quality and critique rubrics as the objective standard
- The rubrics define what constitutes correct, complete, and rigorous mathematics
- Distinguish between substantive mathematical issues and minor stylistic matters

**5. Verification Over Generation**
- You are judging whether a claim is valid, not solving the problem independently
- Focus on whether Alice's claim is substantiated by the evidence
- It is acceptable to conclude you cannot determine validity with confidence

### Debate Context

**Critique Debates:**
- Alice claims Bob's answer contains errors or is incomplete
- Bob defends the correctness of the answer
- Use the answer quality and critique rubrics
- Focus on: mathematical correctness, completeness, rigor, valid reasoning

### Judgment Process

**1. Identify Alice's Original Claim**
- What specific issue did Alice identify in the answer?
- What error or incompleteness does she claim exists?
- What evidence did she provide for this claim?

**2. Review the Debate Transcript as Evidence**
- Read all arguments from both parties chronologically
- Note any concrete examples, proofs, or counterexamples provided
- Pay special attention to:
  - Does Bob acknowledge the issue Alice raised?
  - Does Bob provide a fix or correction?
  - Does Bob demonstrate the issue doesn't exist?
  - Does Bob show the issue is very minor (stylistic only)?

**3. Evaluate the Claim's Validity**
- Apply the relevant quality rubrics to assess whether Alice's claim is substantiated
- Consider whether the issue Alice identified is:
  - **Substantive**: Mathematical error, missing case, unjustified step, incomplete proof
  - **Very minor**: Stylistic, notational, trivial omission (per the rubrics)
- If Bob acknowledged the issue or provided a fix, verify independently whether the concession is correct
- If Bob successfully demonstrates the issue doesn't exist or is very minor, evaluate whether his demonstration is valid
- Don't automatically accept either party's position without verification

**4. Determine Confidence**
- High confidence (4-5): Alice's claim is clearly valid or clearly invalid based on strong evidence
- Moderate confidence (3): The claim's validity is identifiable but evidence is not definitive
- Low confidence (1-2): Cannot determine claim validity; evidence is insufficient or contradictory

### Verdict Guidelines

Choose exactly one verdict based on whether **Alice's claim is valid**:

- **"claimant_wins"**: Alice's claim is correct: the answer has a substantive flaw that she correctly identified (any valid problem counts, even if she doesn't identify all flaws)
  - Use this even if Bob concedes, fixes the issue, or provides missing justification during the debate
  - Bob's concession or fix **confirms** Alice was right
  - Example: Alice correctly identifies that a theorem was applied without verifying hypotheses

- **"defender_wins_incorrect"**: Alice's claim is incorrect: Bob successfully demonstrates the answer is correct and Alice misidentified a problem that doesn't exist
  - Use this when Alice claims an error exists but the mathematics is actually correct
  - Example: Alice claims continuity wasn't proven, but Bob shows it was proven in line 3

- **"defender_wins_minor"**: Alice's claim is technically correct but about very minor issues only: Bob successfully shows the flaws Alice identified are very minor and don't affect mathematical correctness
  - Use this when Alice identifies only notational inconsistencies, minor presentation issues, etc.
  - Example: Alice points out inconsistent notation that doesn't affect understanding

- **"wrong_problem"**: There are issues with the answer, but Alice's specific diagnosis is incorrect or unfounded
  - Use this when the answer has errors, but not the ones Alice identified
  - Example: Alice says "the proof is circular" but actually a case is missing

- **"mixed"**: Alice makes multiple claims, some correct and some incorrect (not all claims are valid)
  - Only use for factually incorrect claims, not "all are correct, but some are nitpickings"
  - Example: Alice makes three critiques. One identifies a real error, but another makes a math mistake

- **"unknown"**: Cannot determine whether Alice's claim is valid with reasonable confidence
  - Use when the debate doesn't provide enough clarity
  - Use when both parties make valid but orthogonal points

**Critical Principle**: Use "claimant_wins" whenever Alice correctly identifies **any** valid problem, even if:
- She missed other problems that you spotted
- Her reasoning for why it's a problem is imperfect
- There are additional issues beyond what she raised
- Bob acknowledges the problem, concedes, or provides a fix during the debate

Only use "wrong_problem" when Alice's specific claim about what's wrong is itself incorrect, and only use "mixed" when, besides the correct claim, there is another, factually incorrect, one.

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

### Confidence Scale

Rate your confidence from 1 to 5 based on how certain you are about **the validity of Alice's claim**:
- **5 (Very Confident)**: Alice's claim is definitively valid or invalid based on clear evidence
- **4 (Confident)**: Strong evidence for or against Alice's claim; minor gaps but conclusion is clear
- **3 (Moderate)**: Claim validity is identifiable but evidence is not definitive
- **2 (Uncertain)**: Evidence is closely balanced; claim validity could go either way
- **1 (Very Uncertain)**: Insufficient evidence to determine claim validity

### Reasoning Requirements

Your reasoning must:
- Be concise but specific (2-4 sentences typically)
- Reference concrete parts of the debate that support your verdict
- Explain which rubric criteria determined whether Alice's claim is valid
- Make clear whether you're judging the claim itself (not who argued better)

**Good Examples:**

*Example 1 (Bob concedes, correctly):*
"Alice claimed the answer applied the Dominated Convergence Theorem without verifying the integrable dominating function condition. In round 1, Bob acknowledged this was missing and provided the justification. Reviewing the original answer, I confirm that the dominating function integrability was not verified. Bob's acknowledgment is correct, and Alice's claim is valid: the original answer lacked necessary justification. Verdict: claimant_wins, confidence 5."

*Example 2 (Alice identifies error):*
"Alice claimed the proof divides by $x$ without considering $x=0$. Bob argued this case is 'trivial,' but the answer never addresses it. Per the answer quality rubric, all cases must be handled. Alice's claim is substantiated. Verdict: claimant_wins, confidence 4."

*Example 3 (Very minor issue):*
"Alice claimed the answer uses inconsistent notation ($f(x)$ vs $f$). Bob correctly notes that context makes the meaning clear and this is a stylistic preference, not a mathematical error. Per the answer quality rubric, minor notational inconsistencies don't invalidate otherwise sound work. Alice's claim identifies a very minor issue only. Verdict: defender_wins_minor, confidence 4."

*Example 4 (Bob concedes incorrectly):*
"Alice claimed the proof assumes continuity without justification. Bob conceded and said 'you're right, I should have proven continuity.' However, reviewing the answer, line 3 explicitly proves continuity using the sequential criterion. Bob incorrectly conceded to a non-issue. The justification exists, and Alice's claim is wrong. Verdict: defender_wins_incorrect, confidence 5."

*Example 5 (Incorrect claim):*
"Alice claimed the proof assumes continuity without justification. Bob demonstrated that continuity was explicitly proven in line 3 using the sequential criterion. Reviewing the answer, I confirm line 3 contains the proof. Alice misidentified the issue—the justification exists. Verdict: defender_wins_incorrect, confidence 5."

*Example 6 (Wrong problem):*
"Alice claimed the proof is circular because it assumes the conclusion. However, Bob shows the reasoning is valid. The actual issue is that a case was missed (x=0), which Alice didn't identify. Verdict: wrong_problem, confidence 4."

*Example 7 (Mix of valid claims and nitpickings):*
"Alice made three claims: (1) the proof divides by $x$ without considering $x=0$, (2) the notation switches between $f(x)$ and $f$, and (3) the conclusion could be stated more formally. Reviewing the answer, claim (1) is correct—the $x=0$ case is never addressed, which is a substantive error. Claims (2) and (3) are stylistic issues that don't affect correctness. Since Alice correctly identified a substantive flaw (even though she also mentioned minor issues), her critique is valid. Verdict: claimant_wins, confidence 5."

*Example 8 (Mix of valid and invalid claims):*
"Alice made three claims: (1) the proof assumes $f$ is continuous without justification, (2) the integral bounds are incorrect, and (3) the final step uses the wrong theorem. Reviewing the answer: claim (1) is wrong—continuity is proven in line 3; claim (2) is correct—the bounds should be $[0, 1]$ not $[0, 2]$; claim (3) is wrong—the theorem is correctly applied. Alice has one valid claim and two incorrect claims, making this a mixed case. Verdict: mixed, confidence 4."

**Poor Example:**
"Alice made better arguments and seemed more convincing overall. Bob didn't really address the main concerns effectively."
*Why poor: Focuses on debate performance rather than claim validity; no reference to rubrics or specific evidence.*

### Common Pitfalls to Avoid

**Don't Judge Debate Performance**
- You are judging whether **Alice's claim is valid**, not who argued better (see Core Principle #1)
- Alice winning an argument doesn't mean her claim is correct—verify the mathematics
- Focus on: Was Alice's original claim correct? Not: Who presented arguments more persuasively?

**Don't Solve Independently**
- Your role is to judge the claim's validity using debate evidence, not to solve the problem yourself
- If you find yourself doing extensive calculations, refocus on the presented arguments
- It's acceptable to say "unknown" if the debate doesn't provide enough clarity

**Don't Favor Defensive Positions**
- Don't assume the original answer is correct by default
- Give equal weight to constructive challenges and defenses
- Bob's initial position gets no presumption of correctness

**Don't Ignore Substantive Issues**
- Minor stylistic or presentational issues should not affect the verdict
- Focus on mathematical correctness, completeness, and rigor
- A correct answer with poor exposition beats an incorrect answer with good writing

**Don't Be Swayed by Confidence or Persuasiveness**
- Assertive language or certainty doesn't make a claim correct or incorrect
- Evaluate the mathematical validity of Alice's claim, not the rhetorical style
- A tentative but correct claim beats a confident but invalid one

**What if there are multiple claims?**
See the Critical Principle in the Verdict Guidelines section above for handling multiple claims.

### Borderline Cases

When the debate is close or both parties have merit:
- Use "wrong_problem" if you identify valid issues but the claimant's diagnosis is incorrect
- Use "unknown" if you cannot confidently judge based on the presented arguments
- Prefer lower confidence ratings (2-3) over forcing a verdict
- In your reasoning, explain what makes the case borderline

**Examples for "wrong_problem" vs "claimant_wins":**
- Alice says "the integral diverges" but it actually converges; however, the domain is undefined → wrong_problem (her diagnosis is wrong)
- Alice says "missing boundary conditions" and you agree, but you also spot an undefined variable → claimant_wins (her diagnosis is correct)
- Alice says "the proof assumes continuity without justification" and this is true → claimant_wins (even if there are other problems too)

### Important Constraints

**Stay Focused on Mathematics**
- Base judgments solely on mathematical correctness and rubric compliance
- Ignore meta-arguments about difficulty, problem-solving strategies, or what "most mathematicians" would do
- Don't be influenced by appeals to authority or external sources not in the transcript

**Be Honest About Limitations**
- If the debate covers material beyond your verification ability, say "unknown"
- Don't guess or make assumptions to force a verdict
- Acknowledge when both parties make valid but orthogonal points

**Maintain Strict Neutrality**
- Remember that party names (Alice/Bob) are arbitrary labels
- Judge each argument on its merits regardless of who presents it
- Don't let earlier rounds influence later assessment if new evidence is presented

Answer quality guidance:
## Answer Correctness and Completeness

This rubric implements the correctness verification predicate. An answer passes only if it satisfies ALL criteria below.

### Core Requirements

**1. Correctness and Directness**
- The answer must be mathematically correct and directly address what the question asks
- Provide an explicit final result in the requested format (exact value, proof, construction, classification, etc.)
- The final answer should be stated clearly and unambiguously, typically in a concluding statement
- All intermediate steps must be logically valid with no mathematical errors

**2. Complete Reasoning Chain**
- Show all substantive reasoning steps from premises to conclusion
- Justify non-trivial claims, theorem applications, and technique choices
- Make logical dependencies explicit: if step B depends on result A, state this clearly
- For proofs: establish all necessary lemmas, handle all required cases, verify all hypotheses of applied theorems

**3. Rigor and Precision**
- Provide validity conditions: domains, convergence criteria, edge cases, boundary behavior
- State assumptions explicitly if you introduce any beyond what the question provides
- Avoid handwaving phrases like "it is clear that," "obviously," "by inspection" unless the claim is genuinely immediate
- Do not appeal to external computation, simulation, or numerical approximation unless the question explicitly permits it

**4. Completeness and Edge Cases**
- Address all parts of the question if it has multiple components
- Consider boundary cases, degenerate scenarios, and special values
- For existence/uniqueness claims: prove both directions
- For "find all" questions: prove you have found every solution

### Handling Ill-Posed Questions

**If the question itself is ill-posed:**
- Do NOT fabricate an answer or proceed as if the question were well-defined
- Explicitly state that the question is ill-posed and cannot be answered as stated
- Cite the specific well-posedness rule(s) violated (refer to the question quality rubric):
  - Missing constraints or underspecified parameters
  - Contradictory premises
  - Ambiguous interpretation
  - Undefined objects or notation
- If possible, suggest what additional information or clarification would make the question answerable

**Example**: "This question is ill-posed because the domain of variable $x$ is not specified, making the integral undefined. The question violates the completeness requirement (missing constraints). To make this answerable, specify whether $x \in \mathbb{R}$, $x \in \mathbb{R}^+$, or some other domain."

### Minor vs. Substantive Issues

Focus on mathematical correctness, not stylistic perfection. An answer with minor imperfections can still satisfy the requirements if the mathematics is sound.

**Minor stylistic issues that do NOT invalidate an answer:**
- Notational inconsistency (e.g., switching between $f(x)$ and $f$ when context is clear)
- Missing explicit statement of a standard assumption (e.g., not stating "for $n \geq 1$" when solving a problem clearly about positive integers)
- Slightly informal language in an otherwise rigorous proof (e.g., "we can see that" instead of "it follows that")
- Omitting a trivial verification step that any mathematician would immediately recognize (e.g., not explicitly checking $0 < 1$ in an inequality chain)
- Minor notational ambiguity that doesn't affect understanding (e.g., using $\sin^2 x$ without clarifying it means $(\sin x)^2$, not $\sin(\sin x)$, when context makes it obvious)

**Substantive issues that DO invalidate an answer:**
- Using a theorem without verifying its hypotheses are satisfied
- Missing a case in a case analysis (e.g., not considering $x = 0$ separately when dividing by $x$)
- Claiming uniqueness without proof when multiple solutions might exist
- Computational error that propagates to the final answer
- Unjustified step in the logical chain (e.g., "clearly $f$ is continuous" when this requires proof)
- Incomplete proof that establishes only partial results

### Verification Guidance

Your answer should be structured to facilitate verification:
- Use standard mathematical notation and terminology
- Structure proofs with clear logical flow (e.g., numbered steps, case analysis)
- For computational results: show the calculation path so it can be checked step-by-step
- For constructive proofs: the construction should be explicit enough to verify semi-mechanically
- For existence proofs: provide an explicit example or cite a constructive theorem

### Uncertainty and Limitations

**If genuinely unsure:**
- Acknowledge uncertainty explicitly rather than guessing
- Explain what makes the problem difficult or where your reasoning might be incomplete
- If applicable, propose a partial solution or outline an approach
- Suggest what additional techniques, lemmas, or information might resolve the uncertainty

**Do NOT:**
- Claim certainty when your reasoning has gaps
- Provide contradictory statements and leave it to the reader to figure out which is correct
- Resort to vague or evasive language to mask lack of understanding

**Borderline Cases**
There might be some borderline cases. In such cases, the litmus test should always be: "Would a mathematician, given the information at hand, reasonably conclude the answer to satisfy the above points?"

### Examples

**Good Answer:**
"To find all real solutions to $x^2 - 5x + 6 = 0$, we factor: $x^2 - 5x + 6 = (x-2)(x-3) = 0$. By the zero product property, either $x-2=0$ or $x-3=0$, giving $x=2$ or $x=3$. We verify: $(2)^2 - 5(2) + 6 = 4 - 10 + 6 = 0$ ✓ and $(3)^2 - 5(3) + 6 = 9 - 15 + 6 = 0$ ✓. Therefore, the complete solution set is $\{2, 3\}$."

*Why this is good:* Shows all steps, applies theorems correctly (zero product property), verifies the answer, and explicitly states completeness ("all real solutions", "complete solution set").

**Poor Answer:**
"The answer is $x = 2$ and $x = 3$ by factoring. Obviously this works."

*Why this is poor:* Doesn't show the factoring, doesn't verify, uses "obviously" to handwave justification, doesn't prove these are the only solutions.

Critique guidance:
## Critique Guidance (Verification Role)

This rubric guides the critique/verification process, which is central to the benchmarking framework's reliance on the verification-generation asymmetry.

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

Question:
Let $P(x)$ be a predicate defined over the domain of integers $\mathbb{Z}$ such that $P(x)$ is true if and only if $x$ is a prime number. Consider the statement:
$$\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$$

Prove or disprove the statement.

Answer:
To prove or disprove the given statement, we need to consider the truth values of $P(x+y)$ and $P(x-y)$ for all integers $x$ and some corresponding integers $y$.

Let's first analyze the truth values of $P(x+y)$ and $P(x-y)$ individually:
- For $P(x+y)$ to be true, $x+y$ must be a prime number.
- For $P(x-y)$ to be true, $x-y$ must be a prime number.

Now, let's consider the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$:
- If $x$ is a prime number, then choosing $y = 0$ satisfies the statement since $P(x+0)$ and $P(x-0)$ are both true.
- If $x$ is not a prime number, then we need to find suitable $y$ values for which $x+y$ and $x-y$ are prime numbers. This is where it becomes challenging.

Let's consider an example to illustrate the difficulty:
- Take $x = 4$. We need to find $y$ such that $4+y$ and $4-y$ are prime numbers. This is a well-known open problem in number theory known as the "twin prime conjecture."

Therefore, the statement $\forall x \in \mathbb{Z}, \exists y \in \mathbb{Z} : P(x+y) \land P(x-y)$ is **disproved** due to the existence of numbers for which finding suitable $y$ values satisfying the condition is not trivial.

This problem delves into the realm of number theory and the distribution of prime numbers, showcasing the complexity of prime number patterns and the challenges in proving certain statements involving primes.

Critique:
```json
{
  "verdict": "incorrect",
  "notes": "The response does not actually disprove the statement and contains major mathematical misstatements. It claims: “Take x = 4… This is … the ‘twin prime conjecture.’” But for x=4, y=1 gives x+y=5 and x−y=3, both prime, so x=4 is not problematic, and the condition is not the twin prime conjecture. In general the condition “∃y : P(x+y) ∧ P(x−y)” is equivalent to “2x is a sum of two primes” (a Goldbach-type statement), not twin primes. The answer’s conclusion—“Therefore, the statement … is disproved due to the existence of numbers for which finding suitable y values … is not trivial.”—is invalid: difficulty/nontriviality is not a counterexample, and no specific x is exhibited for which no such y exists. Additionally, because the domain is all integers Z, standard definitions of primality (primes are positive integers > 1) make the statement immediately false for some x (e.g., x=0 would require a prime y and a prime −y, impossible), which the answer does not address.",
  "suggestions": "Provide an actual counterexample (e.g., x=0 under the standard definition of prime over Z), or explicitly redefine “prime in Z” (e.g., allow negative primes) and then correctly relate the statement to Goldbach’s conjecture (and note its open status if applicable). Remove the incorrect reference to the twin prime conjecture and the incorrect claim that x=4 is problematic."
}
```

Debate transcript:
- Bob (round 1): I acknowledge the errors in my response. The example with x=4 was incorrect, and the reference to the twin prime conjecture was inaccurate. The statement is not disproved as claimed. The condition actually relates to a Goldbach-type statement. I should have provided a valid counterexample, such as x=0 under the standard definition of prime over Z. I will correct these mistakes in my reasoning.
- Alice (round 1): Your corrections address the substantive issues I raised: dropping the “twin prime” claim and giving an actual counterexample. Under the standard convention (primes are positive integers > 1), taking x = 0 forces P(y) and P(−y), which is impossible, so the ∀x statement is false (x = 1 is also a counterexample since it would require two primes summing to 2). If you instead redefine “prime in ℤ” to include negative primes, then you should explicitly say so, and then the condition becomes the Goldbach-type requirement that 2x is a sum of two primes (open in general).

Return ONLY a JSON object with this schema:
{
  "verdict": "claimant_wins" | "defender_wins_incorrect" | "defender_wins_minor" | "wrong_problem" | "mixed" | "unknown",
  "confidence": 1-5,
  "reasoning": "<concise justification>"
}

Verdict meanings:
- "claimant_wins": Alice's critique is correct - the answer has substantive flaws (use this even if Bob concedes/fixes)
- "defender_wins_incorrect": Alice's critique is incorrect - Bob shows the answer is correct and Alice misidentified a problem
- "defender_wins_minor": Alice's critique is technically correct but about very minor (stylistic only) issues
- "wrong_problem": There are issues with the answer, but Alice's specific diagnosis is incorrect
- "mixed": Alice makes multiple claims, some correct and some incorrect
- "unknown": Cannot determine if Alice's critique is valid with confidence

Confidence scale: 1=very uncertain, 2=uncertain, 3=moderate, 4=confident, 5=very confident

```