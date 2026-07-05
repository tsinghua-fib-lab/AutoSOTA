## Answer Correctness and Completeness

This rubric guides the answering process.

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