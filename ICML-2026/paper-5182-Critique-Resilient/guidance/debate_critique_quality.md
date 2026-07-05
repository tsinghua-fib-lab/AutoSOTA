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
