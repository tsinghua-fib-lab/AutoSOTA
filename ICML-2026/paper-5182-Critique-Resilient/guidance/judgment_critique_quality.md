## Judgment Quality Guidance (Critique Debates)

This rubric guides the neutral judgment process for evaluating critique claims.

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
