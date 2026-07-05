## Judgment Quality Guidance (Ill-Posedness Debates)

This rubric guides the neutral judgment process for evaluating ill-posedness claims.

### Primary Goal

Determine whether **Alice's claim that the question is ill-posed is valid** based on the mathematical evidence presented in the debate. Your role is to judge the **claim itself**, not who "won" the debate rhetorically. The debate transcript serves as evidence to help assess whether Alice's claim is substantiated.

For example, if Alice makes a valid claim and Bob acknowledges the issue during the debate, this is evidence that Alice may be correct, but you must independently verify the claim is actually valid. Bob's concession alone does not automatically mean Alice wins; you must agree with the concession based on the evidence.

### Core Principles

**1. Judge the Claim, Not the Debate**
- Your task is to evaluate whether **Alice's claim about ill-posedness is valid**
- The debate is evidence that helps you assess the claim, not a competition to "win"
- Bob's acknowledgment or concession is evidence, but you must independently verify the claim's validity
- A party might incorrectly concede or address irrelevant issues
- Focus on: Was Alice right about the question being ill-posed?

**2. Evidence-Based Assessment**
- Evaluate the validity of Alice's claim using the debate as evidence
- Base verdicts on specific references to the question and rubric criteria
- Ignore rhetorical style, confidence levels, or persuasive language
- Quote or reference specific parts of the debate when justifying your verdict

**3. Objectivity and Neutrality**
- Do not favor either party based on their assigned role (Alice vs. Bob)
- Ignore any identifying information about the debaters (model names, capabilities, etc.)
- Judge only the merit of the claim and supporting evidence

**4. Adherence to Rubrics**
- Use the question quality rubric as the objective standard
- The rubric defines what constitutes a well-posed question
- Focus on: completeness, clarity, non-ambiguity, solvability

**5. Verification Over Generation**
- You are judging whether a claim is valid, not solving the problem independently
- Focus on whether Alice's claim is substantiated by the evidence
- It is acceptable to conclude you cannot determine validity with confidence

### Debate Context

**Ill-Posedness Debates:**
- Alice claims the question is ill-posed
- Bob defends that the question is well-posed
- Use the question quality rubric to assess well-posedness

### Judgment Process

**1. Identify Alice's Original Claim**
- What specific issue did Alice identify about the question?
- What makes the question ill-posed according to her claim?
- What evidence did she provide for this claim?

**2. Review the Debate Transcript as Evidence**
- Read all arguments from both parties chronologically
- Note any concrete examples or demonstrations provided
- Pay special attention to:
  - Does Bob acknowledge the ill-posedness issue Alice raised?
  - Does Bob demonstrate the question actually is well-posed?
  - Does Bob show that all necessary information is present?

**3. Evaluate the Claim's Validity**
- Apply the question quality rubric to assess whether Alice's claim is substantiated
- Consider whether the issue Alice identified is:
  - **Substantive**: Missing constraints, undefined terms, contradictory premises, ambiguous specification
  - **Non-issue**: Information is present, standard convention applies, question is answerable as stated
- If Bob acknowledged the issue, verify independently whether the concession is correct
- If Bob successfully demonstrates the question is well-posed, evaluate whether his demonstration is valid
- Don't automatically accept either party's position without verification

**4. Determine Confidence**
- High confidence (4-5): Alice's claim is clearly valid or clearly invalid based on strong evidence
- Moderate confidence (3): The claim's validity is identifiable but evidence is not definitive
- Low confidence (1-2): Cannot determine claim validity; evidence is insufficient or contradictory

### Verdict Guidelines

Choose exactly one verdict based on whether **Alice's claim is valid**:

- **"claimant_wins"**: Alice's claim is correct: the question is ill-posed per the rubric (any valid problem counts, even if she doesn't identify all problems)
  - Use this even if Bob concedes or acknowledges the issue during the debate
  - Example: Alice correctly identifies that the domain is unspecified

- **"defender_wins_incorrect"**: Alice's claim is incorrect: Bob successfully demonstrates the question is well-posed and Alice misidentified the issue
  - Use this when Alice claims something is missing/ill-posed but it's actually present/correct
  - Example: Alice claims domain is unspecified, but it's explicitly stated in the question

- **"wrong_problem"**: The question has issues, but Alice's specific diagnosis is incorrect or not substantiated
  - Use this when the question is ill-posed, but not for the reason Alice claims
  - Example: Alice says "the integral diverges" but actually the domain is undefined

- **"mixed"**: Alice makes multiple claims, some correct and some incorrect (not all claims are valid)
  - Only use for factually incorrect claims, not "all are correct, but some are nitpickings"
  - Example: Alice makes three claims about ill-posedness, but only one is actually valid

- **"unknown"**: Cannot determine whether Alice's claim is valid with reasonable confidence
  - Use when the debate and available info don't provide enough clarity
  - Use when both parties make valid but orthogonal points

**Critical Principle**: Use "claimant_wins" whenever Alice correctly identifies **any** valid ill-posedness issue, even if:
- She missed other problems that you spotted
- Her reasoning for why it's ill-posed is imperfect
- There are additional issues beyond what she raised
- Bob acknowledges the problem or concedes during the debate

Only use "wrong_problem" when Alice's specific claim about what makes the question ill-posed is itself incorrect.

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
"Alice claimed the domain of variable $x$ is unspecified. In round 1, Bob acknowledged this omission and agreed the question is ill-posed. Reviewing the question, I confirm that no domain is specified anywhere. Bob's acknowledgment is correct, and Alice's claim is valid: the question lacks a necessary constraint. Verdict: claimant_wins, confidence 5."

*Example 2 (Alice identifies ill-posedness):*
"Alice claimed the domain of variable $x$ is unspecified. Bob argued it's 'clearly the reals,' but this assumption appears nowhere in the problem statement. Per the question quality rubric, all constraints must be explicit. Alice's claim is substantiated. Verdict: claimant_wins, confidence 4."

*Example 3 (Bob concedes incorrectly):*
"Alice claimed the domain is unspecified. Bob conceded and said 'you're right, I forgot to specify it.' However, reviewing the question, line 1 explicitly states 'for all real numbers $x$'. Bob incorrectly conceded to a non-issue. The domain is specified, and Alice's claim is wrong. Verdict: defender_wins_incorrect, confidence 5."

*Example 4 (Incorrect claim):*
"Alice claimed the domain is unspecified. Bob demonstrated that line 1 explicitly states 'for all real numbers $x$'. Alice misidentified the issue—the domain constraint exists. Verdict: defender_wins_incorrect, confidence 5."

*Example 5 (Wrong problem):*
"Alice claimed the question is ill-posed because 'the function doesn't converge.' However, convergence is not required for the question to be answerable. The actual issue is that the boundary conditions are undefined, which Alice didn't identify. Verdict: wrong_problem, confidence 4."

**Poor Example:**
"Alice made better arguments and seemed more convincing overall. Bob didn't really address the main concerns effectively."
*Why poor: Focuses on debate performance rather than claim validity; no reference to rubrics or specific evidence.*

### Common Pitfalls to Avoid

**Don't Judge Debate Performance**
- You are judging whether **Alice's claim is valid**, not who argued better
- Alice winning an argument doesn't mean her claim is correct—verify the claim itself
- Focus on: Was Alice's original claim correct? Not: Who presented arguments more persuasively?

**Don't Solve Independently**
- Your role is to judge the claim's validity using debate evidence, not to solve the problem yourself
- If you find yourself doing extensive analysis, refocus on the presented arguments
- It's acceptable to say "unknown" if the debate doesn't provide enough clarity

**Don't Favor Defensive Positions**
- Don't assume the question is well-posed by default
- Give equal weight to challenges and defenses
- Bob's initial position gets no presumption of correctness

**Don't Be Swayed by Confidence or Persuasiveness**
- Assertive language or certainty doesn't make a claim correct or incorrect
- Evaluate the validity of Alice's claim, not the rhetorical style
- A tentative but correct claim beats a confident but invalid one

**What if there are multiple claims?**
- If there are multiple claims and **all are correct** (e.g. Alice correctly spots multiple ill-posedness issues), use "claimant_wins"
- If there are multiple claims and **only some are correct** (mixed validity), use "mixed"
- For example, if Alice makes three claims and only one is correct, this is "mixed" not "claimant_wins"

### Borderline Cases

When the debate is close or both parties have merit:
- Use "wrong_problem" if you identify valid ill-posedness but the claimant's diagnosis is incorrect
- Use "unknown" if you cannot confidently judge based on the presented arguments
- Prefer lower confidence ratings (2-3) over forcing a verdict
- In your reasoning, explain what makes the case borderline

**Examples for "wrong_problem" vs "claimant_wins":**
- Alice says "the function must be integrable" but the question doesn't require that; however, the domain is undefined → wrong_problem (her diagnosis is wrong)
- Alice says "missing boundary conditions" and you agree → claimant_wins (her diagnosis is correct)
- Alice says "the domain is unspecified" and this is true → claimant_wins (even if there are other problems too)

### Important Constraints

**Stay Focused on Question Quality**
- Base judgments solely on question quality rubric compliance
- Ignore meta-arguments about what "most mathematicians" would assume
- Don't be influenced by appeals to authority or external sources not in the transcript

**Be Honest About Limitations**
- If the debate covers material beyond your verification ability, say "unknown"
- Don't guess or make assumptions to force a verdict
- Acknowledge when both parties make valid but orthogonal points

**Maintain Strict Neutrality**
- Remember that party names (Alice/Bob) are arbitrary labels
- Judge each argument on its merits regardless of who presents it
- Don't let earlier rounds influence later assessment if new evidence is presented
