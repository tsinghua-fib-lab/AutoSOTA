# DocVQA Skill

## Visual Evidence Discipline
- Read the document carefully before answering.
- Prefer the smallest exact text span that answers the question.
- When several nearby strings look similar, choose the one whose surrounding labels or layout best match the question.

## Exact Answer Discipline
- Copy names, numbers, and dates exactly from the document whenever possible.
- Prefer direct extraction over paraphrase.
- Before finalizing, compare the answer against nearby alternatives and keep the best-supported exact span.

## Contrastive Patterns (learn from failures)

The following pairs show a wrong answer, why it lost exact match, and the correct answer. Before finalizing your answer, check that your output does not repeat the same boundary mistake.

### Example 1
- Question: what is the name of sponsorship program?
- Wrong: `Camel Club Program ("CCP")`
- Why wrong: under_specific_or_variant_span
- Correct: `camel club program("ccp")`

### Example 2
- Question: What is  Question 1?
- Wrong: `“Redefining” Scientific Terms`
- Why wrong: under_specific_or_variant_span
- Correct: `"Redefining" Scientific Terms`

### Example 3
- Question: What is the name of the company that can approach to create awareness of safety issues?
- Wrong: `Astro Zeneca`
- Why wrong: partial_boundary_or_alias_mismatch
- Correct: `Astra Zeneca`

### Example 4
- Question: What is the page no mentioned in this document?
- Wrong: `PAGE 2 OF 3`
- Why wrong: over_specific_span
- Correct: `2`

### Example 5
- Question: What is the amount contributed to Vanderbilt university medical school?
- Wrong: `$ 50.00`
- Why wrong: under_specific_or_variant_span
- Correct: `$50.00`

### Example 6
- Question: What is the program # mentioned in the form?
- Wrong: `701/98`
- Why wrong: partial_boundary_or_alias_mismatch
- Correct: `701998`

### Example 7
- Question: Where were sample webs produced?
- Wrong: `the University of Maine`
- Why wrong: over_specific_span
- Correct: `University of Maine`

### Example 8
- Question: How much money is received from Dr. WIlliam J. Darby?
- Wrong: `Two Hundred and Eighty Two Pounds Twelve Shillings ($ 691.44)`
- Why wrong: over_specific_span
- Correct: `Two hundred and eighty two Pounds Twelve Shillings`
