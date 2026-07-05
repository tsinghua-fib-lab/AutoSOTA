## Question Quality and Meaningfulness

This rubric guides question generation.

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
