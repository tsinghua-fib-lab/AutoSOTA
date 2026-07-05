# LLM visibility sanity check

This document summarizes what each model sees at each pipeline step and flags
conditions where missing context can invalidate the result.

## Benchmark question generation (generate_benchmark.py)
LLM sees:
- Topic name + question guidance.
- Prior failed/ill-posed questions for the same run (if any), to avoid repeats.

Missing-info risks:
- If prior attempts are not loaded (e.g., missing benchmark file), the model
  does not see prior failures and can repeat bad questions. This degrades
  quality but does not invalidate schema or downstream parsing.

## Self-check + refinement (self_improvement.py)
LLM sees:
- Question, answer, and self-check guidance.
- For answer generation, an optional "ill-posed override" note.

Missing-info risks:
- If the answer text is empty (model failure), the self-check still runs and can
  output a verdict that does not reflect the intended task.

Decision: check if answer text is empty and if so, fail loudly

## Answer generation (generate_answers.py)
LLM sees:
- The final benchmark question text.
- Answer guidance for the initial response and refinement.

Missing-info risks:
- If the question text is missing (corrupt benchmark entry), the answer task is
  skipped, leaving a gap in the dataset.
- If the answer text is empty, later steps may treat it as valid input.
- Ill-posed claim details come from the last self-check evaluation; if that
  evaluation is missing or unparseable, the claim can be empty in later debates.

Decision: check if answer text is empty and if so, fail loudly

## Critique generation (generate_critiques.py)
LLM sees:
- Question, answer, and critique guidance.
- In self-improvement mode: the model's own critique plus the original question
  and answer.

Missing-info risks:
- If the answer text is empty (rare but possible), critiques become meaningless.
- The prompt does not include model identities (question author / answer author);
  this is intentional, but it means the critique lacks provenance context.
- If critique JSON parsing fails, the critique is marked failed and later steps
  skip it, reducing coverage.

## Debate generation: ill-posed (debate.py, mode=ill-posed)
LLM sees:
- The question text.
- The ill-posed claim details (from the answer self-check evaluation).
- Question-quality guidance and debate guidance.

Missing-info risks:
- If the ill-posed claim details are missing or empty, Alice's claim lacks
  substance, and the debate no longer tests the actual issue.
- If the question text is missing, the debate is not meaningful.

## Debate generation: critique (debate.py, mode=critique)
LLM sees:
- Question, answer, and critique text.
- Answer-quality guidance, critique guidance, and debate guidance.

Missing-info risks:
- Empty answer or critique text makes the debate invalid.
- Critiques with "unknown" verdicts are now skipped, which can reduce coverage
  for certain answer pairs.

## Automated judging (automated_judge.py)
LLM sees:
- Redacted question, answer, critique, and debate transcript (model names are
  replaced with Alice/Bob).
- Judgment guidance plus task-specific guidance (question/answer/critique).

Missing-info risks:
- If `--allow-no-debate` is used and debate history is absent, the judge sees no
  debate transcript, which weakens or invalidates the decision relative to
  standard conditions.
- If question/answer/critique fields are missing, the judge can be forced to
  decide without core evidence.
- Redaction can remove content if a model name appears in the problem text; this
  is rare but can alter evidence.

## JSON repair model (utils._repair_with_model, optional)
LLM sees:
- Only the malformed JSON text and a schema hint.

Missing-info risks:
- The repair model has no access to the original prompt context; if it "repairs"
  by guessing, it can introduce subtle inconsistencies that propagate downstream.
