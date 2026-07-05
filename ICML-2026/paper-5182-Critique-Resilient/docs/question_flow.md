# Question flow: topic to automated judgment

This document traces a single question from the run file through generation, improvement, answering, critique, debate, and automated judgment. Every check and activity is linked to the line where it happens.

## Run inputs and topic validation
- The run file is a JSON mapping of run_id to topic slug in `configs/runs.json#L1` (see example topic entries in `configs/runs.json#L2`, `configs/runs.json#L5`, `configs/runs.json#L8`).
- Topic metadata is loaded into a slug->info map by `load_topic_info` in `generate_benchmark.py#L36`, which is called from `generate_benchmark.py#L143`.
- Each run is validated against the topic catalog in `load_runs` (`generate_benchmark.py#L46`). The topic slug is extracted at `generate_benchmark.py#L50`, validated at `generate_benchmark.py#L51`, and converted into a `topic_name` + normalized run payload at `generate_benchmark.py#L53` and `generate_benchmark.py#L54`.
- The optional `--limit` truncates the run list before any generation starts in `generate_benchmark.py#L145`.

## Benchmark question generation (generate_benchmark.py)
- Registry + guidance are loaded before generation: model registry in `generate_benchmark.py#L144`, question guidance in `generate_benchmark.py#L145` (loaded by `prompt_library.py#L9`), answer guidance in `generate_benchmark.py#L146` (loaded by `prompt_library.py#L13`), and self-critique guidance in `generate_benchmark.py#L147` (loaded by `prompt_library.py#L25`).
- Question generator models are selected by role in `generate_benchmark.py#L148`.
- Existing benchmark entries are loaded per model in `generate_benchmark.py#L158` and indexed by run_id in `generate_benchmark.py#L160` to avoid duplicate work.
- A run is skipped if it already succeeded (`generate_benchmark.py#L171`), or if it has already hit the `--max-question-attempts` cap (`generate_benchmark.py#L173`).
- Question generation loops while pending runs remain so failed/ill-posed runs can be rewritten up to the attempt cap (`generate_benchmark.py#L178` and `generate_benchmark.py#L182`).
- `generate_questions` builds prompts per run in `generate_benchmark.py#L91`, and collects prior failed/ill-posed attempts so the next question avoids repeats (`generate_benchmark.py#L105`, `generate_benchmark.py#L106`, `generate_benchmark.py#L109`, `generate_benchmark.py#L111`, `generate_benchmark.py#L112`).
- The question prompt includes the prior attempts only if present via `build_question_prompt` at `generate_benchmark.py#L113` (prompt template in `prompt_library.py#L49` and prior-attempt handling in `prompt_library.py#L55`).
- The model call is single or batched depending on `--disable-batch` or prompt count (`generate_benchmark.py#L115`, `generate_benchmark.py#L117`, `generate_benchmark.py#L120`).
- Responses must include both `[QUESTION]` and `[ANSWER]` tags or generation fails (`generate_benchmark.py#L58`, `generate_benchmark.py#L59`, `generate_benchmark.py#L66`). An empty question is rejected explicitly (`generate_benchmark.py#L63`).
- Question/answer text is cleaned for consistent math delimiters via `clean_math` in `generate_benchmark.py#L198` and `generate_benchmark.py#L199` (normalization logic in `utils.py#L47`).
- Self-check prompts come from `build_self_check_prompt` (`generate_benchmark.py#L204`, template in `prompt_library.py#L107`) and refinement prompts from `build_refine_prompt` (`generate_benchmark.py#L205`, template in `prompt_library.py#L138`).
- The self-improvement loop is invoked with raw initial answers recorded for traceability in `generate_benchmark.py#L207` and `generate_benchmark.py#L217` (implementation in `self_improvement.py#L32`).
- Each attempt becomes a `RefinementAttempt` in `generate_benchmark.py#L224`, the generation round is created in `generate_benchmark.py#L233`, and the entry status is updated in `generate_benchmark.py#L240`.
- Final benchmark entries are written to disk in `generate_benchmark.py#L244`.

## Self-improvement loop (shared by generation, answers, critiques)
- `self_improve_answers` enforces `max_rounds >= 1` and raises on invalid configuration in `self_improvement.py#L50` and `self_improvement.py#L51`.
- Raw answer tracking is initialized from `raw_initial_answers` if provided (`self_improvement.py#L56`, `self_improvement.py#L58`, `self_improvement.py#L60`) or from the cleaned answers otherwise (`self_improvement.py#L62` and `self_improvement.py#L63`).
- Each round builds evaluation prompts via the caller-provided callback in `self_improvement.py#L66` and queries the model via `_batched_query` in `self_improvement.py#L69`.
- Evaluations are parsed with `safe_load_json` (`self_improvement.py#L82`) and fall back to a default "fail" payload if parsing fails (`self_improvement.py#L85`).
- Each evaluation is stored as an `Attempt` record (`self_improvement.py#L93`), with raw answer provenance attached (`self_improvement.py#L92`).
- A "pass" verdict immediately marks success (`self_improvement.py#L103` and `self_improvement.py#L104`), while `ill_posed` marks the entry ill-posed (`self_improvement.py#L107` and `self_improvement.py#L108`).
- If the final round still fails, status becomes failed (`self_improvement.py#L111` and `self_improvement.py#L112`); otherwise a refine prompt is built (`self_improvement.py#L116` and `self_improvement.py#L117`) using the feedback in `self_improvement.py#L120`.
- The loop stops early when there are no items left to refine (`self_improvement.py#L125`), and updates the final answer with the refined output (`self_improvement.py#L135` and `self_improvement.py#L140`).
- `safe_load_json` attempts direct JSON parsing, then cleaned JSON, then optional LLM repair (`utils.py#L176`, `utils.py#L184`, `utils.py#L193`). It uses `configs/parsing.json` if present (`utils.py#L102`).

[Sam's note: auto-retry failures are handled via `--max-question-attempts` in generate_benchmark.]

## Answer generation (generate_answers.py)
- Answer guidance and self-critique guidance are loaded before answering (`generate_answers.py#L182` and `generate_answers.py#L183`), using prompt templates from `prompt_library.py#L13` and `prompt_library.py#L107`.
- Ill-posed overrides are read from a JSON file in `generate_answers.py#L184`. The override note is built in `generate_answers.py#L43` and `generate_answers.py#L47` to instruct the self-check not to mark a question as ill-posed when an override is present.
- Self-answering is blocked unless `--allow-self-answering` is set (`generate_answers.py#L70` and `generate_answers.py#L174`).
- Each benchmark entry is screened: limit check in `generate_answers.py#L74`, status check in `generate_answers.py#L76`, and missing-question check in `generate_answers.py#L81`.
- Only succeeded benchmark entries become answer tasks; others are skipped and logged (`generate_answers.py#L76` and `generate_answers.py#L78`).
- Already-succeeded answers are skipped (`generate_answers.py#L85`) and failed/ill-posed answers are skipped unless `--rerun-failures` is set (`generate_answers.py#L87`).
- The final question used for answering is the latest refinement question in `final_question` (`generate_answers.py#L51` and `generate_answers.py#L58`).
- Answer prompts are built in `generate_answers.py#L106` (template in `prompt_library.py#L88`) and queried single or batched based on settings (`generate_answers.py#L111` and `generate_answers.py#L117`).
- Answers are normalized with `clean_math` in `generate_answers.py#L119` and evaluated with `build_self_check_prompt` in `generate_answers.py#L123`, plus optional override note in `generate_answers.py#L124`.
- The self-improvement loop is run in `generate_answers.py#L127` with raw initial answers preserved in `generate_answers.py#L137`.
- Each `AnswerEntry` is built with run_id/topic_slug from the benchmark entry in `generate_answers.py#L155` and `generate_answers.py#L156`.
- Before saving, the answer list is padded to match the benchmark entry count (`generate_answers.py#L231` and `generate_answers.py#L233`) so indices line up.

## Critique generation (generate_critiques.py)
- Critique guidance is loaded in `generate_critiques.py#L302` (source text in `prompt_library.py#L17`) and critic models are selected by role in `generate_critiques.py#L304`.
- Custom mode requires a mapping file; missing `--custom-map` raises an error (`generate_critiques.py#L309` and `generate_critiques.py#L310`), and incomplete mappings are skipped (`generate_critiques.py#L313` and `generate_critiques.py#L317`).
- For self-answer critique, `load_answer_records` forbids a preexisting self-answer file by raising (`generate_critiques.py#L128` and `generate_critiques.py#L131`), and otherwise builds answers directly from benchmark entries (`generate_critiques.py#L133` and `utils.py#L60`).
- Mode logic in `prepare_pairs`:
- "contradictor" mode skips critics that equal the question author (`generate_critiques.py#L163`) and skips missing self-answers (`generate_critiques.py#L168`); it also respects the `--limit` cap (`generate_critiques.py#L166`).
- "evaluator" mode short-circuits entirely if the question author is not in the critic set (`generate_critiques.py#L173`) and skips self-answers (`generate_critiques.py#L192`).
- "all/custom" mode filters to critic names when provided (`generate_critiques.py#L176`) and requires both a question-author answer and target answer at the same index (`generate_critiques.py#L188` and `generate_critiques.py#L190`).
- Each pair is screened before critique: critic must have the "critique" role (`generate_critiques.py#L361` and `generate_critiques.py#L363`), answer indices must exist (`generate_critiques.py#L373`), failed answers are skipped (`generate_critiques.py#L376`), and missing question text is skipped (`generate_critiques.py#L382`).
- Critique files are extended to the needed index (`generate_critiques.py#L387` and `generate_critiques.py#L388`) and already-succeeded critiques are skipped (`generate_critiques.py#L390`).
- Critique prompts are built in `generate_critiques.py#L213` (template in `prompt_library.py#L159`), and model calls are batched or single in `generate_critiques.py#L217`.
- If self-improvement is disabled, the critique is parsed once via `extract_structured_critique` (`generate_critiques.py#L223`) and stored as a single attempt (`generate_critiques.py#L226`).
- If self-improvement is enabled, self-check prompts are built with `build_critique_self_check` (`generate_critiques.py` and `prompt_library.py`), refine prompts with `build_critique_refine` (`generate_critiques.py` and `prompt_library.py`), and improvements run through `self_improve_critiques` (`generate_critiques.py` and `self_improvement.py`).
- Structured critique parsing validates verdicts against `VALID_CRITIQUE_VERDICTS` (`generate_critiques.py#L85` and `generate_critiques.py#L89`), logs invalid verdicts (`generate_critiques.py#L93`), joins list notes (`generate_critiques.py#L97`), and falls back to raw text when notes are missing (`generate_critiques.py#L102`).
- Final critique status is succeeded only when the verdict is not unknown (`generate_critiques.py#L425` and `generate_critiques.py#L428`), then saved as a `CritiqueEntry` in `generate_critiques.py#L431`.

## Debate generation (debate.py)
- The debate CLI enforces at least one round (`debate.py#L220`) and loads question, answer, critique, and debate guidance (`debate.py#L223`, `debate.py#L224`, `debate.py#L225`, `debate.py#L226`, `debate.py#L227`, `debate.py#L228`).
- Each round is forced to return JSON with "message" and "concede" via a system prompt in `debate.py#L90` and JSON mode in `debate.py#L93`.
- If JSON parsing fails, the raw reply is used and concession is treated as false (`debate.py#L95` and `debate.py#L99`).
- Ill-posed debates only target answers marked ill-posed (`debate.py#L254`) and skip indices that already have a debate entry (`debate.py#L257` and `debate.py#L258`).
- Ill-posed debates also reject existing self-answer files to avoid mixing sources (`debate.py#L246` and `debate.py#L248`).
- `illposed_debate` alternates Bob then Alice each round (`debate.py#L131` and `debate.py#L141`) and stops early on concession (`debate.py#L134` and `debate.py#L143`).
- Debate records store run_id and topic_slug from the answer entry (`debate.py#L321` and `debate.py#L322`) and are saved after each debate (`debate.py#L325`).
- Critique debates only consider critiques with status "succeeded" (`debate.py#L368` and `debate.py#L420`) and skip critiques already labeled "correct" (`debate.py#L370` and `debate.py#L422`).
- Missing answers or missing debate slots are skipped (`debate.py#L425` and `debate.py#L429`).
- Critique text is taken from the raw critique if present or notes otherwise (`debate.py#L50` and `debate.py#L53`).
- `critique_debate` alternates Bob then Alice (`debate.py#L184` and `debate.py#L193`) and allows early concession (`debate.py#L187` and `debate.py#L195`).

## Automated judgment (automated_judge.py)
- Judgment guidance is loaded before task construction (`automated_judge.py#L481`, `automated_judge.py#L482`, `automated_judge.py#L483`, `automated_judge.py#L484`, `automated_judge.py#L485`) via `prompt_library.py#L9`, `prompt_library.py#L13`, `prompt_library.py#L17`, `prompt_library.py#L41`, and `prompt_library.py#L45`.
- Ill-posed tasks are collected from `debates/illposed/*/*.json` in `gather_illposed_tasks` (`automated_judge.py#L132`).
- Each task loads benchmark answers as fallback via `benchmark_answers_from_entries` (`automated_judge.py#L135` and `utils.py#L60`), then skips any answer that is not ill-posed (`automated_judge.py#L152` and `automated_judge.py#L153`).
- Question and answer are sourced from debate/answer/fallback records (`automated_judge.py#L155`, `automated_judge.py#L156`, `automated_judge.py#L158`, `automated_judge.py#L160`), and empty tasks are dropped (`automated_judge.py#L163`).
- Missing debate history is rejected unless `--allow-no-debate` is provided (`automated_judge.py#L165` and `automated_judge.py#L166`).
- Critique tasks skip non-succeeded critiques (`automated_judge.py#L231` and `automated_judge.py#L232`) and skip "correct" verdicts unless forced (`automated_judge.py#L236`).
- Critique tasks also build question/answer with fallback (`automated_judge.py#L239`, `automated_judge.py#L241`, `automated_judge.py#L243`, `automated_judge.py#L245`) and skip empty tasks or missing debates (`automated_judge.py#L249` and `automated_judge.py#L251`).
- Redaction removes model identities before judging by mapping names to Alice/Bob (`automated_judge.py#L82`, `automated_judge.py#L86`, `automated_judge.py#L89`) and replacing in text with boundary-aware regexes (`automated_judge.py#L101` and `automated_judge.py#L102`).
- Prompts embed the redacted question/answer/critique and debate transcript in `build_illposed_prompt` (`automated_judge.py#L277`) and `build_critique_prompt` (`automated_judge.py#L313`), and require a strict JSON response schema (`automated_judge.py#L297` and `automated_judge.py#L338`).
- Confidence must be an integer 1-5 or judgment is marked unknown (`automated_judge.py#L381`, `automated_judge.py#L390`, `automated_judge.py#L409`, `automated_judge.py#L414`).
- Verdicts are normalized by task type (`automated_judge.py#L416` and `automated_judge.py#L419`) and any unknown verdict yields failed status (`automated_judge.py#L420`).
- Judges do not evaluate tasks where they were a debate participant (`automated_judge.py#L506` and `automated_judge.py#L509`).
- Already-judged tasks are skipped unless `--overwrite` is set (`automated_judge.py#L517` and `automated_judge.py#L518`), and batches are sent with `_batched_query` (`automated_judge.py#L535` and `automated_judge.py#L536`).
- Decisions are persisted per judge in `automated_judge.py#L551`.

## Storage, validation, and shared constants
- Status and verdict strings are centralized in `constants.py#L8`, `constants.py#L23`, and `constants.py#L62` and referenced throughout the pipeline.
- Benchmark statuses are validated at load time in `data_models.py#L75` and `data_models.py#L79`.
- Answer statuses are validated in `data_models.py#L129` and `data_models.py#L133`.
- Critique verdicts are validated in `data_models.py#L157` and `data_models.py#L161`.
- Debate histories are checked to be sequential (unless empty) in `data_models.py#L230` and `data_models.py#L239`.
- Automated evaluation types are validated in `data_models.py#L281` and `data_models.py#L285`, and verdicts are validated against the correct set in `data_models.py#L296` and `data_models.py#L301`.
- JSON list loading enforces list structure and validates each entry in `data_models.py#L469`, `data_models.py#L473`, and `data_models.py#L481`.
- Saving writes placeholder objects for None entries in `data_models.py#L487` and `data_models.py#L492`.
- Evaluation files must contain a "decisions" list and each decision is validated in `data_models.py#L533` and `data_models.py#L542`.
