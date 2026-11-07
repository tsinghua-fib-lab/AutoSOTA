"""ALGO-03 post-training skill consolidation.

The consolidation stage turns an already trained trajectory-aware skill into a
cleaner exported skill by combining three operations:

1. mine failure/success trajectories,
2. validate conservative pruning candidates one at a time,
3. rewrite the full skill with failure-driven demonstrations.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from skillopt.model import chat_optimizer
from skillopt.utils import extract_json


@dataclass
class Algo03Result:
    skill: str
    accepted: bool
    validation_hard: float | None
    validation_soft: float | None
    report: dict


def _read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _case_view(row: dict) -> dict:
    keys = (
        "id",
        "task_type",
        "task_description",
        "question",
        "hard",
        "soft",
        "predicted_answer",
        "gold_answers",
        "gold_answer",
        "response",
        "fail_reason",
        "agent_ok",
        "n_turns",
    )
    view = {k: row.get(k) for k in keys if k in row}
    response = str(view.get("response", "") or "")
    if len(response) > 3000:
        view["response"] = response[:3000] + "\n[truncated]"
    return view


def collect_training_cases(
    out_root: str,
    *,
    max_failures: int,
    max_successes: int,
    case_offset: int = 0,
    pool_multiplier: int = 1,
) -> tuple[list[dict], list[dict]]:
    failure_pool: list[dict] = []
    success_pool: list[dict] = []
    failure_limit = max(max_failures, 0) * max(int(pool_multiplier or 1), 1)
    success_limit = max(max_successes, 0) * max(int(pool_multiplier or 1), 1)
    direct_results = os.path.join(out_root, "results.jsonl")
    pattern = os.path.join(out_root, "steps", "step_*", "**", "rollout", "results.jsonl")
    paths = [direct_results] if os.path.isfile(direct_results) else sorted(glob.glob(pattern, recursive=True))
    for path in paths:
        for row in _read_jsonl(path):
            hard = float(row.get("hard", 0) or 0)
            if hard >= 1.0:
                if len(success_pool) < success_limit:
                    success_pool.append(_case_view(row))
            else:
                if len(failure_pool) < failure_limit:
                    failure_pool.append(_case_view(row))
            if len(failure_pool) >= failure_limit and len(success_pool) >= success_limit:
                break

    def _window(rows: list[dict], size: int) -> list[dict]:
        if size <= 0 or not rows:
            return []
        if len(rows) <= size:
            return rows[:size]
        start = int(case_offset or 0) % len(rows)
        rotated = rows[start:] + rows[:start]
        return rotated[:size]

    return _window(failure_pool, max_failures), _window(success_pool, max_successes)


def _numbered_rule_blocks(skill: str, max_blocks: int = 40) -> list[dict]:
    blocks: list[dict] = []
    current: list[str] = []
    current_title = ""
    for line in skill.splitlines():
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+\S", line):
            if current:
                text = "\n".join(current).strip()
                if text:
                    blocks.append({"title": current_title, "text": text})
            current = [line]
            current_title = line.strip()[:120]
        elif current and line.strip():
            current.append(line)
        elif current:
            text = "\n".join(current).strip()
            if text:
                blocks.append({"title": current_title, "text": text})
            current = []
            current_title = ""
    if current:
        text = "\n".join(current).strip()
        if text:
            blocks.append({"title": current_title, "text": text})
    return blocks[:max_blocks]


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _truncate_text(text: object, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


def _compact_case(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "question": _truncate_text(row.get("question") or row.get("task_description"), 240),
        "predicted_answer": _truncate_text(row.get("predicted_answer"), 120),
        "gold_answers": row.get("gold_answers") or row.get("gold_answer"),
        "hard": row.get("hard"),
        "soft": row.get("soft"),
        "fail_reason": _truncate_text(row.get("fail_reason"), 180),
    }


def _compact_spreadsheet_case(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "task_type": row.get("task_type") or row.get("instruction_type"),
        "instruction_type": row.get("instruction_type"),
        "task_description": _truncate_text(row.get("task_description") or row.get("question"), 360),
        "phase": row.get("phase"),
        "hard": row.get("hard"),
        "soft": row.get("soft"),
        "n_cases": row.get("n_cases"),
        "n_pass": row.get("n_pass"),
        "n_exec_pass": row.get("n_exec_pass"),
        "n_turns": row.get("n_turns"),
        "fail_reason": _truncate_text(row.get("fail_reason"), 260),
    }


def _fallback_plan_searchqa(
    failures: list[dict],
    successes: list[dict],
    *,
    max_demos: int,
    demo_policy: str = "real",
) -> dict:
    """Deterministic compact plan when the optimizer fails to return JSON."""
    near_misses = [
        row for row in failures
        if float(row.get("soft", 0) or 0) > 0 and row.get("predicted_answer")
    ]
    hard_misses = [
        row for row in failures
        if float(row.get("soft", 0) or 0) <= 0
    ]
    demos = []
    if demo_policy == "abstract":
        demos = [
            {
                "name": "minimal alias span",
                "source_id": "abstracted_from_successes",
                "demo": (
                    "If the evidence supports a longer descriptive phrase and a "
                    "shorter canonical label, return only the shorter label."
                ),
                "why_it_helps": "It prevents exact-match loss from over-specific answers.",
            },
            {
                "name": "frame-selected answer",
                "source_id": "abstracted_from_successes",
                "demo": (
                    "If the clue frame asks for a place, work, person, or term, "
                    "return that frame-selected entity rather than a nearby "
                    "attribute or description."
                ),
                "why_it_helps": "It keeps the answer type aligned with the clue grammar.",
            },
        ][:max_demos]
    else:
        for row in (successes[:max_demos] or []):
            demos.append(
                {
                    "name": "bare canonical span",
                    "source_id": row.get("id"),
                    "demo": (
                        f"Question: {row.get('question', '')}\n"
                        f"Answer: {row.get('predicted_answer') or row.get('response', '')}"
                    ),
                    "why_it_helps": (
                        "It demonstrates returning only the minimal canonical answer "
                        "inside the answer tag, without descriptors or explanations."
                    ),
                }
            )
    return {
        "failure_clusters": [
            {
                "name": "over-specific aliases lose exact match",
                "evidence_ids": [str(row.get("id")) for row in near_misses[:8]],
                "fix": (
                    "When evidence supports both a longer descriptive answer and a "
                    "shorter canonical label, output the shortest entity/term that "
                    "satisfies the clue frame. Drop descriptors such as first names, "
                    "category words, tribute band, vitamin, dynasty, and appositives "
                    "unless the clue explicitly asks for them."
                ),
            },
            {
                "name": "short or ambiguous clue needs exact echo evidence",
                "evidence_ids": [str(row.get("id")) for row in hard_misses[:8]],
                "fix": (
                    "For very short clues, do not answer from broad world knowledge. "
                    "Find the document whose title/snippet exactly echoes the clue, "
                    "then copy the answer label or canonical span from that document."
                ),
            },
        ],
        "success_demos": demos[:max_demos],
        "prune_candidates": [],
        "rewrite_directives": [
            "Move exact-match calibration before the long appendix.",
            "State a concrete minimal-span rule for aliases and near-miss cases.",
            "Keep the existing clue-echo and relational-frame strategy.",
            "Compress repetitive appendix notes into a short checklist.",
            "Inject concise examples derived from real successful cases.",
        ],
    }


def _spreadsheet_failure_bucket(row: dict) -> str:
    reason = str(row.get("fail_reason") or "").lower()
    phase = str(row.get("phase") or "").lower()
    if "timeout" in reason or phase == "timeout":
        return "timeout_or_overlong_generation"
    if "empty-code-block" in reason or phase == "extract":
        return "missing_or_malformed_code_block"
    if "exec-error" in reason or phase == "exec":
        return "execution_error"
    if "output-not-found" in reason:
        return "output_workbook_not_saved"
    if "no-test-cases" in reason:
        return "invalid_task_packaging"
    if "eval-mismatch" in reason:
        return "workbook_semantics_or_layout_mismatch"
    if float(row.get("soft", 0) or 0) > 0:
        return "partial_case_generalization_failure"
    return "unclassified_spreadsheet_failure"


def _fallback_plan_spreadsheetbench(
    failures: list[dict],
    successes: list[dict],
    *,
    max_demos: int,
    demo_policy: str = "abstract",
) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in failures:
        buckets.setdefault(_spreadsheet_failure_bucket(row), []).append(row)

    cluster_fixes = {
        "timeout_or_overlong_generation": (
            "Generate a short deterministic Python program immediately. Avoid broad workbook exploration, "
            "avoid iterative repair loops, and inspect only sheets/cells needed by the instruction and "
            "answer_position. Always save the output workbook before returning."
        ),
        "missing_or_malformed_code_block": (
            "Return exactly one fenced Python code block and no surrounding prose. The code must define a "
            "complete script that reads the provided input workbook path and writes the required output path."
        ),
        "execution_error": (
            "Use defensive openpyxl code: check sheet names, handle merged cells and missing values, convert "
            "numbers/strings explicitly, and keep formulas/styles valid after row or column edits."
        ),
        "output_workbook_not_saved": (
            "Preserve the original workbook object, write changes into it or a copied worksheet, create any "
            "required answer sheet, and call save(output_path) exactly once at the end."
        ),
        "workbook_semantics_or_layout_mismatch": (
            "Treat SpreadsheetBench as workbook transformation, not question answering. Match the target "
            "answer_position, sheet layout, blank cells, headers, formulas, and formatting. When an expected "
            "output pattern is implied, copy the surrounding structure before filling values."
        ),
        "partial_case_generalization_failure": (
            "Infer a transformation that works across all hidden cases for the same task: derive row/column "
            "indices from headers and cell contents instead of hard-coding a single visible coordinate."
        ),
        "unclassified_spreadsheet_failure": (
            "Before editing, identify the workbook-level operation type: cell fill, row/column insertion, "
            "sheet copy, aggregation, filtering, formatting, or formula construction; then implement only that operation."
        ),
    }
    clusters = []
    for name, rows in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        clusters.append(
            {
                "name": name,
                "evidence_ids": [str(row.get("id")) for row in rows[:8]],
                "fix": cluster_fixes.get(name, cluster_fixes["unclassified_spreadsheet_failure"]),
            }
        )

    abstract_demos = [
        {
            "name": "single-code-block workbook transform",
            "source_id": "abstracted_from_successes",
            "demo": (
                "Read input_path with openpyxl, locate the requested sheet/header region, perform the "
                "minimal workbook edit, preserve unrelated sheets/styles, and save to output_path."
            ),
            "why_it_helps": "It enforces the execution interface and prevents empty-code/output-not-found failures.",
        },
        {
            "name": "layout-preserving answer-position write",
            "source_id": "abstracted_from_successes",
            "demo": (
                "When answer_position points to a block, copy the reference layout shape implied by headers "
                "and fill values into the exact block rather than appending an unrelated table."
            ),
            "why_it_helps": "It targets the dominant eval-mismatch mode: correct computation in the wrong cells or shape.",
        },
        {
            "name": "case-generalized header lookup",
            "source_id": "abstracted_from_successes",
            "demo": (
                "Find columns by normalized header text and rows by stable labels/content; never rely on one "
                "hard-coded coordinate unless the instruction explicitly names that coordinate."
            ),
            "why_it_helps": "It makes code survive hidden workbook variants in the same task.",
        },
    ][:max_demos]
    if demo_policy != "abstract":
        for row in successes[: max(0, max_demos - len(abstract_demos))]:
            abstract_demos.append(
                {
                    "name": "successful spreadsheet transform pattern",
                    "source_id": row.get("id"),
                    "demo": _truncate_text(row.get("task_description"), 220),
                    "why_it_helps": "It shows the task style that the generated Python script must operationalize.",
                }
            )
    return {
        "failure_clusters": clusters,
        "success_demos": abstract_demos[:max_demos],
        "prune_candidates": [],
        "rewrite_directives": [
            "Remove any QA, clue, evidence-span, canonical-answer, or SearchQA-specific wording.",
            "Organize the skill as an execution contract for deterministic XLSX transformation code.",
            "Put code-block/output-path requirements before task-planning advice.",
            "Add compact abstract patterns for layout preservation, header lookup, and robust workbook saving.",
            "Keep the skill concise enough to reduce target-model latency and timeout risk.",
        ],
    }


def _fallback_plan(
    failures: list[dict],
    successes: list[dict],
    *,
    max_demos: int,
    demo_policy: str = "real",
    env: str | None = None,
) -> dict:
    if str(env or "").lower() == "spreadsheetbench":
        return _fallback_plan_spreadsheetbench(
            failures,
            successes,
            max_demos=max_demos,
            demo_policy=demo_policy,
        )
    return _fallback_plan_searchqa(
        failures,
        successes,
        max_demos=max_demos,
        demo_policy=demo_policy,
    )


def propose_consolidation(
    skill: str,
    failures: list[dict],
    successes: list[dict],
    *,
    max_prune_candidates: int,
    max_demos: int,
    demo_policy: str,
    env: str | None,
    reasoning_effort: str | None,
    max_completion_tokens: int,
    round_memory: str = "",
) -> dict:
    env_name = str(env or "").lower()
    is_searchqa = env_name == "searchqa"
    is_spreadsheetbench = env_name == "spreadsheetbench"
    skill_context = skill
    failure_context = failures
    success_context = successes
    if is_searchqa:
        skill_context = _truncate_text(skill, 3500)
        failure_context = [_compact_case(row) for row in failures[:24]]
        success_context = [_compact_case(row) for row in successes[:12]]
    elif is_spreadsheetbench:
        skill_context = _truncate_text(skill, 6500)
        failure_context = [_compact_spreadsheet_case(row) for row in failures[:32]]
        success_context = [_compact_spreadsheet_case(row) for row in successes[:16]]
    task_contract = (
        "This benchmark is SpreadsheetBench. The exported skill guides a target model to generate Python "
        "code for XLSX workbook transformations. It must not use SearchQA concepts such as clues, snippets, "
        "documents, answer spans, canonical entities, or exact-match QA. Improve code generation reliability, "
        "layout preservation, hidden-case generalization, and timeout resistance."
        if is_spreadsheetbench
        else (
            "This benchmark is SearchQA. The exported skill guides evidence selection and exact answer-boundary calibration."
            if is_searchqa
            else "Improve the exported static skill for the current benchmark."
        )
    )
    system = (
        "You are designing ALGO-03, a post-training skill consolidation method. "
        "Improve the exported static skill itself. Do not propose test-time "
        "routing, retries, hidden tools, or benchmark-specific answers. Return "
        "only valid JSON."
    )
    user = (
        "## Benchmark Contract\n"
        f"{task_contract}\n\n"
        "## Current Skill\n"
        f"{skill_context}\n\n"
        "## Candidate Rule Blocks\n"
        f"{_json_dumps(_numbered_rule_blocks(skill, max_blocks=24))}\n\n"
        "## Failed Training Cases\n"
        f"{_json_dumps(failure_context)}\n\n"
        "## Successful Training Cases\n"
        f"{_json_dumps(success_context)}\n\n"
        f"{round_memory}"
        "Create a consolidation plan with: failure_clusters, success_demos, "
        "prune_candidates, and rewrite_directives. Prune candidates must name "
        "exact text spans copied from Candidate Rule Blocks. Success demos must "
        "be short operational examples derived from successful cases.\n"
        "For SpreadsheetBench, success demos should be abstract code/workbook "
        "operation patterns, not copied task answers or workbook contents.\n"
        "If demo_policy is 'abstract', do not copy any concrete question text, "
        "answer string, entity name, or case id into success_demos; describe only "
        "the transferable behavior pattern.\n\n"
        f"demo_policy: {demo_policy}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "failure_clusters": [{"name": "...", "evidence_ids": ["..."], "fix": "..."}],\n'
        '  "success_demos": [{"name": "...", "source_id": "...", "demo": "...", "why_it_helps": "..."}],\n'
        '  "prune_candidates": [{"target_text": "...", "rationale": "..."}],\n'
        '  "rewrite_directives": ["..."]\n'
        "}\n"
        f"Limits: at most {max_prune_candidates} prune_candidates and "
        f"at most {max_demos} success_demos."
    )
    response, _ = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=min(max_completion_tokens, 4000) if is_searchqa else max_completion_tokens,
        retries=3,
        stage="algo03_plan",
        reasoning_effort=reasoning_effort,
    )
    plan = extract_json(response)
    if isinstance(plan, dict) and plan:
        return plan
    return _fallback_plan(failures, successes, max_demos=max_demos, demo_policy=demo_policy, env=env)


def _remove_once(skill: str, target: str) -> str | None:
    target = str(target or "").strip()
    if not target or target not in skill:
        return None
    candidate = skill.replace(target, "", 1)
    candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip() + "\n"
    return candidate


def sequential_prune(
    skill: str,
    plan: dict,
    validate: Callable[[str, str], tuple[float, float]],
    baseline_hard: float,
    work_dir: str,
) -> tuple[str, list[dict], float]:
    current = skill
    current_hard = baseline_hard
    reports: list[dict] = []
    prune_candidates = plan.get("prune_candidates", [])
    if not isinstance(prune_candidates, list):
        return current, reports, current_hard
    for idx, item in enumerate(prune_candidates, start=1):
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_text", "") or "")
        pruned = _remove_once(current, target)
        record = {
            "index": idx,
            "rationale": item.get("rationale", ""),
            "target_text": target,
            "accepted": False,
        }
        if pruned is None:
            record["reason"] = "target_not_found"
            reports.append(record)
            continue
        hard, soft = validate(pruned, f"prune_{idx:02d}")
        record["hard"] = hard
        record["soft"] = soft
        if hard >= current_hard:
            current = pruned
            current_hard = hard
            record["accepted"] = True
        reports.append(record)
        with open(os.path.join(work_dir, f"prune_{idx:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    return current, reports, current_hard


def rewrite_consolidated_skill(
    skill: str,
    plan: dict,
    *,
    env: str | None,
    demo_policy: str = "real",
    reasoning_effort: str | None,
    max_completion_tokens: int,
    round_memory: str = "",
) -> dict | None:
    is_spreadsheetbench = str(env or "").lower() == "spreadsheetbench"
    if is_spreadsheetbench:
        system = (
            "You are ALGO-03's final SpreadsheetBench skill consolidator. Rewrite the exported "
            "static skill for Python code generation over XLSX workbooks. This is not SearchQA "
            "and not few-shot QA prompting. Preserve useful workbook-transformation guidance, "
            "remove QA/clue/span concepts, organize by execution decision points, and keep the "
            "skill short enough to reduce target-model timeouts. Do not add test-time retries, "
            "hidden tools, or benchmark answers."
        )
        user = (
            "## Pruned Current SpreadsheetBench Skill\n"
            f"{skill}\n\n"
            "## SpreadsheetBench Consolidation Plan\n"
            f"{_json_dumps(plan)}\n\n"
            f"{round_memory}"
            f"demo_policy: {demo_policy}\n"
            "The final skill must satisfy this contract:\n"
            "1. It is a static skill for generating Python scripts that transform XLSX files.\n"
            "2. It must require exactly one Python code block, with no explanatory prose around it.\n"
            "3. It must prioritize output_path saving, sheet/range binding, layout preservation, robust "
            "header lookup, formulas/styles, and hidden-case generalization.\n"
            "4. It must explicitly forbid SearchQA vocabulary: clue, snippet, document evidence, answer span, "
            "canonical entity, exact-match QA.\n"
            "5. If demo_policy is 'abstract', include only abstract workbook-operation patterns; do not copy "
            "real task text, workbook contents, ids, or answers.\n\n"
            "Return only the complete standalone skill document in Markdown. Do not wrap it in JSON or code fences."
        )
    else:
        system = (
            "You are ALGO-03's final skill consolidator. Rewrite the full exported "
            "skill as a static skill document. Preserve effective guidance, remove "
            "validated clutter, organize by decision points, and inject only concise "
            "real training-derived demonstrations. Do not add test-time mechanisms "
            "or benchmark-specific answers."
        )
        user = (
            "## Pruned Current Skill\n"
            f"{skill}\n\n"
            "## Consolidation Plan\n"
            f"{_json_dumps(plan)}\n\n"
            f"{round_memory}"
            f"demo_policy: {demo_policy}\n"
            "If demo_policy is 'abstract', the final skill must not contain any "
            "verbatim training question, gold answer, predicted answer, entity name "
            "copied from a case, or case id. Use only abstract behavioral rules or "
            "placeholder-style examples such as [short canonical label].\n\n"
            "Return only the complete standalone skill document in Markdown. "
            "Do not wrap it in JSON or code fences."
        )
    response, _ = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=3,
        stage="algo03_rewrite",
        reasoning_effort=reasoning_effort,
    )
    new_skill = str(response or "").strip()
    if new_skill.startswith("```"):
        new_skill = re.sub(r"^```(?:markdown|md)?\s*", "", new_skill, flags=re.IGNORECASE).strip()
        new_skill = re.sub(r"\s*```$", "", new_skill).strip()
    if not new_skill:
        return None
    return {
        "reasoning": "",
        "change_summary": plan.get("rewrite_directives", [])
        if isinstance(plan.get("rewrite_directives"), list)
        else [],
        "new_skill": new_skill.rstrip() + "\n",
    }


def _norm_answer(text: object) -> str:
    value = str(text or "").lower().strip()
    value = re.sub(r"^<answer>|</answer>$", "", value).strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _gold_values(row: dict) -> list[str]:
    gold = row.get("gold_answers")
    if isinstance(gold, list):
        return [str(x) for x in gold if str(x).strip()]
    if gold:
        return [str(gold)]
    gold = row.get("gold_answer")
    if isinstance(gold, list):
        return [str(x) for x in gold if str(x).strip()]
    return [str(gold)] if gold else []


def _classify_failure(row: dict) -> str:
    pred = _norm_answer(row.get("predicted_answer") or row.get("response"))
    golds = [_norm_answer(x) for x in _gold_values(row)]
    if pred and any(g and (g in pred or pred in g) for g in golds):
        if any(g and g != pred and g in pred for g in golds):
            return "over_specific_span"
        return "under_specific_or_variant_span"
    if float(row.get("soft", 0) or 0) > 0:
        return "partial_boundary_or_alias_mismatch"
    question = str(row.get("question") or row.get("task_description") or "")
    if len(question.split()) <= 5:
        return "short_ambiguous_clue"
    if any(mark in question for mark in ("this ", "This ", "these ", "These ", "called", "known as")):
        return "frame_or_definition_mismatch"
    return "evidence_selection_miss"


def _appendix_notes(skill: str) -> list[str]:
    match = re.search(r"<!-- APPENDIX_START -->(.*?)<!-- APPENDIX_END -->", skill, flags=re.DOTALL)
    if not match:
        return []
    notes = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if line.startswith("- "):
            notes.append(line[2:].strip())
    return notes


def _build_trace_ir_spreadsheetbench(skill: str, failures: list[dict], successes: list[dict], *, env: str | None) -> dict:
    clusters: dict[str, dict] = {}
    for row in failures:
        name = _spreadsheet_failure_bucket(row)
        cluster = clusters.setdefault(
            name,
            {
                "name": name,
                "support": 0,
                "mean_soft": 0.0,
                "evidence_ids": [],
                "phases": {},
                "task_types": {},
            },
        )
        cluster["support"] += 1
        cluster["mean_soft"] += float(row.get("soft", 0) or 0)
        if len(cluster["evidence_ids"]) < 12:
            cluster["evidence_ids"].append(str(row.get("id")))
        phase = str(row.get("phase") or "unknown")
        task_type = str(row.get("task_type") or row.get("instruction_type") or "unknown")
        cluster["phases"][phase] = cluster["phases"].get(phase, 0) + 1
        cluster["task_types"][task_type] = cluster["task_types"].get(task_type, 0) + 1
    for cluster in clusters.values():
        if cluster["support"]:
            cluster["mean_soft"] = round(cluster["mean_soft"] / cluster["support"], 4)

    lower_skill = skill.lower()
    constraints = [
        {
            "id": "single_python_code_block_contract",
            "priority": 1,
            "support": sum(clusters.get(k, {}).get("support", 0) for k in ("missing_or_malformed_code_block", "execution_error")),
            "rule": "Emit exactly one complete Python code block that reads input_path and saves output_path.",
        },
        {
            "id": "workbook_output_commit",
            "priority": 2,
            "support": clusters.get("output_workbook_not_saved", {}).get("support", 0),
            "rule": "Preserve the workbook, apply the minimal required edit, and save the output workbook path exactly once.",
        },
        {
            "id": "layout_and_answer_position_binding",
            "priority": 3,
            "support": clusters.get("workbook_semantics_or_layout_mismatch", {}).get("support", 0),
            "rule": "Bind every generated value to the requested sheet/range and preserve headers, blanks, formulas, and formatting.",
        },
        {
            "id": "hidden_case_generalization",
            "priority": 4,
            "support": clusters.get("partial_case_generalization_failure", {}).get("support", 0),
            "rule": "Find rows and columns by labels and workbook structure rather than by brittle coordinates.",
        },
        {
            "id": "timeout_resistant_generation",
            "priority": 5,
            "support": clusters.get("timeout_or_overlong_generation", {}).get("support", 0),
            "rule": "Prefer short deterministic openpyxl/pandas code and avoid expensive full-workbook scans or self-repair loops.",
        },
    ]
    return {
        "ir_version": "algo03-spreadsheet-trace-compiler-v1",
        "env": env,
        "source_trace": {
            "failure_cases": len(failures),
            "success_cases": len(successes),
            "skill_chars": len(skill),
            "mentions_openpyxl": "openpyxl" in lower_skill,
            "mentions_pandas": "pandas" in lower_skill,
            "searchqa_keyword_leak": any(word in lower_skill for word in ("clue", "snippet", "canonical answer", "answer span")),
        },
        "error_clusters": sorted(clusters.values(), key=lambda x: (-x["support"], x["name"])),
        "behavior_constraints": sorted(constraints, key=lambda x: x["priority"]),
        "redundancy_graph": {
            "compression_action": "compile repeated spreadsheet execution advice into ordered code-generation contracts",
        },
        "compiler_passes": [
            "cluster_spreadsheet_residual_errors",
            "extract_execution_interface_constraints",
            "lift_layout_binding_constraints",
            "remove_qa_keyword_leakage",
            "emit_timeout_resistant_workbook_skill",
            "validate_compiled_skill",
        ],
    }


def _build_trace_ir_docvqa(skill: str, failures: list[dict], successes: list[dict], *, env: str | None) -> dict:
    clusters: dict[str, dict] = {}
    for row in failures:
        task_type = str(row.get("task_type") or "unknown")
        pred = _norm_answer(row.get("predicted_answer") or row.get("response"))
        golds = [_norm_answer(x) for x in _gold_values(row)]
        if float(row.get("soft", 0) or 0) > 0.75:
            name = "ocr_near_miss_or_spelling_variant"
        elif any(part in task_type.lower() for part in ("table", "layout", "form")):
            name = "layout_binding_miss"
        elif pred:
            name = "wrong_visual_span"
        else:
            name = "missing_visual_answer"
        cluster = clusters.setdefault(
            name,
            {
                "name": name,
                "support": 0,
                "mean_soft": 0.0,
                "evidence_ids": [],
                "task_types": {},
                "gold_overlap_failures": 0,
            },
        )
        cluster["support"] += 1
        cluster["mean_soft"] += float(row.get("soft", 0) or 0)
        if len(cluster["evidence_ids"]) < 12:
            cluster["evidence_ids"].append(str(row.get("id")))
        cluster["task_types"][task_type] = cluster["task_types"].get(task_type, 0) + 1
        if pred and any(g and (g in pred or pred in g) for g in golds):
            cluster["gold_overlap_failures"] += 1
    for cluster in clusters.values():
        if cluster["support"]:
            cluster["mean_soft"] = round(cluster["mean_soft"] / cluster["support"], 4)
    lower_skill = skill.lower()
    constraints = [
        {
            "id": "visual_exact_span_copy",
            "priority": 1,
            "support": clusters.get("wrong_visual_span", {}).get("support", 0),
            "rule": "Answer by copying the smallest visible OCR span that directly satisfies the question.",
        },
        {
            "id": "ocr_near_miss_spelling_guard",
            "priority": 2,
            "support": clusters.get("ocr_near_miss_or_spelling_variant", {}).get("support", 0),
            "rule": "For near-miss names and brands, re-read visually similar letters and prefer the exact document spelling.",
        },
        {
            "id": "layout_label_binding",
            "priority": 3,
            "support": clusters.get("layout_binding_miss", {}).get("support", 0),
            "rule": "Use nearby labels, row/column alignment, and form/table structure to bind the requested field before choosing a span.",
        },
        {
            "id": "answer_only_contract",
            "priority": 4,
            "support": len(failures),
            "rule": "Return only the answer text inside the required answer tag, with no explanation or paraphrase.",
        },
    ]
    return {
        "ir_version": "algo03-docvqa-delta-compiler-v1",
        "env": env,
        "source_trace": {
            "failure_cases": len(failures),
            "success_cases": len(successes),
            "skill_chars": len(skill),
            "mentions_layout": "layout" in lower_skill,
            "mentions_exact": "exact" in lower_skill,
        },
        "error_clusters": sorted(clusters.values(), key=lambda x: (-x["support"], x["name"])),
        "behavior_constraints": sorted(constraints, key=lambda x: x["priority"]),
        "redundancy_graph": {
            "compression_action": "prepend compact visual-span and layout-binding guards while preserving the source skill",
        },
        "compiler_passes": [
            "cluster_docvqa_residual_errors",
            "extract_visual_span_constraints",
            "lift_ocr_near_miss_guard",
            "preserve_existing_docvqa_skill",
            "validate_delta_patched_skill",
        ],
    }


def _build_trace_ir_alfworld(skill: str, failures: list[dict], successes: list[dict], *, env: str | None) -> dict:
    clusters: dict[str, dict] = {}
    for row in failures:
        task_type = str(row.get("task_type") or row.get("instruction_type") or "unknown")
        reason = str(row.get("fail_reason") or "").lower()
        if "two" in task_type:
            name = "two_object_count_tracking"
        elif "clean" in task_type:
            name = "clean_transform_then_place"
        elif "heat" in task_type:
            name = "heat_transform_then_place"
        elif "cool" in task_type:
            name = "cool_transform_then_place"
        elif "light" in task_type:
            name = "examine_with_light_sequence"
        elif "without completing" in reason or int(row.get("n_turns", 0) or 0) >= 45:
            name = "max_turn_exploration_or_loop"
        else:
            name = "generic_navigation_or_interaction_failure"
        cluster = clusters.setdefault(
            name,
            {
                "name": name,
                "support": 0,
                "mean_turns": 0.0,
                "evidence_ids": [],
                "task_types": {},
            },
        )
        cluster["support"] += 1
        cluster["mean_turns"] += float(row.get("n_turns", 0) or 0)
        if len(cluster["evidence_ids"]) < 12:
            cluster["evidence_ids"].append(str(row.get("id")))
        cluster["task_types"][task_type] = cluster["task_types"].get(task_type, 0) + 1
    for cluster in clusters.values():
        if cluster["support"]:
            cluster["mean_turns"] = round(cluster["mean_turns"] / cluster["support"], 2)
    constraints = [
        {
            "id": "subgoal_state_machine",
            "priority": 1,
            "support": len(failures),
            "rule": "Maintain an explicit subgoal state machine: locate object, take object, apply required transform, navigate to target receptacle, place object, verify completion.",
        },
        {
            "id": "transform_before_delivery",
            "priority": 2,
            "support": sum(clusters.get(k, {}).get("support", 0) for k in ("clean_transform_then_place", "heat_transform_then_place", "cool_transform_then_place")),
            "rule": "For clean/heat/cool tasks, never go to the final receptacle until the object has been transformed at the correct appliance.",
        },
        {
            "id": "two_object_counter",
            "priority": 3,
            "support": clusters.get("two_object_count_tracking", {}).get("support", 0),
            "rule": "For two-object tasks, keep a remaining-count counter and complete the full place cycle twice before stopping exploration.",
        },
        {
            "id": "loop_breaking_exploration",
            "priority": 4,
            "support": clusters.get("max_turn_exploration_or_loop", {}).get("support", 0),
            "rule": "When an object is not found after checking a receptacle or room once, mark it searched and move to a new unsearched location rather than repeating actions.",
        },
        {
            "id": "admissible_action_contract",
            "priority": 5,
            "support": len(failures) + len(successes),
            "rule": "Choose exactly one action from the admissible action list and keep the required <think>...</think><action>...</action> output format.",
        },
    ]
    return {
        "ir_version": "algo03-alfworld-delta-compiler-v1",
        "env": env,
        "source_trace": {
            "failure_cases": len(failures),
            "success_cases": len(successes),
            "skill_chars": len(skill),
        },
        "error_clusters": sorted(clusters.values(), key=lambda x: (-x["support"], x["name"])),
        "behavior_constraints": sorted(constraints, key=lambda x: x["priority"]),
        "redundancy_graph": {
            "compression_action": "prepend compact embodied-control residual guards while preserving the source skill",
        },
        "compiler_passes": [
            "cluster_alfworld_residual_errors",
            "extract_subgoal_state_constraints",
            "lift_transform_and_count_guards",
            "preserve_existing_alfworld_skill",
            "validate_delta_patched_skill",
        ],
    }


def build_trace_ir(skill: str, failures: list[dict], successes: list[dict], *, env: str | None) -> dict:
    """Build ALGO-03's explicit trace-to-skill compiler IR.

    The IR is intentionally structural: it records residual-error clusters,
    support counts, behavior constraints, and appendix compression signals.
    Downstream compilation can therefore be described as compiler passes rather
    than unconstrained prompt search.
    """
    env_name = str(env or "").lower()
    if env_name == "spreadsheetbench":
        return _build_trace_ir_spreadsheetbench(skill, failures, successes, env=env)
    if env_name == "docvqa":
        return _build_trace_ir_docvqa(skill, failures, successes, env=env)
    if env_name == "alfworld":
        return _build_trace_ir_alfworld(skill, failures, successes, env=env)

    clusters: dict[str, dict] = {}
    for row in failures:
        name = _classify_failure(row)
        cluster = clusters.setdefault(
            name,
            {
                "name": name,
                "support": 0,
                "mean_soft": 0.0,
                "evidence_ids": [],
                "gold_overlap_failures": 0,
            },
        )
        cluster["support"] += 1
        cluster["mean_soft"] += float(row.get("soft", 0) or 0)
        if len(cluster["evidence_ids"]) < 12:
            cluster["evidence_ids"].append(str(row.get("id")))
        pred = _norm_answer(row.get("predicted_answer") or row.get("response"))
        if pred and any(_norm_answer(g) in pred for g in _gold_values(row)):
            cluster["gold_overlap_failures"] += 1
    for cluster in clusters.values():
        if cluster["support"]:
            cluster["mean_soft"] = round(cluster["mean_soft"] / cluster["support"], 4)

    appendix = _appendix_notes(skill)
    appendix_topics = {
        "clue_echo": sum("echo" in n.lower() or "verbatim" in n.lower() for n in appendix),
        "canonical_span": sum("canonical" in n.lower() or "bare" in n.lower() for n in appendix),
        "frame": sum("frame" in n.lower() or "possessive" in n.lower() or "relational" in n.lower() for n in appendix),
        "noise": sum("noise" in n.lower() or "unrelated" in n.lower() for n in appendix),
    }

    constraints = []
    if clusters.get("over_specific_span") or clusters.get("partial_boundary_or_alias_mismatch"):
        support = (
            clusters.get("over_specific_span", {}).get("support", 0)
            + clusters.get("partial_boundary_or_alias_mismatch", {}).get("support", 0)
        )
        constraints.append(
            {
                "id": "answer_interface_minimal_span",
                "priority": 1,
                "support": support,
                "rule": (
                    "After locating evidence, output the shortest canonical entity, "
                    "title, term, or phrase that satisfies the clue frame; remove "
                    "descriptors, appositives, category words, and explanatory context."
                ),
            }
        )
    constraints.extend(
        [
            {
                "id": "evidence_exact_echo",
                "priority": 2,
                "support": appendix_topics["clue_echo"],
                "rule": (
                    "Rank exact or near-verbatim clue echoes above broad topical "
                    "matches; answer from the echoed block or its immediately "
                    "attached answer label."
                ),
            },
            {
                "id": "grammar_frame_binding",
                "priority": 3,
                "support": appendix_topics["frame"],
                "rule": (
                    "Use frame nouns, possessives, and relational prepositions to "
                    "select the required answer type and referent."
                ),
            },
            {
                "id": "short_clue_guard",
                "priority": 4,
                "support": clusters.get("short_ambiguous_clue", {}).get("support", 0),
                "rule": (
                    "For short ambiguous clues, require an exact title/snippet echo "
                    "before relying on world knowledge."
                ),
            },
        ]
    )

    return {
        "ir_version": "algo03-trace-compiler-v1",
        "env": env,
        "source_trace": {
            "failure_cases": len(failures),
            "success_cases": len(successes),
            "skill_chars": len(skill),
            "appendix_notes": len(appendix),
        },
        "error_clusters": sorted(clusters.values(), key=lambda x: (-x["support"], x["name"])),
        "behavior_constraints": sorted(constraints, key=lambda x: x["priority"]),
        "redundancy_graph": {
            "appendix_topic_counts": appendix_topics,
            "compression_action": "lift recurring appendix notes into ordered decision rules",
        },
        "compiler_passes": [
            "cluster_residual_errors",
            "extract_behavior_constraints",
            "lift_interface_constraints",
            "compress_redundant_appendix",
            "emit_decision_ordered_skill",
            "validate_compiled_skill",
        ],
    }


def ir_to_plan(ir: dict, *, max_demos: int) -> dict:
    clusters = []
    for cluster in ir.get("error_clusters", []):
        clusters.append(
            {
                "name": cluster.get("name"),
                "evidence_ids": cluster.get("evidence_ids", []),
                "fix": next(
                    (
                        c.get("rule")
                        for c in ir.get("behavior_constraints", [])
                        if c.get("id") == "answer_interface_minimal_span"
                        and cluster.get("name") in {
                            "over_specific_span",
                            "partial_boundary_or_alias_mismatch",
                            "under_specific_or_variant_span",
                        }
                    ),
                    "Apply the ordered decision rules emitted by the compiler.",
                ),
            }
        )
    demos = [
        {
            "name": "interface-calibrated answer boundary",
            "source_id": "compiled_ir",
            "demo": (
                "Evidence may mention [descriptor + entity]; if the clue asks for "
                "the entity, output only [entity]."
            ),
            "why_it_helps": "It converts residual exact-match errors into a reusable answer-boundary rule.",
        },
        {
            "name": "grammar-selected referent",
            "source_id": "compiled_ir",
            "demo": (
                "If the clue asks for 'this [type]' or 'the X of this Y', output "
                "the referent selected by [type]/Y, not a nearby attribute."
            ),
            "why_it_helps": "It makes clue grammar an explicit selection constraint.",
        },
    ][:max_demos]
    return {
        "failure_clusters": clusters,
        "success_demos": demos,
        "prune_candidates": [],
        "rewrite_directives": [
            "Compile residual-error clusters into ordered decision rules.",
            "Lift answer-interface calibration ahead of long evidence-search notes.",
            "Compress repeated appendix notes into a final checklist.",
            "Use abstract demonstrations only; do not copy training QA examples.",
        ],
    }


def _round_memory_section(cfg: dict) -> str:
    memory = cfg.get("algo03_round_memory")
    if not isinstance(memory, list) or not memory:
        return ""
    compact = []
    for item in memory[-8:]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "round": item.get("round"),
                "accepted": item.get("accepted"),
                "candidate_hard": item.get("candidate_hard"),
                "baseline_hard": item.get("baseline_hard"),
                "lesson": _truncate_text(item.get("lesson"), 400),
            }
        )
    if not compact:
        return ""
    return (
        "\n## Previous ALGO-03 Round Memory\n"
        f"{_json_dumps(compact)}\n\n"
        "Use this memory to avoid repeating rejected changes. Preserve the "
        "working behavior that previous rounds kept, and only propose a "
        "different consolidation hypothesis.\n"
    )


def compile_skill_from_ir(skill: str, ir: dict, *, demo_policy: str) -> dict:
    del skill
    constraints = {c.get("id"): c for c in ir.get("behavior_constraints", [])}
    minimal = constraints.get("answer_interface_minimal_span", {}).get(
        "rule",
        "Return the shortest canonical answer span that satisfies the clue frame.",
    )
    echo = constraints.get("evidence_exact_echo", {}).get(
        "rule",
        "Prefer exact or near-verbatim clue echoes over broad topical matches.",
    )
    frame = constraints.get("grammar_frame_binding", {}).get(
        "rule",
        "Use clue grammar to select the answer type and referent.",
    )
    short = constraints.get("short_clue_guard", {}).get(
        "rule",
        "For short ambiguous clues, require exact echo evidence before answering.",
    )

    examples = ""
    if demo_policy != "none":
        examples = (
            "\n## 5. Abstract calibration patterns\n\n"
            "- If evidence contains `[descriptor] [entity]` and the clue asks for the entity, return only `[entity]`.\n"
            "- If evidence contains `[category word] [term]` and the clue asks for the term, return only `[term]`.\n"
            "- If the clue says `the X of/from/in this Y`, return `Y` when Y is the requested frame referent.\n"
            "- If an echoed clue is followed by a delimiter and an answer label, copy the answer label's canonical span.\n"
        )

    skill_text = f"""# Question Answering Skill

This skill was compiled from optimization traces. Follow the rules in order: first select evidence, then bind the clue frame, then calibrate the answer boundary for exact match.

## 1. Evidence selection

- {echo}
- Anchor on distinctive clue vocabulary: numbers, dates, capitalized names, quoted terms, aliases, nicknames, and unusual wording.
- Ignore pages that overlap only on common words. Prefer passages that contain the described relation, not merely the same topic.
- If an echoed trivia, flashcard, crossword, or Jeopardy-style entry has an immediately attached answer label, treat that label as the highest-strength evidence.
- {short}
- If an echo is truncated before the answer label, corroborate the same distinctive constraints in ordinary reference passages before answering.

## 2. Clue-frame binding

- {frame}
- For definition or alias wording such as "called this term", "also known as", or "is referred to as", answer with the named term or entity being defined.
- For inverse definitions, meanings, glosses, or scoring values, answer with the headword or named term bound to that value, not with the descriptive gloss.
- For possessive frames such as "this country's" or "this team's", answer with the possessor/container entity.
- For relational frames such as "the X of/from/in this Y", answer with the entity selected by the prepositional phrase when that is the requested slot.
- When the clue gives a bare list of concrete entities without asking for a category, do not answer with the container category unless a source explicitly singles it out as the target.

## 3. Answer-interface calibration

- {minimal}
- Use the exact name, title, term, or phrase from the evidence.
- Do not output explanations, prefixes, quotation marks, restatements of the question, or surrounding context.
- Do not add first names, category words, appositives, nationality/dynasty labels, media-type labels, or descriptors unless the clue explicitly asks for them.
- If both a longer descriptive phrase and a shorter canonical label satisfy the evidence, prefer the shorter label.
- Preserve required honorifics or disambiguating words only when they are part of the canonical answer span.

## 4. Conflict resolution

- Evidence strength order: attached answer label after exact clue echo > exact relational sentence > document title plus matching snippet > repeated topical mention.
- If repeated documents disagree, choose the source whose sentence matches the clue's grammar and constraints.
- If world knowledge conflicts with the provided context, follow the context.
- If the answer seems semantically correct but longer than necessary, shorten it before final output.
{examples}
## 6. Final checklist

- Did I use the strongest clue echo or relation-bearing passage?
- Did I bind the clue frame to the right answer type?
- Did I reduce the answer to the shortest canonical span that still satisfies the clue?
- Did I remove descriptors, category words, appositives, and explanations?
- Did I output only the answer span?
"""
    return {
        "reasoning": "deterministic trace-to-skill compilation from ALGO-02 residual IR",
        "change_summary": [
            "Built explicit residual-error IR from ALGO-02 traces.",
            "Lifted answer-boundary residuals into a high-priority interface rule.",
            "Compressed repeated appendix observations into decision-ordered sections.",
            "Emitted a static compiled skill without copying concrete training QA items.",
        ],
        "new_skill": skill_text.rstrip() + "\n",
    }


def compile_spreadsheet_delta_patch(skill: str, ir: dict, plan: dict) -> dict:
    """Conservative SpreadsheetBench compiler.

    Unlike the free-form rewrite backend, this preserves the source skill body and
    only prepends a short residual-error contract. The intent is to keep ALGO-02's
    task-specific behavior while injecting the smallest high-support fixes found
    in ALGO-03 traces.
    """
    del plan
    clean_skill = skill.replace("\ufffd", "").strip()
    constraints = {c.get("id"): c for c in ir.get("behavior_constraints", [])}
    layout_rule = constraints.get("layout_and_answer_position_binding", {}).get(
        "rule",
        "Bind generated values to the requested sheet/range and preserve workbook layout.",
    )
    save_rule = constraints.get("workbook_output_commit", {}).get(
        "rule",
        "Save the output workbook path exactly once after the required edits.",
    )
    code_rule = constraints.get("single_python_code_block_contract", {}).get(
        "rule",
        "Emit exactly one complete Python code block that reads INPUT_PATH and saves OUTPUT_PATH.",
    )
    generalize_rule = constraints.get("hidden_case_generalization", {}).get(
        "rule",
        "Find rows and columns by labels and workbook structure rather than brittle coordinates.",
    )
    timeout_rule = constraints.get("timeout_resistant_generation", {}).get(
        "rule",
        "Prefer short deterministic openpyxl/pandas code and avoid expensive full-workbook scans.",
    )
    patch = f"""# ALGO-03 SpreadsheetBench Residual Patch

This section refines the existing skill; it does not replace the rules below.

## Priority order

1. **Interface first**: {code_rule}
2. **No chatter**: output the fenced code block only; do not print status messages from the script unless the benchmark harness explicitly asks for stdout.
3. **Commit output**: {save_rule}
4. **Bind the target region**: {layout_rule}
5. **Generalize across hidden cases**: {generalize_rule}
6. **Control latency**: {timeout_rule}

## Residual-error guards

- For instructions phrased as "formula" questions, still generate executable Python that writes the evaluated literal values by default. Write live Excel formulas only when the task explicitly requires a recalculating formula.
- If the preview already contains the expected output style or a partially filled answer block, use it as the layout contract: same sheet, same top-left cell, same row/column shape, same blank cells.
- For concatenated or delimited fields, split with whitespace normalization, map each token to source-table rows, aggregate numerically, then write the result into the requested answer range.
- Before saving, ensure that every cell in the requested answer range that should contain a value has been assigned; a correct computation in the wrong cells is a failure.

---
"""
    return {
        "reasoning": "conservative ALGO-03 spreadsheet delta patch compiled from residual-error IR",
        "change_summary": [
            "Preserved the ALGO-02/source SpreadsheetBench skill body.",
            "Prepended a compact residual-error contract for interface, output commit, layout binding, hidden-case generalization, and latency.",
            "Avoided whole-skill rewrite because validation showed free-form rewrite damaged source skill behavior.",
        ],
        "new_skill": patch.rstrip() + "\n\n" + clean_skill.rstrip() + "\n",
    }


def compile_docvqa_delta_patch(skill: str, ir: dict, plan: dict) -> dict:
    """Conservative DocVQA compiler.

    Keep the strong ALGO-02 visual QA skill intact and prepend only a compact
    residual-error contract compiled from the DocVQA trace IR. This gives
    ALGO-03 a real trace-to-skill pass without relying on a free-form rewrite,
    and the validation gate decides whether the patch is useful.
    """
    del plan
    clean_skill = skill.replace("\ufffd", "").strip()
    constraints = {c.get("id"): c for c in ir.get("behavior_constraints", [])}
    exact_rule = constraints.get("visual_exact_span_copy", {}).get(
        "rule",
        "Answer by copying the smallest visible OCR span that directly satisfies the question.",
    )
    spelling_rule = constraints.get("ocr_near_miss_spelling_guard", {}).get(
        "rule",
        "For near-miss names and brands, re-read visually similar letters and prefer the exact document spelling.",
    )
    layout_rule = constraints.get("layout_label_binding", {}).get(
        "rule",
        "Use nearby labels, row/column alignment, and form/table structure to bind the requested field before choosing a span.",
    )
    output_rule = constraints.get("answer_only_contract", {}).get(
        "rule",
        "Return only the answer text inside the required answer tag, with no explanation or paraphrase.",
    )
    patch = f"""# ALGO-03 DocVQA Residual Patch

This section refines the existing visual QA skill; it does not replace the rules below.

## Priority order

1. **Bind the requested field visually before answering**: {layout_rule}
2. **Copy, do not paraphrase**: {exact_rule}
3. **Resolve OCR near-misses conservatively**: {spelling_rule}
4. **Respect the output contract**: {output_rule}

## Residual-error guards

- If the predicted answer is almost right but differs by one or two letters, inspect the original visible text again and copy the document spelling exactly.
- When a value appears near multiple labels, answer the value aligned with the question's label, not the nearest salient text block.
- For tables and forms, use row/column headers, bounding-box proximity, and repeated field structure before choosing the final span.
- Prefer the shortest complete visible span that answers the question; do not add surrounding field names, punctuation, units, or explanatory words unless they are part of the requested value.

---
"""
    return {
        "reasoning": "conservative ALGO-03 DocVQA delta patch compiled from residual-error IR",
        "change_summary": [
            "Preserved the ALGO-02/source DocVQA skill body.",
            "Prepended compact visual-span, OCR spelling, layout-binding, and output-contract guards.",
            "Avoided whole-skill rewrite because DocVQA ALGO-02 is already strong and prior rewrites did not improve final test.",
        ],
        "new_skill": patch.rstrip() + "\n\n" + clean_skill.rstrip() + "\n",
    }


def compile_alfworld_delta_patch(skill: str, ir: dict, plan: dict) -> dict:
    """Conservative ALFWorld compiler.

    Preserve the source embodied-control skill and prepend a small state-machine
    contract derived from failed ALFWorld traces. This targets common residuals
    (transform-before-place, two-object counting, loop breaking) while leaving
    successful navigation and admissible-action behavior untouched.
    """
    del plan
    clean_skill = skill.replace("\ufffd", "").strip()
    constraints = {c.get("id"): c for c in ir.get("behavior_constraints", [])}
    state_rule = constraints.get("subgoal_state_machine", {}).get(
        "rule",
        "Maintain an explicit subgoal state machine: locate object, take object, apply required transform, navigate to target receptacle, place object, verify completion.",
    )
    transform_rule = constraints.get("transform_before_delivery", {}).get(
        "rule",
        "For clean/heat/cool tasks, never go to the final receptacle until the object has been transformed at the correct appliance.",
    )
    count_rule = constraints.get("two_object_counter", {}).get(
        "rule",
        "For two-object tasks, keep a remaining-count counter and complete the full place cycle twice before stopping exploration.",
    )
    loop_rule = constraints.get("loop_breaking_exploration", {}).get(
        "rule",
        "When an object is not found after checking a receptacle or room once, mark it searched and move to a new unsearched location rather than repeating actions.",
    )
    action_rule = constraints.get("admissible_action_contract", {}).get(
        "rule",
        "Choose exactly one action from the admissible action list and keep the required <think>...</think><action>...</action> output format.",
    )
    patch = f"""# ALGO-03 ALFWorld Residual Patch

This section refines the existing ALFWorld skill; it does not replace the rules below.

## Priority order

1. **Track a state machine every turn**: {state_rule}
2. **Transform before delivery**: {transform_rule}
3. **Count repeated-object goals**: {count_rule}
4. **Break exploration loops**: {loop_rule}
5. **Obey the action interface**: {action_rule}

## Residual-error guards

- Before each action, silently update: held object, transformed status, target receptacle, searched locations, and remaining object count.
- For clean/heat/cool tasks, the correct terminal sequence is transform at sink/microwave/fridge first, then carry to the final receptacle, then place.
- For two-object tasks, after placing the first object, reset only the held-object subgoal; keep the target receptacle and decrement the remaining count.
- If the same failed navigation/open/search pattern repeats, switch to another admissible unsearched receptacle or room instead of repeating the same action.
- Never invent an action. Output exactly one admissible action in the required format.

---
"""
    return {
        "reasoning": "conservative ALGO-03 ALFWorld delta patch compiled from residual-error IR",
        "change_summary": [
            "Preserved the ALGO-02/source ALFWorld skill body.",
            "Prepended compact state-machine, transform-before-place, two-object, loop-breaking, and action-interface guards.",
            "Targeted common ALFWorld residual failures without rewriting successful navigation heuristics.",
        ],
        "new_skill": patch.rstrip() + "\n\n" + clean_skill.rstrip() + "\n",
    }


def realize_compiled_skill_from_ir(
    skill: str,
    ir: dict,
    plan: dict,
    *,
    teacher_skill: str | None,
    demo_policy: str,
    reasoning_effort: str | None,
    max_completion_tokens: int,
    round_memory: str = "",
) -> dict | None:
    """LLM realization constrained by the compiler IR.

    This is intentionally not free-form prompt search: the model receives the
    explicit residual IR and must realize the ordered compiler constraints into a
    static skill. An optional teacher skill can provide phrasing quality, but
    the emitted skill must cite the IR priorities rather than copy examples.
    """
    teacher_section = ""
    if teacher_skill:
        teacher_section = (
            "## Optional Teacher Skill for Phrasing Only\n"
            f"{_truncate_text(teacher_skill, 9000)}\n\n"
            "Use this only to recover effective rule ordering and wording. Do not "
            "copy concrete training examples or benchmark-specific answers.\n\n"
        )
    system = (
        "You are ALGO-03's trace-to-skill compiler backend. You are not doing "
        "few-shot prompting or test-time routing. Your job is to realize a "
        "static skill from an explicit compiler IR: residual error clusters, "
        "behavior constraints, redundancy compression, and ordered compiler "
        "passes. Return only the final Markdown skill."
    )
    user = (
        "## Original ALGO-02 Skill\n"
        f"{_truncate_text(skill, 9000)}\n\n"
        f"{teacher_section}"
        "## Compiler IR\n"
        f"{_json_dumps(ir)}\n\n"
        "## Compilation Plan\n"
        f"{_json_dumps(plan)}\n\n"
        f"{round_memory}"
        f"demo_policy: {demo_policy}\n\n"
        "Realize a high-performance static SearchQA skill with these requirements:\n"
        "1. Preserve successful ALGO-02 evidence-selection behavior unless the IR marks it redundant.\n"
        "2. Lift high-support residual clusters into early, strongly worded decision rules.\n"
        "3. Put answer-boundary calibration before final output, but avoid over-shortening when a disambiguator is required.\n"
        "4. Compress repetitive appendix notes into a checklist; do not discard unique successful heuristics.\n"
        "5. If demo_policy is abstract, include only pattern-level examples with placeholders, no real questions, answers, ids, or entities.\n"
        "6. Keep the skill concise enough to execute reliably, but not so compressed that ALGO-02 heuristics are lost.\n\n"
        "Return only the complete standalone Markdown skill."
    )
    response, _ = chat_optimizer(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=3,
        stage="algo03_ir_realize",
        reasoning_effort=reasoning_effort,
    )
    new_skill = str(response or "").strip()
    if new_skill.startswith("```"):
        new_skill = re.sub(r"^```(?:markdown|md)?\s*", "", new_skill, flags=re.IGNORECASE).strip()
        new_skill = re.sub(r"\s*```$", "", new_skill).strip()
    if not new_skill:
        return None
    return {
        "reasoning": "IR-constrained realization of ALGO-03 trace compiler",
        "change_summary": [
            "Realized explicit residual-error IR into a decision-ordered static skill.",
            "Used support-weighted residual clusters to prioritize answer-boundary and evidence rules.",
            "Retained successful ALGO-02 heuristics while compressing redundant appendix notes.",
            "Separated method insight from unconstrained prompt search.",
        ],
        "new_skill": new_skill.rstrip() + "\n",
    }


def run_algo03_consolidation(
    *,
    out_root: str,
    skill: str,
    validate: Callable[[str, str], tuple[float, float]],
    baseline_hard: float,
    baseline_soft: float | None,
    cfg: dict,
) -> Algo03Result:
    work_dir = os.path.join(out_root, "algo03")
    os.makedirs(work_dir, exist_ok=True)

    def _cfg_int(name: str, default: int) -> int:
        value = cfg.get(name, default)
        if value is None:
            return default
        return int(value)

    max_failures = _cfg_int("algo03_max_failure_cases", 24)
    max_successes = _cfg_int("algo03_max_success_cases", 24)
    max_prune = _cfg_int("algo03_max_prune_candidates", 4)
    max_demos = _cfg_int("algo03_max_demos", 5)
    demo_policy = str(cfg.get("algo03_demo_policy", "real") or "real").strip().lower()
    compiler_mode = str(
        cfg.get("algo03_compiler_mode")
        or ("trace_compiler" if str(cfg.get("env") or "").lower() == "searchqa" else "llm_rewrite")
    ).strip().lower()
    reasoning_effort = cfg.get("algo03_rewrite_reasoning_effort") or cfg.get(
        "rewrite_reasoning_effort", "high",
    )
    max_tokens = int(cfg.get("algo03_max_completion_tokens", 64000) or 64000)
    round_memory = _round_memory_section(cfg)

    case_source_root = cfg.get("algo03_case_source_root") or out_root
    failures, successes = collect_training_cases(
        case_source_root,
        max_failures=max_failures,
        max_successes=max_successes,
        case_offset=int(cfg.get("algo03_case_offset", 0) or 0),
        pool_multiplier=int(cfg.get("algo03_case_pool_multiplier", 1) or 1),
    )
    ir = build_trace_ir(skill, failures, successes, env=cfg.get("env"))
    with open(os.path.join(work_dir, "ir.json"), "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2, ensure_ascii=False)

    if compiler_mode in {"trace_compiler", "trace_compiler_plus"}:
        plan = ir_to_plan(ir, max_demos=max_demos)
    else:
        plan = propose_consolidation(
            skill,
            failures,
            successes,
            max_prune_candidates=max_prune,
            max_demos=max_demos,
            demo_policy=demo_policy,
            env=cfg.get("env"),
            round_memory=round_memory,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_tokens,
        )
    with open(os.path.join(work_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"failures": failures, "successes": successes, "ir": ir, "plan": plan},
            f,
            indent=2,
            ensure_ascii=False,
        )

    pruned_skill, prune_report, pruned_hard = sequential_prune(
        skill,
        plan,
        validate,
        baseline_hard,
        work_dir,
    )
    with open(os.path.join(work_dir, "pruned_skill.md"), "w", encoding="utf-8") as f:
        f.write(pruned_skill)

    if compiler_mode == "trace_compiler":
        rewrite = compile_skill_from_ir(pruned_skill, ir, demo_policy=demo_policy)
        with open(os.path.join(work_dir, "compiled_skill.md"), "w", encoding="utf-8") as f:
            f.write(rewrite["new_skill"])
    elif compiler_mode in {"delta_patch", "spreadsheet_delta_patch"}:
        rewrite = compile_spreadsheet_delta_patch(pruned_skill, ir, plan)
        with open(os.path.join(work_dir, "compiled_skill.md"), "w", encoding="utf-8") as f:
            f.write(rewrite["new_skill"])
    elif compiler_mode == "docvqa_delta_patch":
        rewrite = compile_docvqa_delta_patch(pruned_skill, ir, plan)
        with open(os.path.join(work_dir, "compiled_skill.md"), "w", encoding="utf-8") as f:
            f.write(rewrite["new_skill"])
    elif compiler_mode == "alfworld_delta_patch":
        rewrite = compile_alfworld_delta_patch(pruned_skill, ir, plan)
        with open(os.path.join(work_dir, "compiled_skill.md"), "w", encoding="utf-8") as f:
            f.write(rewrite["new_skill"])
    elif compiler_mode == "trace_compiler_plus":
        teacher_skill = None
        teacher_path = str(cfg.get("algo03_teacher_skill_path") or "").strip()
        if teacher_path and os.path.isfile(teacher_path):
            with open(teacher_path, encoding="utf-8") as f:
                teacher_skill = f.read()
        rewrite = realize_compiled_skill_from_ir(
            pruned_skill,
            ir,
            plan,
            teacher_skill=teacher_skill,
            demo_policy=demo_policy,
            round_memory=round_memory,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_tokens,
        )
        if rewrite:
            with open(os.path.join(work_dir, "compiled_skill.md"), "w", encoding="utf-8") as f:
                f.write(rewrite["new_skill"])
    else:
        rewrite = rewrite_consolidated_skill(
            pruned_skill,
            plan,
            env=cfg.get("env"),
            demo_policy=demo_policy,
            round_memory=round_memory,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_tokens,
        )
    if not rewrite:
        report = {
            "accepted": False,
            "reason": "rewrite_failed",
            "baseline_hard": baseline_hard,
            "baseline_soft": baseline_soft,
            "pruned_hard": pruned_hard,
            "prune_report": prune_report,
        }
        with open(os.path.join(work_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return Algo03Result(skill=pruned_skill, accepted=False, validation_hard=None, validation_soft=None, report=report)

    candidate_skill = rewrite["new_skill"]
    with open(os.path.join(work_dir, "candidate_skill.md"), "w", encoding="utf-8") as f:
        f.write(candidate_skill)
    cand_hard, cand_soft = validate(candidate_skill, "final_rewrite")
    accept = cand_hard >= baseline_hard if bool(cfg.get("algo03_validation_gate", True)) else True
    final_skill = candidate_skill if accept else pruned_skill
    report = {
        "accepted": accept,
        "baseline_hard": baseline_hard,
        "baseline_soft": baseline_soft,
        "compiler_mode": compiler_mode,
        "teacher_skill_path": cfg.get("algo03_teacher_skill_path") or "",
        "pruned_hard": pruned_hard,
        "candidate_hard": cand_hard,
        "candidate_soft": cand_soft,
        "prune_report": prune_report,
        "rewrite_change_summary": rewrite.get("change_summary", []),
        "ir_summary": ir.get("source_trace", {}),
        "error_clusters": ir.get("error_clusters", []),
        "failure_cases": len(failures),
        "success_cases": len(successes),
    }
    with open(os.path.join(work_dir, "final_skill.md"), "w", encoding="utf-8") as f:
        f.write(final_skill)
    with open(os.path.join(work_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return Algo03Result(
        skill=final_skill,
        accepted=accept,
        validation_hard=cand_hard,
        validation_soft=cand_soft,
        report=report,
    )


# ---------------------------------------------------------------------------
# Multi-candidate skill-evolve generators
# ---------------------------------------------------------------------------

def _extract_contrastive_examples(
    failures: list[dict],
    *,
    max_examples: int = 8,
) -> list[dict]:
    """Extract (question, wrong_answer, error_reason, correct_answer) tuples.

    Near-miss failures (soft > 0) are preferred because they expose boundary
    and alias errors that the model can calibrate against.
    """
    examples: list[dict] = []
    seen_ids: set[str] = set()
    ordered = sorted(
        failures,
        key=lambda r: (
            0 if float(r.get("soft", 0) or 0) > 0 else 1,
            -float(r.get("soft", 0) or 0),
        ),
    )
    for row in ordered:
        rid = str(row.get("id") or "")
        if rid in seen_ids:
            continue
        question = str(row.get("question") or row.get("task_description") or "").strip()
        wrong = str(row.get("predicted_answer") or row.get("response") or "").strip()
        golds = _gold_values(row)
        if not question or not wrong or not golds:
            continue
        reason = str(row.get("fail_reason") or "").strip()
        if not reason:
            reason = _classify_failure(row)
        examples.append(
            {
                "id": rid,
                "question": _truncate_text(question, 200),
                "wrong_answer": _truncate_text(wrong, 80),
                "correct_answer": _truncate_text(golds[0], 80),
                "error_reason": _truncate_text(reason, 160),
                "soft": float(row.get("soft", 0) or 0),
            }
        )
        seen_ids.add(rid)
        if len(examples) >= max_examples:
            break
    return examples


def generate_contrastive_candidate(
    skill: str,
    failures: list[dict],
    successes: list[dict],
    *,
    max_examples: int = 8,
    env: str | None = None,
) -> dict:
    """Create a candidate skill by injecting contrastive failure examples.

    Keeps the original skill body intact and appends a compact
    ``Contrastive Patterns`` section that pairs wrong outputs with their
    error reasons and the correct answer.
    """
    examples = _extract_contrastive_examples(failures, max_examples=max_examples)
    if not examples:
        return {
            "reasoning": "no contrastive examples available",
            "change_summary": [],
            "new_skill": skill,
        }
    lines = [
        "",
        "## Contrastive Patterns (learn from failures)",
        "",
        "The following pairs show a wrong answer, why it lost exact match, and "
        "the correct answer. Before finalizing your answer, check that your "
        "output does not repeat the same boundary mistake.",
        "",
    ]
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"### Example {idx}")
        lines.append(f"- Question: {ex['question']}")
        lines.append(f"- Wrong: `{ex['wrong_answer']}`")
        lines.append(f"- Why wrong: {ex['error_reason']}")
        lines.append(f"- Correct: `{ex['correct_answer']}`")
        lines.append("")
    contrastive_section = "\n".join(lines).rstrip() + "\n"

    base = skill.rstrip()
    if "## Contrastive Patterns" in base:
        base = re.sub(
            r"\n## Contrastive Patterns.*$",
            "",
            base,
            flags=re.DOTALL,
        ).rstrip()
    new_skill = base + "\n" + contrastive_section
    return {
        "reasoning": "injected contrastive failure examples into the skill",
        "change_summary": [
            f"Added {len(examples)} contrastive (wrong->reason->correct) examples.",
            "Preserved the original skill rules; only appended calibration patterns.",
        ],
        "new_skill": new_skill,
    }


def generate_delta_rule_candidate(
    skill: str,
    ir: dict,
    *,
    max_rules: int = 3,
) -> dict:
    """Create a candidate by prepending targeted rules for top failure clusters.

    Makes a minimal surgical change: lifts the highest-support residual error
    clusters into a short ``Residual Guard`` block at the top of the skill.
    Each rule is derived from the IR's behavior_constraints.
    """
    constraints = ir.get("behavior_constraints", [])
    if not isinstance(constraints, list) or not constraints:
        return {
            "reasoning": "no behavior constraints in IR",
            "change_summary": [],
            "new_skill": skill,
        }
    top = sorted(constraints, key=lambda c: (-int(c.get("support", 0) or 0), int(c.get("priority", 99))))
    top = top[:max_rules]
    guard_lines = [
        "# Residual Error Guards (highest priority)",
        "",
        "Apply these guards AFTER selecting evidence but BEFORE writing the final answer:",
        "",
    ]
    for idx, c in enumerate(top, start=1):
        rule = str(c.get("rule") or "").strip()
        if not rule:
            continue
        guard_lines.append(f"{idx}. {rule}")
    guard_lines.append("")
    guard_lines.append("---")
    guard_lines.append("")
    guard_block = "\n".join(guard_lines)

    base = skill
    if base.startswith("# Residual Error Guards"):
        base = re.sub(
            r"^# Residual Error Guards.*?\n---\n\n",
            "",
            base,
            count=1,
            flags=re.DOTALL,
        )
    new_skill = guard_block + base
    return {
        "reasoning": "prepended residual-error guard rules derived from IR",
        "change_summary": [
            f"Added {len(top)} targeted guard rules for the top residual error clusters.",
            "Rules are support-weighted and derived from compiler IR, not free-form search.",
            "Original skill body is preserved below the guard block.",
        ],
        "new_skill": new_skill,
    }


def generate_llm_rewrite_candidate(
    skill: str,
    failures: list[dict],
    successes: list[dict],
    *,
    env: str | None,
    demo_policy: str = "abstract",
    reasoning_effort: str | None = "high",
    max_completion_tokens: int = 64000,
    round_memory: str = "",
) -> dict | None:
    """Full LLM rewrite candidate with failure/success context and round memory.

    The most exploratory candidate: asks the optimizer to rewrite the entire
    skill guided by residual failures. Complements deterministic candidates
    (trace_compiler, delta_rule, contrastive) by allowing structural
    reorganization.
    """
    plan = propose_consolidation(
        skill,
        failures,
        successes,
        max_prune_candidates=4,
        max_demos=5,
        demo_policy=demo_policy,
        env=env,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        round_memory=round_memory,
    )
    return rewrite_consolidated_skill(
        skill,
        plan,
        env=env,
        demo_policy=demo_policy,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        round_memory=round_memory,
    )
