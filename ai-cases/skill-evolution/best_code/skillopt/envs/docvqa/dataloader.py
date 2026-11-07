from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _parse_answers(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()]


def _extract_document_path(question: str) -> tuple[str, str]:
    marker = "document_path:"
    if marker not in question:
        return question.strip(), ""
    main, tail = question.split(marker, 1)
    return main.strip(), tail.strip()


def _normalize_row(row: dict[str, str]) -> dict:
    question_text, document_path = _extract_document_path(str(row.get("question") or ""))
    answers = _parse_answers(row.get("answer") or row.get("ground_truth") or "")
    image_path = str(row.get("image_path") or document_path or "").strip()
    task_type = str(row.get("topic") or row.get("category") or "docvqa").strip() or "docvqa"
    return {
        "id": str(row.get("questionId") or row.get("id") or "").strip(),
        "question": question_text,
        "answer": answers[0] if answers else "",
        "answers": answers,
        "task_type": task_type,
        "subtask": task_type,
        "image_paths": [image_path] if image_path else [],
        "image_path": image_path,
        "questionId": str(row.get("questionId") or "").strip(),
        "docId": str(row.get("docId") or "").strip(),
        "ucsf_document_id": str(row.get("ucsf_document_id") or "").strip(),
        "ucsf_document_page_no": str(row.get("ucsf_document_page_no") or "").strip(),
        "source_split": str(row.get("source_split") or "").strip(),
    }


def _normalize_json_item(item: dict) -> dict:
    """Normalize the JSON split format emitted by the materializer.

    ``SplitDataLoader.write_split_items`` and ``materialize_docvqa.py`` both
    use ``items.json``.  Keep legacy CSV support below, but do not require a
    lossy JSON-to-CSV conversion merely to load a materialized split.
    """
    answers_raw = item.get("answers") or item.get("ground_truth") or item.get("answer") or []
    if isinstance(answers_raw, str):
        answers = [answers_raw.strip()] if answers_raw.strip() else []
    else:
        answers = [str(value).strip() for value in answers_raw if str(value).strip()]
    image_path = str(item.get("image_path") or "").strip()
    image_paths = item.get("image_paths") or ([image_path] if image_path else [])
    question_types = item.get("question_types") or item.get("task_type") or "docvqa"
    if isinstance(question_types, (list, tuple)):
        task_type = ", ".join(str(value).strip() for value in question_types if str(value).strip()) or "docvqa"
    else:
        task_type = str(question_types).strip() or "docvqa"
    return {
        "id": str(item.get("id") or item.get("questionId") or "").strip(),
        "question": str(item.get("question") or "").strip(),
        "answer": answers[0] if answers else "",
        "answers": answers,
        "task_type": task_type,
        "subtask": task_type,
        "image_paths": [str(path) for path in image_paths if str(path).strip()],
        "image_path": image_path,
        "questionId": str(item.get("questionId") or item.get("id") or "").strip(),
        "docId": str(item.get("docId") or "").strip(),
        "ucsf_document_id": str(item.get("ucsf_document_id") or "").strip(),
        "ucsf_document_page_no": str(item.get("ucsf_document_page_no") or "").strip(),
        "source_split": str(item.get("source_split") or "").strip(),
    }


class DocVQADataLoader(SplitDataLoader):
    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path)
        csv_files = sorted(path.glob("*.csv"))
        if csv_files:
            with csv_files[0].open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return [_normalize_row(row) for row in reader]

        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .csv or .json file found in {split_path}")
        with json_files[0].open(encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError(f"Expected a JSON array of objects in {json_files[0]}")
        return [_normalize_json_item(item) for item in items]
