from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

# Import sentence splitter from shared utils; fallback when unavailable
try:
    from shared_utils import create_sentences, create_sentences_fallback, nlp
except Exception:
    from shared_utils import create_sentences_fallback as create_sentences
    nlp = None


@dataclass
class AttributionExample:
    prompt: str
    target: Optional[str] = None
    indices_to_explain: Optional[List[int]] = None
    attr_mask_indices: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttributionDataset(Iterable[AttributionExample]):
    """Base iterable for attribution-ready datasets."""

    name: str = "dataset"

    def __init__(self) -> None:
        self.examples: List[AttributionExample] = []

    def __iter__(self) -> Iterator[AttributionExample]:
        return iter(self.examples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, item):  # pragma: no cover - convenience
        return self.examples[item]


def _add_dummy_facts_to_prompt(text_sentences: Sequence[str]) -> List[str]:
    """
    Reproduces the original behaviour of interleaving dummy sentences with the
    provided text segments so attribution heads can be masked easily.
    """
    result: List[str] = []
    for sentence in text_sentences:
        result.append(sentence)
        result.append(" Unrelated Sentence.")
    return result


class MathAttributionDataset(AttributionDataset):
    """Dataset wrapper for synthetic math problems with dummy context facts."""

    name = "math"

    def __init__(self, path: str | Path, tokenizer: Any) -> None:
        super().__init__()
        data_path = Path(path)
        with data_path.open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        for entry in raw_examples:
            question_text = entry["question"]
            sentences = create_sentences(question_text, tokenizer)
            if not sentences:
                continue

            context_sentences = sentences[:-1]
            question_sentence = sentences[-1]
            if question_sentence.startswith(" "):
                question_sentence = question_sentence[1:]

            context_with_dummy = _add_dummy_facts_to_prompt(context_sentences)
            question_with_dummy = _add_dummy_facts_to_prompt([question_sentence])

            prompt = "".join(context_with_dummy) + "\n" + "".join(question_with_dummy)
            total_sentences = len(context_with_dummy) + len(question_with_dummy)
            attr_mask_indices = list(range(0, total_sentences, 2))

            self.examples.append(
                AttributionExample(
                    prompt=prompt,
                    target=None,
                    indices_to_explain=[-2],
                    attr_mask_indices=attr_mask_indices,
                    metadata={"raw_question": question_text},
                )
            )


class FactsAttributionDataset(AttributionDataset):
    """Dataset wrapper for curated factual prompts with explicit gold attributions."""

    name = "facts"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        data_path = Path(path)
        with data_path.open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        for entry in raw_examples:
            metadata = {
                key: value
                for key, value in entry.items()
                if key not in {"prompt", "target", "indices_to_explain", "attr_mask_indices"}
            }
            self.examples.append(
                AttributionExample(
                    prompt=entry["prompt"],
                    target=entry.get("target"),
                    indices_to_explain=entry.get("indices_to_explain"),
                    attr_mask_indices=entry.get("attr_mask_indices"),
                    metadata=metadata,
                )
            )


class MoreHopQAAttributionDataset(AttributionDataset):
    """Dataset wrapper for multi-hop QA prompts without explicit gold attribution."""

    name = "morehopqa"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        data_path = Path(path)
        with data_path.open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        for entry in raw_examples:
            context_chunks = ["".join(item[1]) for item in entry.get("context", [])]
            context = " ".join(context_chunks)
            prompt = context + "\n" + entry["question"]

            self.examples.append(
                AttributionExample(
                    prompt=prompt,
                    target=None,
                    indices_to_explain=[-2],
                    attr_mask_indices=None,
                    metadata={
                        "answer": entry.get("answer"),
                        "id": entry.get("_id"),
                        "original_context": entry.get("context"),
                    },
                )
            )


# added
class RulerAttributionDataset(AttributionDataset):
    """Dataset wrapper for raw RULER JSONL files with needle spans.

    Expects a JSONL file produced by repos/RULER (with added `needle_spans`).
    Each line must contain at least: `input`, `answer_prefix`, `outputs`, and
    optionally `needle_spans` with character spans relative to `input`.

    Mapping logic:
    - prompt = input + answer_prefix
    - target = answer_prefix (+ optional space) + ", ".join(outputs)
    - sentence indices computed over " " + prompt (leading space to match evaluator)
    - each span is shifted by +1 to account for that leading space
    - attr_mask_indices = union of all sentences covered by any span
    - indices_to_explain = [0] when target is present
    """

    name = "ruler"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"RULER file not found: {data_path}")

        # Use shared nlp pipeline; fallback to a naive splitter if unavailable
        if nlp is not None:
            def _sentence_bounds(text: str) -> List[tuple[int, int]]:
                doc = nlp(text)
                return [(s.start_char, s.end_char) for s in doc.sents]
        else:
            def _sentence_bounds(text: str) -> List[tuple[int, int]]:
                # Naive fallback: split on newlines, produce contiguous ranges
                bounds: List[tuple[int, int]] = []
                start = 0
                parts = text.split("\n")
                for idx, part in enumerate(parts):
                    end = start + len(part)
                    if end > start:
                        bounds.append((start, end))
                    start = end + 1
                if not bounds:
                    bounds = [(0, len(text))]
                return bounds

        def _map_spans(bounds: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]) -> List[int]:
            indices: set[int] = set()
            for start, end in spans:
                matched = False
                for i, (bs, be) in enumerate(bounds):
                    if start >= bs and end <= be:
                        indices.add(i)
                        matched = True
                        break
                if not matched:
                    # fallback: include all sentences with any overlap
                    for i, (bs, be) in enumerate(bounds):
                        if not (end <= bs or start >= be):
                            indices.add(i)
            return sorted(indices)

        def _read_jsonl(fp: Path) -> Iterator[Dict[str, Any]]:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        for entry in _read_jsonl(data_path):
            input_text: str = entry.get("input", "")
            answer_prefix: str = entry.get("answer_prefix", "")
            outputs = entry.get("outputs", []) or []

            # Build prompt/target
            prompt = input_text + answer_prefix
            if outputs:
                sep = " " if answer_prefix and not answer_prefix.endswith((" ", "\n", "\t")) else ""
                target = answer_prefix + sep + ", ".join(outputs)
            else:
                target = answer_prefix

            # Sentence bounds over leading-space prompt to match evaluator
            prompt_for_seg = " " + prompt
            bounds = _sentence_bounds(prompt_for_seg)

            # Collect spans and shift by +1 for the leading space
            spans_raw = []
            for item in entry.get("needle_spans", []) or []:
                span = item.get("span")
                if isinstance(span, list) and len(span) == 2:
                    spans_raw.append((int(span[0]) + 1, int(span[1]) + 1))

            attr_indices = _map_spans(bounds, spans_raw) if spans_raw else None

            self.examples.append(
                AttributionExample(
                    prompt=prompt,
                    target=target or None,
                    indices_to_explain=[0] if target else None,
                    attr_mask_indices=attr_indices,
                    metadata={
                        "dataset": "ruler",
                        "length": entry.get("length"),
                        "length_w_model_temp": entry.get("length_w_model_temp"),
                        "outputs": outputs,
                        "answer_prefix": answer_prefix,
                        "token_position_answer": entry.get("token_position_answer"),
                        "needle_spans": entry.get("needle_spans"),
                        "prompt_sentence_count": len(bounds),
                    },
                )
            )
