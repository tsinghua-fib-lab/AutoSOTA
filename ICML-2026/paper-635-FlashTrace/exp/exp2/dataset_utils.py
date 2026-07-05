"""Dataset helpers for Experiment 2 (CoT / multi-hop faithfulness).

Named dataset_utils to avoid collision with the HF `datasets` package.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from attribution_datasets import (
    AttributionExample,
    MoreHopQAAttributionDataset,
    RulerAttributionDataset,
)


@dataclass
class CachedExample:
    prompt: str
    target: Optional[str]
    indices_to_explain: Optional[List[int]]
    attr_mask_indices: Optional[List[int]]
    sink_span: Optional[List[int]]
    thinking_span: Optional[List[int]]
    metadata: Dict[str, Any]


def read_cached_jsonl(path: Path) -> List[CachedExample]:
    examples: List[CachedExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(
                CachedExample(
                    prompt=obj["prompt"],
                    target=obj.get("target"),
                    indices_to_explain=obj.get("indices_to_explain"),
                    attr_mask_indices=obj.get("attr_mask_indices"),
                    sink_span=obj.get("sink_span"),
                    thinking_span=obj.get("thinking_span"),
                    metadata=obj.get("metadata", {}),
                )
            )
    return examples


def load_cached(path: Path, sample: Optional[int] = None, seed: int = 42) -> List[CachedExample]:
    ex = read_cached_jsonl(path)
    if sample is not None and sample < len(ex):
        random.Random(seed).shuffle(ex)
        ex = ex[:sample]
    return ex


def load_ruler(path: Path, sample: Optional[int] = None, seed: int = 42) -> List[CachedExample]:
    ds = RulerAttributionDataset(path)
    examples: List[CachedExample] = []
    ex_iter: Iterable[AttributionExample] = ds
    if sample is not None and sample < len(ds):
        ex_iter = list(ds)
        random.Random(seed).shuffle(ex_iter)
        ex_iter = ex_iter[:sample]
    for ex in ex_iter:
        examples.append(
            CachedExample(
                prompt=ex.prompt,
                target=ex.target,
                indices_to_explain=ex.indices_to_explain,
                attr_mask_indices=ex.attr_mask_indices,
                sink_span=None,
                thinking_span=None,
                metadata=ex.metadata,
            )
        )
    return examples


def load_morehopqa(
    path: str | Path = "./data/with_human_verification.json", sample: Optional[int] = None, seed: int = 42
) -> List[CachedExample]:
    ds = MoreHopQAAttributionDataset(path)
    ex_iter: Iterable[AttributionExample] = ds
    if sample is not None and sample < len(ds):
        ex_iter = list(ds)
        random.Random(seed).shuffle(ex_iter)
        ex_iter = ex_iter[:sample]
    examples: List[CachedExample] = []
    for ex in ex_iter:
        examples.append(
            CachedExample(
                prompt=ex.prompt,
                target=None,
                indices_to_explain=ex.indices_to_explain,
                attr_mask_indices=ex.attr_mask_indices,
                sink_span=None,
                thinking_span=None,
                metadata=ex.metadata,
            )
        )
    return examples


def auto_find_ruler(task: str) -> Optional[Path]:
    length_dirs = ["4096", "8192", "16384", "32768", "65536", "131072"]
    base = Path("data/ruler_multihop")
    for ld in length_dirs:
        cand = base / ld / task / "validation.jsonl"
        if cand.exists():
            return cand
    return None


def dataset_from_name(name: str) -> Optional[Path]:
    if name == "hotpotqa_long":
        return auto_find_ruler("hotpotqa_long")
    if name.startswith("vt_"):
        return auto_find_ruler(name)
    if name.startswith("niah"):
        return auto_find_ruler(name)
    p = Path(name)
    if p.exists():
        return p
    return None


_BOX_PATTERN = re.compile(r"\\box(?:ed)?\s*[\{｛](.*?)[\}｝]", flags=re.DOTALL)


def _find_box_span(text: str) -> Optional[tuple[int, int, str]]:
    """Return (start_char, end_char, answer_text) for the last \\boxed block."""
    matches = list(_BOX_PATTERN.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.start(0), m.end(0), m.group(1).strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer string inside the last \\boxed{} block."""
    match = _find_box_span(text)
    return match[2] if match else None


def _find_answer_span(text: str, answer: str) -> Optional[tuple[int, int]]:
    """Return (start_char, end_char) for the last occurrence of `answer` in text."""
    if not answer or not text:
        return None
    start = text.rfind(answer)
    if start == -1:
        return None
    return start, start + len(answer)


def split_boxed_generation(text: str) -> Optional[tuple[str, str, str]]:
    """Return (thinking_text, boxed_segment, boxed_answer) if format matches."""
    if not text:
        return None
    match = _find_box_span(text)
    if not match:
        return None

    start_char, end_char, boxed_inner = match
    boxed_segment = text[start_char:end_char].strip()
    thinking_text = text[:start_char].strip()
    trailing = text[end_char:].strip()

    if not boxed_inner or not boxed_segment:
        return None
    if trailing:
        return None
    if not thinking_text:
        return None

    return thinking_text, boxed_segment, boxed_inner


def attach_spans_from_answer(
    example: CachedExample, tokenizer, answer_text: Optional[str] = None
) -> CachedExample:
    """Attach sink/thinking spans by locating the (plain) answer in `target`.

    `answer_text` should be the extracted boxed answer; falls back to metadata or
    parsing the target when omitted. Works even when the target no longer keeps
    the \\box{} wrapper.
    """
    tgt = example.target or ""
    answer = (answer_text or "").strip()
    if not answer:
        answer = (example.metadata.get("boxed_answer") or extract_boxed_answer(tgt) or "").strip()

    metadata = dict(example.metadata)
    if answer:
        metadata.setdefault("boxed_answer", answer)

    if tokenizer is None or not tgt or not answer:
        return CachedExample(
            prompt=example.prompt,
            target=example.target,
            indices_to_explain=example.indices_to_explain,
            attr_mask_indices=example.attr_mask_indices,
            sink_span=example.sink_span,
            thinking_span=example.thinking_span,
            metadata=metadata,
        )

    span = _find_answer_span(tgt, answer)
    if span is None:
        return CachedExample(
            prompt=example.prompt,
            target=example.target,
            indices_to_explain=example.indices_to_explain,
            attr_mask_indices=example.attr_mask_indices,
            sink_span=example.sink_span,
            thinking_span=example.thinking_span,
            metadata=metadata,
        )

    span_start_char, span_end_char = span
    gen_ids = tokenizer(tgt, add_special_tokens=False, return_offsets_mapping=True)
    sink_tokens: List[int] = []
    for idx, (s, e) in enumerate(gen_ids["offset_mapping"]):
        # include tokens that overlap the answer span
        if s < span_end_char and e > span_start_char:
            sink_tokens.append(idx)
    if not sink_tokens:
        return CachedExample(
            prompt=example.prompt,
            target=example.target,
            indices_to_explain=example.indices_to_explain,
            attr_mask_indices=example.attr_mask_indices,
            sink_span=example.sink_span,
            thinking_span=example.thinking_span,
            metadata=metadata,
        )

    sink_span = [min(sink_tokens), max(sink_tokens)]
    thinking_end = max(0, sink_span[0] - 1)
    thinking_span = [0, thinking_end] if thinking_end >= 0 else sink_span

    return CachedExample(
        prompt=example.prompt,
        target=example.target,
        indices_to_explain=example.indices_to_explain,
        attr_mask_indices=example.attr_mask_indices,
        sink_span=example.sink_span or sink_span,
        thinking_span=example.thinking_span or thinking_span,
        metadata=metadata,
    )


def attach_spans_from_boxed(example: CachedExample, tokenizer) -> CachedExample:
    """Backward-compatible wrapper that first looks for \\box{} then falls back to answer text."""
    tgt = example.target
    match = _find_box_span(tgt) if tgt else None
    boxed_answer = match[2] if match else None
    return attach_spans_from_answer(example, tokenizer, boxed_answer)


def ruler_gold_prompt_token_indices(example: CachedExample, tokenizer) -> List[int]:
    """Return token indices (prompt-side) that overlap RULER `needle_spans` in metadata.

    The returned indices are with respect to `tokenizer(" " + example.prompt, add_special_tokens=False)`,
    matching the attribution pipeline's leading-space convention.
    """
    needle_spans = (example.metadata or {}).get("needle_spans") or []
    if not isinstance(needle_spans, list) or not needle_spans:
        return []

    prompt_text = " " + (example.prompt or "")
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer does not provide offset_mapping; cannot map needle_spans to tokens.")

    spans: List[tuple[int, int]] = []
    for item in needle_spans:
        if not isinstance(item, dict):
            continue
        raw = item.get("span")
        if not (isinstance(raw, list) and len(raw) == 2):
            continue
        try:
            start = int(raw[0]) + 1  # shift for leading space in prompt_text
            end = int(raw[1]) + 1
        except Exception:
            continue
        if end > start:
            spans.append((start, end))

    if not spans:
        return []

    gold: set[int] = set()
    for tok_idx, off in enumerate(offsets):
        if off is None:
            continue
        try:
            s, e = int(off[0]), int(off[1])
        except Exception:
            continue
        if e <= s:
            continue
        for span_start, span_end in spans:
            if s < span_end and e > span_start:
                gold.add(tok_idx)
                break

    return sorted(gold)


class DatasetLoader:
    """Thin loader that resolves and samples datasets for exp2."""

    def __init__(self, seed: int = 42, data_root: Path | str = Path("exp/exp2/data")) -> None:
        self.seed = seed
        self.data_root = Path(data_root)

    def _sample(self, items: List[CachedExample], sample: Optional[int]) -> List[CachedExample]:
        if sample is not None and sample < len(items):
            rnd = random.Random(self.seed)
            rnd.shuffle(items)
            items = items[:sample]
        return items

    def _cached_path(self, name: str) -> Optional[Path]:
        path = self.data_root / f"{name}.jsonl"
        return path if path.exists() else None

    def load(self, name: str, sample: Optional[int] = None) -> List[CachedExample]:
        # 1) Prefer prepared cache under exp/exp2/data
        cached_path = self._cached_path(name)
        if cached_path:
            return self._sample(load_cached(cached_path), sample)

        return self.load_raw(name, sample=sample)

    def load_raw(self, name: str, sample: Optional[int] = None) -> List[CachedExample]:
        def _looks_like_json_array(path: Path) -> bool:
            try:
                with path.open("r", encoding="utf-8") as f:
                    while True:
                        ch = f.read(1)
                        if not ch:
                            return False
                        if ch.isspace():
                            continue
                        return ch == "["
            except OSError:
                return False

        # MoreHopQA
        if name == "morehopqa":
            ex = load_morehopqa()
            for item in ex:
                if "answer" in item.metadata:
                    item.metadata.setdefault("reference_answer", item.metadata["answer"])
            return self._sample(ex, sample)

        # Allow passing the raw MoreHopQA JSON path directly.
        p = Path(name)
        if p.exists() and _looks_like_json_array(p):
            ex = load_morehopqa(p)
            for item in ex:
                if "answer" in item.metadata:
                    item.metadata.setdefault("reference_answer", item.metadata["answer"])
            return self._sample(ex, sample)

        # RULER / HotpotQA / niah / vt (all go through RulerAttributionDataset)
        resolved = dataset_from_name(name)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve dataset {name}")
        ex = load_ruler(resolved)
        for item in ex:
            outputs = item.metadata.get("outputs") or []
            if outputs:
                item.metadata.setdefault("reference_answer", ", ".join(outputs))
            if item.target and "reference_answer" not in item.metadata:
                item.metadata["reference_answer"] = item.target
        return self._sample(ex, sample)
