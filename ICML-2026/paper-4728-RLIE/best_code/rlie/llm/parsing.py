import re
from typing import Dict, List, Sequence

from rlie.datasets.data_loader import _normalize_label


def should_use_responses_api(model_name: str, force_flag: bool = False) -> bool:
    return force_flag or str(model_name or "").lower().startswith("gpt-5")


def extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    output = getattr(response, "output", None) or []
    parts: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def derive_label_aliases(label_text: str) -> set[str]:
    aliases: set[str] = set()
    if not label_text:
        return aliases
    normalized = label_text.lower()

    for pattern in (
        r"headline\s+\d",
        r"argument\s+\d",
        r"headline\s+number\s+\d",
        r"argument\s+number\s+\d",
    ):
        match = re.search(pattern, normalized)
        if match:
            phrase = match.group(0).strip()
            aliases.add(phrase)
            number = re.findall(r"\d+", phrase)
            if number:
                aliases.add(number[0])
            break

    if "first" in normalized:
        aliases.update({"first", "the first", "1"})
    if "second" in normalized:
        aliases.update({"second", "the second", "2"})
    if "third" in normalized:
        aliases.update({"third", "the third", "3"})

    return aliases


def allowed_answers(
    canonical_positive: Sequence[str],
    canonical_negative: Sequence[str],
    *,
    include_not_applicable: bool = True,
) -> List[str]:
    answers = [ans for ans in canonical_positive if ans]
    answers.extend(ans for ans in canonical_negative if ans)
    if include_not_applicable:
        answers.append("not applicable")
    return answers


def append_answer_constraint(messages: List[Dict[str, str]], final_options: Sequence[str]) -> List[Dict[str, str]]:
    allowed = " or ".join(f"`Final answer: {option}`" for option in final_options)
    constraint = (
        "\n\nIMPORTANT: Respond using only "
        f"{allowed}. Do not include <think> tags, reasoning, or any other text."
    )
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "user":
            messages[idx]["content"] += constraint
            break
    return messages


def parse_ollama_model(model_name: str) -> str:
    if ":" in model_name:
        return model_name.split(":", 1)[1]
    if " " in model_name:
        return model_name.split(" ", 1)[1]
    return model_name


def extract_final_answer(text: str) -> str:
    matches = list(
        re.finditer(r"final answer\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    )
    if not matches:
        return text.strip()

    answer = matches[-1].group(1).strip()
    answer = answer.splitlines()[0]
    answer = re.sub(r"^\{?\s*final answer\s*[:\-]?\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip("{}").strip()
    return answer


def parse_prediction_response(
    response: str,
    canonical_positive: Sequence[str],
    canonical_negative: Sequence[str],
    positive_synonyms: set[str],
    negative_synonyms: set[str],
    neutral_synonyms: set[str],
) -> int:
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    normalized_body = cleaned.lower()

    answer_text = extract_final_answer(cleaned)
    if "final answer" not in normalized_body:
        answer_text = ""
    if not answer_text:
        for pattern in (
            r"\{?\s*first answer:\s*([^\}\n]+)\}?",
            r"\{?\s*second answer:\s*([^\}\n]+)\}?",
            r"\{?\s*not applicable\s*\}?",
        ):
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                if match.lastindex:
                    answer_text = match.group(1).strip()
                else:
                    answer_text = "not applicable"
                break
    normalized_answer = _normalize_label(answer_text) if answer_text else ""

    positive_key = _normalize_label(canonical_positive[0]) if canonical_positive else ""
    negative_key = _normalize_label(canonical_negative[0]) if canonical_negative else ""

    if normalized_answer == positive_key:
        return 1
    if normalized_answer == negative_key:
        return -1
    if normalized_answer == "not applicable":
        return 0

    if answer_text:
        answer_lower = answer_text.lower()
        for synonym in positive_synonyms:
            if synonym in answer_lower:
                return 1
        for synonym in negative_synonyms:
            if synonym in answer_lower:
                return -1
        for synonym in neutral_synonyms:
            if synonym in answer_lower:
                return 0

    lines = [line.strip() for line in cleaned.strip().split("\n") if line.strip()]
    if lines:
        last_line = lines[-1].lower()
        for synonym in positive_synonyms:
            if synonym in last_line:
                return 1
        for synonym in negative_synonyms:
            if synonym in last_line:
                return -1
        for synonym in neutral_synonyms:
            if synonym in last_line:
                return 0

    text_lower = cleaned.lower()
    if positive_key and positive_key in text_lower:
        return 1
    if negative_key and negative_key in text_lower:
        return -1

    raise ValueError(f"Unexpected model response: {cleaned.strip() or response}")
