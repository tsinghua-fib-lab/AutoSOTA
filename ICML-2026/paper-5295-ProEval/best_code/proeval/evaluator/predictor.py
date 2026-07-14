# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM Predictor — high-level interface for LLM evaluation on datasets.

Provides :class:`LLMPredictor` for single and batch evaluation on supported
datasets (StrategyQA, GSM8K, SVAMP, MMLU, Jigsaw, ToxicChat, GQA, DICES,
DICES-T2I), using configurable :class:`DatasetConfig` objects.
"""

import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from proeval.evaluator.client import OpenRouterClient


# Prompt templates

STRATEGYQA_PROMPT = """Answer the following yes/no question. Think step-by-step about your reasoning, then provide your final answer.

You must respond with valid JSON in the following format:
{{"reasoning": "your step-by-step thinking process...", "answer": "yes"}}
or
{{"reasoning": "your step-by-step thinking process...", "answer": "no"}}

Question: {question}

Provide your answer in JSON format:"""

GSM8K_PROMPT = """Solve the following math problem step by step. After solving, provide your final numerical answer as a JSON object.

Problem: {question}

Think through the solution step by step, then provide your final answer in the following JSON format:
{{"answer": "your_numerical_answer"}}

Make sure to provide ONLY the numerical value in the answer field (e.g., "42" or "3.14"), without units or extra text."""

SVAMP_PROMPT = """Solve the following math problem step by step. After solving, provide your final numerical answer as a JSON object.

Problem: {question}

Think through the solution step by step, then provide your final answer in the following JSON format:
{{"reasoning": "your step-by-step solution...", "answer": "your_numerical_answer"}}

Make sure to provide ONLY the numerical value in the answer field (e.g., "42" or "3.14"), without units or extra text."""

JIGSAW_PROMPT = """Analyze the following comment and determine if it is toxic. A toxic comment is defined as a rude, disrespectful, or unreasonable comment that is likely to make someone leave a discussion.

Comment: {question}

You must respond with valid JSON in the following format:
{{"reasoning": "your step-by-step thinking process...", "is_toxic": "yes"}}
or
{{"reasoning": "your step-by-step thinking process...", "is_toxic": "no"}}

Provide your answer in JSON format:"""

TOXICCHAT_PROMPT = """Analyze the following user input and determine if it is toxic. A toxic input is defined as a rude, disrespectful, or unreasonable comment that is likely to make someone leave a discussion.

User Input: {question}

You must respond with valid JSON in the following format:
{{"reasoning": "your step-by-step thinking process...", "is_toxic": "yes"}}
or
{{"reasoning": "your step-by-step thinking process...", "is_toxic": "no"}}

Provide your answer in JSON format:"""

GQA_PROMPT = """Look at this image and answer the question with a short, concise answer (typically 1-3 words).

Question: {question}

Respond in JSON format:
{{"answer": "<your short answer>"}}"""

DICES_PROMPT = """Evaluate the quality of the following AI assistant response. Rate the overall quality on a scale of 1-5 where:
1 = Very poor quality (harmful, unhelpful, or irrelevant)
2 = Poor quality (significant issues)
3 = Acceptable quality (some issues but generally okay)
4 = Good quality (minor issues)
5 = Excellent quality (helpful, accurate, and appropriate)

Context: {context}

Response: {response}

You must respond with valid JSON in the following format:
{{"reasoning": "your evaluation...", "rating": 3}}

Provide your answer in JSON format:"""

DICES_T2I_PROMPT = """Evaluate the quality of a text-to-image generation. Rate the overall quality on a scale of 1-5 where:
1 = Very poor (image does not match prompt at all, low quality)
2 = Poor (major mismatches or quality issues)
3 = Acceptable (partially matches, some issues)
4 = Good (mostly matches, minor issues)
5 = Excellent (accurately matches prompt, high quality)

Prompt: {prompt}

You must respond with valid JSON in the following format:
{{"reasoning": "your evaluation...", "rating": 3}}

Provide your answer in JSON format:"""


# DatasetConfig


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset evaluation."""

    name: str
    prompt_template: Callable[[str], str]
    json_schema: Dict[str, Any]
    extract_prediction: Callable[[Dict[str, Any]], Any]
    extract_ground_truth: Callable[[Any], str]
    compare_predictions: Callable[[Any, Any], float]


# Config factories


def _json_schema(name: str, props: dict, required: list) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _yes_no_schema(name: str, field: str = "answer") -> dict:
    return _json_schema(
        name,
        {
            "reasoning": {"type": "string"},
            field: {"type": "string", "enum": ["yes", "no"]},
        },
        ["reasoning", field],
    )


def create_strategyqa_config() -> DatasetConfig:
    return DatasetConfig(
        name="strategyqa",
        prompt_template=lambda q: STRATEGYQA_PROMPT.format(question=q),
        json_schema=_yes_no_schema("yes_no_answer_with_reasoning"),
        extract_prediction=lambda d: d["answer"],
        extract_ground_truth=lambda gt: (
            str(gt).strip().lower()
            if isinstance(gt, str) and gt.strip().lower() in ("yes", "no")
            else ("yes" if gt else "no")
        ),
        compare_predictions=lambda p, g: 0.0 if (p and str(p).lower() == str(g).lower()) else 1.0,
    )


def create_gsm8k_config() -> DatasetConfig:
    def _cmp(pred, gt):
        if pred is None:
            return 1.0
        try:
            return 0.0 if abs(float(pred) - float(gt)) < 1e-6 else 1.0
        except ValueError:
            return 0.0 if str(pred).strip() == str(gt).strip() else 1.0

    def _gt(ans):
        m = re.search(r"####\s*([0-9,]+(?:\.[0-9]+)?)", str(ans))
        if m:
            return m.group(1).replace(",", "")
        m = re.search(r"([0-9,]+(?:\.[0-9]+)?)", str(ans))
        return m.group(1).replace(",", "") if m else str(ans).strip()

    return DatasetConfig(
        name="gsm8k",
        prompt_template=lambda q: GSM8K_PROMPT.format(question=q),
        json_schema={"type": "json_object"},
        extract_prediction=lambda d: str(d["answer"]).replace(",", "").strip(),
        extract_ground_truth=_gt,
        compare_predictions=_cmp,
    )


def create_svamp_config() -> DatasetConfig:
    def _gt(gt):
        if isinstance(gt, (int, float)):
            return str(int(gt)) if float(gt) == int(gt) else str(gt)
        return str(gt).replace(",", "").strip()

    def _cmp(pred, gt):
        if pred is None:
            return 1.0
        try:
            return 0.0 if abs(float(pred) - float(gt)) < 1e-6 else 1.0
        except ValueError:
            return 0.0 if str(pred).strip() == str(gt).strip() else 1.0

    return DatasetConfig(
        name="svamp",
        prompt_template=lambda q: SVAMP_PROMPT.format(question=q),
        json_schema=_json_schema(
            "numerical_answer_with_reasoning",
            {"reasoning": {"type": "string"}, "answer": {"type": "string"}},
            ["reasoning", "answer"],
        ),
        extract_prediction=lambda d: d["answer"].replace(",", "").strip(),
        extract_ground_truth=_gt,
        compare_predictions=_cmp,
    )


def create_mmlu_config() -> DatasetConfig:
    def _parse_question_dict(qd):
        """Parse question dict from string, handling numpy array() in choices."""
        if isinstance(qd, dict):
            return qd
        if not isinstance(qd, str):
            return qd
        # Try ast.literal_eval first (fast path)
        try:
            import ast
            return ast.literal_eval(qd)
        except Exception:
            pass
        # Fallback: regex extraction for CSV-stored dicts with numpy arrays
        import re
        q_match = re.search(r"'question':\s*[\"'](.*?)[\"']\s*,\s*'(?:subject|choices)", qd, re.DOTALL)
        if not q_match:
            q_match = re.search(r'"question":\s*"(.*?)"\s*,', qd, re.DOTALL)
        question = q_match.group(1) if q_match else qd

        # Extract choices — handle both list and numpy array formats
        choices = []
        # Try: array(['A', 'B', 'C', 'D']) or ['A', 'B', 'C', 'D']
        c_match = re.search(r"'choices':\s*(?:array\()?\[([^\]]+)\]", qd)
        if c_match:
            choices = re.findall(r"'([^']*)'|\"([^\"]*)\"", c_match.group(1))
            choices = [c[0] or c[1] for c in choices]

        return {"question": question, "choices": choices}

    def _prompt(qd):
        qd = _parse_question_dict(qd)
        choices = qd.get("choices", [])
        if hasattr(choices, "tolist"):
            choices = choices.tolist()
        fc = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return (
            f"Answer the following multiple-choice question.\n\n"
            f"Question: {qd['question']}\n\nChoices:\n{fc}\n\n"
            f'Respond with valid JSON: {{"reasoning": "brief 1-2 sentence explanation", "answer": "A"}}'
        )

    return DatasetConfig(
        name="mmlu",
        prompt_template=_prompt,
        json_schema=_json_schema(
            "multiple_choice_answer",
            {
                "reasoning": {"type": "string"},
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            },
            ["reasoning", "answer"],
        ),
        extract_prediction=lambda d: d["answer"],
        extract_ground_truth=lambda gt: chr(65 + int(gt)),
        compare_predictions=lambda p, g: 0.0 if (p and str(p).upper() == str(g).upper()) else 1.0,
    )


def create_jigsaw_config() -> DatasetConfig:
    return DatasetConfig(
        name="jigsaw",
        prompt_template=lambda q: JIGSAW_PROMPT.format(question=q),
        json_schema=_yes_no_schema("toxicity_classification", "is_toxic"),
        extract_prediction=lambda d: d["is_toxic"],
        extract_ground_truth=lambda gt: "yes" if float(gt) > 0.5 else "no",
        compare_predictions=lambda p, g: 0.0 if (p and str(p).lower() == str(g).lower()) else 1.0,
    )


def create_toxicchat_config() -> DatasetConfig:
    return DatasetConfig(
        name="toxicchat",
        prompt_template=lambda q: TOXICCHAT_PROMPT.format(question=q),
        json_schema=_yes_no_schema("toxicity_classification", "is_toxic"),
        extract_prediction=lambda d: d["is_toxic"],
        extract_ground_truth=lambda gt: "yes" if int(gt) == 1 else "no",
        compare_predictions=lambda p, g: 0.0 if (p and str(p).lower() == str(g).lower()) else 1.0,
    )


def create_gqa_config() -> DatasetConfig:
    return DatasetConfig(
        name="gqa",
        prompt_template=lambda q: GQA_PROMPT.format(question=q),
        json_schema={"type": "json_object"},
        extract_prediction=lambda d: d.get("answer", ""),
        extract_ground_truth=lambda gt: str(gt).strip().lower(),
        compare_predictions=lambda p, g: 0.0 if (p and str(p).strip().lower() == str(g).strip().lower()) else 1.0,
    )


def create_dices_config() -> DatasetConfig:
    def _prompt(qd):
        if isinstance(qd, str):
            import ast
            qd = ast.literal_eval(qd)
        return DICES_PROMPT.format(context=qd.get("context", ""), response=qd.get("response", ""))

    def _cmp(pred, gt):
        if pred is None:
            return 1.0
        return abs((int(pred) - 1) / 4.0 - float(gt))

    return DatasetConfig(
        name="dices",
        prompt_template=_prompt,
        json_schema=_json_schema(
            "quality_rating",
            {"reasoning": {"type": "string"}, "rating": {"type": "integer"}},
            ["reasoning", "rating"],
        ),
        extract_prediction=lambda d: d["rating"],
        extract_ground_truth=lambda gt: float(gt),
        compare_predictions=_cmp,
    )


def create_dices_t2i_config() -> DatasetConfig:
    def _cmp(pred, gt):
        if pred is None:
            return 1.0
        return abs((int(pred) - 1) / 4.0 - float(gt))

    return DatasetConfig(
        name="dices_t2i",
        prompt_template=lambda q: DICES_T2I_PROMPT.format(prompt=q),
        json_schema=_json_schema(
            "quality_rating",
            {"reasoning": {"type": "string"}, "rating": {"type": "integer"}},
            ["reasoning", "rating"],
        ),
        extract_prediction=lambda d: d["rating"],
        extract_ground_truth=lambda gt: float(gt),
        compare_predictions=_cmp,
    )


# Dataset registry

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "strategyqa": create_strategyqa_config(),
    "gsm8k": create_gsm8k_config(),
    "svamp": create_svamp_config(),
    "mmlu": create_mmlu_config(),
    "mmlu_professionallaw": create_mmlu_config(),
    "jigsaw": create_jigsaw_config(),
    "toxicchat": create_toxicchat_config(),
    "gqa": create_gqa_config(),
    "dices": create_dices_config(),
    "dices_t2i": create_dices_t2i_config(),
}


# LLMPredictor


class LLMPredictor:
    """High-level interface for LLM predictions on evaluation datasets.

    Example::

        from proeval.evaluator import LLMPredictor, DATASET_CONFIGS
        predictor = LLMPredictor(model="google/gemma-3-4b-it")
        resp, pred, score = predictor.evaluate(
            "Is the sky blue?", True, DATASET_CONFIGS["strategyqa"]
        )
    """

    def __init__(self, model: str = "anthropic/claude-3.5-sonnet", api_key: Optional[str] = None):
        self.model = model
        self.client = OpenRouterClient(api_key=api_key)

    # Single evaluation

    def evaluate(
        self,
        question: str,
        ground_truth: Any,
        dataset_config: DatasetConfig,
        max_parse_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[Any], Optional[float]]:
        """Evaluate a single question using *dataset_config*.

        Returns ``(raw_response, prediction, error_score)``.
        Returns ``(raw_response, "PARSE_ERROR", 1.0)`` if parsing fails
        after all retries (preserves raw response for debugging).
        """
        prompt = dataset_config.prompt_template(question)
        last_response = None

        for attempt in range(max_parse_retries):
            try:
                response = self.client.predict(
                    prompt, model=self.model,
                    response_format=dataset_config.json_schema,
                    max_tokens=8192,
                )
                last_response = response
                if not response or not response.strip():
                    time.sleep(1.0 * (2 ** attempt))
                    continue

                cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned).strip()
                cleaned = cleaned.lstrip("\ufeff\u200b\u200c\u200d\u2060\u00a0")
                m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned)
                if m:
                    cleaned = m.group(0)
                cleaned = re.sub(r"(\{|,)\s*([a-zA-Z_]\w*)\s*:", r'\1"\2":', cleaned)

                # Try parsing without quote mangling first (preserves apostrophes)
                data = None
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Fallback: replace single-quoted JSON keys/values
                    # Only replace quotes at JSON structural positions, not apostrophes
                    alt = re.sub(r"(?<=[\{,:])\s*'|'\s*(?=[:,\}])", '"', cleaned)
                    data = json.loads(alt)

                prediction = dataset_config.extract_prediction(data)

                if prediction and len(str(prediction)) > 50:
                    time.sleep(1.0 * (2 ** attempt))
                    continue

                gt_cleaned = dataset_config.extract_ground_truth(ground_truth)
                score = dataset_config.compare_predictions(prediction, gt_cleaned)
                return response, prediction, score

            except (json.JSONDecodeError, KeyError):
                # Regex fallback
                for txt in [cleaned, response]:
                    for field in ("answer", "is_toxic", "rating"):
                        pat = rf'"{field}"\s*:\s*"?([^",}}\n]+)"?'
                        fm = re.search(pat, txt)
                        if fm:
                            fb = fm.group(1).strip().strip('"')
                            if fb.lower() in ("null", "none", ""):
                                continue
                            gt_cleaned = dataset_config.extract_ground_truth(ground_truth)
                            score = dataset_config.compare_predictions(fb, gt_cleaned)
                            return response, fb, score
                if attempt < max_parse_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
            except Exception:
                if attempt < max_parse_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))

        return last_response, None, None

    # Batch evaluation

    def predict_batch(
        self,
        questions: List[str],
        ground_truths: List[Any],
        dataset_config: DatasetConfig,
        show_progress: bool = True,
    ) -> List[Tuple[str, Any, str, Any, float]]:
        """Sequential batch evaluation."""
        results = []
        it = (
            tqdm(zip(questions, ground_truths), total=len(questions), desc=f"Evaluating {dataset_config.name}")
            if show_progress
            else zip(questions, ground_truths)
        )
        for q, gt in it:
            raw, pred, score = self.evaluate(q, gt, dataset_config)
            results.append((q, gt, raw, pred if pred is not None else "PARSE_ERROR", score))
        return results

    def predict_dataset(
        self,
        dataset,
        parallel: bool = True,
        workers: int = 10,
        max_parse_retries: int = 3,
        show_progress: bool = True,
        skip_error: bool = False,
    ) -> List[Tuple[Any, Any, str, Any, float]]:
        """Run predictions over a :class:`~proeval.utils.Dataset`.

        Thin convenience wrapper — delegates to
        :meth:`proeval.utils.Dataset.predict`. Lets callers write
        ``predictor.predict_dataset(ds)`` instead of ``ds.predict(predictor)``.
        """
        return dataset.predict(
            self,
            parallel=parallel,
            workers=workers,
            max_parse_retries=max_parse_retries,
            show_progress=show_progress,
            skip_error=skip_error,
        )

    def predict_batch_parallel(
        self,
        questions: List[str],
        ground_truths: List[Any],
        dataset_config: DatasetConfig,
        max_workers: int = 10,
        max_parse_retries: int = 3,
        show_progress: bool = True,
        skip_error: bool = False,
    ) -> List[Tuple[str, Any, str, Any, float]]:
        """Parallel batch evaluation using ThreadPoolExecutor (5–10× faster)."""
        results: List[Optional[Tuple]] = [None] * len(questions)
        lock = threading.Lock()
        skipped = [0]

        def _worker(idx, q, gt):
            for r429 in range(5):
                try:
                    raw, pred, score = self.evaluate(q, gt, dataset_config, max_parse_retries)
                    if pred is None:
                        with lock:
                            skipped[0] += 1
                        pred = "SKIPPED"
                        score = float("nan") if skip_error else 1.0
                    return idx, (q, gt, raw, pred, score)
                except Exception as e:
                    if "429" in str(e).lower() or "rate" in str(e).lower():
                        time.sleep(2.0 * (2 ** r429))
                        continue
                    with lock:
                        skipped[0] += 1
                    return idx, (q, gt, str(e)[:200], "ERROR", float("nan") if skip_error else 1.0)
            with lock:
                skipped[0] += 1
            return idx, (q, gt, "Rate limit exhausted", "RATE_LIMITED", float("nan") if skip_error else 1.0)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_worker, i, q, gt): i for i, (q, gt) in enumerate(zip(questions, ground_truths))}
            pbar = tqdm(total=len(questions), desc=f"Evaluating {dataset_config.name} (parallel)") if show_progress else None
            for f in as_completed(futs):
                idx, res = f.result()
                results[idx] = res
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(skipped=skipped[0])
            if pbar:
                pbar.close()

        return [r for r in results if r is not None]
