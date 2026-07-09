"""MathVista dataset."""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Callable, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset


DEFAULT_DATASET_DIR = "./.dataset"
DEFAULT_OUTPUT_DIR = "./output"


class MathVistaDataset(BaseDataset):
    """MathVista benchmark dataset.

    Mathematical reasoning with visual context.
    Dataset: AI4Math/MathVista
    Supports both multiple choice and free-form math questions.
    """

    def __init__(self, dataset_dir: str = None, output_dir: str = None):
        self.data = None
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def load(self, split: str = "testmini"):
        ds = load_dataset(
            "AI4Math/MathVista",
            cache_dir=self.dataset_dir,
        )
        # MathVista uses 'testmini' as a common split
        available_splits = list(ds.keys())
        if split not in available_splits:
            split = available_splits[0]
        self.data = ds[split]
        return self

    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages for MathVista sample."""
        content = []

        # Handle image - could be 'image' or 'decoded_image'
        image = sample.get("image") or sample.get("decoded_image")
        if image is not None:
            content.append({"type": "image", "image": image})

        # Build question with choices if multiple choice
        question = sample.get("question", "")
        choices = sample.get("choices", [])

        if choices and len(choices) > 0:
            # Multiple choice format
            choice_text = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
            prompt = f"{question}\n{choice_text}\nAnswer with the letter of the correct choice."
        else:
            # Free-form format
            prompt = f"{question}\nProvide a numerical or short answer."

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def get_answer(self, sample: Dict) -> str:
        return str(sample.get("answer", ""))

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text."""
        text = text.replace(',', '').replace('$', '').replace('%', '')
        # Find numbers including decimals and negatives
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches:
            try:
                return float(matches[-1])  # Take last number (usually the answer)
            except ValueError:
                return None
        return None

    def _extract_choice(self, prediction: str, num_choices: int) -> Optional[str]:
        """Extract choice letter from prediction."""
        choices = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D...
        raw = prediction.strip().upper()

        # Direct letter
        if raw in choices:
            return raw
        if len(raw) >= 3 and raw[0] == '(' and raw[2] == ')' and raw[1] in choices:
            return raw[1]

        # Look for pattern like "(A)" or "A."
        for c in choices:
            if f"({c})" in prediction.upper() or f"{c}." in prediction.upper():
                return c
            if prediction.upper().startswith(c + " ") or prediction.upper().startswith(c + "."):
                return c

        # Check last line
        last = raw.splitlines()[-1] if raw else ""
        for c in choices:
            if c in last.split():
                return c

        return None

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check correctness for both MC and free-form."""
        answer = self.get_answer(sample)
        choices = sample.get("choices", [])

        if choices and len(choices) > 0:
            # Multiple choice
            extracted = self._extract_choice(prediction, len(choices))
            # Answer could be letter or index
            if answer.upper() in [chr(65 + i) for i in range(len(choices))]:
                return extracted == answer.upper()
            # Answer might be the text of the choice
            try:
                ans_idx = choices.index(answer)
                return extracted == chr(65 + ans_idx)
            except (ValueError, IndexError):
                pass
            return extracted == answer.upper()
        else:
            # Free-form numerical
            pred_num = self._extract_number(prediction)
            ans_num = self._extract_number(answer)

            if pred_num is not None and ans_num is not None:
                # Allow small tolerance for floating point
                if ans_num == 0:
                    return abs(pred_num) < 0.01
                relative_diff = abs(pred_num - ans_num) / abs(ans_num)
                return relative_diff < 0.01

            # Fallback to string match
            return answer.lower().strip() in prediction.lower()

    def get_result_dict(self, sample: Dict, prediction: str) -> Dict:
        """Build result dict for a sample."""
        choices = sample.get("choices", [])
        extracted = None
        if choices:
            extracted = self._extract_choice(prediction, len(choices))

        return {
            "pid": sample.get("pid", ""),
            "question": sample.get("question", ""),
            "choices": choices,
            "answer": self.get_answer(sample),
            "prediction": prediction,
            "prediction_choice": extracted,
            "question_type": sample.get("question_type", ""),
            "answer_type": sample.get("answer_type", ""),
            "category": sample.get("category", ""),
            "task": sample.get("task", ""),
        }

    def evaluate(
        self,
        inference_fn: Callable[[Dict], Dict],
        num_samples: int = None,
    ) -> Dict:
        """Evaluate on this dataset using provided inference function."""
        data = self.data
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))

        results = []
        correct = 0

        # Track per-category accuracy
        cat_correct = {}
        cat_total = {}

        print(f"\nEvaluating MathVista")
        print(f"Total samples: {len(data)}")

        for i in tqdm(range(len(data)), desc="Processing"):
            sample = data[i]
            output = inference_fn(sample)
            pred = output["prediction"]

            is_correct = self.check_correct(sample, pred)
            if is_correct:
                correct += 1

            # Track per-category
            category = sample.get("category", "unknown")
            cat_total[category] = cat_total.get(category, 0) + 1
            if is_correct:
                cat_correct[category] = cat_correct.get(category, 0) + 1

            results.append({
                **self.get_result_dict(sample, pred),
                "correct": is_correct,
                **{k: v for k, v in output.items() if k != "prediction"},
            })

            if i < 3 or (i + 1) % 50 == 0:
                print(f"\n[{i}] Answer: {self.get_answer(sample)} | Pred: {pred[:60]}...")

        accuracy = correct / len(data) * 100 if len(data) > 0 else 0

        # Per-category accuracy
        cat_accuracy = {}
        for c in cat_total:
            cat_accuracy[c] = cat_correct.get(c, 0) / cat_total[c] * 100

        metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens")]
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        summary = {
            "dataset": "MathVista",
            "total_samples": len(data),
            "correct": correct,
            "accuracy": accuracy,
            "category_accuracy": cat_accuracy,
            **avg_metrics,
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Dataset: MathVista")
        print(f"Overall Accuracy: {correct}/{len(data)} = {accuracy:.2f}%")
        print("\nPer-category accuracy:")
        for c, acc in sorted(cat_accuracy.items()):
            print(f"  {c}: {acc:.2f}%")
        for key, val in avg_metrics.items():
            print(f"{key}: {val:.2f}")

        return {"results": results, "summary": summary}

    def save_results(self, output: Dict, label: str) -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"mathvista_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir
