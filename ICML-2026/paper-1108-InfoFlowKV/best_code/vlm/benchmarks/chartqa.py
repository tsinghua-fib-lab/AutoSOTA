"""ChartQA dataset."""

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


class ChartQADataset(BaseDataset):
    """ChartQA benchmark dataset.

    Chart understanding with short answer questions.
    Dataset: HuggingFaceM4/ChartQA
    Uses relaxed accuracy: exact match or within 5% for numeric answers.
    """

    def __init__(self, dataset_dir: str = None, output_dir: str = None):
        self.data = None
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def load(self, split: str = "test"):
        ds = load_dataset(
            "HuggingFaceM4/ChartQA",
            cache_dir=self.dataset_dir,
        )
        self.data = ds[split]
        return self

    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages for ChartQA sample."""
        content = []
        if sample.get("image") is not None:
            content.append({"type": "image", "image": sample["image"]})

        question = sample["query"]
        prompt = f"{question}\nAnswer with a short phrase or number."
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def get_answer(self, sample: Dict) -> str:
        # ChartQA may have list of answers
        answer = sample["label"]
        if isinstance(answer, list):
            return answer[0]
        return answer

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        # Remove punctuation except decimal points in numbers
        text = re.sub(r'[^\w\s.]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text."""
        text = text.replace(',', '').replace('%', '').replace('$', '')
        # Find numbers including decimals
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return None
        return None

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check correctness with relaxed accuracy for numbers."""
        answer = self.get_answer(sample)

        # Normalize both
        norm_pred = self._normalize_text(prediction)
        norm_ans = self._normalize_text(answer)

        # Exact match
        if norm_pred == norm_ans or norm_ans in norm_pred:
            return True

        # Numeric relaxed match (within 5%)
        pred_num = self._extract_number(prediction)
        ans_num = self._extract_number(answer)

        if pred_num is not None and ans_num is not None:
            if ans_num == 0:
                return pred_num == 0
            relative_diff = abs(pred_num - ans_num) / abs(ans_num)
            if relative_diff <= 0.05:
                return True

        return False

    def get_result_dict(self, sample: Dict, prediction: str) -> Dict:
        """Build result dict for a sample."""
        return {
            "question": sample["query"],
            "answer": self.get_answer(sample),
            "prediction": prediction,
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

        print(f"\nEvaluating ChartQA")
        print(f"Total samples: {len(data)}")

        for i in tqdm(range(len(data)), desc="Processing"):
            sample = data[i]
            output = inference_fn(sample)
            pred = output["prediction"]

            is_correct = self.check_correct(sample, pred)
            if is_correct:
                correct += 1

            results.append({
                **self.get_result_dict(sample, pred),
                "correct": is_correct,
                **{k: v for k, v in output.items() if k != "prediction"},
            })

            if i < 3 or (i + 1) % 50 == 0:
                ans = self.get_answer(sample)
                print(f"\n[{i}] Answer: {ans} | Pred: {pred[:60]}...")

        accuracy = correct / len(data) * 100 if len(data) > 0 else 0

        metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens")]
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        summary = {
            "dataset": "ChartQA",
            "total_samples": len(data),
            "correct": correct,
            "accuracy": accuracy,
            **avg_metrics,
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Dataset: ChartQA")
        print(f"Relaxed Accuracy: {correct}/{len(data)} = {accuracy:.2f}%")
        for key, val in avg_metrics.items():
            print(f"{key}: {val:.2f}")

        return {"results": results, "summary": summary}

    def save_results(self, output: Dict, label: str) -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"chartqa_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir
