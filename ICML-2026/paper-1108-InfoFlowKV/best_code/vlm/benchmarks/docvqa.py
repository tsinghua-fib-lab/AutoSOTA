"""DocVQA dataset."""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Callable

from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDataset


DEFAULT_DATASET_DIR = "./.dataset"
DEFAULT_OUTPUT_DIR = "./output"


class DocVQADataset(BaseDataset):
    """DocVQA benchmark dataset.

    Document Visual Question Answering.
    Dataset: lmms-lab/DocVQA
    Uses ANLS (Average Normalized Levenshtein Similarity) metric.
    """

    def __init__(self, dataset_dir: str = None, output_dir: str = None):
        self.data = None
        self.dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def load(self, split: str = "validation"):
        ds = load_dataset(
            "lmms-lab/DocVQA", 'DocVQA',
            cache_dir=self.dataset_dir,
        )
        self.data = ds[split]
        return self

    def build_messages(self, sample: Dict) -> List[Dict]:
        """Build messages for DocVQA sample."""
        content = []
        if sample.get("image") is not None:
            content.append({"type": "image", "image": sample["image"]})

        question = sample["question"]
        prompt = f"{question}\nAnswer briefly based on the document."
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def get_answer(self, sample: Dict) -> str:
        # DocVQA has multiple acceptable answers
        answers = sample.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            return answers[0]
        return str(answers)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for ANLS computation."""
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _anls_score(self, prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
        """
        Compute ANLS score.
        ANLS = 1 - NL if NL < threshold else 0
        where NL = levenshtein_distance / max_len
        """
        prediction = self._normalize_text(prediction)

        max_score = 0.0
        for gt in ground_truths:
            gt = self._normalize_text(gt)
            if len(prediction) == 0 and len(gt) == 0:
                max_score = 1.0
                continue

            max_len = max(len(prediction), len(gt))
            if max_len == 0:
                continue

            distance = self._levenshtein_distance(prediction, gt)
            nl = distance / max_len

            if nl < threshold:
                score = 1 - nl
            else:
                score = 0.0

            max_score = max(max_score, score)

        return max_score

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check if prediction matches any ground truth (ANLS >= 0.5)."""
        answers = sample.get("answers", [])
        if not isinstance(answers, list):
            answers = [str(answers)]

        anls = self._anls_score(prediction, answers)
        return anls >= 0.5

    def get_result_dict(self, sample: Dict, prediction: str) -> Dict:
        """Build result dict for a sample."""
        answers = sample.get("answers", [])
        if not isinstance(answers, list):
            answers = [str(answers)]

        anls = self._anls_score(prediction, answers)

        return {
            "question_id": sample.get("questionId", ""),
            "question": sample["question"],
            "answers": answers,
            "prediction": prediction,
            "anls_score": anls,
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
        total_anls = 0.0

        print(f"\nEvaluating DocVQA")
        print(f"Total samples: {len(data)}")

        for i in tqdm(range(len(data)), desc="Processing"):
            sample = data[i]
            output = inference_fn(sample)
            pred = output["prediction"]

            result_dict = self.get_result_dict(sample, pred)
            total_anls += result_dict["anls_score"]

            results.append({
                **result_dict,
                **{k: v for k, v in output.items() if k != "prediction"},
            })

            if i < 3 or (i + 1) % 50 == 0:
                print(f"\n[{i}] Answers: {result_dict['answers'][:2]} | Pred: {pred[:60]}... | ANLS: {result_dict['anls_score']:.3f}")

        avg_anls = total_anls / len(data) * 100 if len(data) > 0 else 0

        metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens")]
        avg_metrics = {}
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)

        summary = {
            "dataset": "DocVQA",
            "total_samples": len(data),
            "avg_anls": avg_anls,
            **avg_metrics,
        }

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Dataset: DocVQA")
        print(f"Average ANLS: {avg_anls:.2f}%")
        for key, val in avg_metrics.items():
            print(f"{key}: {val:.2f}")

        return {"results": results, "summary": summary}

    def save_results(self, output: Dict, label: str) -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"docvqa_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir
