"""LongBench datasets (2wikimqa, hotpotqa, musique)."""

import os
import json
import re
import string
import collections
import gc
from datetime import datetime
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
import torch
from rouge_score import rouge_scorer
import time
from .base import BaseDataset


class LongBenchDataset(BaseDataset):
    """LongBench benchmark datasets for long-context QA."""

    DATASET_PROMPTS = {
        "2wikimqa": "{context}Answer the question based on the given passages. Only give me the answer and do not output any other words. The answer should be within 5 words.\nQuestion: {input}\nAnswer:",
        "hotpotqa": "{context}Answer the question based on the given passages. Only give me the answer and do not output any other words. The answer should be within 5 words.\nQuestion: {input}\nAnswer:",
        "musique": "{context}Answer the question based on the given passages. Only give me the answer and do not output any other words. The answer should be within 5 words.\nQuestion: {input}\nAnswer:",
        "narrativeqa": "{context}\nAnswer the question based on the given story. Only give me the answer and do not output any other words.\nQuestion: {input}\nAnswer:",
        "qasper": "{context}\nAnswer the question based on the given paper. Only give me the answer and do not output any other words.\nQuestion: {input}\nAnswer:",
        "multifieldqa_en": "{context}\nAnswer the question based on the given context. Only give me the answer and do not output any other words.\nQuestion: {input}\nAnswer:",
    }

    def __init__(self, name: str, input_dir: str = "inputs",device: str = "cpu"):
        super().__init__()
        self.name = name
        self.input_dir = input_dir
        self.device = device
        self.data = None
        if name not in self.DATASET_PROMPTS:
            raise ValueError(f"Unknown LongBench dataset: {name}. Supported: {list(self.DATASET_PROMPTS.keys())}")

    def load(self, num_samples: Optional[int] = None, **kwargs) -> "LongBenchDataset":
        """Load LongBench dataset samples."""
        filepath = os.path.join(self.input_dir, f"{self.name}.jsonl")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                samples.append(json.loads(line))

        self.data = samples
        return self

    def build_prompt(self, sample: Dict) -> str:
        """Build prompt with context and question."""
        template = self.DATASET_PROMPTS[self.name]
        return template.format(
            context=sample['context'],
            input=sample['input']
        )

    def get_answer(self, sample: Dict) -> str:
        """Get ground truth answer."""
        answers = sample.get('answers', sample.get('answer', ''))
        if isinstance(answers, list) and len(answers) > 0:
            answers = answers[0]
        return answers if answers else ""

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check if prediction matches any answer using F1 score."""
        answers = sample.get('answers', sample.get('answer', ''))
        if not answers:
            return False
        
        # Ensure answers is a list
        if not isinstance(answers, list):
            answers = [answers]
        
        # Compute F1 against all answers, take max
        f1_scores = [self._compute_f1_internal(prediction, ans) for ans in answers]
        return max(f1_scores) > 0.5  # Consider correct if F1 > 0.5


    def parse_generation(self, text):
        """
        Parse generation output to extract the answer.
        Handles common prefixes and splits by punctuation.
        """
        if text is None:
            return ""
        s = str(text).strip()
        
        # Remove common answer prefixes
        prefixes = ["answer:", "final answer:", "the answer is"]
        lower = s.lower()
        for p in prefixes:
            if lower.startswith(p):
                s = s[len(p):].strip()
                break
        
        # Take only first line
        s = s.split("\n")[0].strip()
        
        # Split by common sentence delimiters and take first part
        parts = re.split(r"[,;，。；]", s)
        parts = [p.strip(" \"'") for p in parts if p.strip(" \"'")]
        if parts:
            s = parts[0]
        
        # Remove quotes
        s = s.strip(" \"'")
        
        # Normalize Yes/No answers (only exact matches, not substrings)
        if not s:
            return s
        first_word = s.split()[0].lower() if s.split() else ""
        if first_word in ["yes", "no"]:
            s = first_word.capitalize()
        
        return s


    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))




    def _compute_f1_internal(self, a_pred, a_gold):
        """
        Internal F1 computation between two strings (TOKEN-LEVEL).
        This matches LongBench's official implementation for language-agnostic evaluation.
        """
        a_pred = self.parse_generation(a_pred)

        # Normalize answers (remove articles, punctuation, lowercase, etc.)
        gold_toks = self.normalize_answer(a_gold)
        pred_toks = self.normalize_answer(a_pred)
        gold_toks = gold_toks.split()
        pred_toks = pred_toks.split()
        # Token-level F1: count common tokens
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_rl(self, pred, gold):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
        return rougeL
    
    def compute_f1(self, sample: Dict, prediction: str) -> float:
        """Compute F1 score for a sample."""
        answers = sample.get('answers', sample.get('answer', ''))
        if not answers:
            return 0.0
        
        # Ensure answers is a list
        if not isinstance(answers, list):
            answers = [answers]
        
        # Compute F1 against all answers, take max
        f1_scores = [self._compute_f1_internal(prediction, ans) for ans in answers]
        return max(f1_scores)

    def get_result_dict(self, sample: Dict, prediction: str, f1_score: float) -> Dict:
        """Build result dict for a sample."""
        return {
            'context': sample['context'],
            'input': sample['input'],
            'answer': self.get_answer(sample),
            'prediction': prediction,
            'f1_score': f1_score,
        }
    


    def evaluate(
        self,
        inference_fns: Dict[str, Callable[[Dict], Dict]],
        num_samples: Optional[int] = None,
        warmup_samples: int = 1,
    ) -> Dict[str, Dict]:
        """
        Evaluate on this dataset using multiple inference functions.
        Runs all strategies on each sample before moving to the next sample.

        Args:
            inference_fns: Dict of {strategy_key: inference_function}
            num_samples: Number of samples to run (None = all)
            warmup_samples: Number of warmup samples to run first (not counted in results)

        Returns:
            Dict of {strategy_key: {'results': [...], 'summary': {...}}}
        """
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        data = self.data
        if num_samples:
            data = data[:min(num_samples, len(data))]

        # Initialize results for all strategies
        all_results = {key: {'results': [], 'total_f1': 0.0, 'correct': 0} for key in inference_fns.keys()}

        print(f"\nEvaluating {self.name}")
        print(f"Total samples: {len(data)} ({warmup_samples} warmup + {len(data) - warmup_samples} timed)")
        print(f"Strategies: {list(inference_fns.keys())}")

        for i in tqdm(range(len(data)), desc="Processing samples"):
            is_warmup = i < warmup_samples
            sample = data[i]

            if is_warmup:
                tqdm.write(f"  [Warmup {i+1}/{warmup_samples}] Running all strategies (not counted in results)")

            # Run all strategies on this sample
            for strategy_key, infer_fn in inference_fns.items():
                output = infer_fn(sample)
                pred = output["prediction"]

                # Skip recording results for warmup samples
                if is_warmup:
                    continue

                # Compute F1 score
                f1_score = self.compute_f1(sample, pred)
                all_results[strategy_key]['total_f1'] += f1_score

                is_correct = self.check_correct(sample, pred)
                if is_correct:
                    all_results[strategy_key]['correct'] += 1

                all_results[strategy_key]['results'].append({
                    **self.get_result_dict(sample, pred, f1_score),
                    **{k: v for k, v in output.items() if k != "prediction"},
                })

            # Memory cleanup after each sample (ensure clean state for next sample)
            # Synchronize first so all async CUDA ops finish before freeing memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Compute summaries for all strategies (exclude warmup samples from count)
        num_timed_samples = len(data) - warmup_samples
        output_dict = {}
        for strategy_key in inference_fns.keys():
            results = all_results[strategy_key]['results']
            correct = all_results[strategy_key]['correct']
            total_f1 = all_results[strategy_key]['total_f1']

            accuracy = correct / num_timed_samples * 100 if num_timed_samples > 0 else 0
            avg_f1 = total_f1 / num_timed_samples if num_timed_samples > 0 else 0

            # Compute average metrics from results
            metric_keys = [k for k in results[0].keys() if k.endswith("_ms") or k.endswith("_tokens") or k == "ttft"]
            avg_metrics = {}
            for key in metric_keys:
                values = [r[key] for r in results if key in r and r[key] is not None]
                if values:
                    avg_metrics[f"avg_{key}"] = sum(values) / len(values)

            summary = {
                "dataset": self.name,
                "total_samples": len(data),
                "correct": correct,
                "accuracy": accuracy,
                "avg_f1": avg_f1,
                **avg_metrics,
            }

            output_dict[strategy_key] = {
                'results': results,
                'summary': summary,
            }

        return output_dict


    def save_results(self, output: Dict, label: str, output_dir: str = "results") -> str:
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(output_dir, f"{self.name}_{label}", timestamp)
        os.makedirs(output_subdir, exist_ok=True)

        with open(os.path.join(output_subdir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        summary = {**output["summary"], "timestamp": datetime.now().isoformat()}
        with open(os.path.join(output_subdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir
