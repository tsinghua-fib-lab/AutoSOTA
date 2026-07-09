"""LongBench v2 dataset (multiple-choice, zai-org/LongBench-v2)."""

import os
import json
import re
from typing import Dict, List, Optional
from .base import BaseDataset
from .longbench import LongBenchDataset


class LongBenchV2Dataset(LongBenchDataset):
    """LongBench v2 multiple-choice benchmark dataset.

    Loads from HuggingFace zai-org/LongBench-v2 or local JSONL.
    Each sample has a long context and a 4-option multiple-choice question (A/B/C/D).
    """

    def __init__(self, input_dir: str = "inputs", device: str = "cpu"):
        BaseDataset.__init__(self)  # Skip LongBenchDataset name validation
        self.name = "longbenchv2"
        self.input_dir = input_dir
        self.device = device
        self.data = None

    def load(self, num_samples: Optional[int] = None, length_filter: Optional[List[str]] = None, **kwargs) -> "LongBenchV2Dataset":
        """Load LongBench v2 dataset from local JSONL or HuggingFace.

        Args:
            num_samples: Max number of samples to load (None = all).
            length_filter: List of allowed length categories, e.g. ["short", "medium"].
                           None means no filtering.
        """
        filepath = os.path.join(self.input_dir, "longbenchv2.jsonl")

        if os.path.exists(filepath):
            all_items = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    all_items.append(json.loads(line))
        else:
            from datasets import load_dataset
            ds = load_dataset("zai-org/LongBench-v2", split="train")
            all_items = list(ds)

        # Filter by length category if specified
        if length_filter:
            allowed = set(length_filter)
            all_items = [item for item in all_items if item.get('length', '') in allowed]

        # Apply num_samples limit after filtering
        if num_samples is not None:
            all_items = all_items[:num_samples]

        self.data = [self._format_sample(item) for item in all_items]
        return self

    def _format_sample(self, item: Dict) -> Dict:
        """Convert raw LongBench v2 item to internal sample format."""
        question = item['question']
        formatted_input = (
            f"Question: {question}\n"
            f"A. {item['choice_A']}\n"
            f"B. {item['choice_B']}\n"
            f"C. {item['choice_C']}\n"
            f"D. {item['choice_D']}"
        )
        return {
            'context': item['context'],
            'input': formatted_input,
            'answer': item['answer'],
            'answers': [item['answer']],
            '_id': item.get('_id', ''),
            'domain': item.get('domain', ''),
            'sub_domain': item.get('sub_domain', ''),
            'difficulty': item.get('difficulty', ''),
            'length': item.get('length', ''),
        }

    def build_prompt(self, sample: Dict) -> str:
        """Build MC prompt with context, question, and choices."""
        return f"{sample['context']}\n{sample['input']}\nAnswer:"

    def get_answer(self, sample: Dict) -> str:
        """Get ground truth answer letter (A/B/C/D)."""
        return sample.get('answer', '')

    def parse_generation(self, text: str) -> str:
        """Extract the answer letter (A/B/C/D) from model output."""
        if not text:
            return ""
        text = text.strip()
        # Direct letter match at start
        if text and text[0].upper() in ('A', 'B', 'C', 'D'):
            return text[0].upper()
        # Search for standalone letter pattern like "A.", "A)", "A:"
        match = re.search(r'\b([A-D])[.):,\s]', text)
        if match:
            return match.group(1).upper()
        # Last resort: first A-D letter found
        for ch in text:
            if ch.upper() in ('A', 'B', 'C', 'D'):
                return ch.upper()
        return text[:1].upper() if text else ""

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Exact match on parsed letter vs gold answer."""
        pred_letter = self.parse_generation(prediction)
        gold_letter = sample.get('answer', '').strip().upper()
        return pred_letter == gold_letter

    def compute_f1(self, sample: Dict, prediction: str) -> float:
        """For MC, return 1.0 if correct, 0.0 otherwise."""
        return 1.0 if self.check_correct(sample, prediction) else 0.0

    def get_result_dict(self, sample: Dict, prediction: str, f1_score: float) -> Dict:
        """Build result dict with MC-specific fields."""
        return {
            'context': sample['context'][:500],  # Truncate for readability in results
            'input': sample['input'],
            'answer': self.get_answer(sample),
            'prediction': prediction,
            'parsed_prediction': self.parse_generation(prediction),
            'correct': self.check_correct(sample, prediction),
            'f1_score': f1_score,
            'domain': sample.get('domain', ''),
            'sub_domain': sample.get('sub_domain', ''),
            'difficulty': sample.get('difficulty', ''),
            'length': sample.get('length', ''),
        }
