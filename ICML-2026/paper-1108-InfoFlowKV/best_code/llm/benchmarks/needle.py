"""Needle-in-a-Haystack dataset for long-context evaluation."""

import os
import json
import random
import re
import collections
import string
import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
from transformers import AutoTokenizer

from .base import BaseDataset


class NeedleDatasetAdapter:
    """Generates Needle-in-a-Haystack samples."""

    RANDOM_NEEDLE_CITIES = [
        "Chicago", "Yangon", "Antananarivo", "Colombo", "Almaty", "Sydney",
        "Chicago", "Mexico City", "Seattle", "Lagos", "Amsterdam", "Belgrade",
        "Cairo", "Baghdad", "Damascus", "Kigali", "Dakar", "Dakar", "Sofia",
        "Kigali", "Victoria", "Tashkent", "Mumbai", "Barcelona", "Almaty",
        "Amman", "Toronto", "Bratislava", "Johannesburg", "Thimphu", "Bangkok",
        "Santiago", "Cairo", "San Francisco", "Lagos", "Amsterdam", "Paris",
        "Rabat", "Santiago", "Copenhagen", "Madrid", "Kigali",
        "Ho Chi Minh City", "Sarajevo", "Delhi", "Istanbul",
        "Ho Chi Minh City", "Khartoum", "Helsinki", "Doha", "Istanbul",
        "Kuala Lumpur", "Budapest", "Shanghai", "Moscow", "Los Angeles",
        "Oslo", "Johannesburg", "Berlin", "Bangalore", "Tokyo", "Melbourne",
        "Barcelona", "Chicago", "Port Louis", "Lisbon", "Nairobi", "Kampala",
        "Lima", "Maputo", "Vancouver", "Dubai", "Khartoum", "Jakarta",
        "Madrid", "Yerevan", "Beirut", "Athens", "Chicago", "Paris",
        "Bucharest", "Copenhagen", "Brussels", "Damascus", "Seattle",
        "Los Angeles", "Yerevan", "Victoria", "Tunis", "Astana", "Seoul",
        "Buenos Aires", "Bangkok", "Colombo", "Brussels", "Khartoum", "Doha",
        "San Francisco", "Vienna", "Jakarta",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        haystack_file: str,
        *,
        digits: int = 7,
        context_buffer: int = 200,
        retrieval_question: str = "What is the special magic {} number?",
    ):
        self.tokenizer = tokenizer
        self.haystack_file = haystack_file
        self.final_context_length_buffer = context_buffer
        self.retrieval_question = retrieval_question
        self.needle_template = "\nThe special magic {city} number is: {rnd_number}\n"
        self.num_digits = digits

    def _generate_random_number(self) -> str:
        lower = 10 ** (self.num_digits - 1)
        upper = 10 ** self.num_digits - 1
        return str(random.randint(lower, upper))

    def _read_context_files(self, rounds: int, max_context_length: int) -> List[str]:
        contexts = []
        with open(self.haystack_file, "r", encoding="utf-8") as handle:
            for _ in range(rounds):
                context = ""
                toks = 0
                while toks < max_context_length:
                    line = handle.readline()
                    if not line:
                        break
                    text = json.loads(line)["text"]
                    context += text
                    toks += len(self.tokenizer.encode(text))
                if not context:
                    break
                contexts.append(context)
        return contexts

    def _insert_needle(self, context: str, needle: str, depth_percent: float, context_length: int) -> str:
        tokens_context = self.tokenizer.encode(context)
        tokens_needle = self.tokenizer.encode(needle)[1:]

        context_length -= self.final_context_length_buffer
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]
            period_tokens = [13]
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        return self.tokenizer.decode(tokens_new_context)

    def _create_context_record(
        self,
        trimmed_context: str,
        context_length: int,
        depth_percent: float,
        city: str,
        needle_number: str,
        seed: int,
    ) -> Dict:
        needle = self.needle_template.format(city=city, rnd_number=needle_number)
        question = self.retrieval_question.format(city)
        full_context = self._insert_needle(trimmed_context, needle, depth_percent, context_length)
        return {
            "context": full_context,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "needle": needle,
            "question": question,
            "needle_rnd_number": needle_number,
            "seed": seed,
        }

    def build_samples(
        self,
        *,
        context_lengths: List[int],
        depth_percents: List[int],
        rounds: int,
        limit: Optional[int],
        seed: int,
    ) -> List[Dict]:
        if not os.path.exists(self.haystack_file):
            raise FileNotFoundError(f"Needle haystack file not found: {self.haystack_file}")

        max_len = max(context_lengths) if context_lengths else 0
        base_contexts = self._read_context_files(rounds, max_len)
        if len(base_contexts) < rounds:
            raise RuntimeError(
                f"Haystack file {self.haystack_file} does not have enough content for {rounds} rounds."
            )

        tokenized = [self.tokenizer.encode(text) for text in base_contexts]
        rng = random.Random(seed)
        samples: List[Dict] = []

        for context_length in context_lengths:
            trimmed_contexts = [self.tokenizer.decode(tokens[:context_length]) for tokens in tokenized]
            for depth_percent in depth_percents:
                for round_idx in range(min(rounds, len(trimmed_contexts))):
                    city = rng.choice(self.RANDOM_NEEDLE_CITIES)
                    needle_number = self._generate_random_number()
                    record = self._create_context_record(
                        trimmed_context=trimmed_contexts[round_idx],
                        context_length=context_length,
                        depth_percent=depth_percent,
                        city=city,
                        needle_number=needle_number,
                        seed=round_idx,
                    )
                    sample = {
                        "input": record["question"],
                        "context": record["context"],
                        "answers": [record["needle_rnd_number"]],
                        "_needle_meta": {
                            "context_length": record["context_length"],
                            "depth_percent": record["depth_percent"],
                            "seed": record["seed"],
                            "needle_rnd_number": record["needle_rnd_number"],
                        },
                    }
                    samples.append(sample)
                    if limit is not None and len(samples) >= limit:
                        return samples
        return samples


class NeedleDataset(BaseDataset):
    """Needle-in-a-Haystack benchmark dataset."""

    def __init__(
        self,
        name: str = "needle",
        input_dir: str = 'inputs',
        device: torch.device = None,
        min_length: int = 1000,
        max_length: int = 100000,
        context_intervals: int = 15,
        depth_intervals: int = 10,
        rounds: int = 3,
        digits: int = 7,
        context_buffer: int = 200,
        retrieval_question: str = "What is the special magic {} number?",
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.input_dir = input_dir
        self.device = device
        self.min_length = min_length
        self.max_length = max_length
        self.context_intervals = context_intervals
        self.depth_intervals = depth_intervals
        self.rounds = rounds
        self.digits = digits
        self.context_buffer = context_buffer
        self.retrieval_question = retrieval_question
        self.seed = seed
        self.data = None
        
        # Load tokenizer from Qwen model (consistent with longbench)
        from transformers import AutoTokenizer
        model_path = "/data1/cz3000/Qwen/Qwen3-14B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set haystack file path
        haystack_file = kwargs.get('haystack_file', None)
        if haystack_file is None:
            haystack_file = os.path.join(input_dir, "PaulGrahamEssays.jsonl")
        self.haystack_file = haystack_file

        self.adapter = NeedleDatasetAdapter(
            tokenizer=self.tokenizer,
            haystack_file=self.haystack_file,
            digits=digits,
            context_buffer=context_buffer,
            retrieval_question=retrieval_question,
        )

    def load(self, num_samples: Optional[int] = None, **kwargs) -> "NeedleDataset":
        """Load needle samples."""
        context_lengths = np.round(
            np.linspace(self.min_length, self.max_length, num=self.context_intervals, endpoint=True)
        ).astype(int).tolist()

        depth_percents = np.round(
            np.linspace(0, 100, num=self.depth_intervals, endpoint=True)
        ).astype(int).tolist()

        limit = num_samples if (num_samples is not None and num_samples > 0) else None
        self.data = self.adapter.build_samples(
            context_lengths=context_lengths,
            depth_percents=depth_percents,
            rounds=self.rounds,
            limit=limit,
            seed=self.seed,
        )
        return self

    def build_prompt(self, sample: Dict) -> str:
        """Build prompt with context and question."""
        return f"{sample['context']}\n\n{sample['input']}"

    def get_answer(self, sample: Dict) -> str:
        """Get ground truth answer."""
        return sample['answers'][0]

    def check_correct(self, sample: Dict, prediction: str) -> bool:
        """Check if prediction contains the exact answer number."""
        answer = sample['answers'][0]
        # Simple check: does the answer appear in the prediction?
        return answer in prediction
    
    def parse_generation(self, text: str) -> str:
        """Parse generation to extract answer (just return text for needle)."""
        if text is None:
            return ""
        return str(text).strip()
    
    def get_result_dict(self, sample: Dict, prediction: str, is_correct: bool) -> Dict:
        """Build result dict for a sample."""
        meta = sample.get('_needle_meta', {})
        return {
            'input': sample['input'],
            'answer': self.get_answer(sample),
            'prediction': prediction,
            'correct': is_correct,
            'context_length': meta.get('context_length'),
            'depth_percent': meta.get('depth_percent'),
            'seed': meta.get('seed'),
        }
    
    def evaluate(
        self,
        inference_fns: Dict[str, Callable[[Dict], Dict]],
        num_samples: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate on needle dataset using multiple inference functions.
        Runs all strategies on each sample before moving to the next sample.
        
        Args:
            inference_fns: Dict of {strategy_key: inference_function}
            num_samples: Number of samples to run (None = all)
        
        Returns:
            Dict of {strategy_key: {'results': [...], 'summary': {...}}}
        """
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        data = self.data
        if num_samples:
            data = data[:min(num_samples, len(data))]
        
        # Initialize results for all strategies
        all_results = {
            key: {'results': [], 'correct': 0} 
            for key in inference_fns.keys()
        }
        
        print(f"\nEvaluating needle-in-haystack")
        print(f"Total samples: {len(data)}")
        print(f"Strategies: {list(inference_fns.keys())}")
        
        for i in tqdm(range(len(data)), desc="Processing samples"):
            sample = data[i]
            
            # Run all strategies on this sample
            for strategy_key, infer_fn in inference_fns.items():
                output = infer_fn(sample)
                pred = output["prediction"]
                
                # Check if correct (simple string match)
                is_correct = self.check_correct(sample, pred)
                if is_correct:
                    all_results[strategy_key]['correct'] += 1
                
                all_results[strategy_key]['results'].append({
                    **self.get_result_dict(sample, pred, is_correct),
                    **{k: v for k, v in output.items() if k != "prediction"},
                })
            
            # Memory cleanup after each sample (ensure clean state for next sample)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Compute summaries for all strategies
        output_dict = {}
        for strategy_key in inference_fns.keys():
            results = all_results[strategy_key]['results']
            correct = all_results[strategy_key]['correct']
            
            accuracy = correct / len(results) if results else 0
            
            # Compute average TTFT and throughput
            ttfts = [r['ttft_ms'] for r in results if 'ttft_ms' in r]
            avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
            
            total_times = [r['total_time_ms'] for r in results if 'total_time_ms' in r]
            avg_total_time = sum(total_times) / len(total_times) if total_times else 0
            
            summary = {
                'dataset': 'needle',
                'num_samples': len(results),
                'accuracy': accuracy,
                'correct': correct,
                'avg_ttft_ms': avg_ttft,
                'avg_total_time_ms': avg_total_time,
            }
            
            output_dict[strategy_key] = {
                'results': results,
                'summary': summary,
            }
        
        return output_dict
