"""Simple baseline inference runner for evaluation."""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from benchmarks.base import BaseDataset


@dataclass
class EvaluationConfig:
    """Configuration for baseline evaluation run."""
    
    # Model settings
    large_model_path: str
    device: str = "auto"
    
    # Dataset settings
    dataset_name: str
    num_samples: int
    
    # Output settings
    output_dir: str
    
    # Generation settings
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


class InferenceRunner:
    """Simple baseline inference runner for evaluation."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._get_device(config.device)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.large_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = self._load_model(config.large_model_path)
        
        # Results storage
        self.results = []

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_path: str):
        """Load the model."""
        if self.device.startswith("cuda"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
                use_cache=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_cache=True
            )
            model = model.to(self.device)
        
        model.eval()
        return model

    def run_evaluation(self, dataset: BaseDataset):
        """
        Run baseline evaluation on the dataset.
        
        Args:
            dataset: Loaded dataset to evaluate on
        """
        print(f"\n{'='*60}")
        print(f"Baseline Evaluation: {self.config.dataset_name}")
        print(f"Model: {self.config.large_model_path}")
        print(f"Samples: {len(dataset)}")
        print(f"{'='*60}\n")
        
        results = []
        correct = 0
        total = 0
        
        for idx in tqdm(range(len(dataset)), desc="Processing"):
            sample = dataset[idx]
            
            # Build prompt
            prompt = dataset.build_prompt(sample)
            
            # Run inference (simplified - implement actual KV cache logic)
            prediction = self._run_inference(prompt)
            
            # Check correctness
            is_correct = dataset.check_correct(sample, prediction)
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            result = {
                "sample_id": idx,
                "input": sample.get("input", ""),
                "prediction": prediction,
                "answer": dataset.get_answer(sample),
                "correct": is_correct,
            }
            results.append(result)
            
            # Print some samples
            if idx < 3 or (idx + 1) % 50 == 0:
                print(f"\n[{idx}] Pred: {prediction[:100]}... | Answer: {dataset.get_answer(sample)}")
        
        # Compute summary
        accuracy = correct / total * 100 if total > 0 else 0
        summary = {
            "dataset": self.config.dataset_name,
            "model": self.config.large_model_path,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
        }
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print("="*60)
        print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        return {"results": results, "summary": summary}

    def _run_inference(self, prompt: str) -> str:
        """
        Run simple baseline inference.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated prediction
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32000
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        
        # Decode
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return prediction

    def save_results(self, output: Dict, label: str = ""):
        """Save results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = self.config.dataset_name
        
        if label:
            output_subdir = os.path.join(self.config.output_dir, f"{dataset_name}_{label}", timestamp)
        else:
            output_subdir = os.path.join(self.config.output_dir, dataset_name, timestamp)
        
        os.makedirs(output_subdir, exist_ok=True)

        # Save detailed results
        with open(os.path.join(output_subdir, "results.json"), "w") as f:
            json.dump(output["results"], f, indent=2, ensure_ascii=False)

        # Save summary
        with open(os.path.join(output_subdir, "summary.json"), "w") as f:
            json.dump(output["summary"], f, indent=2)

        print(f"Results saved to {output_subdir}")
        return output_subdir
