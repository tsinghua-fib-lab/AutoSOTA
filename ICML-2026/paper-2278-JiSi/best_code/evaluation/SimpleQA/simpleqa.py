import os
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Any

from datasets import Dataset, disable_progress_bars

from evaluation.SimpleQA.prompts import GRADER_TEMPLATE
from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/SimpleQA"

PROMPT = """{question}""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-D])[.\s\n]?"

class SimpleQAEvaluator(BaseEvaluator):
    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        super().__init__(grader_cache_config)
        self.task = "SimpleQA"
        self.seed = 42
        
    def load_data(self, split: str = "subset_500"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.jsonl"))
            
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('problem', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test", "subset_500"]
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            question = item["problem"],
        )
        return {"prompt": prompt}
    
    
    def _get_grader_result(self, question: str, answer: str, predicted_answer: str) -> str:
        """Get grading result with automatic caching and retry"""
        prompt = GRADER_TEMPLATE.format(
            question=question,
            target=answer,
            predicted_answer=predicted_answer
        )

        # Call grader (DirectGenerator) - automatic caching + retry
        result = self.grader.generate(prompt)

        # Parse the response
        match = re.search(r"(A|B|C)", result.output)
        return match.group(0) if match else "C"

    def get_grader_result(self, question: str, answer: str, raw_data: str) -> str:
        return self._get_grader_result(question=question, answer=answer, predicted_answer=raw_data)
    
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['answer']
        question = data['problem']
        
        # Use grader to evaluate the answer
        grader_result = self.get_grader_result(question=question, answer=answer, raw_data=output_text)
        
        is_correct = grader_result == "A"
        
        return {
            "prediction": grader_result,
            "ground_truth": answer,
            "is_correct": is_correct
        }
    
