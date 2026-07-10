import os
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/GPQA"

PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {candidates}. Think step by step before answering.

{question}

{options}"""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?"

class GPQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "GPQA"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"gpqa_diamond.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["test"]  # GPQA always loads "gpqa_diamond.json" regardless of split
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            subject = item["High-level domain"],
            question = item["question"],
            candidates = "".join(list(item["options"].keys())),
            options = "\n".join([f"{key}) {value}" for key, value in item["options"].items()])
        )
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        return self.extract_normal_answer(text=raw_data, answer_pattern=ANSWER_PATTERN)
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['answer']
        
        prediction = self.extract_raw_answer(raw_data=output_text)
        
        is_correct = answer == prediction
        
        return {
            "prediction": prediction,
            "ground_truth": answer,
            "is_correct": is_correct
        }
    
