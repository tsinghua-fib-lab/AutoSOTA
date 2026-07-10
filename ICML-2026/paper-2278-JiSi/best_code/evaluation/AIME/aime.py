import os
import re
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.deepscaler_rm import (extract_answer, grade_answer_mathd,
                                    grade_answer_sympy)

disable_progress_bars()


DATA_DIR = "data/AIME"

PROMPT = """Solve the following math problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.

{Question}

Remember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class AIMEEvaluator(BaseEvaluator):
    def __init__(self, split: str):
        super().__init__()
        self.split = split
        self.task = f"AIME-{split}"
        self.seed = 42
    
    def load_data(self, split: str = "2024"):
        assert split in ["2024", "2025", "hybrid", "total"], f"Invalid split: {split}"
        
        if split == "2024":
            data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        elif split == "2025":
            data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        elif split == "hybrid":
            aime2024 = self.load_jsonl(os.path.join(DATA_DIR, "2024.json"))
            aime2025 = self.load_jsonl(os.path.join(DATA_DIR, "2025.json"))
            data = aime2024 + aime2025
        elif split == "total":
            data = self.load_jsonl(os.path.join(DATA_DIR, f"{split}.json"))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('Problem', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["2024", "2025", "hybrid", "total"]
    
    def format_prompt(self, item: Dict):
        prompt = PROMPT.replace("{Question}", item["Problem"])
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        if "Final Answer" in raw_data and "\\boxed" not in raw_data:
            answer = self.extract_normal_answer(text=raw_data, answer_pattern=ANSWER_PATTERN)
        else:
            answer = extract_answer(passage=raw_data)

        # Fallback mechanism: if answer is None, try to extract the last integer gemini-2.5-pro
        if answer is None:
            # Try to extract the last integer in the text (e.g., 123)
            integers = re.findall(r'\b\d+\b', raw_data)
            if integers:
                answer = integers[-1]
            else:
                answer = ""

        return answer
    
    def evaluate(self, data, output_text, **kwargs):
        answer = data['Answer']
        
        prediction = self.extract_raw_answer(raw_data=output_text)
        
        if "\\boxed" in answer:
            ground_truth = extract_answer(passage=answer)
        else:
            ground_truth = answer
        
        if ground_truth is None or prediction is None or prediction == "":
            is_correct = False
        else:
            is_correct = grade_answer_mathd(prediction, ground_truth) \
                or grade_answer_sympy(prediction, ground_truth)
        
        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        }
    
