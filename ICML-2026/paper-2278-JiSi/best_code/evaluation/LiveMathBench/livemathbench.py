import os
from collections import Counter
from typing import Dict

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.deepscaler_rm import (extract_answer, grade_answer_mathd,
                                    grade_answer_sympy)

disable_progress_bars()


DATA_DIR = "data/livemathbench"

PROMPT = """Solve the following math problem step by step. The last line of your response should only contain your final answer inside a \\boxed{} command.

{Question}

Remember to put your final answer on the last line using the format \\boxed{$ANSWER} where $ANSWER is the answer to the problem.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class LiveMathBenchEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.task = "LIVE_MATHBENCH"
        self.seed = 42
    
    def load_data(self, split: str = "test"):
        data = self.load_jsonl(os.path.join(DATA_DIR, f"en_{split}.json"))
        
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
            
        return data
    
    def get_valid_splits(self):
        return ["test"]
    
    def format_prompt(self, item: Dict):
        # answer key: Answer
        prompt = PROMPT.replace("{Question}", item["question"])
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        import re
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
        answer = data['answer']
        
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
    
