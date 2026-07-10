import os
import json
import base64
import zlib
import pickle
from pathlib import Path
from typing import Dict

from datasets import Dataset, disable_progress_bars, load_dataset

from evaluation.base_evaluator import BaseEvaluator
from evaluation.LiveCodeBench.compute_code_generation_metrics import evaluate_generation

disable_progress_bars()

DATA_DIR = os.getenv(
    "LIVECODEBENCH_DATA_DIR",
    str(Path("data/LiveCodeBench/code_generation_lite")),
)

PROMPT = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.
### Question:
{question}
""".strip()

FORMATTING_WITH_STARTER_CODE = "\n\n### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
FORMATTING_WITHOUT_STARTER_CODE = "\n\n### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"

class LiveCodeBenchEvaluator(BaseEvaluator):
    def __init__(self, split: str = "test"):
        super().__init__()
        self.task = f"LiveCodeBench_{split}"
        self.seed = 42
        self.debug = False
        self.split = split
        self._raw_data = []  # keep raw rows to lazily decode heavy fields during evaluation
        
    def load_data(self, split: str = "test"):
        # Load all JSONL files from the data directory
        all_data = []
        data_dir_path = Path(DATA_DIR)
        jsonl_files = sorted(data_dir_path.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            file_data = self.load_jsonl(jsonl_file)
            all_data.extend(file_data)

        # Save full raw rows (heavy fields stay here, not in the arrow dataset)
        self._raw_data = all_data

        # Build a lightweight list for HF Dataset to avoid Arrow offset overflows
        light_rows = []
        for idx, row in enumerate(all_data):
            light_rows.append({
                "question_content": row.get("question_content", ""),
                "starter_code": row.get("starter_code", ""),
                "metadata": row.get("metadata", "{}"),
                "__idx__": idx,
            })

        data = Dataset.from_list(light_rows)

        data = data.map(lambda x: self.format_prompt(x))

        # Add origin_query field
        data = data.map(lambda x: {**x, 'origin_query': x.get('question_content', '')})

        return data
    
    def get_valid_splits(self):
        return ["test"]  # Now loads all JSONL files regardless of split parameter
    
    def format_prompt(self, item: Dict):
        # Only build the prompt here; defer expensive private test decoding to evaluation time
        # answer key: Answer
        prompt = PROMPT.format(
            question=item["question_content"],
        )
        
        # Check if starter code exists
        starter_code = item.get("starter_code", "")
        
        if starter_code and starter_code.strip():
            # With starter code
            prompt += FORMATTING_WITH_STARTER_CODE
            prompt += f"```python\n{starter_code}\n```\n\n### Answer: (use the provided format with backticks)\n\n"
        else:
            # Without starter code
            prompt += FORMATTING_WITHOUT_STARTER_CODE
            prompt += "```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n"
        return {"prompt": prompt}
    
    def extract_raw_answer(self, raw_data: str) -> str:
        answer = self.extract_code_answer(text=raw_data)
        if answer is None:
            answer = ""
        return answer
    
    def extract_code_answer(self, text: str) -> str:
        outputlines = text.split("\n")
        if not outputlines:
            return ""
        try:
            indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
            
            if len(indexlines) < 2:
                indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
                
            if len(indexlines) >= 2:
                code = "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
                return code.strip()
                
            return ""
            
        except Exception as e:
            print(f"Error extracting code: {str(e)}")
            return ""
        
    def evaluate(self, data, output_text, **kwargs):
        prediction = self.extract_raw_answer(raw_data=output_text)

        # Build tests lazily (combine public + private) to avoid slow initialization
        idx = data["__idx__"]
        raw_item = self._raw_data[idx]
        public_test_cases = json.loads(raw_item["public_test_cases"])  # type: ignore
        # Decode private_test_cases: may be JSON string or base64(zlib(pickle)) encoded JSON string
        try:
            private_test_cases = json.loads(raw_item["private_test_cases"])  # type: ignore
        except Exception:
            try:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(raw_item["private_test_cases"].encode("utf-8"))  # type: ignore
                        )
                    )
                )
            except Exception:
                private_test_cases = []

        fn_name = json.loads(raw_item["metadata"]).get("func_name", None)
        all_tests = public_test_cases + private_test_cases
        inputs = [t["input"] for t in all_tests]
        outputs = [t["output"] for t in all_tests]
        sample = {"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": fn_name})}

        # Align timeout with the official runner (default 6s)
        results, metadata = evaluate_generation(generations=[prediction], sample=sample, debug=False, timeout=6)
        real_results = results[0]
        
        correct_number = 0
        for result in real_results:
            if result is True:
                correct_number += 1
        is_correct = True if correct_number == len(real_results) else False
        
        return {
            "prediction": prediction,
            "ground_truth": None,
            "is_correct": is_correct
        }
    
