import json
import re
import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Any, Optional

from loguru import logger

BOXED_PATTERN = r"\\boxed\{([^}]*)\}"

class BaseEvaluator(ABC):
    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        self.prompt_tokens = 0
        self.completion_tokens = 0

        # Initialize grader as DirectGenerator with caching support
        from generators.generator import DirectGenerator
        self.grader = DirectGenerator(
            model_name=os.getenv("GRADER_MODEL_NAME", "gpt-4.1-mini"),
            base_url=os.getenv("GRADER_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("GRADER_API_KEY", ""),
            temperature=0.2,
            top_p=1.0,
            timeout=500,
            cache_config=grader_cache_config
        )
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data
    
    def extract_boxed_content(self, text: str) -> str:
        start_tag = r"\boxed{"
        start = text.find(start_tag)
        if start == -1:
            return ""

        start += len(start_tag)
        brace_count = 1
        result = []

        for char in text[start:]:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            result.append(char)

        return ''.join(result).strip() 

    def extract_normal_answer(self, text: str, answer_pattern: str) -> str:
        """
        Extract the answer from the text using the answer pattern.
        Like:
        - Answer: 123 -> 123
        - Answer:123 -> 123
        - Final Answer\n\nxxx -> xxx
        if failed, try to parse \\boxed{answer}
        """
        if text is None:
            return ""

        if len(text) <= 10 and 'Answer' not in text and 'box' not in text:
            return text.lstrip()
        
        # First, try to match using the provided answer_pattern
        matches = re.findall(answer_pattern, text)
        if matches:
            extracted_answer = matches[-1].strip()
            if extracted_answer.lower().startswith("answer: "):
                extracted_answer = extracted_answer[len("answer:"):].strip().lstrip()
            return extracted_answer
        
        # If no match is found, check for "Final Answer" format
        answer_pattern = answer_pattern.replace(r"Answer\s*:\s", r"Final Answer\s\n+\s")
        final_answer_match = re.search(answer_pattern, text)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # If both patterns fail, try to extract boxed content
        return self.extract_boxed_content(text)
    
    @abstractmethod
    def load_data(self, split: str):
        pass

    @abstractmethod
    def get_valid_splits(self) -> List[str]:
        """Return list of valid split identifiers for this evaluator"""
        pass

    @abstractmethod
    def evaluate(self, data: Dict[str, Any], output_text: str, **kwargs) -> Dict[str, Any]:
        pass
