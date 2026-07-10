import os
import json
import re
from typing import Dict, Optional, Any

from datasets import Dataset, disable_progress_bars

from evaluation.HLE.prompts import GRADER_TEMPLATE, HLE_PROMPT
from evaluation.base_evaluator import BaseEvaluator

disable_progress_bars()

DATA_DIR = "data/HLE"


class HLEEvaluator(BaseEvaluator):
    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        super().__init__(grader_cache_config)
        self.task = "HLE"
        self.seed = 42
        
    def load_data(self, split: str = "subset_500"):
        """Load HLE data from local JSONL file"""
        data_file = os.path.join(DATA_DIR, f"{split}.jsonl")
        
        data = self.load_jsonl(data_file)
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        
        # Add origin_query field for compatibility
        data = data.map(lambda x: {**x, 'origin_query': x.get('question', '')})
        
        return data
    
    def get_valid_splits(self):
        return ["test", "subset_500"]
    
    def format_prompt(self, item: Dict) -> Dict:
        """Format the prompt for HLE questions"""
        prompt = HLE_PROMPT.format(
            question=item["question"],
        )
        return {"prompt": prompt}
    
    def _get_grader_result(self, question: str, correct_answer: str, predicted_answer: str) -> Dict:
        """
        Get grading result from grader model with automatic caching and retry

        Returns:
            Dict with keys: extracted_final_answer, reasoning, correct, confidence
        """
        prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=predicted_answer
        )

        # Call grader (DirectGenerator) - automatic caching + retry
        result = self.grader.generate(prompt)

        # Parse the structured response
        return self._parse_grader_response(result.output)
    
    def _parse_grader_response(self, response_text: str) -> Dict:
        """Parse the grader response to extract structured information"""
        
        # Initialize default values
        result = {
            "extracted_final_answer": "None",
            "reasoning": "",
            "correct": "no",
            "confidence": 0
        }
        
        try:
            # Extract extracted_final_answer
            answer_match = re.search(r'extracted_final_answer:\s*(.*?)(?=\n|$)', response_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                result["extracted_final_answer"] = answer_match.group(1).strip()
            
            # Extract reasoning
            reasoning_match = re.search(r'reasoning:\s*(.*?)(?=\ncorrect:|$)', response_text, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract correct (yes/no)
            correct_match = re.search(r'correct:\s*(yes|no)', response_text, re.IGNORECASE)
            if correct_match:
                result["correct"] = correct_match.group(1).lower()
            
            # Extract confidence (percentage)
            confidence_match = re.search(r'confidence:\s*(\d+)', response_text, re.IGNORECASE)
            if confidence_match:
                result["confidence"] = int(confidence_match.group(1))
            
        except Exception as e:
            print(f"Error parsing grader response: {e}")
            result["reasoning"] = f"Parse error: {str(e)}"
        
        return result
    
    def get_grader_result(self, question: str, answer: str, raw_data: str) -> Dict:
        """
        Public interface for getting grader result
        
        Args:
            question: The original question
            answer: The correct answer
            raw_data: The model's response to be evaluated
            
        Returns:
            Dict with grading information
        """
        return self._get_grader_result(question=question, correct_answer=answer, predicted_answer=raw_data)
    
    def evaluate(self, data, output_text, **kwargs):
        """
        Evaluate a single response using the grader
        
        Args:
            data: Dictionary containing question and answer
            output_text: The model's response
            
        Returns:
            Dictionary with evaluation results
        """
        answer = data['answer']
        question = data['question']
        
        # Use grader to evaluate the answer
        grader_result = self.get_grader_result(question=question, answer=answer, raw_data=output_text)
        
        is_correct = grader_result.get("correct", "no") == "yes"
        
        return {
            "prediction": grader_result.get("extracted_final_answer", "None"),
            "ground_truth": answer,
            "is_correct": is_correct,
            "grader_result": grader_result  # Include full grader response
        }