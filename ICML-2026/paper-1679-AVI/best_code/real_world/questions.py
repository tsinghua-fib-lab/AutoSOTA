

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any


def format_mmlu_question(
    question: str,
    choices: List[str],
    include_answer_hint: bool = False,
    answer_idx: Optional[int] = None,
) -> str:
    """
    Format the MMLU question and options into a complete prompt text.
    Args:
        question: The question text
        choices: The list of options (usually 4 options)
        include_answer_hint: Whether to add an answer hint at the end
        answer_idx: The correct answer index (0-3), only used for debugging
    Returns:
        The formatted complete question text
    Example output:
        Question: Which is the largest ocean on Earth?
        A. Atlantic Ocean
        B. Indian Ocean
        C. Pacific Ocean
        D. Arctic Ocean
            
        Please select the correct answer (A/B/C/D):
    """
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:len(choices)]
    
    formatted_parts = [f"Question: {question}"]
    for label, choice in zip(option_labels, choices):
        formatted_parts.append(f"{label}. {choice}")
    
    if include_answer_hint:
        formatted_parts.append("")
        formatted_parts.append(f"Please select the correct answer ({'/'.join(option_labels)}):")
    
    return "\n".join(formatted_parts)


class QuestionProvider:


    def __init__(
        self,
        questions: Optional[List[str]] = None,
        question_file: Optional[str] = None,
        shuffle: bool = True,
    ) -> None:
        loaded_questions = list(questions or [])
        if question_file:
            loaded_questions.extend(self._load_from_file(question_file))

        cleaned = [q.strip() for q in loaded_questions if q and q.strip()]
        if not cleaned:
            raise ValueError("QuestionProvider requires at least one question.")

        self.questions = cleaned
        self.shuffle = shuffle
        self._cursor = 0

        if self.shuffle:
            random.shuffle(self.questions)

    @staticmethod
    def _load_from_file(path: str) -> List[str]:
        if path.startswith("hf:"):
            return QuestionProvider._load_from_hf(path[3:])
        
        if path.startswith("hf-mmlu:"):
            return QuestionProvider._load_from_hf_mmlu(path[8:])

        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Question file not found: {file_path}")

        suffix = file_path.suffix.lower()
        with file_path.open("r", encoding="utf-8") as f:
            if suffix in {".json", ".jsonl"}:
                lines = []
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if isinstance(record, dict) and "question" in record:
                        lines.append(record["question"])
                    elif isinstance(record, str):
                        lines.append(record)
                return lines
            if suffix == ".txt":
                return [line.strip() for line in f if line.strip()]

        raise ValueError("Question file only supports .json / .jsonl / .txt format.")

    @staticmethod
    def _load_from_hf(hf_path_str: str) -> List[str]:
        """
        Load a dataset from Hugging Face.
        Format: dataset_name[:config][:split][:column]
        - dataset_name: Dataset name (required)
        - config: Configuration name/subset (optional, e.g. 'rc' for trivia_qa)
        - split: Dataset split (optional, default 'train')
        - column: Column name (optional, default 'question')
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Need to install datasets to load Hugging Face dataset: pip install datasets")

        parts = hf_path_str.split(':')
        dataset_name = parts[0]
        
        config_name = None
        split = "train"
        column = "question"
        
        if len(parts) == 2:
            if parts[1] in ['train', 'validation', 'test']:
                split = parts[1]
            else:
                config_name = parts[1]
        elif len(parts) == 3:
            if parts[1] in ['train', 'validation', 'test']:
                split = parts[1]
                column = parts[2]
            else:
                config_name = parts[1]
                split = parts[2]
        elif len(parts) == 4:
            config_name = parts[1]
            split = parts[2]
            column = parts[3]

        print(f"Loading dataset from Hugging Face: {dataset_name}" + 
              (f", config={config_name}" if config_name else "") + 
              f", split={split}, column={column}...")
        
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        if column not in ds.column_names:
             raise ValueError(f"Column '{column}' not in dataset '{dataset_name}'. Available columns: {ds.column_names}")

        return [str(item) for item in ds[column]]

    @staticmethod
    def _load_from_hf_mmlu(hf_path_str: str) -> List[str]:
        """
        Load a MMLU format dataset from Hugging Face.
        Format: dataset_name[:config][:split]
        Example:
        - "cais/mmlu:anatomy:test" -> load test split of anatomy subset
        - "cais/mmlu:all:test" -> load test split of all subsets
        - "cais/mmlu:test" -> load test split (need to specify config)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Need to install datasets to load Hugging Face dataset: pip install datasets")

        parts = hf_path_str.split(':')
        dataset_name = parts[0]
        
        config_name = None
        split = "test"
        
        if len(parts) == 2:
            if parts[1] in ['train', 'validation', 'test', 'dev', 'auxiliary_train']:
                split = parts[1]
            else:
                config_name = parts[1]
        elif len(parts) >= 3:
            config_name = parts[1]
            split = parts[2]

        print(f"Loading MMLU dataset from Hugging Face: {dataset_name}" + 
              (f", config={config_name}" if config_name else "") + 
              f", split={split}...")
        
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        
        required_cols = ['question', 'choices']
        missing_cols = [col for col in required_cols if col not in ds.column_names]
        if missing_cols:
            raise ValueError(
                f"MMLU dataset missing required columns: {missing_cols}. "
                f"Available columns: {ds.column_names}"
            )
        
        formatted_questions = []
        for item in ds:
            question = item['question']
            choices = item['choices']
            formatted_q = format_mmlu_question(
                question=question,
                choices=choices,
                include_answer_hint=True,
            )
            formatted_questions.append(formatted_q)
        
        print(f"Loaded {len(formatted_questions)} MMLU formatted questions")
        return formatted_questions

    def sample(self) -> str:
        if not self.questions:
            raise RuntimeError("QuestionProvider not initialized with questions list.")

        question = self.questions[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.questions)

        if self.shuffle and self._cursor == 0:
            random.shuffle(self.questions)

        return question

    def __len__(self) -> int:
        return len(self.questions)


