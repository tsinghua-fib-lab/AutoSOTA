
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from dataset.gsm8k_dataset import gsm_get_predict


class MultiArithDataset:


    def __init__(
        self,
        split: str = "test",
        data_path: Optional[str] = None,
        seed: int = 888,
        max_samples: Optional[int] = None,
    ):

        if data_path is None:

            current_dir = Path(__file__).parent.parent

            split_file_map = {
                "train": "train.json",
                "test": "test.json",
            }
            file_name = split_file_map.get(split, "test.json")
            self.file_path = current_dir / "dataset" / "MultiArith" / file_name
        else:
            self.file_path = Path(data_path)

        if not self.file_path.exists():
            error_msg = (
                f"❌ Dataset file not found: {self.file_path}\n"
                f"💡 Please make sure the MultiArith dataset file exists at:\n"
                f"   {self.file_path}"
            )
            raise FileNotFoundError(error_msg)


        with open(self.file_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)


        if max_samples is not None:
            all_records = all_records[:max_samples]


        for idx, record in enumerate(all_records):
            record.setdefault("id", f"multiarith_{idx}")
            record.setdefault("name", f"multiarith_{record.get('id', idx)}")

        self.records = all_records
        self._split = split
        print(f"✅ Loaded {len(self.records)} MultiArith samples from {self.file_path} (split: {split})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: Optional[int] = None) -> Dict[str, Any]:

        if idx is None:
            idx = random.randint(0, len(self.records) - 1)
        return self.records[idx]

    @property
    def split(self) -> str:

        return self._split

    @staticmethod
    def get_domain() -> str:

        return "multiarith"

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, Any]:


        question = record.get("question", "").strip()

        return {
            "task": question,
            "name": record.get("name", ""),
            "question": question,
        }

    def record_to_target_answer(self, record: Dict[str, Any]) -> str:

        return str(record.get("final_ans", "")).strip()

    @staticmethod
    def record_to_target_answer_content(record: Dict[str, Any]) -> str:

        return str(record.get("final_ans", "")).strip()

    def postprocess_answer(self, raw_answer: Any) -> str:

        if isinstance(raw_answer, list):
            raw_answer = raw_answer[-1] if raw_answer else ""
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)


        cleaned = gsm_get_predict(raw_answer)
        return cleaned

    def evaluate_answer(self, actor_output: str, record: Dict[str, Any]) -> Tuple[bool, str]:

        predicted = self.postprocess_answer(actor_output)
        correct = str(record.get("final_ans", "")).strip()


        predicted = predicted.replace(",", "")
        correct = correct.replace(",", "")

        is_correct = predicted == correct
        feedback = f"Predicted: {predicted}, Correct: {correct}"
        return is_correct, feedback

    def evaluate_candidate(
        self,
        candidate_answer: str,
        record: Dict[str, Any],
        timeout: int = 5,
    ) -> Tuple[float, bool, str, Tuple[bool, ...], List[str]]:

        predicted = self.postprocess_answer(candidate_answer)
        correct = str(record.get("final_ans", "")).strip()


        predicted = predicted.replace(",", "")
        correct = correct.replace(",", "")

        is_correct = predicted == correct
        pass_ratio = 1.0 if is_correct else 0.0

        feedback = f"Predicted: {predicted}, Correct: {correct}"
        state = (is_correct,)
        tests = [f"Expected: {correct}, Got: {predicted}"]

        return pass_ratio, is_correct, feedback, state, tests

    @staticmethod
    def format_feedback_summary(
        pass_ratio: float,
        state: Tuple[bool, ...],
        feedback: str,
    ) -> str:

        passed = sum(state)
        total = len(state)
        summary = [
            f"pass_ratio={pass_ratio:.3f}",
            f"tests_passed={passed}/{total}",
            "feedback:",
            feedback,
        ]
        return "\n".join(summary)

    def record_to_critic_inputs(self, record: Dict[str, Any], actor_output: str) -> Dict[str, Any]:

        return {
            "name": record.get("name", ""),
            "question": record.get("question", ""),
            "final_ans": record.get("final_ans", ""),
            "actor_output": actor_output
        }

    def compute_accuracy(self, actor_output: str, record: Dict[str, Any]) -> float:

        ok, _ = self.evaluate_answer(actor_output, record)
        return 1.0 if ok else 0.0

