
import json
import os
import random
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from AgentTailor.tools.coding.python_executor import PyExecutor
class HumanEvalDataset:

    def __init__(
        self,
        split: str = "test",
        data_path: Optional[str] = None,
        seed: int = 888,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):

        if data_path is None:

            current_dir = Path(__file__).parent.parent
            self.file_path = current_dir / "dataset" / "humaneval" / "humaneval-py.jsonl"
        else:
            self.file_path = Path(data_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"❌ Dataset file not found: {self.file_path}")


        with open(self.file_path, "r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]



        if split == "test":
            self.records = all_records
        else:
            assert split in {"train", "val"}, f"split must be 'train', 'val', or 'test', got {split}"
            rng = random.Random(seed)
            rng.shuffle(all_records)

            total = len(all_records)
            train_end = int(total * split_ratio[0])
            val_end = train_end + int(total * split_ratio[1])

            if split == "train":
                self.records = all_records[:train_end]
            else:
                self.records = all_records[train_end:val_end]

        self._split = split
        self.executor = PyExecutor()
        print(f"✅ Loaded {len(self.records)} HumanEval samples from {self.file_path} (split: {split})")

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

        return "humaneval"

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, Any]:

        prompt = record["prompt"]
        entry_point = record["entry_point"]

        return {
            "task": prompt,
            "name": record.get("name", ""),
            "entry_point": entry_point,
        }

    def record_to_target_answer(self, record: Dict[str, Any]) -> str:
        return record.get("prompt", "")

    @staticmethod
    def record_to_target_answer_content(record: Dict[str, Any]) -> str:

        return "PASS"

    def postprocess_answer(self, raw_answer: Any) -> str:

        if isinstance(raw_answer, list):
            raw_answer = raw_answer[-1] if raw_answer else ""
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)

        cleaned = raw_answer.strip()
        if "```" in cleaned:

            parts = cleaned.split("```")
            if len(parts) >= 3:
                code = parts[1]
                if code.startswith("python"):
                    code = code[len("python"):]
                cleaned = code.strip()
        return cleaned

    def evaluate_answer(self, actor_output: str, record: Dict[str, Any]) -> Tuple[bool, str]:

        entry = record["entry_point"]
        func_code = actor_output.strip()
        test_code = record["test"]

        try:

            local_env = {}
            exec(func_code, local_env)
            exec(test_code, local_env)
            return True, "Pass"
        except Exception as e:
            err = traceback.format_exc(limit=2)
            return False, err

    def evaluate_candidate(
        self,
        candidate_code: str,
        record: Dict[str, Any],
        timeout: int = 5,
    ) -> Tuple[float, bool, str, Tuple[bool, ...], List[str]]:
        tests = self._extract_unit_tests(record)
        if len(tests) == 0:
            return 0.0, False, "No unit tests found.", tuple(), tests

        is_passing, feedback, state = self.executor.execute(
            candidate_code, tests, timeout=timeout, verbose=False
        )
        pass_count = sum(state)
        pass_ratio = pass_count / len(state) if len(state) > 0 else 0.0
        return pass_ratio, is_passing, feedback, state, tests

    @staticmethod
    def _extract_unit_tests(record: Dict[str, Any]) -> List[str]:

        entry_point = record["entry_point"]
        tests: List[str] = []
        for line in record["test"].splitlines():
            stripped = line.strip()
            if not stripped.startswith("assert "):
                continue
            tests.append(stripped.replace("candidate", entry_point))
        return tests

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
            "prompt": record["prompt"],
            "entry_point": record["entry_point"],
            "test_code": record["test"],
            "actor_output": actor_output
        }

    def compute_accuracy(self, actor_output: str, record: Dict[str, Any]) -> float:

        ok, _ = self.evaluate_answer(actor_output, record)
        return 1.0 if ok else 0.0

