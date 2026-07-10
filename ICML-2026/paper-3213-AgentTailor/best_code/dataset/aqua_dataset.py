
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional


def aqua_data_process(dataset):

    list_data_dict = []
    for data in dataset:
        task = data["question"] + ' ' + 'Choices:'
        for option in data["options"]:
            task = task + ' ' + option
        item = {"task": task}
        item["step"] = data["rationale"]
        item["answer"] = data["correct"]
        list_data_dict.append(item)

    return list_data_dict


def aqua_get_predict(pred_str):
    if not isinstance(pred_str, str):
        pred_str = str(pred_str)

    # 1) Prefer common answer templates; only accept A–E
    # - "The answer is A"
    # - "Final answer: B"
    # - "Answer: C"
    m = re.search(r"(?i)\b(the answer is|final answer is|final answer|answer)\b\s*[:：]?\s*([A-E])\b", pred_str)
    if m:
        pred = m.group(2).strip().upper()
    # 2) Handle LaTeX \\boxed{...} (e.g. boxed{A})
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a
    else:
        # 3) Fallback: scan last lines for standalone A–E choice letters
        tail = pred_str.strip().splitlines()[-1] if pred_str.strip() else ""
        m2 = re.search(r"\b([A-E])\b", tail)
        if m2:
            pred = m2.group(1).strip().upper()
        else:
            m3 = re.search(r"\b([A-E])\b", pred_str[-200:])
            pred = m3.group(1).strip().upper() if m3 else ""

    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]

    pred = _strip_string(pred)

    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a

    pred = _strip_string(pred)
    if len(pred) == 1 and pred in {"A", "B", "C", "D", "E"}:
        return pred
    # Final fallback: only return A–E or empty string (empty counts wrong)
    matches = re.findall(r"[A-E]", pred.upper())
    return matches[-1] if matches else ""


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def delete_extra_zero(n):
    try:
        n = float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n = str(n)
        return n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string


    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]


    string = _fix_sqrt(string)


    string = string.replace(" ", "")


    string = _fix_fracs(string)


    if string == "0.5":
        string = "\\frac{1}{2}"


    string = _fix_a_slash_b(string)

    return string


class AQuADataset:

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
                "train": "test.json",
                "val": "dev.jsonl",
                "test": "test.jsonl",
            }
            file_name = split_file_map.get(split, "test.jsonl")
            self.file_path = current_dir / "dataset" / "aqua" / file_name
        else:
            self.file_path = Path(data_path)

        if not self.file_path.exists():
            error_msg = (
                f"❌ Dataset file not found: {self.file_path}\n"
                f"💡 Please download the AQuA dataset:\n"
                f"   1. Run the download script: python dataset/aqua/download.py\n"
                f"   2. Or download manually from: https://github.com/google-deepmind/AQuA\n"
                f"   3. Convert test.json, dev.json, test.json to JSONL format\n"
                f"   4. Place them in the directory: {self.file_path.parent}\n"
                f"   5. Use the filenames: test.json, dev.jsonl, test.jsonl"
            )
            raise FileNotFoundError(error_msg)


        with open(self.file_path, "r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]


        if max_samples is not None:
            all_records = all_records[:max_samples]

        self.records = all_records
        self._split = split
        print(f"✅ Loaded {len(self.records)} AQuA samples from {self.file_path} (split: {split})")

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

        return "aqua"

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, Any]:


        question = record.get("question", "")
        options = record.get("options", [])


        task = question + " Choices:"
        for option in options:
            task = task + " " + option

        return {
            "task": task,
            "name": record.get("name", ""),
            "question": question,
            "options": options,
        }

    def record_to_target_answer(self, record: Dict[str, Any]) -> str:

        return record.get("correct", "")

    @staticmethod
    def record_to_target_answer_content(record: Dict[str, Any]) -> str:

        return record.get("correct", "")

    def postprocess_answer(self, raw_answer: Any) -> str:

        if isinstance(raw_answer, list):
            raw_answer = raw_answer[-1] if raw_answer else ""
        if not isinstance(raw_answer, str):
            raw_answer = str(raw_answer)


        cleaned = aqua_get_predict(raw_answer)
        return cleaned

    def evaluate_answer(self, actor_output: str, record: Dict[str, Any]) -> Tuple[bool, str]:

        predicted = self.postprocess_answer(actor_output)
        correct = record.get("correct", "").strip()

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
        correct = record.get("correct", "").strip()

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
            "options": record.get("options", []),
            "rationale": record.get("rationale", ""),
            "correct": record.get("correct", ""),
            "actor_output": actor_output
        }

    def compute_accuracy(self, actor_output: str, record: Dict[str, Any]) -> float:

        ok, _ = self.evaluate_answer(actor_output, record)
        return 1.0 if ok else 0.0
