import json
import re
from pathlib import Path
from typing import Any, Callable
from monica_tools import *
import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessor



def save_monitor_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_response(
    full_text: str,
    token_ids,
    tokenizer,
    ques_len: int = 0,
    model_name: str = "",
    generation_type: str = "monica",
) -> dict[str, Any]:
    think_end_token_ids = tokenizer.encode("</think>", add_special_tokens=False)

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    def find_sublist(lst: list[int], sub: list[int]) -> int:
        for i in range(len(lst) - len(sub) + 1):
            if lst[i : i + len(sub)] == sub:
                return i
        return -1

    think_pos = full_text.find("<think>")
    if think_pos != -1:
        full_text = full_text[think_pos:]

    thinking_match = re.search(r"<think>\\n?(.*?)</think>", full_text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        response_text = full_text.split("</think>", 1)[-1].strip()
    else:
        thinking = full_text.replace("<think>", "").strip()
        response_text = ""

    think_end_idx = find_sublist(token_ids, think_end_token_ids)
    if think_end_idx == -1:
        thinking_length = max(len(token_ids) - ques_len, 0)
        response_length = 0
    else:
        thinking_length = max(think_end_idx + len(think_end_token_ids) - ques_len, 0)
        response_length = max(len(token_ids) - thinking_length - ques_len, 0)

    return {
        "thinking_text": thinking,
        "response_text": response_text,
        "thinking_length": thinking_length,
        "response_length": response_length,
        "thinking_answer": extract_answer(thinking),
        "response_answer": extract_answer(response_text),
        "model_name": model_name,
        "generation_type": generation_type,
    }

class DynamicMonitorSteerProcessor(LogitsProcessor):
    def __init__(
        self,
        model_wrapper,
        monitor_vec: dict[int, Any],
        calibrator_vec: dict[int, np.ndarray],
        lrm_config: dict[str, Any],
        monitor_layers: list[int],
        prompt_len: int,
        punctuation_token_ids: list[int],
        hs_tokens: int,
        tokenizer=None,
        question_id: int | None = None,
        monitor_log_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.model_wrapper = model_wrapper
        self.monitor_vec = monitor_vec
        self.calibrator_vec = calibrator_vec
        self.cfg = lrm_config
        self.monitor_layers = sorted(int(l) for l in monitor_layers)
        self.prompt_len = int(prompt_len)
        self.punctuation_token_ids = set(punctuation_token_ids)
        self.hs_tokens = int(hs_tokens)
        self.tokenizer = tokenizer
        self.question_id = question_id
        # Per-layer steering weights
        raw_weights = lrm_config.get("steer_layer_weights", None)
        self.steer_layer_weights = None
        if raw_weights is not None:
            steer_layers = [int(x) for x in lrm_config.get("steer_layers", [])]
            if isinstance(raw_weights, list):
                self.steer_layer_weights = {l: float(w) for l, w in zip(steer_layers, raw_weights)}
            elif isinstance(raw_weights, dict):
                self.steer_layer_weights = {int(k): float(v) for k, v in raw_weights.items()}
        self.monitor_log_callback = monitor_log_callback

        self.step_count = 0
        self.last_input_len = self.prompt_len
        self.last_punctuation_pos = self.prompt_len
        self.punctuation_count = 0

    def set_prompt_len(self, prompt_len: int) -> None:
        self.prompt_len = int(prompt_len)
        self.last_input_len = self.prompt_len
        self.last_punctuation_pos = self.prompt_len
        self.punctuation_count = 0

    def set_question_id(self, question_id: int | None) -> None:
        self.question_id = question_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        cur_steps = input_ids.shape[1] - self.prompt_len
        if cur_steps <= 0:
            return scores

        if input_ids.shape[1] <= self.last_input_len:
            return scores

        self.last_input_len = input_ids.shape[1]
        self.step_count += 1

        last_token_id = input_ids[0, -1].item()
        if last_token_id not in self.punctuation_token_ids:
            return scores

        self.punctuation_count += 1
        if self.punctuation_count % 3 != 0:
            return scores

        with torch.no_grad():
            base_model = self.model_wrapper.model if hasattr(self.model_wrapper, "model") else self.model_wrapper

            current_pos = input_ids.shape[1]
            window_start = self.last_punctuation_pos
            window_input = input_ids[:, window_start:current_pos]

            monitored_text = ""
            if self.tokenizer is not None:
                monitored_text = self.tokenizer.decode(window_input[0], skip_special_tokens=True)

            self.last_punctuation_pos = current_pos

            outputs = base_model(input_ids=window_input, output_hidden_states=True, use_cache=False)
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                return scores

            hs = outputs.hidden_states
            p0_list = []
            layer_scores: dict[int, float] = {}

            for layer_id in self.monitor_layers:
                if layer_id >= len(hs):
                    continue
                try:
                    h_gpu = hs[layer_id][:, -min(self.hs_tokens, hs[layer_id].shape[1]) :, :].mean(dim=(0, 1))
                    h = h_gpu.detach().cpu().float().numpy().reshape(1, -1)
                    clf = self.monitor_vec[layer_id]
                    proba = clf.predict_proba(h)
                    p0_value = float(proba[0, 0])
                    p0_list.append(p0_value)
                    layer_scores[layer_id] = p0_value
                except Exception:
                    continue

            if not p0_list:
                return scores
            p0_mean = float(np.mean(p0_list))
            p0_max = float(np.max(p0_list))
            if p0_max > 0.5:
                new_w = float(self.cfg["steer_min"]) + p0_mean * float(self.cfg["steer_min"])
                self.model_wrapper.reset()
                self.model_wrapper.set_control(self.calibrator_vec, new_w, normalize=True,
                                               layer_weights=self.steer_layer_weights)

            if self.monitor_log_callback is not None:
                print("tmp log....")

        return scores
