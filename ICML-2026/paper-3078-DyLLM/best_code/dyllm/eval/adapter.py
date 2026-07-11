from __future__ import annotations
from typing import List, Iterable, Tuple
import itertools
import re
import time

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

from dyllm.config import Config
from dyllm.dllm import dLLM
from dyllm.sampling_params import SamplingParams


def _cut_on_first_stop(text: str, stops: list[str]) -> str:
    if not stops:
        return text
    cut = min([text.find(s) for s in stops if s in text] + [len(text)])
    return text[:cut]


@register_model("dyllm")
class DyLLMAdapter(LM):
    """
    Minimal lm-eval-harness adapter that implements generate_until using dLLM.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_new_toks: int = 256,
        tensor_parallel_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        ignore_eos: bool = False,
        trust_remote_code: bool = True,
        num_steps: int = 256,
        num_full_steps: int = 16,
        block_size: int = 32,
        threshold: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self._batch_size = int(batch_size)
        self._max_new_toks = int(max_new_toks)

        def to_float(x, default):
            if x is None or str(x) == "None":
                return default
            return float(x)

        self.temperature = to_float(temperature, None)
        self.top_p = to_float(top_p, 1.0)
        self.ignore_eos = ignore_eos
        self.num_steps = int(num_steps)
        self.num_full_steps = int(num_full_steps)
        self.block_size = int(block_size)
        self.threshold = float(threshold)
        trust_remote_code = trust_remote_code
        self.model_path = model_path

        # Tokenizer (CPU)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, local_files_only=True, use_fast=True
        )

        # Engine (GPU)
        self.engine = dLLM(
            model_path,
            threshold=threshold,
            enforce_eager=True,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.is_instruct = "instruct" in model_path.lower()

    # ---- LM required properties ----
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def eot_token_id(self) -> int:
        eid = self.tokenizer.eos_token_id
        return int(eid) if eid is not None else -1

    @property
    def max_gen_toks(self) -> int:
        return self._max_new_toks

    @property
    def max_length(self) -> int:
        mlen = getattr(self.tokenizer, "model_max_length", 4096)
        return int(mlen if mlen and mlen != int(1e30) else 4096)

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    # ---- Token helpers (lm-eval uses these in some paths) ----
    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ):
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    def tok_encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def tok_decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ---- Not needed for GSM8K tasks; leave unimplemented ----
    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not implemented for DyLLMAdapter")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("loglikelihood_rolling not implemented for DyLLMAdapter")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        total_time = 0.0
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]

            # prompts = [inst.args[0] for inst in batch]
            prompts = []
            for inst in batch:
                raw_prompt = inst.args[0]

                # Check if this is a HumanEval task
                doc = getattr(inst, "doc", {})
                task_id = str(doc.get("task_id", "")).lower() if doc else ""
                is_humaneval = task_id.startswith("humaneval")

                # Apply chat template only for instruct models on non-HumanEval tasks
                if self.is_instruct and not is_humaneval:
                    messages = [{"role": "user", "content": raw_prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(formatted_prompt)
                else:
                    prompts.append(raw_prompt)

            batch_max_toks = self._max_new_toks

            sp = SamplingParams(
                max_new_tokens=batch_max_toks,
                temperature=self.temperature,
                top_p=self.top_p,
                steps=self.num_steps,
                num_full_steps=self.num_full_steps,
                block_size=self.block_size,
                ignore_eos=self.ignore_eos,
            )

            start_time = time.perf_counter_ns()
            outs = self.engine.generate(prompts, sp)  # [{"text": ..., "token_ids": ...}, ...]
            end_time = time.perf_counter_ns()
            total_time += (end_time - start_time) / 1e9

            for inst, o in zip(batch, outs):
                stops = inst.args[1].get("until", [])
                doc = getattr(inst, "doc", {})
                task_id = str(doc.get("task_id", "")).lower() if doc else ""

                if self.is_instruct and task_id.startswith("humaneval"):
                    stops = []
                all_stops = stops + ["<|eot_id|>", "<|endoftext|>", "</s>"]

                trimmed = _cut_on_first_stop(o["text"], all_stops).strip()
                results.append(trimmed)
                # print(f"Prompt: {inst.args[0]}\nGenerated: {trimmed}\n")
            print(f"time for batch {i // self._batch_size + 1}: {(end_time - start_time) / 1e9:.2f} seconds")
        print(f"Total generation time for all batches in generate_until: {total_time:.2f} seconds")
        return results
