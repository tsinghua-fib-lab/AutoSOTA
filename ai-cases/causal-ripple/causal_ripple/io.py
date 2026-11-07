"""Validated JSONL inputs, completion cleanup, and inference-cost accounting."""

import os
import re
import subprocess
import sys

import torch

from .adapters import clean_completion, static_contract
from .guards import StaticCertificate, _certificate_for_text, local_lexical_risk
from .jsonl import file_digest as file_digest
from .jsonl import read_jsonl as read_jsonl
from .jsonl import write_jsonl as write_jsonl
from .refinement import visible_assertions


def clean_legacy(raw, row):
    stops = ["<|endoftext|>", "<|endofmask|>", "</s>", "[DONE]"]
    if row["cleaning_mode"] == "legacy_mbpp":
        stops += ['\n"""', "\nassert"]
    else:
        stops += [
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
            "\ndef ",
            "\nclass ",
            "\nimport ",
            "\nfrom ",
            "\nassert ",
        ]
    offsets = [raw.find(stop) for stop in stops if raw.find(stop) >= 0]
    body = raw[: min(offsets)] if offsets else raw
    fence = "`" * 3
    if body.lstrip().startswith(fence):
        body = body.lstrip()
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.split(fence, 1)[0]
    completion = body.replace("\t", "    ")
    return (
        completion,
        row["official_prefix"] + completion,
        dict(
            first_stop=min(offsets) if offsets else None,
            removed_trailing_lines=0,
            removed_marker_lines=0,
        ),
    )


def clean_tokens(tokens, prompt_length, tokenizer, row):
    ids = tokens[prompt_length:].detach().cpu().tolist()
    if row["cleaning_mode"] == "corrected_adapted":
        raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return clean_completion(raw, row)
    raw = tokenizer.decode(ids, skip_special_tokens=True)
    return clean_legacy(raw, row)


def benchmark_diagnostics(row, tokenizer):
    def certificate(tokens, prompt_length, tok):
        _, program, _ = clean_tokens(tokens, prompt_length, tok, row)
        if row["contract"]["benchmark"] != "ds1000":
            return _certificate_for_text(program)
        parsed, missing, undefined = static_contract(program, row["contract"])
        lexical = local_lexical_risk(re.findall(r"\w+|==|!=|<=|>=|:=|->|\S", program))
        return StaticCertificate(parsed, undefined, missing, -lexical)

    def visible_score(tokens, prompt_length, prompt, tok, timeout=1.0):
        if row["contract"]["benchmark"] == "ds1000":
            return 0, 0
        assertions = visible_assertions(prompt, tok)
        if not assertions:
            return 0, 0
        _, program, _ = clean_tokens(tokens[0], prompt_length, tok, row)
        harness = [program, "\n__dlm_passed = 0"]
        for statement in assertions:
            harness += [
                "try:",
                f"    {statement}",
                "    __dlm_passed += 1",
                "except BaseException:",
                "    pass",
            ]
        harness.append(f"print('__DLM_VISIBLE__', __dlm_passed, {len(assertions)})")
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", "\n".join(harness)],
                text=True,
                capture_output=True,
                timeout=timeout,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except (subprocess.TimeoutExpired, OSError):
            return 0, len(assertions)
        match = re.search(r"__DLM_VISIBLE__\s+(\d+)\s+(\d+)", result.stdout)
        return (int(match[1]), int(match[2])) if match else (0, len(assertions))

    return certificate, visible_score


class CostCounter:
    def __init__(self, model):
        self.hook = model.register_forward_pre_hook(self._count, with_kwargs=True)
        self.reset()

    def reset(self):
        self.values = dict(
            serial_calls=0, sequence_forwards=0, processed_tokens=0, max_forward_batch=0
        )

    def _count(self, module, inputs, kwargs):
        tokens = inputs[0] if inputs else kwargs.get("input_ids")
        if isinstance(tokens, torch.Tensor):
            self.values["serial_calls"] += 1
            self.values["sequence_forwards"] += int(tokens.shape[0])
            self.values["processed_tokens"] += int(tokens.numel())
            self.values["max_forward_batch"] = max(
                self.values["max_forward_batch"], int(tokens.shape[0])
            )

    def close(self):
        self.hook.remove()
