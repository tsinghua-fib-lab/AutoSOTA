"""Vendored data collators removed from trl >= 0.14.

DataCollatorForCompletionOnlyLM was removed from trl in favor of
``completion_only_loss=True`` in ``SFTConfig`` with prompt-completion datasets.
This codebase uses single-text formatting with a response template marker, so
we vendor the original class here to avoid restructuring the data pipeline.

See https://github.com/huggingface/trl/discussions/3826
"""

from typing import Any, List, Union

import torch
from transformers import DataCollatorForLanguageModeling


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Data collator that masks prompt tokens so loss is only on completions.

    Finds ``response_template`` token IDs in each input and sets labels to -100
    for every token before the template. Tokens after the template keep their
    original IDs so the cross-entropy loss is computed only on the completion.

    Usage::

        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        tokenizer=None,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        if isinstance(response_template, str):
            self.response_template_ids: List[int] = self.tokenizer.encode(
                response_template, add_special_tokens=False
            )
        else:
            self.response_template_ids = list(response_template)
        if not self.response_template_ids:
            raise ValueError("response_template must encode to at least one token")

    def torch_call(self, examples: List[Union[List[int], Any, dict]]) -> dict:
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            response_start = self._find_response_start(batch["input_ids"][i])
            if response_start is not None:
                batch["labels"][i, :response_start] = self.ignore_index
            else:
                batch["labels"][i, :] = self.ignore_index
        return batch

    def _find_response_start(self, input_ids: torch.Tensor) -> int | None:
        """Return the index of the first token *after* the response template."""
        template_len = len(self.response_template_ids)
        for idx in range(len(input_ids) - template_len + 1):
            if (
                input_ids[idx : idx + template_len].tolist()
                == self.response_template_ids
            ):
                return idx + template_len
        return None
