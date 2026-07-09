import os
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import TRAINER_STATE_NAME
from transformers.utils import logging as transformers_logging
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (
    EvalLoopOutput,
    PREFIX_CHECKPOINT_DIR,
    rotate_checkpoints,
)

from trl.trainer.utils import disable_dropout_in_model

logger = transformers_logging.get_logger(__name__)

if is_deepspeed_available():
    import deepspeed


def _pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: int | float, dim: int = -1
) -> torch.Tensor:
    """Pad a tensor to the specified length along the given dimension."""
    if tensor.size(dim) >= length:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.size(dim)
    return torch.cat(
        [
            tensor,
            pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
        ],
        dim=dim,
    )


@dataclass
class MargeDataCollatorWithPadding:
    r"""
    MargE DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`bool | None`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool | None = False
    input_field_name: str = "input"
    target_field_name: str = "target"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith(self.input_field_name)) and (
                        k.endswith("input_ids")
                    ):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(
                        (self.input_field_name, self.target_field_name)
                    ) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if self.input_field_name in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    # for the prompt, flip back so padding is on left side
                    if self.input_field_name in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


@dataclass
class MargeConfig(TrainingArguments):
    """Configuration for MARGE (Margin-based Reward Guided Exploration) training.

    Extends TrainingArguments with MARGE-specific fields. Unlike the previous
    version which inherited from DPOConfig, this config only contains fields
    that MARGE actually uses.
    """

    alpha: float = 1.0
    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 128
    disable_dropout: bool = True
    input_field_name: str = "input"
    target_field_name: str = "target"
    input_score_field_name: str = "score_input"
    target_score_field_name: str = "score_target"
    self_normalize_weights: bool = False
    reinforce_style: bool = False
    label_pad_token_id: int = -100
    precompute_ref_log_probs: bool = False
    generate_during_eval: bool = False
    truncation_mode: str = "keep_end"


class MargeTrainer(Trainer):
    """MARGE (Margin-based Reward Guided Exploration) trainer.

    Extends Trainer with weighted NLL + KL regularization loss on
    single (input, target) pairs with explicit rewards.
    """

    def __init__(
        self,
        *,
        metrics_fn: Callable | None = None,
        rewards_fn: Callable | None = None,
        num_generate_batches: int = 1,
        threshold_percent_valid: float = 0.9,
        pretokenized: bool = False,
        model: PreTrainedModel | nn.Module | None = None,
        ref_model: PreTrainedModel | nn.Module | None = None,
        args: MargeConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
        compute_metrics: Callable | None = None,
        # Legacy alias
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        # Support legacy 'tokenizer' kwarg
        if tokenizer is not None and processing_class is None:
            processing_class = tokenizer

        if processing_class is None:
            raise ValueError("processing_class (tokenizer) must be provided.")

        # MARGE-specific attributes
        self.metrics_fn = metrics_fn
        self.rewards_fn = rewards_fn
        self.num_generate_batches = num_generate_batches
        self.threshold_percent_valid = threshold_percent_valid
        self.reward_sum = 0.0
        self._peft_has_been_casted_to_bf16 = False
        self.is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)

        # Store ref model before super().__init__
        self.ref_model = ref_model

        # Config-derived attributes
        self.beta = args.beta
        self.alpha = args.alpha
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.max_target_length = getattr(args, "max_target_length", None)
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = processing_class.pad_token_id
        self.truncation_mode = args.truncation_mode
        self.generate_during_eval = args.generate_during_eval
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.loss_type = getattr(args, "loss_type", "sigmoid")
        self.tokenizer = processing_class
        self.dataset_num_proc = getattr(args, "dataset_num_proc", None)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # Disable dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Create data collator if not provided
        if data_collator is None:
            data_collator = MargeDataCollatorWithPadding(
                pad_token_id=processing_class.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
                input_field_name=args.input_field_name,
                target_field_name=args.target_field_name,
            )
            if args.remove_unused_columns:
                args.remove_unused_columns = False
            self.use_marge_data_collator = True
        else:
            self.use_marge_data_collator = False

        # Set args early so tokenize_row can access self.args before super().__init__()
        self.args = args

        # Tokenize dataset if not pretokenized
        with PartialState().local_main_process_first():
            self.pretokenized = pretokenized
            if not self.pretokenized:
                train_dataset = train_dataset.map(
                    self.tokenize_row, num_proc=self.dataset_num_proc
                )
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(
                        self.tokenize_row, num_proc=self.dataset_num_proc
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Prepare ref model with accelerator (must happen after super().__init__)
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    def _prepare_deepspeed(self, model: PreTrainedModel):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(self.train_dataset, **dataloader_params)
            )

            reference_target_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Train dataset reference log probs"
            ):
                reference_target_logp = self.compute_reference_log_probs(padded_batch)
                reference_target_logp = self.accelerator.gather_for_metrics(
                    reference_target_logp
                )
                reference_target_logps.append(reference_target_logp.cpu())

            all_reference_target_logps = (
                torch.cat(reference_target_logps).float().numpy()
            )

            self.train_dataset = self.train_dataset.add_column(
                name="reference_target_logps", column=all_reference_target_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(eval_dataset, **dataloader_params)
            )

            reference_target_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Eval dataset reference log probs"
            ):
                reference_target_logp = self.compute_reference_log_probs(padded_batch)
                reference_target_logp = self.accelerator.gather_for_metrics(
                    reference_target_logp
                )
                reference_target_logps.append(reference_target_logp.cpu())

            all_reference_target_logps = (
                torch.cat(reference_target_logps).float().numpy()
            )

            eval_dataset = eval_dataset.add_column(
                name="reference_target_logps", column=all_reference_target_logps
            )

            # Save calculated reference_target_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if (
            prompt_input_ids
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][
            response_token_ids_start_idx:
        ]

        return {
            f"{self.args.input_field_name}_input_ids": prompt_input_ids,
            f"{self.args.input_field_name}_attention_mask": prompt_attention_mask,
            "input_ids": answer_input_ids,
            "attention_mask": answer_attention_mask,
        }

    def tokenize_row(
        self, feature, model: PreTrainedModel | nn.Module | None = None
    ) -> dict:
        """Tokenize a single row from a MargE specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the target responses, which are of length equal to
            the sum of the length of the prompt and the target response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature[self.args.input_field_name]
        target = feature[self.args.target_field_name]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {
                f"{self.args.input_field_name}_{k}": v for k, v in prompt_tokens.items()
            }

            if not isinstance(target, str):
                raise ValueError(f"target should be an str but got {type(target)}")
            target_tokens = self.build_tokenized_answer(prompt, target)

            target_prompt_len_input_ids = len(
                target_tokens[f"{self.args.input_field_name}_input_ids"]
            )
            prompt_len_input_ids = target_prompt_len_input_ids

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            longer_response_length = len(target_tokens["input_ids"])

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [target_tokens, prompt_tokens]:
                if (
                    len(answer_tokens[f"{self.args.input_field_name}_input_ids"])
                    + longer_response_length
                    > self.max_length
                ):
                    if self.truncation_mode == "keep_start":
                        for k in [
                            f"{self.args.input_field_name}_input_ids",
                            f"{self.args.input_field_name}_attention_mask",
                        ]:
                            answer_tokens[k] = answer_tokens[k][
                                : self.max_prompt_length
                            ]
                    elif self.truncation_mode == "keep_end":
                        for k in [
                            f"{self.args.input_field_name}_input_ids",
                            f"{self.args.input_field_name}_attention_mask",
                        ]:
                            answer_tokens[k] = answer_tokens[k][
                                -self.max_prompt_length :
                            ]
                    else:
                        raise ValueError(
                            f"Unknown truncation mode: {self.truncation_mode}"
                        )

            # if that's still too long, truncate the response
            for answer_tokens in [target_tokens]:
                if (
                    len(answer_tokens[f"{self.args.input_field_name}_input_ids"])
                    + longer_response_length
                    > self.max_length
                ):
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][
                            : self.max_length - self.max_prompt_length
                        ]

            # Create labels
            target_sequence_tokens = {
                k: target_tokens[f"{self.args.input_field_name}_{k}"] + target_tokens[k]
                for k in ["input_ids", "attention_mask"]
            }
            target_sequence_tokens["labels"] = target_sequence_tokens["input_ids"][:]
            target_sequence_tokens["labels"][
                : len(target_tokens[f"{self.args.input_field_name}_input_ids"])
            ] = [self.label_pad_token_id] * len(
                target_tokens[f"{self.args.input_field_name}_input_ids"]
            )

            for k, toks in {
                f"{self.args.target_field_name}_": target_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            target_tokens = self.tokenizer(
                target,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch[f"{self.args.target_field_name}_labels"] = target_tokens["input_ids"]
            batch[f"{self.args.input_field_name}_input_ids"] = prompt_tokens[
                "input_ids"
            ]
            batch[f"{self.args.input_field_name}_attention_mask"] = prompt_tokens[
                "attention_mask"
            ]

            if model is not None and hasattr(
                model, "prepare_decoder_input_ids_from_labels"
            ):
                batch[f"{self.args.target_field_name}_decoder_input_ids"] = (
                    model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(
                            batch[f"{self.args.target_field_name}_labels"]
                        )
                    )
                )

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: dict) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_target_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_target_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_target_logps

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: torch.device | None = None,
        target_field_name: str = "target",
        input_field_name: str = "prompt",
    ) -> dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys '{self.args.target_field_name}_input_ids', which contains tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.
            target_field_name: The name of the target field.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = batch[f"{target_field_name}_labels"].shape[1]
        else:
            max_length = batch[f"{target_field_name}_input_ids"].shape[1]

        for k in batch:
            if k.startswith(target_field_name) and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace(target_field_name, "concatenated")
                concatenated_batch[concatenated_key] = _pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch[f"{input_field_name}_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch[f"{input_field_name}_attention_mask"]
                .repeat(2, 1)
                .to(device=device)
            )
        return concatenated_batch

    def marge_loss(
        self,
        policy_target_logps: torch.FloatTensor,
        policy_reference_target_prob_ratios: torch.FloatTensor,
        target_rewards: torch.FloatTensor,
        target_sizes: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the MargE loss for a batch of policy and reference model log probabilities.

        Args:
            policy_target_logps: Log probabilities of the policy model for the targets. Shape: (batch_size,)
            policy_reference_target_prob_ratios: Ratios of pi_theta(y) / pi_ref(y). Shape: (batch_size,)
            target_rewards: Rewards for the targets. Shape: (batch_size,)
            target_sizes: Number of tokens in each y. Shape: (batch_size,)

        Returns:
            losses: Tensor containing loss value for each example in the batch. Shape: (batch_size,)
            kl_div_to_target
            kl_div_to_ref
        """
        target_sizes = target_sizes.to(policy_target_logps.device)
        kl_div_to_ref = -policy_target_logps / target_sizes

        if self.args.self_normalize_weights:
            policy_reference_target_prob_ratios = (
                policy_reference_target_prob_ratios
                / policy_reference_target_prob_ratios.sum()
            )
        kl_div_to_target = policy_reference_target_prob_ratios * (
            policy_target_logps / target_sizes
            - (target_rewards.to(policy_target_logps.device))
        )
        if self.args.reinforce_style:
            avg_reward_baseline = self.reward_sum / (self.state.global_step + 1)
            losses = (
                -policy_target_logps * ((target_rewards).to(policy_target_logps.device))
                + self.beta * kl_div_to_ref
            )
            self.reward_sum += target_rewards.mean().detach().cpu().item()
            return losses, target_rewards, avg_reward_baseline
        else:
            losses = self.alpha * kl_div_to_target + self.beta * kl_div_to_ref
        return losses, kl_div_to_target, kl_div_to_ref

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs."""
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            target_field_name=self.args.target_field_name,
            input_field_name=self.args.input_field_name,
        )

        labels = concatenated_batch["concatenated_labels"]
        model_kwargs = (
            {
                "labels": labels,
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        return (all_logps, all_logits, size_completion)

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (policy_target_logps, policy_target_logits, target_num_tokens) = (
            self.concatenated_forward(model, batch)
        )

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_target_logps" in batch:
            reference_target_logps = batch["reference_target_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_target_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_target_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        policy_reference_target_ratios = torch.exp(
            policy_target_logps - reference_target_logps
        )
        rewards = self.rewards_fn(batch)
        prefix = "eval_" if train_eval == "eval" else ""
        if self.args.reinforce_style:
            losses, target_rewards, avg_reward_baseline = self.marge_loss(
                policy_target_logps,
                policy_reference_target_ratios,
                rewards,
                target_num_tokens,
            )
            metrics[f"{prefix}rewards/target"] = target_rewards.mean().cpu().item()
            metrics[f"{prefix}rewards/running_average"] = avg_reward_baseline
        else:
            losses, kl_div_to_target, kl_div_to_ref = self.marge_loss(
                policy_target_logps,
                policy_reference_target_ratios,
                rewards,
                target_num_tokens,
            )
            metrics[f"{prefix}kl_div/reverse_to_target"] = (
                kl_div_to_target.mean().cpu().item()
            )
            metrics[f"{prefix}kl_div/forward_to_ref"] = (
                kl_div_to_ref.mean().cpu().item()
            )

        metrics[f"{prefix}likelihoods/policy_reference_ratio"] = (
            policy_reference_target_ratios.mean().cpu().item()
        )

        metrics[f"{prefix}logps/policy"] = (
            policy_target_logps.detach().mean().cpu().item()
        )
        metrics[f"{prefix}logits/target"] = (
            policy_target_logits.detach().mean().cpu().item()
        )

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.use_marge_data_collator:
            warnings.warn(
                "compute_loss is only implemented for MargeDataCollatorWithPadding, and you passed a datacollator that is different than "
                "MargeDataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="train"
            )

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def generate_eval_samples(
        self, model: nn.Module, batch: dict[str, torch.LongTensor]
    ) -> tuple[list[str], list[str]]:
        """
        Greedily decode samples from the model and reference model for the given batch of inputs.
        """

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            nullcontext
            if not self._peft_has_been_casted_to_bf16
            else torch.cuda.amp.autocast
        )

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch[f"{self.args.input_field_name}_input_ids"],
                attention_mask=batch[f"{self.args.input_field_name}_attention_mask"],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=None,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch[f"{self.args.input_field_name}_input_ids"],
                            attention_mask=batch[
                                f"{self.args.input_field_name}_attention_mask"
                            ],
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            max_length=None,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch[f"{self.args.input_field_name}_input_ids"],
                        attention_mask=batch[
                            f"{self.args.input_field_name}_attention_mask"
                        ],
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_length=None,
                    )

        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output[
                :, batch[f"{self.args.input_field_name}_input_ids"].shape[-1] :
            ],
            skip_special_tokens=True,
        )

        reference_output_decoded = self.tokenizer.batch_decode(
            reference_output[
                :, batch[f"{self.args.input_field_name}_input_ids"].shape[-1] :
            ],
            skip_special_tokens=True,
        )

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        if not self.use_marge_data_collator:
            warnings.warn(
                "prediction_step is only implemented for MargeDataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/target": metrics["eval_logits/target"],
        }
        logits = tuple(
            v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys
        )
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(
        self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # self.get_eval_dataloader(dataset) # TODO: also eval on best examples from overall dataset?

        # Sample and save to game log if requested (for |self.num_generate_batches| batches to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            if self.num_generate_batches is not None:
                num_samples = len(dataloader.dataset)
                random.seed(self.args.seed)
                random_indices = random.sample(
                    range(num_samples),
                    k=min(
                        num_samples,
                        self.args.eval_batch_size * self.num_generate_batches,
                    ),
                )

                # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
                random_batches_dataset = dataloader.dataset.select(random_indices)
            else:
                random_indices = list(
                    np.arange(0, len(dataloader.dataset), self.args.eval_batch_size)
                )
                random_batches_dataset = dataloader.dataset
            [d[self.args.input_field_name] for d in random_batches_dataset]
            [d[self.args.input_score_field_name] for d in random_batches_dataset]
            policy_output_decoded = []
            ref_output_decoded = []
            for i in range(0, len(random_indices), self.args.eval_batch_size):
                random_batch = random_batches_dataset.select(
                    range(
                        i,
                        min(i + self.args.eval_batch_size, len(random_batches_dataset)),
                    )
                )
                random_batch = self._prepare_inputs(self.data_collator(random_batch))
                policy_output_decoded_batch, ref_output_decoded_batch = (
                    self.generate_eval_samples(self.model, random_batch)
                )
                policy_output_decoded.extend(policy_output_decoded_batch)
                ref_output_decoded.extend(ref_output_decoded_batch)

            policy_metrics = self.metrics_fn(
                random_batches_dataset,
                policy_output_decoded,
            )
            policy_metrics = {f"eval_policy_{k}": v for k, v in policy_metrics.items()}
            self.store_metrics(policy_metrics, train_eval="eval")
            ref_metrics = self.metrics_fn(
                random_batches_dataset,
                ref_output_decoded,
            )
            ref_metrics = {f"eval_ref_{k}": v for k, v in ref_metrics.items()}
            self.store_metrics(ref_metrics, train_eval="eval")

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Average stored metrics and merge, then reset
        averaged_metrics = {
            k: np.mean(v) for k, v in self._stored_metrics["eval"].items()
        }
        self._stored_metrics["eval"].clear()

        initial_output = EvalLoopOutput(
            predictions=initial_output.predictions,
            label_ids=initial_output.label_ids,
            metrics={**initial_output.metrics, **averaged_metrics},
            num_samples=initial_output.num_samples,
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log `logs` on the various objects watching training, including stored metrics."""
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        self._stored_metrics[train_eval].clear()
        return super().log(logs, start_time)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        commit_message: str | None = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        return super().push_to_hub(
            commit_message=commit_message, blocking=blocking, **kwargs
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # Calculate a monotonically increasing checkpoint number that never resets across epochs
        # This prevents checkpoint name collisions when step numbers reset each epoch
        # We use: (epoch_number * 10000) + step_within_epoch to ensure uniqueness
        checkpoint_number = int(self.state.epoch) * 10000 + self.state.global_step
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{checkpoint_number}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                # also check that the checkpoint produces mostly valid outputs
                validity_higher_metrics = [
                    "eval_policy_%_parsable",
                    "eval_policy_%_correct_length",
                    "eval_policy_%_feasible",
                    "eval_policy_%_values_in_range",
                ]
                validity_lower_metrics = [
                    "eval_policy_%_repeated_input",
                ]
                passed_validity_check = True
                for m in validity_higher_metrics:
                    if m not in metrics:
                        logger.warning(f"Metric {m} not found in metrics: {metrics}")
                        continue
                    elif metrics[m] < self.threshold_percent_valid:
                        logger.warning(
                            f"Checkpoint {self.state.global_step} has best {metric_to_check} "
                            + f"but {m} is too low: {metrics[m]} < {self.threshold_percent_valid}"
                        )
                        passed_validity_check = False
                        break
                for m in validity_lower_metrics:
                    if m not in metrics:
                        logger.warning(f"Metric {m} not found in metrics: {metrics}")
                        continue
                    elif metrics[m] > 1.0 - self.threshold_percent_valid:
                        logger.warning(
                            f"Checkpoint {self.state.global_step} has best {metric_to_check} "
                            + f"but {m} is too high: {metrics[m]} > {1.0 - self.threshold_percent_valid}"
                        )
                        passed_validity_check = False
                        break
                if passed_validity_check:
                    self.state.best_metric = float(metric_value)
                    self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            rotate_checkpoints(
                output_dir=run_dir,
                save_total_limit=self.args.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,
                use_mtime=False,
            )
