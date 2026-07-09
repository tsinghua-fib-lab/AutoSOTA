"""DPO trainer with Ehrlich test function validation and checkpoint gating."""

import json
import numpy as np
import os
import random
import torch
import torch.nn as nn

from botorch.test_functions import SyntheticTestFunction
from collections import defaultdict
from contextlib import nullcontext
from ..test_functions.finetune_utils import (
    remove_integer_padding_from_list,
    truncate_after_right_bracket,
)
from holo.test_functions.closed_form import Ehrlich
from torch.utils.data import DataLoader
from transformers.trainer_utils import (
    EvalLoopOutput,
    PREFIX_CHECKPOINT_DIR,
    rotate_checkpoints,
)
from transformers.trainer import TRAINER_STATE_NAME
from transformers.utils import logging as transformers_logging
from trl import DPOTrainer

logger = transformers_logging.get_logger(__name__)


class DPOTrainerWithLogging(DPOTrainer):
    """Extends trl DPOTrainer with Ehrlich-specific evaluation and checkpoint validation.

    Custom functionality beyond base DPOTrainer:
    - Generation during eval with metrics from Ehrlich test function
    - Validity-gated checkpoint saving (% parsable, % feasible, etc.)
    """

    def __init__(
        self,
        *,
        test_fn: SyntheticTestFunction | None = None,
        num_generate_batches: int = 1,
        threshold_percent_valid: float = 0.9,
        generate_during_eval: bool = False,
        pretokenized: bool = False,
        # Legacy alias — accept but forward as processing_class
        tokenizer=None,
        **kwargs,
    ):
        self.test_fn = test_fn
        self.num_generate_batches = num_generate_batches
        self.threshold_percent_valid = threshold_percent_valid
        self.generate_during_eval = generate_during_eval
        self.pretokenized = pretokenized
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Support legacy 'tokenizer' kwarg
        if tokenizer is not None and "processing_class" not in kwargs:
            kwargs["processing_class"] = tokenizer

        super().__init__(**kwargs)

    def store_metrics(
        self,
        metrics: dict[str, float],
        train_eval: str = "train",
    ) -> None:
        """Store metrics for later aggregation in evaluation_loop."""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def get_metrics_for_outputs(
        self,
        outputs: list[str],
        input_scores: list[float],
        input_particles: list[list[int]],
    ) -> tuple[dict[str, float], list[float | None]]:
        """Compute Ehrlich test function metrics for generated outputs.

        Returns both a dictionary of computed metrics and the list of scores.
        """
        total_chars_truncated = 0
        num_parsable = 0
        num_correct_length = 0
        num_values_in_range = 0
        num_feasible = 0
        num_repeated_input = 0
        all_scores_including_nulls: list[float | None] = []
        num_decreased_score = 0

        for i, o in enumerate(outputs):
            o, num_chars_truncated = truncate_after_right_bracket(
                o, return_num_chars_truncated=True
            )
            total_chars_truncated += num_chars_truncated
            try:
                o_list = json.loads(o)
                assert isinstance(o_list, list)
                assert all([int(x) == x for x in o_list])
                particle = [int(x) for x in o_list]
            except (json.JSONDecodeError, ValueError, TypeError, AssertionError):
                logger.warning(
                    f"Prediction is not parsable. Continuing. Prediction:\n{o}"
                )
                all_scores_including_nulls.append(None)
                continue
            num_parsable += 1
            if len(o_list) != self.test_fn.dim:
                all_scores_including_nulls.append(None)
                continue
            num_correct_length += 1
            if isinstance(self.test_fn, Ehrlich):
                if any([x < 0 or x >= self.test_fn.num_states for x in particle]):
                    all_scores_including_nulls.append(None)
                    continue
                num_values_in_range += 1
            if particle == input_particles[i]:
                num_repeated_input += 1
            particle = torch.FloatTensor(particle).unsqueeze(0)
            score = self.test_fn(particle).item()
            if score < input_scores[i]:
                num_decreased_score += 1
            if score == float("inf"):
                all_scores_including_nulls.append(None)
                continue
            num_feasible += 1
            all_scores_including_nulls.append(score)
        num_total = len(outputs)
        scores_wo_nulls = [s for s in all_scores_including_nulls if s is not None]
        output_metrics = {
            "%_parsable": num_parsable / num_total if num_total > 0 else -1,
            "%_correct_length": (
                num_correct_length / num_parsable if num_parsable > 0 else np.nan
            ),
            "%_decreased_score": (
                num_decreased_score / num_values_in_range
                if num_values_in_range > 0
                else np.nan
            ),
            "%_repeated_input": (
                num_repeated_input / num_values_in_range
                if num_values_in_range > 0
                else np.nan
            ),
            "%_feasible": (
                num_feasible / num_values_in_range
                if num_values_in_range > 0
                else np.nan
            ),
            "avg_score": (
                np.mean(scores_wo_nulls) if len(scores_wo_nulls) > 0 else np.nan
            ),
            "avg_num_chars_truncated": total_chars_truncated / num_total
            if num_total > 0
            else -1,
        }
        if isinstance(self.test_fn, Ehrlich):
            output_metrics["%_values_in_range"] = (
                num_values_in_range / num_correct_length
                if num_correct_length > 0
                else np.nan
            )
        return output_metrics, all_scores_including_nulls

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Override evaluation loop to add generation-based metrics."""

        # Generate and evaluate outputs if requested
        if self.generate_during_eval:
            num_samples = len(dataloader.dataset)
            random.seed(self.args.seed)
            if self.num_generate_batches is not None:
                random_indices = random.sample(
                    range(num_samples),
                    k=min(
                        num_samples,
                        self.args.eval_batch_size * self.num_generate_batches,
                    ),
                )
                random_batches_dataset = dataloader.dataset.select(random_indices)
            else:
                random_indices = list(
                    np.arange(0, len(dataloader.dataset), self.args.eval_batch_size)
                )
                random_batches_dataset = dataloader.dataset

            input_particles = [d["prompt"] for d in random_batches_dataset]
            input_scores = [d["prompt_score"] for d in random_batches_dataset]
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

            policy_metrics, policy_scores = self.get_metrics_for_outputs(
                policy_output_decoded, input_scores, input_particles
            )
            policy_metrics = {f"eval_policy_{k}": v for k, v in policy_metrics.items()}
            self.store_metrics(policy_metrics, train_eval="eval")
            ref_metrics, ref_scores = self.get_metrics_for_outputs(
                ref_output_decoded, input_scores, input_particles
            )
            ref_metrics = {f"eval_ref_{k}": v for k, v in ref_metrics.items()}
            self.store_metrics(ref_metrics, train_eval="eval")

        # Base evaluation — skip DPOTrainer's evaluation_loop, go to Trainer's
        initial_output = super(DPOTrainer, self).evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Average stored metrics and merge into output, then reset
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

    def generate_eval_samples(
        self, model: nn.Module, batch: dict[str, torch.LongTensor]
    ) -> tuple[list[str], list[str]]:
        """Generate samples from policy and reference model for the given batch."""
        generate_context_manager = (
            nullcontext
            if not getattr(self, "_peft_has_been_casted_to_bf16", False)
            else torch.cuda.amp.autocast
        )

        # Determine the tokenizer for decoding
        tokenizer = getattr(self, "processing_class", None) or getattr(
            self, "tokenizer", None
        )

        # Modern trl uses input_ids with completion_mask; extract prompt portion
        if "prompt_input_ids" in batch:
            prompt_ids = batch["prompt_input_ids"]
            prompt_mask = batch["prompt_attention_mask"]
        else:
            # Modern format: reconstruct prompt from input_ids + completion_mask
            input_ids = batch["input_ids"]
            completion_mask = batch["completion_mask"]
            # The batch is [chosen, rejected] concatenated; take first half (chosen)
            half = input_ids.shape[0] // 2
            chosen_ids = input_ids[:half]
            chosen_completion_mask = completion_mask[:half]
            chosen_attention_mask = batch["attention_mask"][:half]
            # Prompt = attended tokens where completion_mask is 0
            prompt_lens = (chosen_attention_mask * (1 - chosen_completion_mask)).sum(
                dim=1
            )
            max_prompt_len = prompt_lens.max().item()
            prompt_ids = chosen_ids[:, :max_prompt_len]
            prompt_mask = (prompt_ids != tokenizer.pad_token_id).long()

        prompt_texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        prompt_texts = [
            remove_integer_padding_from_list(text) for text in prompt_texts
        ]
        reencoded = tokenizer(
            prompt_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = reencoded["input_ids"].to(prompt_ids.device)
        prompt_mask = reencoded["attention_mask"].to(prompt_ids.device)

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            if self.ref_model is not None:
                reference_output = self.ref_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                reference_output = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

        policy_output_decoded = tokenizer.batch_decode(
            policy_output[:, prompt_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        reference_output_decoded = tokenizer.batch_decode(
            reference_output[:, prompt_ids.shape[-1] :],
            skip_special_tokens=True,
        )

        return policy_output_decoded, reference_output_decoded

    def _save_checkpoint(
        self, model: nn.Module, trial, metrics: dict[str, float] | None = None
    ) -> None:
        """Save checkpoint with validity gating based on Ehrlich metrics."""
        # Calculate a monotonically increasing checkpoint number
        checkpoint_number = int(self.state.epoch) * 10000 + self.state.global_step
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{checkpoint_number}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
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
                # Also check that the checkpoint produces mostly valid outputs
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
