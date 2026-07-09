import logging
import os
import torch
import warnings

from copy import deepcopy
from dataclasses import dataclass, field
from datasets import Dataset
from pathlib import Path
from transformers import (
    EvalPrediction,
    GenerationConfig,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import SFTConfig, SFTTrainer
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from transformers import (
        DataCollator,
        EvalPrediction,
        PredictionOutput,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    from peft import PeftConfig

from ..infrastructure.file_handler import LocalOrS3Client


@dataclass
class Seq2SeqSFTConfig(SFTConfig):
    """Adds new generation params to SFTConfig."""

    predict_with_generate: bool = field(
        default=False,
        metadata={
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."
        },
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d


class S3Callback(TrainerCallback):
    def __init__(self, s3_output_dir: str, logger: logging.Logger = None):
        self.s3_output_dir = s3_output_dir
        if not self.s3_output_dir.endswith("/"):
            self.s3_output_dir += "/"

        # Use LocalOrS3Client instead of direct s3fs
        # Detect if it's an S3 path
        self.is_s3 = self.s3_output_dir.startswith("s3://")
        self.fs = LocalOrS3Client(init_s3=self.is_s3)
        self.logger = logger

    def on_save(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        checkpoint_dir = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
        )

        # Check if checkpoint still exists before trying to copy
        # It may have been deleted due to save_total_limit or failing validity checks
        if not os.path.exists(checkpoint_dir):
            if self.logger is not None:
                self.logger.warning(
                    f"Checkpoint {checkpoint_dir} does not exist. "
                    "It may have been deleted due to save_total_limit or failing validity checks. "
                    "Skipping S3 upload."
                )
            return

        self.fs.put(checkpoint_dir, self.s3_output_dir, recursive=True)
        if self.logger is not None:
            self.logger.info(
                f"Successfully copied checkpoint in {checkpoint_dir} to {self.s3_output_dir}."
            )


class Seq2SeqSFTTrainer(SFTTrainer):
    """Overrides the evaluate and prediction methods in SFTTrainer with the ones in
    Seq2SeqTrainer so that we can generate during eval.
    (Not currently compatible with FSDP...)
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module] = None,
        args: Seq2SeqSFTConfig = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional["PreTrainedTokenizerBase"] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
        # Legacy aliases — accept but ignore for backwards compat with callers
        max_length: Optional[int] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ):
        # Support legacy 'tokenizer' kwarg
        if tokenizer is not None and processing_class is None:
            processing_class = tokenizer
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
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config

    @staticmethod
    def load_generation_config(
        gen_config_arg: Union[str, GenerationConfig],
    ) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # GenerationConfig provided, nothing to do
        if isinstance(gen_config_arg, GenerationConfig):
            gen_config = deepcopy(gen_config_arg)
        else:
            # str or Path
            pretrained_model_name = (
                Path(gen_config_arg)
                if isinstance(gen_config_arg, str)
                else gen_config_arg
            )
            config_file_name = None

            # Figuring if it is path pointing to a file, pointing to a directory or else a model id or URL
            # This step is required in order to determine config_file_name
            if pretrained_model_name.is_file():
                config_file_name = pretrained_model_name.name
                pretrained_model_name = pretrained_model_name.parent
            # dir path
            elif pretrained_model_name.is_dir():
                pass
            # model id or URL
            else:
                pretrained_model_name = gen_config_arg

            gen_config = GenerationConfig.from_pretrained(
                pretrained_model_name, config_file_name
            )

        # Strict validation to fail early. `GenerationConfig.save_pretrained()`, run at the end of training, throws
        # an exception if there are warnings at validation time.
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                gen_config.validate()
            if len(caught_warnings) > 0:
                raise ValueError(str([w.message for w in caught_warnings]))
        except ValueError as exc:
            raise ValueError(
                "The loaded generation config instance is invalid -- `GenerationConfig.validate()` throws warnings "
                "and/or exceptions. Fix these issues to train your model.\n\nThrown during validation:\n"
                + str(exc)
            )
        return gen_config

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # WORKAROUND: transformers 5.x removed include_inputs_for_metrics from
        # TrainingArguments and no longer passes inputs to compute_metrics via
        # EvalPrediction.  The Ehrlich evaluators (EvaluatorEditPairs /
        # EvaluatorPlainPairs) need access to input_ids and labels to decode
        # prompts and score generated sequences.
        #
        # We stash raw inputs during each prediction_step call (see below),
        # then re-inject them into EvalPrediction here.
        #
        # The proper fix is to refactor the evaluators so they don't depend on
        # raw inputs — e.g. by decoding within prediction_step and passing
        # decoded strings through a different channel.  Tracked in issue #19.
        self._eval_inputs = []
        original_compute_metrics = self.compute_metrics

        def _compute_metrics_with_inputs(eval_pred, **kwargs):
            if self._eval_inputs:
                batched = {}
                for k in self._eval_inputs[0]:
                    vals = [inp[k] for inp in self._eval_inputs]
                    if hasattr(vals[0], "shape"):
                        batched[k] = torch.cat(vals, dim=0)
                    else:
                        batched[k] = vals
                eval_pred = EvalPrediction(
                    predictions=eval_pred.predictions,
                    label_ids=eval_pred.label_ids,
                    inputs=batched,
                )
            return original_compute_metrics(eval_pred, **kwargs)

        self.compute_metrics = _compute_metrics_with_inputs
        try:
            outputs = super().evaluate(
                eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = original_compute_metrics
            self._eval_inputs = []
        torch.cuda.empty_cache()
        return outputs

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # Same include_inputs_for_metrics workaround as evaluate() — see
        # the comment there for full explanation.
        self._eval_inputs = []
        original_compute_metrics = self.compute_metrics

        def _compute_metrics_with_inputs(eval_pred, **kwargs):
            if self._eval_inputs:
                batched = {}
                for k in self._eval_inputs[0]:
                    vals = [inp[k] for inp in self._eval_inputs]
                    if hasattr(vals[0], "shape"):
                        batched[k] = torch.cat(vals, dim=0)
                    else:
                        batched[k] = vals
                eval_pred = EvalPrediction(
                    predictions=eval_pred.predictions,
                    label_ids=eval_pred.label_ids,
                    inputs=batched,
                )
            return original_compute_metrics(eval_pred, **kwargs)

        self.compute_metrics = _compute_metrics_with_inputs
        try:
            result = super().predict(
                test_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = original_compute_metrics
            self._eval_inputs = []
        return result

    @torch.no_grad()
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        # Part of the include_inputs_for_metrics workaround (see evaluate()).
        if not hasattr(self, "_eval_inputs"):
            self._eval_inputs = []
        self._eval_inputs.append(
            {
                k: v.detach().cpu() if hasattr(v, "detach") else v
                for k, v in inputs.items()
            }
        )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        generation_inputs = {}
        # Generation inputs should not contain the input_ids from the labels. Also padding should be on the left side, not right
        generation_inputs["input_ids"] = torch.where(
            inputs["labels"] == -100,
            inputs["input_ids"],
            self.processing_class.pad_token_id,
        )
        generation_inputs["attention_mask"] = torch.where(
            generation_inputs["input_ids"] != self.processing_class.pad_token_id, 1, 0
        )
        if self.processing_class.padding_side == "right":
            # Change padding side to left for generation
            padded_gen_inputs = []
            for i in range(generation_inputs["input_ids"].shape[0]):
                gen_input_ids = generation_inputs["input_ids"][i]
                # get length of non-pad-token IDs
                len_input = sum(generation_inputs["attention_mask"][i]).item()
                pad_tensor = self.processing_class.pad_token_id * torch.ones(
                    generation_inputs["input_ids"].shape[-1] - len_input,
                    dtype=gen_input_ids.dtype,
                    device=gen_input_ids.device,
                )
                new_input_ids = torch.concat([pad_tensor, gen_input_ids[:len_input]])
                padded_gen_inputs.append(new_input_ids)
            generation_inputs["input_ids"] = torch.stack(padded_gen_inputs)
            generation_inputs["attention_mask"] = torch.flip(
                generation_inputs["attention_mask"], dims=(1,)
            )
            padded_gen_inputs = None

        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
        max_gen_len = generated_tokens.shape[-1]
        # Now remove left padding from generated_tokens and pad on right side (for batching)
        new_gen_tokens = []
        for i in range(generated_tokens.shape[0]):
            len_padding = (
                generation_inputs["attention_mask"].shape[-1]
                - generation_inputs["attention_mask"][i].sum().item()
            )
            new_gens = generated_tokens[i][len_padding:]
            new_gen_tokens.append(new_gens)
        new_gen_tokens_padded = []
        for gen_tokens in new_gen_tokens:
            if len(gen_tokens) < max_gen_len:
                padded_gen_tokens = torch.concat(
                    [
                        gen_tokens,
                        self.processing_class.pad_token_id
                        * torch.ones(
                            max_gen_len - len(gen_tokens),
                            dtype=gen_tokens.dtype,
                            device=gen_tokens.device,
                        ),
                    ]
                )
                new_gen_tokens_padded.append(padded_gen_tokens)
            else:
                new_gen_tokens_padded.append(gen_tokens)
        generated_tokens = torch.stack(new_gen_tokens_padded)
        new_gen_tokens = None
        new_gen_tokens_padded = None

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        if (
            gen_config.max_new_tokens is not None
            and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1
            )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if (
                gen_config.max_new_tokens is not None
                and labels.shape[-1] < gen_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1
                )
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.processing_class is not None and hasattr(
            self.processing_class, "pad_token_id"
        ):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.processing_class.pad_token_id
                if self.processing_class.pad_token_id is not None
                else self.processing_class.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
