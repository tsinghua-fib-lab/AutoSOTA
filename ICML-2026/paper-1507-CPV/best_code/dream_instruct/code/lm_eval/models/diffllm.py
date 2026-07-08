import logging
import gc
import json
import os
import random
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, get_dtype
from .cover.modeling_dream import DreamModel
# Import COVER generation utilities.
from .cover import generation_utils as cover_generation_utils

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def empty_cache_by_memory(threshold_gb=70):
    """
    Empty CUDA cache if allocated memory exceeds threshold
    Args:
        threshold_gb: Memory threshold in GB
    """
    if torch.cuda.is_available():
        # Get current memory allocated
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB

        if allocated > threshold_gb:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cache cleared. Memory freed: {allocated:.2f} GB")

@register_model("diffllm")
class DiffLLM(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_prompt_len: Optional[int] = 1024,
        max_new_tokens: Optional[int] = 128,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        classifier_free_guidance: Optional[float] = 1.0,
        pad_to_max_len: Optional[bool] = False,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 32,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)

        # Get algorithm type before model creation (needed to select correct model variant)
        self.alg = kwargs.get("alg", "entropy")
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        # generation params
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", 0.9)
        self.alg = kwargs.get("alg", "entropy")
        self.alg_temp = kwargs.get("alg_temp", 0.0)
        self.top_k = kwargs.get("top_k", None)

        # COVER-specific params (named to match llada version2)
        self.block_length = int(kwargs.get("block_length", 64))
        self.tau_draft = float(kwargs.get("tau_draft", 0.8))
        # Boolean params need string parsing
        # NOTE: For Dream model, KV-cache reverify has known issues due to shifted logits alignment.
        # Default to False until the implementation is fixed.
        use_low_conf_reverify = kwargs.get("version2_use_low_conf_reverify", False)
        if isinstance(use_low_conf_reverify, str):
            self.version2_use_low_conf_reverify = use_low_conf_reverify.lower() == "true"
        else:
            self.version2_use_low_conf_reverify = bool(use_low_conf_reverify)
        self.version2_max_unmask_per_step = int(kwargs.get("version2_max_unmask_per_step", 15))
        self.version2_max_reverify_per_step = int(kwargs.get("version2_max_reverify_per_step", 8))
        self.version2_max_reverify_times = int(kwargs.get("version2_max_reverify_times", 30))
        use_kv_cache_for_reverify = kwargs.get("version2_use_kv_cache_for_reverify", False)
        if isinstance(use_kv_cache_for_reverify, str):
            self.version2_use_kv_cache_for_reverify = use_kv_cache_for_reverify.lower() == "true"
        else:
            self.version2_use_kv_cache_for_reverify = bool(use_kv_cache_for_reverify)
        use_attention_score = kwargs.get("version2_use_attention_score", True)
        if isinstance(use_attention_score, str):
            self.version2_use_attention_score = use_attention_score.lower() == "true"
        else:
            self.version2_use_attention_score = bool(use_attention_score)
        version2_debug = kwargs.get("version2_debug", False)
        if isinstance(version2_debug, str):
            self.version2_debug = version2_debug.lower() == "true"
        else:
            self.version2_debug = bool(version2_debug)
        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.classifier_free_guidance = classifier_free_guidance
        self.pad_to_max_len = pad_to_max_len
        self.sampling_eps = sampling_eps

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        # Always use the standard DreamModel
        # For COVER, monkey-patch the generation methods.
        self.model = (
            DreamModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)

        # For COVER, replace the generation methods with COVER versions.
        if hasattr(self, 'alg') and self.alg == 'cover':
            # Monkey-patch the generation methods from cover_generation_utils.
            import types
            self.model.diffusion_generate = types.MethodType(
                cover_generation_utils.DreamGenerationMixin.diffusion_generate,
                self.model
            )
            self.model._sample = types.MethodType(
                cover_generation_utils.DreamGenerationMixin._sample,
                self.model
            )
            self.model._prepare_generation_config = types.MethodType(
                cover_generation_utils.DreamGenerationMixin._prepare_generation_config,
                self.model
            )
            self.model._prepare_generated_length = types.MethodType(
                cover_generation_utils.DreamGenerationMixin._prepare_generated_length,
                self.model
            )
            self.model._prepare_special_tokens = types.MethodType(
                cover_generation_utils.DreamGenerationMixin._prepare_special_tokens,
                self.model
            )
            self.model._validate_generated_length = types.MethodType(
                cover_generation_utils.DreamGenerationMixin._validate_generated_length,
                self.model
            )
            self.model._expand_inputs_for_generation = cover_generation_utils.DreamGenerationMixin._expand_inputs_for_generation

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str]) -> Union[Tuple[List[str], Optional[int], Optional[float], Optional[dict]], Tuple[List[str], Union[int, List[int]], Union[float, List[float]], Optional[dict]]]:
        # tokenize
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        prompt_ids = prompt_ids[:, -self.max_prompt_len:]
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        # Check if using COVER algorithm.
        if self.alg == "cover":
            return self._generate_batch_cover(prompt_ids, prompts)

        # generate using standard algorithms
        generation_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
        )
        steps_used = getattr(generation_ids, "steps_used", None)
        if steps_used is None:
            steps_used = self.diffusion_steps

        # decode
        responses = [
            self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_ids.sequences)
        ]

        # Return None for inference_times and flip_flop_stats for standard algorithms
        return responses, steps_used, None, None

    def _generate_batch_cover(self, prompt_ids: torch.Tensor, prompts: List[str]) -> Tuple[List[str], Union[int, List[int]], Union[float, List[float]], Optional[List[dict]]]:
        """
        Generate using COVER.

        Args:
            prompt_ids: Tokenized prompt tensor of shape (batch_size, seq_len)
            prompts: Original prompt strings

        Returns:
            Tuple of (responses list, steps_used, inference_times, flip_flop_stats_list)
            Note: steps_used, inference_times, and flip_flop_stats_list are lists of per-sample values
        """
        responses = []
        steps_list = []
        time_list = []
        flip_flop_stats_list = []

        # COVER only supports batch_size=1, so we process each sample sequentially
        for i in range(prompt_ids.shape[0]):
            single_prompt = prompt_ids[i:i+1]  # Keep 2D shape (1, seq_len)

            # Synchronize before timing
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            sample_start_time = time.perf_counter()

            # Call model.diffusion_generate directly with use_kv_cache_reverify=True (now returns 3-tuple)
            result = self.model.diffusion_generate(
                inputs=single_prompt,
                attention_mask=None,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg="threshold",  # COVER uses threshold-based decoding internally
                alg_temp=self.alg_temp,
                # Block-wise generation params
                block_length=self.block_length,
                # Threshold params (tau_draft is used as the primary threshold)
                tau_draft=self.tau_draft,
                threshold=self.tau_draft,
                max_unmask_per_step=self.version2_max_unmask_per_step,
                # Re-verification params
                use_low_conf_reverify=self.version2_use_low_conf_reverify,
                use_kv_cache_for_reverify=self.version2_use_kv_cache_for_reverify,
                use_attention_score=self.version2_use_attention_score,
                max_reverify_per_step=self.version2_max_reverify_per_step,
                max_reverify_times=self.version2_max_reverify_times,
                # Debug
                log_step_stats=self.version2_debug,
            )

            # Handle 3-tuple return from diffusion_generate
            if isinstance(result, tuple) and len(result) == 3:
                output, steps_used, flip_flop_stats = result
            else:
                output = result
                steps_used = getattr(output, "steps_used", None) if hasattr(output, "steps_used") else self.diffusion_steps
                flip_flop_stats = None

            # Extract steps_used from output if not already set
            if steps_used is None:
                steps_used = getattr(output, "steps_used", None)
            if steps_used is None:
                steps_used = self.diffusion_steps

            # Synchronize after generation for accurate timing
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            sample_time = time.perf_counter() - sample_start_time

            # Get output sequences
            if hasattr(output, 'sequences'):
                output_sequences = output.sequences
            else:
                output_sequences = output

            # Decode the generated tokens (excluding the prompt)
            prompt_len = single_prompt.shape[1]
            generated_tokens = output_sequences[0, prompt_len:].tolist()
            response = self.tokenizer.decode(generated_tokens).split(self.tokenizer.eos_token)[0]
            responses.append(response)
            steps_list.append(steps_used)
            time_list.append(sample_time)
            # Always append flip_flop_stats (can be None for individual samples)
            flip_flop_stats_list.append(flip_flop_stats)

        # Return per-sample flip_flop_stats list (not aggregated)
        # For batch_size=1, return single values; otherwise return lists
        if len(steps_list) == 1:
            return responses, steps_list[0], time_list[0], flip_flop_stats_list
        return responses, steps_list, time_list, flip_flop_stats_list

    def _save_flip_flop_stats(self, stats: dict, gen_kwargs: dict, method_name: str) -> None:
        """
        Save flip-flop statistics to a JSON file.

        Args:
            stats: Aggregated flip-flop statistics dictionary
            gen_kwargs: Generation kwargs for parameter logging
            method_name: Name of the generation method.
        """
        if stats is None:
            return

        # Determine output directory
        output_dir = os.environ.get("FLIP_FLOP_OUTPUT_DIR", "./flip_flop_stats")
        os.makedirs(output_dir, exist_ok=True)

        method_params = {
            'tau_draft': self.tau_draft,
            'max_unmask_per_step': self.version2_max_unmask_per_step,
            'max_reverify_times': self.version2_max_reverify_times,
            'use_attention_score': self.version2_use_attention_score,
            'block_length': self.block_length,
        }

        # Create output data
        output_data = {
            'method': method_name,
            'total_flip_flops': stats.get('total_flip_flops', 0),
            'total_remasked': stats.get('total_remasked', 0),
            'flip_flop_rate': stats.get('flip_flop_rate', 0.0),
            'generation_params': {
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
            },
            'method_params': method_params,
        }

        # Generate filename with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"flip_flop_stats_{method_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        eval_logger.info(f"Flip-flop stats saved to {filepath}")

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif self.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()

            start_time = time.perf_counter()
            responses, steps_used, inference_times, flip_flop_stats_list = self._generate_batch(contexts)

            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif self.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()

            batch_time_s = time.perf_counter() - start_time
            per_request_time_s = batch_time_s / max(len(batch_requests), 1)

            # Handle steps_used, inference_times, and flip_flop stats.
            if inference_times is not None:
                # Per-sample times available
                if isinstance(inference_times, list):
                    for idx, req in enumerate(batch_requests):
                        req.inference_time_s = inference_times[idx] if idx < len(inference_times) else inference_times[-1]
                        req.inference_steps = steps_used[idx] if isinstance(steps_used, list) and idx < len(steps_used) else steps_used
                else:
                    # Single sample case
                    for idx, req in enumerate(batch_requests):
                        req.inference_time_s = inference_times
                        req.inference_steps = steps_used
            else:
                # Standard algorithms: use batch average time
                if isinstance(steps_used, list):
                    for idx, req in enumerate(batch_requests):
                        req.inference_time_s = per_request_time_s
                        req.inference_steps = steps_used[idx] if idx < len(steps_used) else steps_used[-1]
                else:
                    for req in batch_requests:
                        req.inference_time_s = per_request_time_s
                        req.inference_steps = steps_used

            # Save per-sample flip-flop stats to each request
            if flip_flop_stats_list is not None and isinstance(flip_flop_stats_list, list):
                for idx, req in enumerate(batch_requests):
                    if idx < len(flip_flop_stats_list) and flip_flop_stats_list[idx] is not None:
                        stats = flip_flop_stats_list[idx]
                        # Extract flip-flop counts from the stats dict
                        # Handle both single-batch (lists with 1 element) and multi-batch stats
                        if 'flip_flop_count' in stats:
                            ff_count = stats['flip_flop_count']
                            req.flip_flop_count = ff_count[0] if isinstance(ff_count, list) else ff_count
                        elif 'total_flip_flops' in stats:
                            req.flip_flop_count = stats['total_flip_flops']
                        else:
                            req.flip_flop_count = 0

                        if 'total_remask_count' in stats:
                            remask_count = stats['total_remask_count']
                            req.flip_flop_remask_count = remask_count[0] if isinstance(remask_count, list) else remask_count
                        elif 'total_remasked' in stats:
                            req.flip_flop_remask_count = stats['total_remasked']
                        else:
                            req.flip_flop_remask_count = 0

                        if 'total_unmask_count' in stats:
                            unmask_count = stats['total_unmask_count']
                            req.flip_flop_unmask_count = unmask_count[0] if isinstance(unmask_count, list) else unmask_count
                        else:
                            req.flip_flop_unmask_count = 0

                        if 'replace_count' in stats:
                            rc = stats['replace_count']
                            req.replace_count = rc[0] if isinstance(rc, list) else rc
                        else:
                            req.replace_count = 0


                        if 'changed_after_remask_count' in stats:
                            cc = stats['changed_after_remask_count']
                            req.changed_after_remask_count = cc[0] if isinstance(cc, list) else cc
                        else:
                            req.changed_after_remask_count = 0

                        if 'keep_count' in stats:
                            kc = stats['keep_count']
                            req.keep_count = kc[0] if isinstance(kc, list) else kc
                        else:
                            req.keep_count = 0
                    else:
                        req.flip_flop_count = None
                        req.flip_flop_remask_count = None
                        req.flip_flop_unmask_count = None
                        req.replace_count = None
                        req.changed_after_remask_count = None
                        req.keep_count = None

            for i, r in enumerate(responses):
                for s in gen_args[0]['until']:
                    r = r.split(s)[0]
                responses[i] = r

            res.extend(responses)
            pbar.update(len(contexts))

        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        if self.pad_to_max_len:
            raise NotImplementedError
        else:
            input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input, 'full').logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        mc_num = self.diffusion_steps
        for _ in range(max(mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
            del logits, loss, perturbed_seq, perturbed_seq_, p_mask, mask_indices
            empty_cache_by_memory(threshold_gb=70)

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError
