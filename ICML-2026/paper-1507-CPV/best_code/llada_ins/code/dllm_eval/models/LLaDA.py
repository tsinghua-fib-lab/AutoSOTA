import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeVar
import torch
import torch.nn.functional as F
import numpy as np
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from packaging import version
from tqdm import tqdm
import torch.distributed as dist
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from dllm_eval.api.instance import Instance
from dllm_eval.api.model import LM, TemplateLM
from dllm_eval.api.registry import register_model
from dllm_eval.models.utils import get_dtype, configure_pad_token


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")

def add_gumbel_noise(logits, temperature):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

_generate_ori_impl = generate


@ torch.no_grad()
def generate_ori(
    model,
    tokenizer,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    tau_draft=None,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    **kwargs,
):
    """
    Compatibility wrapper for the original LLaDA `generate` implementation.

    Notes:
      - Keeps a unified call signature with our custom generators (accepts `tokenizer`, `tau_draft`, etc.).
      - Returns `(tokens, nfe)` to match the evaluator expectations.
    """
    if gen_length % block_length != 0:
        raise ValueError(
            f"`generate_ori` requires `gen_length % block_length == 0`, got gen_length={gen_length}, block_length={block_length}."
        )
    num_blocks = gen_length // block_length
    if steps % max(num_blocks, 1) != 0:
        raise ValueError(
            f"`generate_ori` requires `steps % (gen_length//block_length) == 0`, got steps={steps}, gen_length={gen_length}, block_length={block_length}."
        )

    out = _generate_ori_impl(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
    )
    return out, int(steps)

_GENERATE_METHODS = {
    "generate_ori": generate_ori,
    "ori": generate_ori,
    "original": generate_ori,
}


def get_generate_fn(method: str):
    if method is None:
        return generate_ori
    key = str(method).strip().lower()
    if not key:
        return generate_ori
    if key in _GENERATE_METHODS:
        return _GENERATE_METHODS[key]
    raise ValueError(
        f"Unknown generate method '{method}'. Available: {sorted(_GENERATE_METHODS.keys())}"
    )

# `generate` remains the original implementation defined above.

@register_model("LLaDA")
class LLaDA(TemplateLM):
    AUTO_MODEL_CLASS = transformers.AutoModel
    _DEFAULT_MAX_LENGTH = 20480
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        escape_until:Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        mc_num: int = 1024,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        is_check_greedy : bool =True,
        assistant_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mc_num = mc_num
        self.mask_id = mask_id
        self.remasking = remasking
        self.pretrained = pretrained
        self.is_check_greedy = is_check_greedy
        self.assistant_prefix = assistant_prefix
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
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
            if not (parallelize or accelerator.num_processes > 1):
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
            else:
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )
            revision = str(revision)
            revision = revision + ("/" + subfolder if subfolder is not None else "")
            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                **kwargs,
            )
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )
        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > self.accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {self.accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{self.accelerator.device}")
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
            else:
                self._rank = 0
                self._world_size = 1
        else:
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        self.is_first_inference = True

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        if hasattr(self, "accelerator"):
            return self.accelerator.local_process_index
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def world_size(self):
        if hasattr(self, "_world_size"):
            return self._world_size
        if hasattr(self, "accelerator"):
            return self.accelerator.num_processes
        return int(os.environ.get("WORLD_SIZE", 1))

    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if parallelize is None and gpus is not None and gpus > 1:
            parallelize = True
        args = {}
        if parallelize:
            max_memory_all_gpus = get_max_memory()
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            max_memory_per_gpu_map = {
                device_idx: max_memory_per_gpu for device_idx in range(len(max_memory_all_gpus))
            } if max_memory_per_gpu is not None else {k: v for k, v in max_memory_all_gpus.items()}
            if hasattr(self, "accelerator"):
                max_memory_per_gpu_map = {
                    k: v for k, v in max_memory_all_gpus.items() if k % num_local_processes == self.accelerator.process_index % num_local_processes
                }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto"
            args["offload_folder"] = offload_folder
            if max_cpu_memory is not None:
                args["max_memory"]["cpu"] = max_cpu_memory
            eval_logger.info(
                f"Model parallel set to True. Max memory per GPU: {args['max_memory']}, Device map: {args['device_map']}"
            )
        else:
            args["device_map"] = {"": str(self.device)}
            eval_logger.info(
                f"Model parallel set to False. Device map: {args['device_map']}"
            )
        return args

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length > 1e10:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        assert backend in ["default", "causal", "seq2seq"]
        if backend != "default":
            self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "seq2seq"
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
            else:
                eval_logger.warning(
                    "HF model type is neither CausalLM nor Seq2SeqLM. Assuming CausalLM."
                )
                self.backend = "causal"

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        if autogptq or gptqmodel:
            raise NotImplementedError("Quantization options are not implemented for this custom class.")

        from dllm_eval.models.cover.modeling_llada_kv_cover import LLaDAModelLM

        model_dtype = get_dtype(dtype)
        eval_logger.info(f"Loading original LLaDA baseline model with dtype: {model_dtype}")

        model_kwargs = kwargs if kwargs else {}
        if not parallelize:
            model_kwargs.update(
                self._get_accelerate_args(
                    parallelize=parallelize,
                    gpus=gpus,
                    max_memory_per_gpu=max_memory_per_gpu,
                    max_cpu_memory=max_cpu_memory,
                    offload_folder=offload_folder,
                )
            )

        self._model = LLaDAModelLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )

        if peft:
            from peft import PeftModel

            eval_logger.info(f"Loading PEFT model from {peft}")
            self._model = PeftModel.from_pretrained(self._model, peft, torch_dtype=model_dtype)

        if not parallelize:
            self._model = self._model.to(self.device)
        self._model.eval()

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast_tokenizer
        }
        if add_bos_token:
            kwargs["add_bos_token"] = True
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, **kwargs)
            else:
                self.tokenizer = tokenizer
        else:
            model_name = pretrained if isinstance(pretrained, str) else self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs["add_special_tokens"] = self.add_bos_token
        else:
            special_tokens_kwargs["add_special_tokens"] = add_special_tokens
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        add_special_tokens = {"add_special_tokens": self.add_bos_token} if self.backend == "causal" else {}
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len and encoding["input_ids"].size(1) > left_truncate_len:
            eval_logger.warning(
                f"Left-truncating from {encoding['input_ids'].size(1)} to {left_truncate_len} tokens."
            )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if self.backend == "seq2seq":
                return self.model(input_ids=inps, attention_mask=attn_mask, labels=labels).logits
            else:
                return self.model(inps, attention_mask=attn_mask).logits

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running generate_until requests")
        ds_data = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds_data)
        gen_kwargs = requests[0].args[1]
        generate_fn = get_generate_fn(gen_kwargs.get("generate_method", None))
        tau_draft = gen_kwargs.get("tau_draft", None)
        if tau_draft is None:
            tau_draft = gen_kwargs.get("threshold", None)
        num_nfe=0
        req_offset = 0
        for batch in ds.iter(batch_size=int(self.batch_size)):
            t0 = time.perf_counter()
            contexts = batch["text"]
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
            )
            prompt_length = context_enc.shape[1]
            generate_call_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": context_enc,
                "steps": gen_kwargs.get("steps", gen_kwargs.get("gen_length", 128)),
                "gen_length": gen_kwargs.get("gen_length", 128),
                "block_length": gen_kwargs.get("block_length", 32),
                "temperature": gen_kwargs.get("temperature", 0.0),
                "mask_id": self.mask_id,
            }
            if tau_draft is not None:
                generate_call_kwargs["tau_draft"] = tau_draft
            out_full, nfe = generate_fn(
                **generate_call_kwargs,
            )
            generated_tokens = out_full[:, prompt_length:]
            cont_toks_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            num_nfe+=nfe
            t1 = time.perf_counter()
            batch_time_s = t1 - t0
            batch_reqs = requests[req_offset : req_offset + len(cont_toks_list)]
            per_req_time_s = (batch_time_s / len(batch_reqs)) if batch_reqs else None
            for req in batch_reqs:
                req.inference_steps = int(nfe) if nfe is not None else None
                req.inference_time_seconds = per_req_time_s
            req_offset += len(batch_reqs)
            # if self.rank == 0 and self.is_first_inference:
            #     eval_logger.info("\n--- First Batch Inference (Rank 0) ---")
            #     for i, (question, answer) in enumerate(zip(contexts, cont_toks_list)):
            #         eval_logger.info(f"Question {i+1}: {question}")
            #         eval_logger.info(f"Answer   {i+1}: {answer}\n")
            #     eval_logger.info("------------------------------------")
            #     self.is_first_inference = False
            for s in cont_toks_list:
                if not self.escape_until:
                    stop_sequences = gen_kwargs.get("until", [])
                    if stop_sequences:
                        for term in stop_sequences:
                            if len(term) > 0:
                                s = s.split(term)[0]
                res.append(s)
                bar.update(1)
        bar.close()
        return res

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.assistant_prefix:
            chat_templated += self.assistant_prefix
        return chat_templated
