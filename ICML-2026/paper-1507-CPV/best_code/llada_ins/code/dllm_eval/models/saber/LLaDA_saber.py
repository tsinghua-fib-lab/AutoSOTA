import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from tqdm import tqdm

from dllm_eval.api.instance import Instance
from dllm_eval.api.registry import register_model
from dllm_eval.models.LLaDA import LLaDA as BaseLLaDA
from dllm_eval.models.utils import get_dtype

from .decoding import generate_with_saber
from .inference import _normalize_saber_gen_kwargs, init_saber_model


eval_logger = logging.getLogger(__name__)


@register_model("LLaDA_saber")
class LLaDA_saber(BaseLLaDA):
    """
    LLaDA wrapper that uses SABER decoding (`generate_with_saber`).

    Notes:
      - Current SABER implementation allocates tensors with batch_size == 1. This wrapper
        enforces per-request batching to keep evaluation correct.
    """

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
        **kwargs: Any,
    ) -> None:
        if autogptq or gptqmodel:
            raise NotImplementedError("Quantization options are not implemented for this custom class.")

        model_dtype = get_dtype(dtype)
        eval_logger.info(f"Loading SABER model with dtype: {model_dtype}")

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

        self._model = init_saber_model(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
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

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running saber generate_until requests")

        ds_data = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds_data)

        gen_kwargs: Dict[str, Any] = requests[0].args[1] or {}
        defaults: Dict[str, Any] = {
            "n": gen_kwargs.get("n", 2),
            "mu": gen_kwargs.get("mu", 8),
            "gen_length": gen_kwargs.get("gen_length", 256),
            "block_length": gen_kwargs.get("block_length", 256),
            "temperature": gen_kwargs.get("temperature", 0.0),
            "mask_id": self.mask_id,
            "track_flip_flop": gen_kwargs.get("track_flip_flop", True),
        }
        call_kwargs = _normalize_saber_gen_kwargs({**defaults, **gen_kwargs})

        gen_length = int(call_kwargs.get("gen_length", defaults["gen_length"]))
        block_length = int(call_kwargs.get("block_length", defaults["block_length"]))
        if gen_length % block_length != 0:
            raise ValueError(
                f"SABER requires `gen_length % block_length == 0`, got gen_length={gen_length}, block_length={block_length}."
            )

        req_offset = 0
        # Enforce batch_size=1 for correctness (SABER decoding is implemented for a single sample).
        for batch in ds.iter(batch_size=1):
            t0 = time.perf_counter()
            contexts = batch["text"]
            context_enc, _attn_masks = self.tok_batch_encode(contexts, truncation=self.truncation)
            prompt_length = int(context_enc.shape[1])

            result = generate_with_saber(self.model, context_enc, **call_kwargs)
            if isinstance(result, tuple) and len(result) == 3:
                out_full, nfe, flip_flop_stats = result
            else:
                out_full, nfe = result
                flip_flop_stats = None
            generated_tokens = out_full[:, prompt_length:]
            cont_toks_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

            t1 = time.perf_counter()
            batch_time_s = t1 - t0
            batch_reqs = requests[req_offset : req_offset + len(cont_toks_list)]
            per_req_time_s = (batch_time_s / len(batch_reqs)) if batch_reqs else None
            for i, req in enumerate(batch_reqs):
                req.inference_steps = int(nfe) if nfe is not None else None
                req.inference_time_seconds = per_req_time_s
                if flip_flop_stats is not None:
                    ff_count = flip_flop_stats.get("flip_flop_count", [0])
                    remask_count = flip_flop_stats.get("total_remask_count", [0])
                    unmask_count = flip_flop_stats.get("total_unmask_count", [0])

                    req.flip_flop_count = (
                        ff_count[i]
                        if isinstance(ff_count, list) and i < len(ff_count)
                        else (ff_count[0] if isinstance(ff_count, list) else ff_count)
                    )
                    req.flip_flop_remask_count = (
                        remask_count[i]
                        if isinstance(remask_count, list) and i < len(remask_count)
                        else (remask_count[0] if isinstance(remask_count, list) else remask_count)
                    )
                    req.flip_flop_unmask_count = (
                        unmask_count[i]
                        if isinstance(unmask_count, list) and i < len(unmask_count)
                        else (unmask_count[0] if isinstance(unmask_count, list) else unmask_count)
                    )
                else:
                    req.flip_flop_count = None
                    req.flip_flop_remask_count = None
                    req.flip_flop_unmask_count = None
            req_offset += len(batch_reqs)

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
