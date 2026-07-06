import os
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image

from ..base import BaseModel


try:
    from maa.checkpoint import load_maa_adapter_state
    from maa.modeling import prepare_maa_model
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from maa.checkpoint import load_maa_adapter_state
    from maa.modeling import prepare_maa_model


def _patch_llava_forward_for_transformers() -> None:
    try:
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
    except ImportError:
        return

    def patch_model_class(model_class):
        if hasattr(model_class, "_maa_forward_patch_applied"):
            return
        original_forward = model_class.forward

        def patched_forward(self, *args, **kwargs):
            kwargs.pop("cache_position", None)
            kwargs.pop("num_logits_to_keep", None)
            return original_forward(self, *args, **kwargs)

        model_class.forward = patched_forward
        model_class._maa_forward_patch_applied = True

    patch_model_class(LlavaLlamaForCausalLM)
    patch_model_class(LlavaMistralForCausalLM)


def _infer_conv_mode(model_path: str) -> str:
    model_path = model_path.lower()
    if "llama-3" in model_path or "llama3" in model_path:
        return "llava_llama_3"
    if "mistral" in model_path:
        return "mistral_instruct"
    if "yi" in model_path:
        return "chatml_direct"
    if "vicuna" in model_path or "v1.5" in model_path:
        return "llava_v1"
    return "llava_v0"


class LLaVA_MAA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(
        self,
        model_path=None,
        adapter_path=None,
        kernel_size=3,
        conv_mode=None,
        force_anyres=True,
        **kwargs,
    ):
        _patch_llava_forward_for_transformers()

        model_path = model_path or os.environ.get(
            "MAA_BASE_MODEL",
            "liuhaotian/llava-v1.6-mistral-7b",
        )
        adapter_path = adapter_path or os.environ.get("MAA_ADAPTER_PATH")

        self.model, self.tokenizer, self.image_processor, _ = prepare_maa_model(
            model_path,
            kernel_size=kernel_size,
            trainable=False,
            with_teacher=False,
        )

        if adapter_path:
            missing_keys, unexpected_keys = load_maa_adapter_state(
                self.model,
                adapter_path,
                map_location="cpu",
                strict=False,
            )
            if unexpected_keys:
                warnings.warn(f"Unexpected keys while loading MAA adapter: {len(unexpected_keys)}")
            self._adapter_missing_keys = missing_keys
        else:
            warnings.warn("MAA_ADAPTER_PATH is not set; adapters will use their initialization.")

        self.model.eval()
        self.model.cuda()
        self.model._validate_model_kwargs = lambda *args, **kwargs: None

        if force_anyres:
            if not getattr(self.model.config, "image_aspect_ratio", None):
                self.model.config.image_aspect_ratio = "anyres"
            if not hasattr(self.model.config, "image_grid_pinpoints"):
                self.model.config.image_grid_pinpoints = [
                    [336, 672],
                    [672, 336],
                    [672, 672],
                    [1008, 336],
                    [336, 1008],
                ]

        self.conv_mode = conv_mode or _infer_conv_mode(model_path)
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        content = ""
        images = []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
                else:
                    content += DEFAULT_IMAGE_TOKEN + "\n"

        image_sizes = [image.size for image in images]
        image_tensor = process_images(images, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [
                image.to(self.model.device, dtype=torch.float16) for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        temperature = self.kwargs.get("temperature", 0.2)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=int(os.environ.get("MAA_MAX_NEW_TOKENS", self.kwargs.get("max_new_tokens", 1024))),
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # In transformers >= 4.46, generate() may return only new tokens
        # rather than the full input+output sequence.
        if output_ids.shape[1] >= input_ids.shape[1]:
            new_tokens = output_ids[:, input_ids.shape[1]:]
        else:
            new_tokens = output_ids
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
