import sys
import os.path as osp
import torch
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusion3Pipeline
from types import SimpleNamespace
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    GemmaTokenizerFast,
    Gemma2ForCausalLM,
)

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _get_clip_prompt_embeds(
    prompt: Union[str, List[str]],
    text_encoder,
    tokenizer,
    tokenizer_max_length: int=77,
    num_images_per_prompt: int = 1,
    clip_skip: Optional[int] = None,
):
    device = text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer_max_length} tokens: {removed_text}"
        )
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


def _get_t5_prompt_embeds(
    prompt: Union[str, List[str]],
    text_encoder,
    tokenizer,
    tokenizer_max_length: int = 256,
    num_images_per_prompt: int = 1,
):
    device = text_encoder.device
    dtype = text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {tokenizer_max_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    prompt: Union[str, List[str]],
    text_encoders,
    tokenizers,
    clip_tokenizer_max_length: int = 77,
    t5_tokenizer_max_length: int = 256,
    num_images_per_prompt: int = 1,
    clip_skip: Optional[int] = None,
    text_encoder_dropout: Optional[float] = None,
):
    # clip, clip, t5
    text_encoder_one, text_encoder_two, text_encoder_three = text_encoders
    tokenizer_one, tokenizer_two, tokenizer_three = tokenizers

    prompt_embed_one, pooled_prompt_embed_one = _get_clip_prompt_embeds(
        prompt=prompt,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        tokenizer_max_length=clip_tokenizer_max_length,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )
    if text_encoder_dropout is not None:
        if random.random() < text_encoder_dropout:
            logger.debug("Dropping text_encoder_one")
            prompt_embed_one = torch.zeros_like(prompt_embed_one)
            pooled_prompt_embed_one = torch.zeros_like(pooled_prompt_embed_one)

    prompt_embed_two, pooled_prompt_embed_two = _get_clip_prompt_embeds(
        prompt=prompt,
        text_encoder=text_encoder_two,
        tokenizer=tokenizer_two,
        tokenizer_max_length=clip_tokenizer_max_length,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )
    if text_encoder_dropout is not None:
        if random.random() < text_encoder_dropout:
            logger.debug("Dropping text_encoder_two")
            prompt_embed_two = torch.zeros_like(prompt_embed_two)
            pooled_prompt_embed_two = torch.zeros_like(pooled_prompt_embed_two)

    clip_prompt_embeds = torch.cat([prompt_embed_one, prompt_embed_two], dim=-1)

    prompt_embed_three = _get_t5_prompt_embeds(
        prompt=prompt,
        text_encoder=text_encoder_three,
        tokenizer=tokenizer_three,
        tokenizer_max_length=t5_tokenizer_max_length,
        num_images_per_prompt=num_images_per_prompt,
    )
    if text_encoder_dropout is not None:
        if random.random() < text_encoder_dropout:
            logger.debug("Dropping text_encoder_three")
            prompt_embed_three = torch.zeros_like(prompt_embed_three)

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, prompt_embed_three.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embed_three], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed_one, pooled_prompt_embed_two], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


def compute_text_embeddings(
    prompt: Union[str, List[str]],
    text_encoders,
    tokenizers,
    clip_tokenizer_max_length: int = 77,
    t5_tokenizer_max_length: int = 256,
    num_images_per_prompt: int = 1,
    clip_skip: Optional[int] = None,
    text_encoder_dropout: Optional[float] = None,
    device: Optional[torch.device] = None,
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt,
            text_encoders,
            tokenizers,
            clip_tokenizer_max_length,
            t5_tokenizer_max_length,
            num_images_per_prompt,
            clip_skip,
            text_encoder_dropout,
        )
        if device is not None:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


class SD3Infer:
    def __init__(
        self,
        pretrained_path: str,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[torch.device] = torch.device("cuda"),
    ):
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(pretrained_path, torch_dtype=torch_dtype)
        self.pipeline = self.pipeline.to(device)

    def combine_prompts_embeds(
        self,
        clip_l14_prompt_embeds: torch.FloatTensor,  # [bs, 77, 768]
        clip_l14_pooled_prompt_embeds: torch.FloatTensor,  # [bs, 768]
        openclip_bigg14_prompt_embeds: torch.FloatTensor,  # [bs, 77, 1280]
        openclip_bigg14_pooled_prompt_embeds: torch.FloatTensor,  # [bs, 1280]
        t5_v1_1_xxl_prompt_embeds: torch.FloatTensor,  # [bs, 256, 4096]
    ):
        clip_prompt_embeds = torch.cat([clip_l14_prompt_embeds, openclip_bigg14_prompt_embeds], dim=-1)
        t5_prompt_embeds = t5_v1_1_xxl_prompt_embeds
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
        pooled_prompt_embeds = torch.cat([clip_l14_pooled_prompt_embeds, openclip_bigg14_pooled_prompt_embeds], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def inference_with_prompt_embeds(
        self,
        prompt_embeds: torch.FloatTensor,  # [bs, 333, 4096]
        pooled_prompt_embeds: torch.FloatTensor,  # [bs, 2048]
    ):
        generator = torch.Generator(device=self.pipeline.device).manual_seed(0)
        image = self.pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]
        return image

    def inference_with_separate_prompt_embeds(
        self,
        clip_l14_prompt_embeds: torch.FloatTensor,  # [bs, 77, 768]
        clip_l14_pooled_prompt_embeds: torch.FloatTensor,  # [bs, 768]
        openclip_bigg14_prompt_embeds: torch.FloatTensor,  # [bs, 77, 1280]
        openclip_bigg14_pooled_prompt_embeds: torch.FloatTensor,  # [bs, 1280]
        t5_v1_1_xxl_prompt_embeds: torch.FloatTensor,  # [bs, 256, 4096]
    ):
        prompt_embeds, pooled_prompt_embeds = self.combine_prompts_embeds(
            clip_l14_prompt_embeds,
            clip_l14_pooled_prompt_embeds,
            openclip_bigg14_prompt_embeds,
            openclip_bigg14_pooled_prompt_embeds,
            t5_v1_1_xxl_prompt_embeds,
        )
        image = self.inference_with_prompt_embeds(prompt_embeds, pooled_prompt_embeds)
        return image

class CLIPTextEncoder:
    def __init__(
        self,
        pretrained_path,
        model_revision,
        model_variant,
        tokenizer_subfolder,
        encoder_subfolder,
    ):
        self.config = SimpleNamespace(
            pretrained_path=pretrained_path,
            revision=model_revision,
            variant=model_variant,
            tokenizer_subfolder=tokenizer_subfolder,
            encoder_subfolder=encoder_subfolder,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_path,
            subfolder=tokenizer_subfolder,
            revision=model_revision,
        )
        self.encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_path,
            subfolder=encoder_subfolder,
            revision=model_revision,
            variant=model_variant,
        )

    def freeze(self):
        logger.info("Freezing CLIPTextEncoder")
        self.encoder.requires_grad_(False)
        logger.info("CLIPTextEncoder frozen")

    def to(self, device, dtype=None):
        logger.info(f"Moving CLIPTextEncoder to {device} and {dtype}")
        if dtype is not None:
            self.encoder.to(device, dtype=dtype)
        else:
            self.encoder.to(device)
        logger.info(f"CLIPTextEncoder moved to {device} and {dtype}")

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        tokenizer_max_length: int=77,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
    ):
        device = self.encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, (tokenizer_max_length - 1):-1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def __call__(
        self,
        prompt: Union[str, List[str]],
        tokenizer_max_length: int=77,
        num_images_per_prompt: int = 1,
        clip_skip: Optional[int] = None,
    ):
        prompt_embeds, pooled_prompt_embeds = self.get_prompt_embeds(
            prompt=prompt,
            tokenizer_max_length=tokenizer_max_length,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
        )
        return prompt_embeds, pooled_prompt_embeds

class T5TextEncoder:
    def __init__(
        self,
        pretrained_path,
        model_revision,
        model_variant,
        tokenizer_subfolder,
        encoder_subfolder,
    ):
        self.config = SimpleNamespace(
            pretrained_path=pretrained_path,
            revision=model_revision,
            variant=model_variant,
            tokenizer_subfolder=tokenizer_subfolder,
            encoder_subfolder=encoder_subfolder,
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(
            pretrained_path,
            subfolder=tokenizer_subfolder,
            revision=model_revision,
        )
        self.encoder = T5EncoderModel.from_pretrained(
            pretrained_path,
            subfolder=encoder_subfolder,
            revision=model_revision,
            variant=model_variant,
        )

    def freeze(self):
        logger.info("Freezing T5TextEncoder")
        self.encoder.requires_grad_(False)
        logger.info("T5TextEncoder frozen")

    def to(self, device, dtype=None):
        logger.info(f"Moving T5TextEncoder to {device} and {dtype}")
        if dtype is not None:
            self.encoder.to(device, dtype=dtype)
        else:
            self.encoder.to(device)
        logger.info(f"T5TextEncoder moved to {device} and {dtype}")

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        tokenizer_max_length: int = 256,
        num_images_per_prompt: int = 1,
    ):
        device = self.encoder.device
        dtype = self.encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, (tokenizer_max_length - 1):-1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {tokenizer_max_length} tokens: {removed_text}"
            )

        prompt_embeds = self.encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def __call__(
        self,
        prompt: Union[str, List[str]],
        tokenizer_max_length: int = 256,
        num_images_per_prompt: int = 1,
    ):
        prompt_embeds = self.get_prompt_embeds(
            prompt=prompt,
            tokenizer_max_length=tokenizer_max_length,
            num_images_per_prompt=num_images_per_prompt,
        )
        return prompt_embeds

class Gemma2TextEncoder:
    def __init__(
        self,
        pretrained_path,
        model_revision,
        model_variant,
        tokenizer_subfolder,
        encoder_subfolder,
    ):
        self.config = SimpleNamespace(
            pretrained_path=pretrained_path,
            revision=model_revision,
            variant=model_variant,
            tokenizer_subfolder=tokenizer_subfolder,
            encoder_subfolder=encoder_subfolder,
        )
        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            pretrained_path,
            subfolder=tokenizer_subfolder,
            revision=model_revision,
        )
        self.tokenizer.padding_side = 'right'
        assert self.tokenizer.padding_side == "right"
        self.model = Gemma2ForCausalLM.from_pretrained(
            pretrained_path,
            revision=model_revision,
            variant=model_variant,
        )

    def freeze(self):
        logger.info("Freezing Gemma2TextEncoder")
        self.model.requires_grad_(False)
        logger.info("Gemma2TextEncoder frozen")

    def to(self, device, dtype=None):
        logger.info(f"Moving Gemma2TextEncoder to {device} and {dtype}")
        if dtype is not None:
            self.model.to(device, dtype=dtype)
        else:
            self.model.to(device)
        logger.info(f"Gemma2TextEncoder moved to {device} and {dtype}")

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        tokenizer_max_length: int = 256,
        num_images_per_prompt: int = 1,
        add_special_tokens=False,
    ):
        device = self.model.device
        dtype = self.model.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, (tokenizer_max_length - 1):-1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {tokenizer_max_length} tokens: {removed_text}"
            )

        prompt_embeds = self.model.model.embed_tokens(text_input_ids.to(device))
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def __call__(
        self,
        prompt: Union[str, List[str]],
        tokenizer_max_length: int = 256,
        num_images_per_prompt: int = 1,
        add_special_tokens=False,
    ):
        prompt_embeds = self.get_prompt_embeds(
            prompt=prompt,
            tokenizer_max_length=tokenizer_max_length,
            num_images_per_prompt=num_images_per_prompt,
            add_special_tokens=add_special_tokens,
        )
        return prompt_embeds
