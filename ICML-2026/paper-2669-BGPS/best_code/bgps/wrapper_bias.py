import os
import torch
# torch.backends.cuda.enable_cudnn_sdp(False)
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_math_sdp(True)
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional
from copy import deepcopy
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import cProfile
import pstats
import types
from utils.utils import custom_unet_forward, custom_unet_forward_new

try:
    from diffusers.models.lora import (
        PatchedLoraProjection,
        text_encoder_attn_modules,
        text_encoder_mlp_modules,
    )
except ImportError:  # diffusers < 0.30 fallback
    from diffusers.loaders import (
        PatchedLoraProjection,
        text_encoder_attn_modules,
        text_encoder_mlp_modules,
    )


def _remove_text_encoder_lora_layers(text_encoder) -> None:
    """
    Detach legacy PatchedLoraProjection modules so we can re-apply them with a
    fresh configuration. Mirrors the behavior of diffusers<=0.20.
    """
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        for proj_attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            proj = getattr(attn_module, proj_attr, None)
            if isinstance(proj, PatchedLoraProjection):
                setattr(attn_module, proj_attr, proj.regular_linear_layer)

    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        for proj_attr in ("fc1", "fc2"):
            proj = getattr(mlp_module, proj_attr, None)
            if isinstance(proj, PatchedLoraProjection):
                setattr(mlp_module, proj_attr, proj.regular_linear_layer)


def _modify_text_encoder_for_lora(text_encoder, *, lora_scale=1.0, rank=4, dtype=None, patch_mlp=False):
    """
    Recreates the legacy `_modify_text_encoder` helper so we can keep loading the
    LoRA checkpoints that were trained with PatchedLoraProjection layers.
    """
    _remove_text_encoder_lora_layers(text_encoder)

    for _, attn_module in text_encoder_attn_modules(text_encoder):
        attn_module.q_proj = PatchedLoraProjection(
            attn_module.q_proj, lora_scale, network_alpha=None, rank=rank, dtype=dtype
        )
        attn_module.k_proj = PatchedLoraProjection(
            attn_module.k_proj, lora_scale, network_alpha=None, rank=rank, dtype=dtype
        )
        attn_module.v_proj = PatchedLoraProjection(
            attn_module.v_proj, lora_scale, network_alpha=None, rank=rank, dtype=dtype
        )
        attn_module.out_proj = PatchedLoraProjection(
            attn_module.out_proj, lora_scale, network_alpha=None, rank=rank, dtype=dtype
        )

    if patch_mlp:
        for _, mlp_module in text_encoder_mlp_modules(text_encoder):
            mlp_module.fc1 = PatchedLoraProjection(
                mlp_module.fc1, lora_scale, network_alpha=None, rank=rank, dtype=dtype
            )
            mlp_module.fc2 = PatchedLoraProjection(
                mlp_module.fc2, lora_scale, network_alpha=None, rank=rank, dtype=dtype
            )

def load_text_encoder_lora_weights(
    text_encoder,
    checkpoint_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rank: int = 4,
    patch_mlp: bool = True,
    lora_scale: float = 1.0,
) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    _modify_text_encoder_for_lora(
        text_encoder,
        lora_scale=lora_scale,
        rank=rank,
        dtype=dtype,
        patch_mlp=patch_mlp,
    )
    state_dict = {
        k: v.to(device=text_encoder.device, dtype=text_encoder.dtype) for k, v in state_dict.items()
    }
    load_result = text_encoder.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(
            f"Unexpected keys while loading text encoder LoRA: {load_result.unexpected_keys}"
        )

# --- your classifier classes (as provided) ---
class CustomMLP(nn.Module):
    def __init__(self, input_size: int = 1*1280*8*8, output_size: int = 2, num_timesteps: int = 50):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_timesteps)])
        self.forward_timesteps = list(range(1, num_timesteps+1)) # 1 to 49
        self.reversed_timesteps = list(reversed(self.forward_timesteps))

    @torch.autocast(device_type="cuda")
    def forward(self, input_tensor: torch.Tensor, timestep_index: int):
        batch_size = input_tensor.shape[0]
        reshaped_input = input_tensor.reshape(batch_size, -1)
        reversed_t_idx = self.reversed_timesteps[timestep_index-1]
        output_tensor = self.linears[reversed_t_idx](reshaped_input)
        return output_tensor

class MLPClassifierIF(nn.Module):
    def __init__(self, num_classes: int, num_timesteps: int, feature_shape: tuple):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.forward_timesteps = list(range(1, num_timesteps + 1))
        self.reversed_timesteps = list(reversed(self.forward_timesteps))
        c, h, w = feature_shape
        in_dim = c * h * w
        self.linears = nn.ModuleList([nn.Linear(in_dim, num_classes) for _ in range(self.num_timesteps)])
    
    def forward(self, x, t):
        # x: [1, num_steps, B, C, H, W]
        reversed_t = self.reversed_timesteps[self.forward_timesteps.index(t)]
        x_t = x[:, reversed_t - 1, ...]      # [1, B, C, H, W]
        x_t = x_t.reshape(-1, *x_t.size()[-3:])  # [B, C, H, W]
        x_t = x_t.reshape(x_t.size(0), -1)          # [B, in_dim]
        return self.linears[t - 1](x_t)          # [B, num_classes]
    

class AttributeClassifier:
    def __init__(self, model_path, output_size, device, num_timesteps=50, input_size=1*1280*8*8):
        self.model = self.make_model(model_path, device, output_size, num_timesteps, input_size)

    def make_model(self, path: os.PathLike, device: torch.device, output_size: int, num_timesteps: int, input_size: int) -> CustomMLP:
        model = CustomMLP(output_size=output_size, num_timesteps=num_timesteps, input_size=input_size).to(device)
        sd = torch.load(path, map_location=device)
        model.load_state_dict(sd)
        model.eval()
        return model

class AttributeClassifierIF:
    def __init__(self, model_path, output_size, device, num_timesteps=48, feature_shape=(2816, 8, 8)):
        self.model = self.make_model_if(model_path, device, output_size, num_timesteps, feature_shape)

    def make_model_if(self, path: os.PathLike, device: torch.device, output_size: int, num_timesteps: int, feature_shape: tuple) -> nn.Module:
        model = MLPClassifierIF(num_classes=output_size, num_timesteps=num_timesteps, feature_shape=feature_shape).to(device)
        sd = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(sd)
        model.eval()
        return model


class BGPS(nn.Module):
    """
    BGPS that *maximizes gender bias* according to a pretrained linear classifier
    on SD UNet mid-block activations. Uses LLM beam search; scores each partial
    prompt by -(cross_entropy) of classifier logits at a chosen timestep.

    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        # --- LLM ---
        model_ckpt = cfg["model"]["model_ckpt"]
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            load_in_4bit=cfg["model"].get("load_in_4bit", False),
            load_in_8bit=cfg["model"].get("load_in_8bit", False),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        trust_remote = ("Qwen" in model_ckpt) or ("deepseek" in model_ckpt.lower())
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_ckpt,
            use_fast=False,
            trust_remote_code=trust_remote,
        )
        # prompts / chat template
        self.system_prompt = cfg["model"].get("system_prompt", "You are a helpful prompt writer.")
        self.user_prompt = cfg["model"].get("user_prompt", "Write a short, vivid prompt describing a single person.")
        self.model_prompt = cfg["model"].get("model_prompt", "")
        self.model_prompt_primer = cfg["model"].get("model_prompt_primer", "")
        self.model_occupation_template = cfg["model"].get("model_occupation_template", "")
        self.use_occupation_prompt = cfg.get("use_occupation_template", False)

        if model_ckpt == "mistralai/Mistral-7B-Instruct-v0.3":
            self.model_prompt_primer += " "
        self.prompt_template = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
            {"role": "assistant", "content": self.model_prompt + self.model_prompt_primer},
        ]
        # self.llm_prompt = self.llm_tokenizer.apply_chat_template(
        #     self.prompt_template, return_tensors="pt"
        # )[:, :-1].to(self.device)
        self.llm_prompt = self.llm_tokenizer.apply_chat_template(
            self.prompt_template, return_tensors="pt",continue_final_message=True,
        ).to(self.device)

        # beam search settings
        self.beam_expand_factor = cfg["model"].get("beam_expand_factor", 1)
        self.beam_size = cfg["model"].get("num_beams", 10)
        self.llm_beam_size = self.beam_size * self.beam_expand_factor
        self.beam_offset = torch.arange(
            0, 1 * self.beam_size, step=self.beam_size, dtype=torch.long, device=self.device
        )
        self.sampling_temperature = cfg["model"].get("sampling_temperature", 1.0)
        self.candidate_pool_multiplier = cfg["model"].get("candidate_pool_multiplier", 2)

        self.max_length = cfg["model"].get("max_length", 32)
        self.min_length = cfg["model"].get("min_length", 8)
        self.length_cutoff = cfg.get("length_cutoff", False)

        # scoring weights
        self.llm_alpha = cfg["model"].get("llm_alpha", 1.0)
        self.clf_alpha = cfg["model"].get("clf_alpha", 1.0)
        self.clf2_alpha = cfg["model"].get("clf2_alpha", 1.0)

        # --- Stable Diffusion ---
        self.sd_version = cfg["model"]["sd_version"]
        self.eval_set_use_full_if_pipeline = cfg["model"].get("eval_set_use_full_if_pipeline", False)
        self.if_stage_2_model = cfg["model"].get("if_stage_2_model", "DeepFloyd/IF-II-L-v1.0")
        self.if_stage_2_num_inference_steps = cfg["model"].get("if_stage_2_num_inference_steps", 40)
        self.if_stage_2_guidance_scale = cfg["model"].get("if_stage_2_guidance_scale", 4.0)
        self.if_stage_3_model = cfg["model"].get("if_stage_3_model", "stabilityai/stable-diffusion-x4-upscaler")
        self.if_stage_3_num_inference_steps = cfg["model"].get("if_stage_3_num_inference_steps", 75)
        self.if_stage_3_guidance_scale = cfg["model"].get("if_stage_3_guidance_scale", 9.0)
        self.if_stage_2_pipeline = None
        self.if_stage_3_pipeline = None
        if "DeepFloyd" in self.sd_version:
            print("Loading IF model:", self.sd_version)
            self.deepfloyd_if = True
            self.sd_pipeline = DiffusionPipeline.from_pretrained(self.sd_version, torch_dtype=torch.float16).to(self.device)
            feature_shape = (2816, 8, 8)  # hardcode to match IF classifier training
            print("Using IF feature shape:", feature_shape)
            self.sd_pipeline.safety_checker = None
        elif self.sd_version == "stabilityai/stable-diffusion-xl-base-1.0":
            self.deepfloyd_if = False
            self.using_sdxl = True
            print("Loading SDXL model:", self.sd_version)
            self.sd_pipeline = StableDiffusionXLPipeline.from_pretrained(self.sd_version, torch_dtype=torch.float16).to(self.device)
            self.sd_pipeline.unet.forward = types.MethodType(custom_unet_forward, self.sd_pipeline.unet)
        else:
            self.deepfloyd_if = False
            print("Loading SD model:", self.sd_version) 
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(self.sd_version).to(self.device)
            self.sd_pipeline.unet.forward = types.MethodType(custom_unet_forward, self.sd_pipeline.unet)
        if cfg["model"].get("use_ddim_scheduler", False):
            print("Using DDIM scheduler")
            self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        if cfg["use_finetuned_text_encoder"]:
            print("Loading finetuned text encoder LoRA from:", cfg["model"]["load_text_encoder_lora_from"])
            load_text_encoder_lora_weights(
                self.sd_pipeline.text_encoder,
                cfg["model"]["load_text_encoder_lora_from"],
                device=self.device,
                dtype=torch.float32,
                rank=50,
                patch_mlp=True,
            )
        if cfg["use_finetuned_text_encoder_race"]:
            print("Loading finetuned text encoder LoRA from:", cfg["model"]["load_text_encoder_lora_race_from"])
            load_text_encoder_lora_weights(
                self.sd_pipeline.text_encoder,
                cfg["model"]["load_text_encoder_lora_race_from"],
                device=self.device,
                dtype=torch.float32,
                rank=50,
                patch_mlp=True,
            )
        if cfg["use_finetuned_text_encoder_gender_race"]:
            print("Loading finetuned text encoder LoRA from:", cfg["model"]["load_text_encoder_lora_gender_race_from"])
            load_text_encoder_lora_weights(
                self.sd_pipeline.text_encoder,
                cfg["model"]["load_text_encoder_lora_gender_race_from"],
                device=self.device,
                dtype=torch.float32,
                rank=50,
                patch_mlp=True,
            )
        if cfg["use_finetuned_text_encoder_multiconcepts"]:
            print("Loading finetuned text encoder LoRA from:", cfg["model"]["load_text_encoder_lora_multiconcepts_from"])
            load_text_encoder_lora_weights(
                self.sd_pipeline.text_encoder,
                cfg["model"]["load_text_encoder_lora_multiconcepts_from"],
                device=self.device,
                dtype=torch.float32,
                rank=50,
                patch_mlp=True,
            )
        self.sd_pipeline.set_progress_bar_config(disable=True)

        # self.sd_pipeline.enable_attention_slicing()
        self.sd_batch_size = cfg["model"].get("sd_batch_size", 1)
        self.if_chunk_size = cfg["model"].get("if_chunk_size", 25)
        self.if_latent_chunk_size = cfg["model"].get("if_latent_chunk_size", 2)

        self.height = cfg["model"].get("height", 512)
        self.width  = cfg["model"].get("width", 512)

        sampling = cfg["model"]["sampling"]
        self.num_inference_steps = sampling["steps"]
        self.guidance_scale = sampling["scale"]
        self.gen_prompt_only = cfg["model"].get("gen_prompt_only", False)

        self.latents_cache_dir = "latents_cache"
        os.makedirs(self.latents_cache_dir, exist_ok=True)


        # --- Attribute classifier ---
        self.target_attribute = int(cfg["attributes"][cfg["bias_attribute"]]["target"])
        
        if self.deepfloyd_if:
            self.classifier = AttributeClassifierIF(
                cfg["attributes"][cfg["bias_attribute"]]["classifier_path"],
                cfg["attributes"][cfg["bias_attribute"]]["num_classes"],
                self.device,
                num_timesteps=48,
                feature_shape=feature_shape
            )
        else:   
            if cfg["model"]["sd_version"] == "stabilityai/stable-diffusion-2-1":
                input_size = 1*1280*12*12
            elif cfg["model"]["sd_version"] == "stabilityai/stable-diffusion-xl-base-1.0":
                input_size = 1280*32*32 # 1310720
            else:
                input_size = 1*1280*8*8
            if cfg["use_clf_sae_bias"]:
                self.classifier = AttributeClassifier(cfg["attributes"][cfg["bias_attribute"]]["classifier_path"], cfg["attributes"][cfg["bias_attribute"]]["num_classes"], 
                                                    self.device,
                                                    num_timesteps=48,
                                                    input_size=input_size)
            else:
                self.classifier = AttributeClassifier(cfg["attributes"][cfg["bias_attribute"]]["classifier_alt_path"], cfg["attributes"][cfg["bias_attribute"]]["num_classes"], 
                                                    self.device,
                                                    num_timesteps=50)
                
        self.maximize_attribute = cfg.get("maximize_attribute", False)

        if cfg["bias_attribute2"] is not None:
            self.target_attribute2 = int(cfg["attributes"][cfg["bias_attribute2"]]["target"])
            if self.deepfloyd_if:
                self.classifier2 = AttributeClassifierIF(
                    cfg["attributes"][cfg["bias_attribute2"]]["classifier_path"],
                    cfg["attributes"][cfg["bias_attribute2"]]["num_classes"],
                    self.device,
                    num_timesteps=48,
                    feature_shape=feature_shape
                )
            else:
                self.classifier2 = AttributeClassifier(cfg["attributes"][cfg["bias_attribute2"]]["classifier_path"], cfg["attributes"][cfg["bias_attribute2"]]["num_classes"], 
                                                  self.device,
                                                  num_timesteps=48)
        else:
            self.classifier2 = None
        # step selection
        self.timestep_index_override = cfg["attributes"][cfg["bias_attribute"]].get("timestep_index", None)

        # misc
        self.seed = cfg.get("seed", 0)
        self.initial_prompt_str = cfg.get("initial_prompt", "a photo of a person")

        # internal state for beam finalization
        self.reset()
        self.freeze()

    # --------------- small utilities --------------- #
    def reset(self):
        self.candidate = [[]]
        self.candidate_score = [[]]
        self.done_cnt = 0


    def freeze(self):
        self.llm.eval()
        self.llm.requires_grad_(False)
        for p in self.llm.parameters():
            p.requires_grad_(False)

    def is_done(self):
        return 1 if self.done_cnt > self.beam_size else 0

    def length_penalty(self, length, alpha=1.2, min_length=5):
        return ((min_length + length) / (min_length + 1)) ** alpha

    def eos_check(self, indices: torch.Tensor, scores: torch.Tensor):
        mask = torch.eq(indices[:, -1], self.llm_tokenizer.eos_token_id)
        mask |= torch.eq(indices[:, -1], 13)  # common extra newline token id in some LLMs

        if mask.sum() > 0 and indices.shape[1] > self.min_length:
            for i, (s, m, ids) in enumerate(zip(scores, mask.reshape(scores.shape), indices.reshape(scores.shape[0],scores.shape[1],-1))):
                ss = s[m].tolist()
                if len(self.candidate_score[i]) > 0:
                    self.candidate_score[i] += ss
                    self.candidate[i] += [id for id in ids[m]]
                else:
                    self.candidate_score[i] = ss
                    self.candidate[i] = [id for id in ids[m]]
            self.done_cnt += mask.reshape(scores.shape).sum()
            scores[mask.reshape(scores.shape)] = -torch.inf
        return scores

    # --------------- SD helpers --------------- #
    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        tok = self.sd_pipeline.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.sd_pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        enc = self.sd_pipeline.text_encoder(tok.input_ids)[0]  # (B, seq, dim)
        return enc

    @torch.no_grad()
    def _encode_prompt_tensor(
        self,
        prompt,
        *,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
    ) -> torch.Tensor:
        """
        Compatibility wrapper that mirrors the legacy `_encode_prompt` tensor output.
        Works across diffusers versions by switching to `encode_prompt` when available.
        """
        pipe = self.sd_pipeline
        encode_fn = getattr(pipe, "encode_prompt", None)
        if encode_fn is None:
            return pipe._encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )

        prompt_embeds = encode_fn(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        if isinstance(prompt_embeds, tuple):
            cond_embeds, negative_embeds = prompt_embeds
            if do_classifier_free_guidance:
                if negative_embeds is None:
                    raise ValueError(
                        "diffusers returned `None` for negative prompt embeds while CFG was requested."
                    )
                return torch.cat([negative_embeds, cond_embeds])
            return cond_embeds

        # diffusers < 0.30 returns the concatenated tensor already
        return prompt_embeds

    def _safe_batch_decode(self, token_ids, skip_special_tokens: bool = True) -> List[str]:
        """
        Robust decoding helper that filters out invalid / None tokens that can
        appear with some chat tokenizers (e.g., Qwen) before string conversion.
        """
        if isinstance(token_ids, torch.Tensor):
            ids_list = token_ids.detach().cpu().tolist()
        else:
            ids_list = []
            for seq in token_ids:
                if isinstance(seq, torch.Tensor):
                    ids_list.append(seq.detach().cpu().tolist())
                else:
                    ids_list.append(list(seq))
        decoded_texts = []
        for seq in ids_list:
            tokens = self.llm_tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=skip_special_tokens)
            tokens = [tok for tok in tokens if tok is not None]
            if not tokens:
                decoded_texts.append("")
                continue
            try:
                decoded_texts.append(self.llm_tokenizer.convert_tokens_to_string(tokens))
            except TypeError:
                decoded_texts.append("".join(tok for tok in tokens if isinstance(tok, str)))
        return decoded_texts

    def _batch_decode(self, token_ids, skip_special_tokens: bool = True) -> List[str]:
        try:
            return self.llm_tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        except TypeError:
            return self._safe_batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def _prepare_latents(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        if not getattr(self, "using_sdxl", False):
            latents = torch.randn(
                (batch_size, self.sd_pipeline.unet.in_channels, self.height // 8, self.width // 8),
                generator=generator,
                device=self.device,
                dtype=self.sd_pipeline.unet.dtype,
            )
            latents = latents * self.sd_pipeline.scheduler.init_noise_sigma
            return latents
        
        pipe = self.sd_pipeline
        latents = pipe.prepare_latents(
            batch_size,
            pipe.unet.config.in_channels,
            self.height,
            self.width,
            pipe.unet.dtype,
            self.device,
            generator,
        )
        return latents

    def _sdxl_conditioning(
        self,
        prompt,
        *,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
        original_size=None,
        target_size=None,
        crops_coords_top_left=(0, 0),
    ):
        if not getattr(self, "using_sdxl", False):
            raise RuntimeError("_sdxl_conditioning should only be used with SDXL models.")

        pipe = self.sd_pipeline
        original_size = original_size or (self.height, self.width)
        target_size = target_size or (self.height, self.width)
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        add_text_embeds = pooled_prompt_embeds
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if do_classifier_free_guidance:
            negative_add_time_ids = add_time_ids
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_text_embeds = add_text_embeds.to(device=self.device)
        add_time_ids = add_time_ids.to(device=self.device)

        repeat_factor = prompt_embeds.shape[0] // add_time_ids.shape[0]
        add_time_ids = add_time_ids.repeat(repeat_factor, 1)

        return prompt_embeds, add_text_embeds, add_time_ids

    def _sdxl_timestep_cond(self, batch_size: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
        pipe = self.sd_pipeline
        if pipe.unet.config.time_cond_proj_dim is None:
            return None

        guidance_scale_tensor = torch.tensor(
            self.guidance_scale - 1, device=self.device, dtype=torch.float32
        ).repeat(batch_size)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
            dtype=dtype,
        )
        return timestep_cond.to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def _half_run_get_latents(self, sd_batch_size=1, cache_latents=False) -> (torch.Tensor, int, torch.Tensor):
        """
        Run half of the scheduler steps with a neutral initial prompt to obtain
        a latent state and return the *next* timestep index and the scaled latent input.
        """
        if cache_latents:
            latents_path = os.path.join(self.latents_cache_dir, f"latents_{self.seed}_{sd_batch_size}.pt")
            if os.path.exists(latents_path):
                print(f"Loading cached latents from {latents_path}")
                cached = torch.load(latents_path)
                return cached["latents"], cached["timestep_index"], cached["scheduler_t"]
        pipe = self.sd_pipeline
        pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = pipe.scheduler.timesteps  # len = steps

        t_half_idx = self.timestep_index_override
        if t_half_idx is None:
            t_half_idx = len(timesteps) // 2  # half way

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        latents = self._prepare_latents(batch_size=sd_batch_size, generator=generator)

        if not getattr(self, "using_sdxl", False):
            text_embeds = self._encode_prompt_tensor(
                prompt=self.initial_prompt_str,
                num_images_per_prompt=sd_batch_size,
                do_classifier_free_guidance=True,
            )
            for _, t in enumerate(timesteps[:t_half_idx]):
                latent_model_input = torch.cat([latents] * 2)
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        else:
            prompt_embeds, add_text_embeds, add_time_ids = self._sdxl_conditioning(
                prompt=self.initial_prompt_str,
                num_images_per_prompt=sd_batch_size,
                do_classifier_free_guidance=True,
            )
            timestep_cond = self._sdxl_timestep_cond(sd_batch_size, latents.dtype)

            for _, t in enumerate(timesteps[:t_half_idx]):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs={
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # for scoring we use the very next step's scaled latent input
        next_t = timesteps[t_half_idx]
        if cache_latents:
            torch.save({
                "latents": latents,
                "timestep_index": t_half_idx,
                "scheduler_t": next_t,
            }, latents_path)
            print(f"Saved cached latents to {latents_path}")
        return latents, t_half_idx, next_t

    @torch.no_grad()
    def _classifier_score_batch(self, classifier, 
                                target_attribute,
                                candidate_texts: List[str],
                                latent_model_input: torch.Tensor,
                                timestep_index: int,
                                scheduler_t,
                                maximize_attribute=False) -> torch.Tensor:
        """
        For a list of candidate prompts, compute -(cross_entropy) against target attribute.
        Returns a tensor of shape (B*batch_size,) with *higher is better*.
        """

      
        if not getattr(self, "using_sdxl", False):
            text_embeds = self._encode_prompt_tensor(
                prompt=candidate_texts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            text_embeddings = text_embeds
            added_cond_kwargs = None
            timestep_cond = None
        else:
            prompt_embeds, add_text_embeds, add_time_ids = self._sdxl_conditioning(
                prompt=candidate_texts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            text_embeddings = prompt_embeds
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            timestep_cond = self._sdxl_timestep_cond(text_embeddings.shape[0], latent_model_input.dtype)

        # Duplicate latent to B
        B = text_embeddings.shape[0]
        # expand to B*latent_model_input.shape[0]
        latents_b = latent_model_input.repeat(B, 1, 1, 1)
        model_input = latents_b
        if getattr(self, "using_sdxl", False):
            model_input = self.sd_pipeline.scheduler.scale_model_input(latents_b, scheduler_t)

        # Single UNet forward at the chosen timestep, requesting mid-level features.
        unet_out = self.sd_pipeline.unet(
            model_input,
            scheduler_t,
            encoder_hidden_states=text_embeddings,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            return_h=True,
        )

        # Handle different return conventions for return_h
        if isinstance(unet_out, tuple):
            # assume (noise_pred, h_sample)
            _, h_sample = unet_out
        elif hasattr(unet_out, "h"):
            h_sample = unet_out.h
        else:
            # assume direct h is returned
            h_sample = unet_out

        logits = classifier.model(h_sample.float(), timestep_index=timestep_index)  # (B*bs, num_classes)
        if self.cfg["use_clf_sae_bias"]:

            temperature = 1.0  # FIXED: was 1000 (disabling classifier)
            logits = logits / temperature
        log_probs = F.log_softmax(logits, dim=-1)  # (B, num_classes)
        if maximize_attribute:
            # Take the maximum log probability across all classes
            log_prob_target, _ = log_probs.max(dim=-1)  # (B*bs,)
        else:
            # Original behavior: use specific target
            target = torch.full((logits.size(0),), target_attribute, device=self.device, dtype=torch.long)
            log_prob_target = log_probs[torch.arange(log_probs.size(0)), target]  # (B*bs,)
        log_prob_target = log_prob_target.view(latent_model_input.shape[0], -1)
       
        return log_prob_target  # (bs,B)

    # --------------- main: classifier-guided decoding --------------- #
    @torch.no_grad()
    def generate_prompt(self,llm_only=False,sampling_generator=None) -> Dict:
        """
        LLM beam search guided by classifier score on SD mid-block activations.
        """
        # SD half-run to set up latent & timestep
        if not llm_only:
            latent_model_input, t_idx, scheduler_t = self._half_run_get_latents(sd_batch_size=self.sd_batch_size, cache_latents=False)
       
        # @torch.no_grad()
        # def complete_from_latents_and_save(
        #     self,
        #     half_scaled_latent_input: torch.Tensor,  # from _half_run_get_latents (kept for debugging)
        #     start_idx: int,
        #     start_t: torch.Tensor,                   # from _half_run_get_latents
        #     latents: torch.Tensor,                   # <-- unscaled latents returned in step (1)
        #     out_path: str = "sd_check.png",
        #     prompt: str | None = None
        # ):
        #     """
        #     Continues denoising from the halfway latent state and saves the final image.
        #     If `prompt` is None, it keeps using self.initial_prompt_str.
        #     """
        #     pipe = self.sd_pipeline
        #     device = self.device
        #     timesteps = pipe.scheduler.timesteps

        #     # use the same prompt (or let you try a different one mid-run)
        #     text_embeds = self._encode_text([prompt or self.initial_prompt_str])

        #     # continue the diffusion steps from start_idx .. end
        #     for t in timesteps[start_idx:]:
        #         model_in = pipe.scheduler.scale_model_input(latents, t)
        #         noise_pred = pipe.unet(model_in, t, encoder_hidden_states=text_embeds).sample
        #         latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        #     # decode with VAE and save
        #     # diffusers convention: divide by VAE scaling factor before decode
        #     scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)  # SDXL is typically 0.13025
        #     latents = latents / scaling_factor

        #     image = pipe.vae.decode(latents).sample  # (B, 3, H, W), in [-1, 1]
        #     image = (image / 2 + 0.5).clamp(0, 1)    # to [0, 1]
        #     image = image[0].detach().cpu().permute(1, 2, 0).numpy()
        #     Image.fromarray((image * 255).astype(np.uint8)).save(out_path)
        #     return out_path
        # sanity_image_path = complete_from_latents_and_save(
        #     self,
        #     half_scaled_latent_input=latent_model_input,
        #     start_idx=t_idx,
        #     start_t=scheduler_t,
        #     latents=latents,
        #     out_path="sd_check_initial.png",
        #     prompt=self.initial_prompt_str
        # )
        # print(f"Sanity check image (initial prompt) saved to {sanity_image_path}")
        input_ids = self.llm_prompt.to(self.device)
        llm_input_len = input_ids.shape[-1]
        llm_vocab_size = self.llm.vocab_size
        cummulative_scores = torch.zeros(1, device=self.device)

        # Initialize past_key_values storage
        past_key_values = None

        for curr_len in range(1, self.max_length + 1):
            # ---- LLM step with caching
            llm_outputs = self.llm(input_ids, use_cache=False, return_dict=True)
            # print("-------------USING CACHE----------------")
            # if past_key_values is None:
            #     # First iteration: process full prompt
            #     llm_outputs = self.llm(
            #         input_ids,
            #         use_cache=True,  # Enable caching
            #         return_dict=True
            #     )
            # else:
            #     # Subsequent iterations: only process new tokens
            #     llm_outputs = self.llm(
            #         input_ids[:, -1:],  # Only the last token(s)
            #         past_key_values=past_key_values,
            #         use_cache=True,
            #         return_dict=True
            #     )
            
            # # Store past_key_values for next iteration
            # past_key_values = llm_outputs.past_key_values

            llm_logits = llm_outputs.logits[:, -1, :]             
            llm_scores = nn.functional.log_softmax(llm_logits, dim=-1)
            # llm_scores = nn.functional.log_softmax(llm_outputs.logits[:, -1, :], dim=-1)
            llm_scores += cummulative_scores.view(input_ids.shape[0], -1)

            # NEW CODE - Corrected temperature-based sampling:
            larger_k = min(self.llm_beam_size * self.candidate_pool_multiplier, llm_scores.view(1, -1).numel())
            if curr_len == 1 and self.cfg["model"].get("first_token_full_dist", False):
                # --- FIRST TOKEN: sample from the full [B, V] distribution ---
                temp_scores = llm_scores / self.sampling_temperature
                probs = torch.softmax(temp_scores, dim=-1)  # [B, V]
                sampled_token_idx = torch.multinomial(
                    probs, self.llm_beam_size, replacement=False, generator=sampling_generator
                )  # [B, beam]

                # accumulate log-scores for chosen tokens
                cummulative_scores = llm_scores.gather(-1, sampled_token_idx)  # [B, beam]

                # build flattened "global" indices consistent with later // and % logic
                beam_offsets = (torch.arange(input_ids.shape[0], device=sampled_token_idx.device)
                                .unsqueeze(-1) * llm_vocab_size)  # [B, 1]
                llm_topk_indices_raw = (beam_offsets + sampled_token_idx).to(torch.long)  # [B, beam]

            else:
                # --- Subsequent tokens: use the flattened top-k pool ---
                llm_topk_large = torch.topk(llm_scores.view(1, -1), dim=-1, k=larger_k)
                temp_scores = llm_topk_large.values / self.sampling_temperature
                probs = torch.softmax(temp_scores, dim=-1)  # [1, K]
                sampled_indices = torch.multinomial(
                    probs, self.llm_beam_size, replacement=False, generator=sampling_generator
                )  # [1, beam]

                cummulative_scores = llm_topk_large.values.gather(-1, sampled_indices)  # [1, beam]
                llm_topk_indices_raw = llm_topk_large.indices.gather(-1, sampled_indices).to(torch.long)  # [1, beam]


            # Extract the final indices and beam IDs (same as before)
            llm_topk_indices = llm_topk_indices_raw % llm_vocab_size
            llm_topk_beam_id = llm_topk_indices_raw // llm_vocab_size

            # llm_topk_indices = llm_topk.indices % llm_vocab_size
            # cummulative_scores = llm_topk.values
            # llm_topk_beam_id = llm_topk.indices // llm_vocab_size
            llm_topk_batch_id = (llm_topk_beam_id + self.beam_offset.unsqueeze(-1)).view(-1)

            if input_ids.shape[0] == 1:
                input_ids = input_ids.unsqueeze(1).expand(-1, self.beam_size, -1).reshape(self.beam_size, -1)

            # build new sequences
            llm_beam_ids = torch.cat(
                (input_ids[llm_topk_batch_id].view(1, self.llm_beam_size, -1), llm_topk_indices.unsqueeze(-1)),
                dim=-1
            )
            llm_beam_ids = llm_beam_ids.view(-1, llm_beam_ids.shape[-1])
            # # Handle beam expansion changes - reorder past_key_values
            # if llm_beam_ids.shape[0] != input_ids.shape[0]:
            #     past_key_values = self._reorder_cache(past_key_values, llm_topk_batch_id)

            # decode candidate texts for this step
            llm_gen_texts = self._batch_decode(
                llm_beam_ids[:, llm_input_len:],
                skip_special_tokens=True,
            )

            if not llm_only and not (self.cfg["skip_first_steps"] and curr_len > 1):
                # llm_gen_texts = [self.model_prompt_primer + t for t in llm_gen_texts]
                if getattr(self, 'use_occupation_prompt', False):
                    # Use occupation template and insert candidates into {occupation} slot
                    model_occupation_template = getattr(self, 'model_occupation_template', 'A photo of the face of a {occupation}, a person')
                    llm_gen_texts = [model_occupation_template.format(occupation=t.strip()) for t in llm_gen_texts]
                else:
                    # Use original model_prompt_primer method
                    llm_gen_texts = [self.model_prompt_primer + t for t in llm_gen_texts]
                all_clf_log_probs = []
                if self.classifier2 is not None:
                    all_clf_log_probs2 = []
                print("latent_model_input shape:", latent_model_input.shape)
                # clf_scores = self._classifier_score_batch(
                #         classifier=self.classifier,
                #         target_attribute=self.target_attribute,
                #         candidate_texts=llm_gen_texts,
                #         latent_model_input=latent_model_input,
                #         timestep_index=t_idx,
                #         scheduler_t=scheduler_t
                #     )  # (K,)
                for i in range(latent_model_input.shape[0]):
                    
                    latent = latent_model_input[i].unsqueeze(0)
                    # print("latent shape:", latent.shape)
                    # ---- Classifier score for all candidates in batch
                    clf_log_probs = self._classifier_score_batch(
                        classifier=self.classifier,
                        target_attribute=self.target_attribute,
                        candidate_texts=llm_gen_texts,
                        latent_model_input=latent,
                        timestep_index=t_idx,
                        scheduler_t=scheduler_t,
                        maximize_attribute=self.maximize_attribute
                    )  # (K,)
                    all_clf_log_probs.append(clf_log_probs)
                    if self.classifier2 is not None:
                        clf_log_probs2 = self._classifier_score_batch(
                            classifier=self.classifier2,
                            target_attribute=self.target_attribute2,
                            candidate_texts=llm_gen_texts,
                            latent_model_input=latent,
                            timestep_index=t_idx,
                            scheduler_t=scheduler_t,
                            maximize_attribute=self.maximize_attribute
                        )  # (K,)
                        all_clf_log_probs2.append(clf_log_probs2)

                # clf_probs = torch.stack(all_clf_probs, dim=0)
                # clf_probs = torch.mean(clf_probs, dim=0)  # (K,)
                # clf_scores = torch.log(clf_probs + 1e-10)  # avoid log(0)
                clf_log_probs = torch.stack(all_clf_log_probs, dim=0)
                clf_scores = torch.logsumexp(clf_log_probs, dim=0) - torch.log(torch.tensor(clf_log_probs.shape[0], device=clf_log_probs.device, dtype=clf_log_probs.dtype))  # log-mean-exp trick
                if self.classifier2 is not None:
                    clf_log_probs2 = torch.stack(all_clf_log_probs2, dim=0)
                    clf_scores2 = torch.logsumexp(clf_log_probs2, dim=0) - torch.log(torch.tensor(clf_log_probs2.shape[0], device=clf_log_probs2.device, dtype=clf_log_probs2.dtype))  # log-mean-exp trick
                    total_score = self.llm_alpha * cummulative_scores + self.clf_alpha * clf_scores.view(1, -1) + self.clf2_alpha * clf_scores2.view(1,-1)
                else:
                    #combine scores
                    total_score = self.llm_alpha * cummulative_scores + self.clf_alpha * clf_scores.view(1, -1)
            else:
                clf_scores = torch.zeros_like(cummulative_scores)
                print("Skipping classifier scoring at step", curr_len)
                total_score = self.llm_alpha * cummulative_scores

            # optional hard length control
            if self.length_cutoff:
                # approximate token length by tokenizer attention mask on the fly if desired
                pass

            total_score = self.eos_check(llm_beam_ids[:, llm_input_len:], total_score)
            beam_index = (total_score.topk(self.beam_size, dim=-1).indices + self.beam_expand_factor * self.beam_offset.unsqueeze(-1)).view(-1)
            cummulative_scores = cummulative_scores.reshape(1 * self.llm_beam_size, -1)
            cummulative_scores = cummulative_scores[beam_index].reshape(1 * self.beam_size, -1)
            input_ids = llm_beam_ids[beam_index].reshape(1 * self.beam_size, -1)

            # # Reorder past_key_values for selected beams
            # past_key_values = self._reorder_cache(past_key_values, beam_index)

            output_ids = input_ids.reshape(1* self.beam_size, -1)[ :, llm_input_len:]
            output_text = self._batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            print(output_text)

            # preview (optional)
            # print(self._batch_decode(input_ids[:, llm_input_len:], skip_special_tokens=True)[:5])

            if self.is_done():
                break

        # finalize
        output_score = total_score[:, 0]
        output_ids = input_ids.reshape(1, self.beam_size, -1)[:, 0, llm_input_len:]

        # include best-last decoded candidate
        for b_i in range(1):
            self.candidate_score[b_i].append(output_score[b_i].item())
            self.candidate[b_i].append(output_ids[b_i])
            self.candidate_score[b_i] = [
                s / self.length_penalty(self.candidate[b_i][s_i].shape[-1])
                for s_i, s in enumerate(self.candidate_score[b_i])
            ]
            b_i_best = self.candidate_score[b_i].index(max(self.candidate_score[b_i]))
            self.candidate[b_i] = self.candidate[b_i][b_i_best]

        output_text = self._batch_decode(self.candidate, skip_special_tokens=True)
        input_prompt = self._batch_decode(input_ids, skip_special_tokens=False)
        llm_prompt_text = self.llm_tokenizer.decode(self.llm_prompt[0], skip_special_tokens=False)
        final_candidates = self._batch_decode(
            llm_beam_ids[beam_index, llm_input_len:],
            skip_special_tokens=True,
        )

        if (self.cfg.get("oops", False) and not llm_only):
            if getattr(self, 'use_occupation_prompt', False):
                # Use occupation template for final output
                model_occupation_template = getattr(self, 'model_occupation_template', 'A photo of the face of a {occupation}, a person')
                output_text = [model_occupation_template.format(occupation=t.strip()) for t in output_text]
            else:
                # Use original model_prompt_primer method
                output_text = [self.model_prompt_primer + t for t in output_text]

        #     all_clf_scores = []
        #     for i in range(latent_model_input.shape[0]):
                
        #         latent = latent_model_input[i].unsqueeze(0)
        #         print("latent shape:", latent.shape)
        #         # ---- Classifier score for all candidates in batch
        #         clf_scores = self._classifier_score_batch(
        #             classifier = self.classifier,
        #             target_attribute=self.target_attribute,
        #             candidate_texts=[output_text[0]],
        #             latent_model_input=latent,
        #             timestep_index=t_idx,
        #             scheduler_t=scheduler_t
        #         )  # (K,)
        #         all_clf_scores.append(clf_scores)
        #     clf_scores = torch.stack(all_clf_scores, dim=0)
        #     clf_scores = torch.mean(clf_scores, dim=0)  # (K,)
        #     final_score = clf_scores[0].item()
        # else:
        final_score = 0.0

        return {"text": output_text[0], 
                "input_prompt": llm_prompt_text, 
                "alt_prompts": final_candidates, 
                "bias_score": final_score,
                }

    @torch.no_grad()
    def _half_run_get_latents_if(self, sd_batch_size=1, cache_latents=False):
        """
        Precompute IF latents up to the scoring timestep using the same default prompt
        that _half_run_get_latents uses (self.initial_prompt_str). Returns:
        latents:           latent tensor at step t_idx (unscaled, ready for next step)
        t_idx:             integer index into scheduler.timesteps
        scheduler_t:       the scheduler timestep tensor/value for that index
        """
        pipe = self.sd_pipeline
        device = self.device

        # one-time: ensure UNet uses the custom forward that supports return_h
        if getattr(pipe.unet, "_bgps_patched_forward", False) is not True:
            pipe.unet.forward = types.MethodType(custom_unet_forward, pipe.unet)
            pipe.unet._bgps_patched_forward = True

        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps  # len = steps

        # choose the same scoring index policy as SD path
        t_idx = self.timestep_index_override
        if t_idx is None:
            t_idx = len(timesteps) // 2
        t_idx = int(t_idx)

        # caching (separate key for IF)
        if cache_latents:
            latents_path = os.path.join(
                self.latents_cache_dir, f"latents_if_{self.seed}_{sd_batch_size}.pt"
            )
            if os.path.exists(latents_path):
                cached = torch.load(latents_path, map_location=device)
                return cached["latents"], cached["timestep_index"], cached["scheduler_t"]

        # init latents
        generator = torch.Generator(device=device).manual_seed(self.seed)
        latents = torch.randn(
            (sd_batch_size, pipe.unet.in_channels, self.height , self.width ),
            generator=generator,
            device=device,
            dtype=pipe.unet.dtype,
        )
        if hasattr(pipe.scheduler, "init_noise_sigma"):
            latents = latents * pipe.scheduler.init_noise_sigma

        # text embeds for neutral prompt, with CFG
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=self.initial_prompt_str,
            device=device,
            num_images_per_prompt=sd_batch_size,
            do_classifier_free_guidance=True,
        )
        do_cfg = True

        latent_chunk_size = getattr(self, "if_latent_chunk_size", 1)
        latent_chunk_size = max(1, int(latent_chunk_size))
        # advance up to (but not including) t_idx
        for i, t in enumerate(timesteps[:t_idx]):
            for start in range(0, sd_batch_size, latent_chunk_size):
                end = min(start + latent_chunk_size, sd_batch_size)
                chunk_latents = latents[start:end]

                if do_cfg:
                    chunk_prompt = prompt_embeds[start:end]
                    chunk_negative = negative_prompt_embeds[start:end]
                    model_in = torch.cat([chunk_latents, chunk_latents], dim=0)
                    chunk_text = torch.cat([chunk_negative, chunk_prompt], dim=0)
                else:
                    model_in = chunk_latents
                    chunk_text = prompt_embeds[start:end]

                noise_pred = pipe.unet(
                    model_in,
                    t,
                    encoder_hidden_states=chunk_text,
                ).sample

                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                chunk_latents = pipe.scheduler.step(noise_pred, t, chunk_latents).prev_sample
                latents[start:end] = chunk_latents
        next_t = timesteps[t_idx]
        if cache_latents:
            torch.save(
                {"latents": latents, "timestep_index": t_idx, "scheduler_t": next_t},
                latents_path,
            )
        return latents, t_idx, next_t


    @torch.no_grad()
    def _classifier_score_batch_if(
        self,
        classifier,
        target_attribute,
        candidate_texts: List[str],
        latent_model_input: torch.Tensor,   # shape: [bs, C, H, W] for the next step
        timestep_index: int,
        scheduler_t,
        maximize_attribute: bool = False,
    ) -> torch.Tensor:
        """
        IF-fast scoring: single UNet forward (with return_h) per candidate at the chosen timestep.
        Uses precomputed latents and candidate-specific text embeddings.
        Returns log-prob scores of shape (bs, K) where K=len(candidate_texts).
        """
        pipe = self.sd_pipeline
        device = self.device

        # ensure custom forward (return_h) is active
        if getattr(pipe.unet, "_bgps_patched_forward", False) is not True:
            pipe.unet.forward = types.MethodType(custom_unet_forward, pipe.unet)
            pipe.unet._bgps_patched_forward = True

        # encode candidate prompts (conditional only; no CFG duplication here)
        try:
            text_embeddings = self._encode_prompt_tensor(
                prompt=candidate_texts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        except Exception:
            tok = pipe.tokenizer(
                candidate_texts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_embeddings = pipe.text_encoder(tok.input_ids)[0]

        K = len(candidate_texts)
        bs = latent_model_input.shape[0]
        target_batch = K * bs

        if text_embeddings.shape[0] != target_batch:
            if text_embeddings.shape[0] == K:
                text_embeddings = (
                    text_embeddings.unsqueeze(0)
                    .expand(bs, -1, *text_embeddings.shape[1:])
                    .reshape(target_batch, *text_embeddings.shape[1:])
                )
            else:
                repeat_factor = target_batch // text_embeddings.shape[0]
                if repeat_factor * text_embeddings.shape[0] != target_batch:
                    raise ValueError(
                        f"Unexpected text embedding shape {text_embeddings.shape}; cannot match latents batch {target_batch}"
                    )
                text_embeddings = text_embeddings.repeat_interleave(repeat_factor, dim=0)

        latents_b = (
            latent_model_input.unsqueeze(1)
            .expand(bs, K, *latent_model_input.shape[1:])
            .reshape(target_batch, *latent_model_input.shape[1:])
        )

        chunk_size = max(1, int(getattr(self, "if_chunk_size", 25)))
        log_prob_target = torch.empty(target_batch, device=device, dtype=torch.float32)
        num_steps = classifier.model.num_timesteps
        reversed_t = classifier.model.reversed_timesteps[
            classifier.model.forward_timesteps.index(timestep_index)
        ]

        for start in range(0, target_batch, chunk_size):
            end = min(start + chunk_size, target_batch)
            unet_out = pipe.unet(
                latents_b[start:end],
                scheduler_t,
                encoder_hidden_states=text_embeddings[start:end],
                return_h=True,
            )
            if isinstance(unet_out, tuple):
                h = unet_out[1]
            elif hasattr(unet_out, "h"):
                h = unet_out.h
            else:
                h = unet_out

            B_chunk = h.shape[0]
            C, H, W = h.shape[-3:]
            feats = torch.zeros(
                (1, num_steps, B_chunk, C, H, W), device=h.device, dtype=h.dtype
            )
            feats[0, reversed_t - 1, :, :, :, :] = h
            feats = feats.to(torch.float32)

            logits = classifier.model(feats, t=timestep_index)
            temperature = 1.0  # FIXED: was 1000 (disabling classifier)
            logits = logits / temperature
            log_probs = F.log_softmax(logits.to(device), dim=-1)

            if maximize_attribute:
                log_prob_target_chunk, _ = log_probs.max(dim=-1)
            else:
                target = torch.full(
                    (log_probs.size(0),),
                    target_attribute,
                    device=device,
                    dtype=torch.long,
                )
                log_prob_target_chunk = log_probs[
                    torch.arange(log_probs.size(0), device=device), target
                ]
            log_prob_target[start:end] = log_prob_target_chunk

        # h: [K*bs, C, H, W]

        # reshape back to (bs, K)
        return log_prob_target.view(bs, K).to(device)


    @torch.no_grad()
    def generate_prompt_if(self, llm_only=False, sampling_generator=None) -> Dict:
        """
        LLM beam search guided by IF Stage-I mid-block classifier, but FAST:
        - Precompute IF latents up to scoring timestep (neutral prompt)
        - One UNet forward per candidate with return_h=True
        """
        # prepare IF latents & scoring timestep
        if not llm_only:
            latent_model_input, t_idx, scheduler_t = self._half_run_get_latents_if(
                sd_batch_size=self.sd_batch_size, cache_latents=False
            )

        pipe = self.sd_pipeline
        device = self.device

        # ensure custom forward is patched (safety)
        if getattr(pipe.unet, "_bgps_patched_forward", False) is not True:
            pipe.unet.forward = types.MethodType(custom_unet_forward, pipe.unet)
            pipe.unet._bgps_patched_forward = True

        input_ids = self.llm_prompt.to(device)
        llm_input_len = input_ids.shape[-1]
        llm_vocab_size = self.llm.vocab_size
        cummulative_scores = torch.zeros(1, device=device)

        for curr_len in range(1, self.max_length + 1):
            llm_outputs = self.llm(input_ids, use_cache=False, return_dict=True)
            llm_logits = llm_outputs.logits[:, -1, :]
            llm_scores = F.log_softmax(llm_logits, dim=-1)
            llm_scores += cummulative_scores.view(input_ids.shape[0], -1)

            larger_k = min(
                self.llm_beam_size * self.candidate_pool_multiplier,
                llm_scores.view(1, -1).numel(),
            )
            if curr_len == 1 and self.cfg["model"].get("first_token_full_dist", False):
                temp_scores = llm_scores / self.sampling_temperature
                probs = torch.softmax(temp_scores, dim=-1)
                sampled_token_idx = torch.multinomial(
                    probs, self.llm_beam_size, replacement=False, generator=sampling_generator
                )
                cummulative_scores = llm_scores.gather(-1, sampled_token_idx)
                beam_offsets = (
                    torch.arange(input_ids.shape[0], device=sampled_token_idx.device).unsqueeze(-1)
                    * llm_vocab_size
                )
                llm_topk_indices_raw = (beam_offsets + sampled_token_idx).to(torch.long)
            else:
                llm_topk_large = torch.topk(llm_scores.view(1, -1), dim=-1, k=larger_k)
                temp_scores = llm_topk_large.values / self.sampling_temperature
                probs = torch.softmax(temp_scores, dim=-1)
                sampled_indices = torch.multinomial(
                    probs, self.llm_beam_size, replacement=False, generator=sampling_generator
                )
                cummulative_scores = llm_topk_large.values.gather(-1, sampled_indices)
                llm_topk_indices_raw = llm_topk_large.indices.gather(-1, sampled_indices).to(torch.long)

            llm_topk_indices = llm_topk_indices_raw % llm_vocab_size
            llm_topk_beam_id = llm_topk_indices_raw // llm_vocab_size
            llm_topk_batch_id = (llm_topk_beam_id + self.beam_offset.unsqueeze(-1)).view(-1)

            if input_ids.shape[0] == 1:
                input_ids = (
                    input_ids.unsqueeze(1)
                    .expand(-1, self.beam_size, -1)
                    .reshape(self.beam_size, -1)
                )

            llm_beam_ids = torch.cat(
                (input_ids[llm_topk_batch_id].view(1, self.llm_beam_size, -1), llm_topk_indices.unsqueeze(-1)),
                dim=-1,
            )
            llm_beam_ids = llm_beam_ids.view(-1, llm_beam_ids.shape[-1])

            # decode candidate strings
            llm_gen_texts = self._batch_decode(
                llm_beam_ids[:, llm_input_len:], skip_special_tokens=True
            )

            # classifier scoring (fast IF path) unless llm_only
            if not llm_only and not (self.cfg["skip_first_steps"] and curr_len > 1):
                if getattr(self, "use_occupation_prompt", False):
                    tmpl = getattr(
                        self, "model_occupation_template", "A photo of the face of a {occupation}, a person"
                    )
                    llm_gen_texts = [tmpl.format(occupation=t.strip()) for t in llm_gen_texts]
                else:
                    llm_gen_texts = [self.model_prompt_primer + t for t in llm_gen_texts]

                all_clf_log_probs = []
                if self.classifier2 is not None:
                    all_clf_log_probs2 = []

                # average scores across precomputed latent batch (bs)
                for i in range(latent_model_input.shape[0]):
                    latent = latent_model_input[i].unsqueeze(0)  # [1, C, H, W]
                    clf_log_probs = self._classifier_score_batch_if(
                        classifier=self.classifier,
                        target_attribute=self.target_attribute,
                        candidate_texts=llm_gen_texts,
                        latent_model_input=latent,
                        timestep_index=t_idx,
                        scheduler_t=scheduler_t,
                        maximize_attribute=self.maximize_attribute,
                    )  # [1, K]
                    all_clf_log_probs.append(clf_log_probs)

                    if self.classifier2 is not None:
                        clf_log_probs2 = self._classifier_score_batch_if(
                            classifier=self.classifier2,
                            target_attribute=self.target_attribute2,
                            candidate_texts=llm_gen_texts,
                            latent_model_input=latent,
                            timestep_index=t_idx,
                            scheduler_t=scheduler_t,
                            maximize_attribute=self.maximize_attribute,
                        )  # [1, K]
                        all_clf_log_probs2.append(clf_log_probs2)

                # log-mean-exp across latent batch
                clf_log_probs = torch.stack(all_clf_log_probs, dim=0)  # [bs, 1, K]
                clf_scores = torch.logsumexp(clf_log_probs, dim=0) - torch.log(
                    torch.tensor(clf_log_probs.shape[0], device=clf_log_probs.device, dtype=clf_log_probs.dtype)
                )  # [1, K]

                if self.classifier2 is not None:
                    clf_log_probs2 = torch.stack(all_clf_log_probs2, dim=0)
                    clf_scores2 = torch.logsumexp(clf_log_probs2, dim=0) - torch.log(
                        torch.tensor(clf_log_probs2.shape[0], device=clf_log_probs2.device, dtype=clf_log_probs2.dtype)
                    )  # [1, K]
                    total_score = (
                        self.llm_alpha * cummulative_scores
                        + self.clf_alpha * clf_scores.view(1, -1)
                        + self.clf2_alpha * clf_scores2.view(1, -1)
                    )
                else:
                    total_score = self.llm_alpha * cummulative_scores + self.clf_alpha * clf_scores.view(1, -1)
            else:
                clf_scores = torch.zeros_like(cummulative_scores)
                total_score = self.llm_alpha * cummulative_scores

            # beam step
            total_score = self.eos_check(llm_beam_ids[:, llm_input_len:], total_score)
            beam_index = (
                total_score.topk(self.beam_size, dim=-1).indices
                + self.beam_expand_factor * self.beam_offset.unsqueeze(-1)
            ).view(-1)
            cummulative_scores = cummulative_scores.reshape(1 * self.llm_beam_size, -1)
            cummulative_scores = cummulative_scores[beam_index].reshape(1 * self.beam_size, -1)
            input_ids = llm_beam_ids[beam_index].reshape(1 * self.beam_size, -1)

            # debug preview
            output_ids = input_ids.reshape(1 * self.beam_size, -1)[:, llm_input_len:]
            output_text = self._batch_decode(output_ids, skip_special_tokens=True)
            print(output_text)

            if self.is_done():
                break

        # finalize (same as your SD path)
        output_score = total_score[:, 0]
        output_ids = input_ids.reshape(1, self.beam_size, -1)[:, 0, llm_input_len:]

        for b_i in range(1):
            self.candidate_score[b_i].append(output_score[b_i].item())
            self.candidate[b_i].append(output_ids[b_i])
            self.candidate_score[b_i] = [
                s / self.length_penalty(self.candidate[b_i][s_i].shape[-1])
                for s_i, s in enumerate(self.candidate_score[b_i])
            ]
            b_i_best = self.candidate_score[b_i].index(max(self.candidate_score[b_i]))
            self.candidate[b_i] = self.candidate[b_i][b_i_best]

        output_text = self._batch_decode(self.candidate, skip_special_tokens=True)
        llm_prompt_text = self.llm_tokenizer.decode(self.llm_prompt[0], skip_special_tokens=False)
        final_candidates = self._batch_decode(
            llm_beam_ids[beam_index, llm_input_len:], skip_special_tokens=True
        )

        
        if getattr(self, "use_occupation_prompt", False):
            tmpl = getattr(self, "model_occupation_template", "A photo of the face of a {occupation}, a person")
            output_text = [tmpl.format(occupation=t.strip()) for t in output_text]
        else:
            output_text = [self.model_prompt_primer + t for t in output_text]

        final_score = 0.0

        return {
            "text": output_text[0],
            "input_prompt": llm_prompt_text,
            "alt_prompts": final_candidates,
            "bias_score": final_score,
        }


    # --------------- public API --------------- #
    @torch.no_grad()
    def inference(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        skip_inference: bool = False,
        sampling_generator: torch.Generator = None,
        **_
    ) -> List[Dict]:
        """
        Generates a prompt that maximizes target attribute bias (per classifier),
        then (optionally) generates images for that prompt.
        """
        # generate the best prompt
        if skip_inference:
            best_prompt = self.model_prompt_primer
        else:
            if self.cfg["llm_only"]:
                print("Generating prompt with LLM only (no classifier guidance).")
                # gen = self.generate_prompt_llm_only()
                if self.deepfloyd_if:
                    gen = self.generate_prompt_if(llm_only=True, sampling_generator=sampling_generator)
                else:
                    gen = self.generate_prompt(llm_only=True,sampling_generator=sampling_generator)
            else:
                print("Generating prompt with classifier-guided beam search.")
                # profiler = cProfile.Profile()
                # profiler.enable()
                if self.deepfloyd_if:
                    gen = self.generate_prompt_if(sampling_generator=sampling_generator)
                else:
                    gen = self.generate_prompt(sampling_generator=sampling_generator)
                # profiler.disable()
                # stats = pstats.Stats(profiler).sort_stats('cumulative')
                # stats.print_stats(20)
            best_prompt = gen["text"]

        # optionally synthesize images with SD
        images = None
        if not self.gen_prompt_only:
            pipe = self.sd_pipeline
            steps = num_inference_steps or self.num_inference_steps
            scale = guidance_scale or self.guidance_scale
            images = pipe(
                prompt=best_prompt,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=scale,
                num_inference_steps=steps,
                height=self.height,
                width=self.width,
            ).images

        if skip_inference:
            gen = {"input_prompt": [self.model_prompt_primer], 
                   "bias_score": 0.0,
                   "alt_prompts": [],
                   }
        # For compatibility with your runner, return "similarity" = bias score.
        results = [{
            "image": images,
            "initial_condition": None,
            "prompt": best_prompt,
            "alt_prompts": gen.get("alt_prompts", []),
            "input_prompt": gen.get("input_prompt", []),
            "similarity": gen.get("bias_score", 0.0),   # <- classifier-based score
            "seed": self.seed,
        }]
        return results
    
    def _ensure_full_if_eval_pipelines(self) -> None:
        """Lazy-load IF Stage-II and Stage-III pipelines for eval-set generation."""
        if not self.deepfloyd_if:
            return
        if self.if_stage_2_pipeline is None:
            self.if_stage_2_pipeline = DiffusionPipeline.from_pretrained(
                self.if_stage_2_model,
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16
            ).to(self.device)
            self.if_stage_2_pipeline.set_progress_bar_config(disable=True)
        if self.if_stage_3_pipeline is None:
            self.if_stage_3_pipeline = DiffusionPipeline.from_pretrained(
                self.if_stage_3_model,
                torch_dtype=torch.float16
            ).to(self.device)
            self.if_stage_3_pipeline.set_progress_bar_config(disable=True)

    def _maybe_upscale_eval_images(self, prompt: str, images: List[Image.Image]) -> List[Image.Image]:
        """
        Optionally run DeepFloyd IF Stage-II + Stage-III on eval images before saving.
        """
        if not images or not (self.deepfloyd_if and self.eval_set_use_full_if_pipeline):
            return images

        self._ensure_full_if_eval_pipelines()

        # IF-II expects embeddings because its text encoder is removed.
        encode_outputs = self.sd_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        if not isinstance(encode_outputs, tuple) or len(encode_outputs) < 2:
            raise RuntimeError("IF pipeline encode_prompt must return prompt and negative embeddings.")
        prompt_embeds, negative_prompt_embeds = encode_outputs[:2]
        prompt_embeds = prompt_embeds.to(self.device)
        negative_prompt_embeds = negative_prompt_embeds.to(self.device)

        upscaled_images: List[Image.Image] = []
        with torch.no_grad():
            for image in images:
                stage2_kwargs = {
                    "image": image,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "output_type": "pil",
                }
                if self.if_stage_2_guidance_scale is not None:
                    stage2_kwargs["guidance_scale"] = self.if_stage_2_guidance_scale
                if self.if_stage_2_num_inference_steps is not None:
                    stage2_kwargs["num_inference_steps"] = self.if_stage_2_num_inference_steps
                stage2_image = self.if_stage_2_pipeline(**stage2_kwargs).images[0]

                stage3_kwargs = {
                    "image": stage2_image,
                    "prompt": prompt,
                }
                if self.if_stage_3_guidance_scale is not None:
                    stage3_kwargs["guidance_scale"] = self.if_stage_3_guidance_scale
                if self.if_stage_3_num_inference_steps is not None:
                    stage3_kwargs["num_inference_steps"] = self.if_stage_3_num_inference_steps
                stage3_image = self.if_stage_3_pipeline(**stage3_kwargs).images[0]
                upscaled_images.append(stage3_image)

        return upscaled_images

    def create_eval_set(self, prompt : str, set_size: int, out_dir: str, batch_size: int = 5, num_run: int = 0) -> str:
        """
        Generates an evaluation set of prompts and images for the chosen target
        """
        print(f"Creating eval set of size {set_size} in {out_dir}")
        pipe = self.sd_pipeline
        steps = self.num_inference_steps
        scale = self.guidance_scale

        batches=set_size // batch_size
        remainder = set_size % batch_size
        os.makedirs(out_dir, exist_ok=True)
        for b in range(batches):
            images = pipe(
                prompt=prompt,
                num_images_per_prompt=batch_size,
                guidance_scale=scale,
                num_inference_steps=steps,
                height=self.height,
                width=self.width,
            ).images
            images = self._maybe_upscale_eval_images(prompt, images)
            for i, img in enumerate(images):
                img.save(os.path.join(out_dir, f"eval_{b*batch_size+i:04d}_{num_run}.png"))
        if remainder > 0:
            images = pipe(
                prompt=prompt,
                num_images_per_prompt=remainder,
                guidance_scale=scale,
                num_inference_steps=steps,
                height=self.height,
                width=self.width,
            ).images
            images = self._maybe_upscale_eval_images(prompt, images)
            for i, img in enumerate(images):
                img.save(os.path.join(out_dir, f"eval_{batches*batch_size+i:04d}_{num_run}.png"))
        return out_dir
