#!/usr/bin/env python3
"""
Training External Heads for Concept Learning in FLUX.1-dev
"""

import os
import json
import zipfile
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers import FluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# CombinedTimestepGuidanceTextProjEmbeddings import removed as it is no longer needed for patching
from types import MethodType

from utils_data import TrainingDataset


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    model_id: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None
    resolution: int = 512
    # ↓↓↓ Reduce batch size to avoid OOM on FLUX.1-dev
    batch_size: int = 1
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    mixed_precision: str = "bf16"
    output_dir: str = "out_model"
    seed: int = 42

    train_data_dir: str = "datasets/."
    target_layers: Tuple[int, ...] = tuple(range(19))
    num_heads: int = 24
    importance_ema_decay: float = 0.99
    save_every: int = 1000000
    save_every_epochs: int = 5

    # Added guidance scale config
    guidance_scale: float = 3.5


# ============================================================
# ExternalHeads module
# ============================================================

class ExternalHeads(nn.Module):
    def __init__(
            self,
            target_layers: List[int],
            num_heads: int,
            seq_len: int,
            head_dim: int,
            init_scale: float = 0.02,
            param_dtype: torch.dtype = torch.float32,
    ):
        """
        param_dtype: dtype of the learnable heads (match transformer dtype,
        e.g., bfloat16) to cut memory usage.
        """
        super().__init__()
        self.target_layers_sorted: List[int] = sorted(target_layers)
        self.layer_to_index: Dict[int, int] = {
            l: i for i, l in enumerate(self.target_layers_sorted)
        }
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim

        self.external_heads = nn.Parameter(
            torch.randn(
                len(self.target_layers_sorted),
                num_heads,
                seq_len,
                head_dim,
                dtype=param_dtype,
            )
            * init_scale
        )

        # Flag to enable/disable during forward
        self.training_enabled = True

    def get_params_for_layer(
            self, layer_idx: int, seq_len_current: int, batch_size: int
    ) -> torch.Tensor:
        l_local = self.layer_to_index[layer_idx]
        assert seq_len_current <= self.seq_len

        layer_heads = self.external_heads[l_local, :, :seq_len_current, :]
        return layer_heads.unsqueeze(0).expand(batch_size, -1, -1, -1)


# ============================================================
#    *** forward-hook attention modification  ***
# ============================================================

def install_external_head_processors(pipe, external_heads, target_layers):
    """
    FLUX ATTENTION NO LONGER SUPPORTS PROCESSOR-REPLACEMENT.
    """

    print("Installing ExternalHeadProcessor via forward hooks...")

    tr = pipe.transformer
    layer_processors = {}

    for layer_idx in target_layers:
        block = tr.transformer_blocks[layer_idx]
        attn = block.attn

        # Save original forward
        orig_forward = attn.forward

        def make_wrapper(orig_forward, attn, layer_idx):
            def wrapped_forward(hidden_states, *args, **kwargs):
                # original FLUX attention (may return Tensor or tuple)
                out = orig_forward(hidden_states, *args, **kwargs)

                # baseline pass → no modification
                if not external_heads.training_enabled:
                    return out

                # Handle both Tensor and tuple outputs
                if isinstance(out, tuple):
                    attn_hidden = out[0]  # [B, N, C]
                    extra = out[1:]
                else:
                    attn_hidden = out
                    extra = None

                # attn_hidden = [B, N, C]
                B, N, C = attn_hidden.shape
                H = external_heads.num_heads
                D = external_heads.head_dim

                # external heads: [1, H, N, D] → [B, H, N, D]
                ext = external_heads.get_params_for_layer(layer_idx, N, B)
                # reshape to FLUX layout: [B, N, H*D]
                ext = ext.permute(0, 2, 1, 3).reshape(B, N, H * D)
                # Match attention hidden dtype (e.g., bf16)
                ext = ext.to(attn_hidden.dtype)

                # Add external heads
                attn_hidden = attn_hidden + ext

                # Re-wrap in original structure
                if isinstance(out, tuple):
                    return (attn_hidden, *extra)
                else:
                    return attn_hidden

            return wrapped_forward

        # Replace attention forward with wrapped version
        attn.forward = make_wrapper(orig_forward, attn, layer_idx)
        layer_processors[layer_idx] = attn

        print(f"  ✓ Patched attention forward for layer {layer_idx}")

    print("Done.\n")
    return layer_processors


# ============================================================
# Probe sequence length
# ============================================================

@torch.no_grad()
def probe_flux_sequence_length(pipe, resolution, device):
    pipe = pipe.to(device)
    pipe.vae.to(device)

    # correct dtype from VAE
    vae_dtype = getattr(pipe.vae, "dtype", next(pipe.vae.parameters()).dtype)

    dummy = torch.zeros(
        1, 3, resolution, resolution,
        device=device,
        dtype=vae_dtype,
    )

    latents = pipe.vae.encode(dummy).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

    B, C, H_lat, W_lat = latents.shape

    # FLUX expects pack using original spatial dims
    height_tokens = H_lat
    width_tokens = W_lat

    packed = pipe._pack_latents(latents, B, C, height_tokens, width_tokens)
    seq_len = packed.shape[1]

    print(f"Latents shape: {latents.shape}")
    print(f"Height tokens: {height_tokens} Width tokens: {width_tokens}")
    print(f"Sequence length = {seq_len}")

    return int(seq_len)


# ============================================================
# Trainer
# ============================================================

class FluxExternalHeadsTrainer(nn.Module):
    def __init__(
            self,
            pipe,
            noise_scheduler,
            external_heads,
            layer_processors,
            cfg,
            device,
    ):
        super().__init__()
        self.pipe = pipe
        self.noise_scheduler = noise_scheduler
        self.external_heads = external_heads
        self.layer_processors = layer_processors
        self.cfg = cfg
        self.device = device

        self.vae = pipe.vae
        self.transformer = pipe.transformer

        # freeze base model
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False

        L = len(self.external_heads.target_layers_sorted)
        H = self.external_heads.num_heads

        self.register_buffer(
            "head_importance",
            torch.zeros(L, H, dtype=torch.float32, device=device),
            persistent=False,
        )

    def set_external_heads_enabled(self, enabled: bool):
        self.external_heads.training_enabled = enabled

    def encode_prompts(self, prompts, max_sequence_length=512):
        device = self.device
        out = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        return out[0], out[1], out[2]

    def forward_step(self, pixel_values, prompts_person, prompts_woman):

        device = self.device
        B = pixel_values.shape[0]

        # auto dtype
        pixel_values = pixel_values.to(device=device, dtype=self.vae.dtype)

        # Encode images to latents (x_0)
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor  # [B, C, H, W]

        # Sample noise and timesteps
        noise = torch.randn_like(latents)

        # For FlowMatchEulerDiscreteScheduler we MUST use values from `self.noise_scheduler.timesteps`
        if isinstance(self.noise_scheduler, FlowMatchEulerDiscreteScheduler):
            all_timesteps = self.noise_scheduler.timesteps.to(device)
            # Randomly sample indices into the scheduler's timestep schedule
            idx = torch.randint(
                0,
                all_timesteps.shape[0],
                (B,),
                device=device,
            )
            timesteps = all_timesteps[idx]
        else:
            # Fallback for other schedulers: integer timesteps 0..num_train_timesteps-1
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            )

        # ---- FlowMatchEulerDiscreteScheduler forward process ----
        if hasattr(self.noise_scheduler, "add_noise"):
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        elif hasattr(self.noise_scheduler, "scale_noise"):
            noisy_latents = self.noise_scheduler.scale_noise(latents, timesteps, noise)
        else:
            raise AttributeError(
                f"{type(self.noise_scheduler).__name__} has neither `add_noise` nor `scale_noise`"
            )
        # ---------------------------------------------------------------------- #

        B, C_lat, H_lat, W_lat = noisy_latents.shape
        height_tokens = H_lat
        width_tokens = W_lat

        # Pack latents exactly like FluxPipeline
        packed = self.pipe._pack_latents(
            noisy_latents, B, C_lat, height_tokens, width_tokens
        )

        # Prepare latent image ids consistent with packed latents
        latent_ids = self.pipe._prepare_latent_image_ids(
            batch_size=B,
            height=height_tokens // 2,  # H_lat // 2
            width=width_tokens // 2,  # W_lat // 2
            device=device,
            dtype=packed.dtype,
        )

        # same scaling as before for transformer timestep input
        t_scaled = timesteps.to(device, dtype=packed.dtype) / float(
            self.noise_scheduler.config.num_train_timesteps
        )

        txt_person, pooled_person, txt_ids_person = self.encode_prompts(prompts_person)
        txt_woman, pooled_woman, txt_ids_woman = self.encode_prompts(prompts_woman)

        # 🔧 GLOBAL DTYPE ALIGNMENT FOR TRANSFORMER INPUTS
        tdtype = getattr(self.transformer, "dtype", packed.dtype)

        packed = packed.to(device=device, dtype=tdtype)
        latent_ids = latent_ids.to(device=device, dtype=tdtype)
        t_scaled = t_scaled.to(device=device, dtype=tdtype)

        txt_person = txt_person.to(device=device, dtype=tdtype)
        pooled_person = pooled_person.to(device=device, dtype=tdtype)
        txt_ids_person = txt_ids_person.to(device=device, dtype=tdtype)

        txt_woman = txt_woman.to(device=device, dtype=tdtype)
        pooled_woman = pooled_woman.to(device=device, dtype=tdtype)
        txt_ids_woman = txt_ids_woman.to(device=device, dtype=tdtype)

        # ============================================================
        # FIX: Create Guidance Tensor
        # FLUX.1-dev expects 'guidance'. Standard inference uses ~3.5
        # ============================================================
        guidance_tensor = torch.full(
            (B,),
            self.cfg.guidance_scale,
            device=device,
            dtype=tdtype
        )

        # BASELINE (heads OFF) - NO GRAD
        self.set_external_heads_enabled(False)
        with torch.no_grad():
            noise_ref = self.transformer(
                hidden_states=packed,
                timestep=t_scaled,
                encoder_hidden_states=txt_woman,
                pooled_projections=pooled_woman,
                txt_ids=txt_ids_woman,
                img_ids=latent_ids,
                guidance=guidance_tensor,  # <--- FIX: Pass guidance
                return_dict=False,
            )[0]

        # EDITED (heads ON) - GRAD ENABLED for external heads
        self.set_external_heads_enabled(True)
        noise_pred = self.transformer(
            hidden_states=packed,
            timestep=t_scaled,
            encoder_hidden_states=txt_person,
            pooled_projections=pooled_person,
            txt_ids=txt_ids_person,
            img_ids=latent_ids,
            guidance=guidance_tensor,  # <--- FIX: Pass guidance
            return_dict=False,
        )[0]

        loss = F.mse_loss(noise_pred.float(), noise_ref.float())
        return loss

    @torch.no_grad()
    def update_head_importance(self, ema_decay):
        param = self.external_heads.external_heads
        if param.grad is None:
            return

        grad = param.grad.detach()
        per_head_norm = grad.pow(2).sum(dim=(2, 3)).sqrt()  # stay CUDA

        if self.head_importance.device != grad.device:
            self.head_importance = self.head_importance.to(grad.device)

        self.head_importance.mul_(ema_decay).add_(
            per_head_norm * (1 - ema_decay)
        )


# ============================================================
# Training loop
# ============================================================

def train(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device = {device}")

    hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
    )
    print("HF_TOKEN visible inside Python:", "YES" if hf_token else "NO")

    print(f"Loading FLUX pipeline: {cfg.model_id}")
    dtype = (
        torch.bfloat16 if cfg.mixed_precision == "bf16"
        else torch.float16 if cfg.mixed_precision == "fp16"
        else torch.float32
    )

    pipe = FluxPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        revision=cfg.revision,
        token=hf_token,
    ).to(device)
    print(f"Loadede FLUX config: {pipe.config}")

    # Try to reduce memory via gradient checkpointing / efficient attention
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        print("Enabling gradient checkpointing on Flux transformer...")
        pipe.transformer.enable_gradient_checkpointing()
    if hasattr(pipe.transformer, "gradient_checkpointing"):
        pipe.transformer.gradient_checkpointing = True
    if hasattr(pipe.transformer, "set_use_memory_efficient_attention_xformers"):
        try:
            pipe.transformer.set_use_memory_efficient_attention_xformers(True)
            print("Enabled xFormers memory-efficient attention (if available).")
        except Exception as e:
            print(f"Could not enable xFormers attention: {e}")



    pipe.scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
        shift=pipe.scheduler.config.shift,
    )
    noise_scheduler = pipe.scheduler

    # Initialize scheduler timesteps so scale_noise has a non-empty schedule
    if hasattr(noise_scheduler, "set_timesteps"):
        noise_scheduler.set_timesteps(
            noise_scheduler.config.num_train_timesteps,
            device=device,
        )

    seq_len = probe_flux_sequence_length(pipe, cfg.resolution, device)

    head_dim = pipe.transformer.config.attention_head_dim
    head_dtype = next(pipe.transformer.parameters()).dtype

    external_heads = ExternalHeads(
        target_layers=list(cfg.target_layers),
        num_heads=cfg.num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        param_dtype=head_dtype,  # match transformer dtype (e.g., bf16)
    ).to(device)

    layer_processors = install_external_head_processors(
        pipe, external_heads, list(cfg.target_layers)
    )

    trainer = FluxExternalHeadsTrainer(
        pipe=pipe,
        noise_scheduler=noise_scheduler,
        external_heads=external_heads,
        layer_processors=layer_processors,
        cfg=cfg,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        trainer.external_heads.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_tfms = transforms.Compose(
        [
            transforms.Resize((cfg.resolution, cfg.resolution)),
            transforms.CenterCrop(cfg.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def flux_tokenizer_wrapper(text_list):
        out = pipe.tokenizer(
            text_list,
            padding=False,
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
        )
        return out["input_ids"]

    train_dataset = TrainingDataset(
        image_folder=cfg.train_data_dir,
        transform=train_tfms,
        tokenizer=flux_tokenizer_wrapper,
        max_concept_length=128,
        select="random",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda samples: {
            "pixel_values": torch.stack([x[0] for x in samples]).float()
        },
    )

    global_step = 0
    decay = cfg.importance_ema_decay

    for epoch in range(cfg.num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.num_epochs} =====")

        for batch in train_loader:
            pixel_values = batch["pixel_values"]

            prompts_person = ["a photo of a person"] * cfg.batch_size
            prompts_woman = ["a photo of a woman"] * cfg.batch_size

            optimizer.zero_grad(set_to_none=True)
            loss = trainer.forward_step(pixel_values, prompts_person, prompts_woman)
            loss.backward()

            trainer.update_head_importance(decay)
            optimizer.step()

            global_step += 1

            if global_step % 50 == 0:
                print(f"Step {global_step} | loss = {loss.item():.4f}")

        # PixArt-style epoch-based checkpoint saving
        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_heads_pixart_style(
                trainer,
                cfg,
                f"checkpoint_epoch_{epoch + 1}",
            )
            print(f"Saved PixArt-style checkpoint at epoch {epoch + 1}")

    # Final PixArt-style save with name 'external_heads_final'
    save_heads_pixart_style(trainer, cfg, "external_heads_final")
    print("Training complete.")


# ============================================================
# Save checkpoints (step-based; now unused in training loop but kept)
# ============================================================

def save_heads_pixart_style(trainer, cfg, name):
    """
    PixArt-style saver: saves full external_heads state, per-layer per-head tensors,
    and head importance into a folder, then zips it and removes the folder.
    """
    output_dir = cfg.output_dir
    save_dir = os.path.join(output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    # Full external_heads weights
    torch.save(
        trainer.external_heads.state_dict(),
        os.path.join(save_dir, "external_heads_full.pt"),
    )

    # Restore missing variables
    layer_ids = trainer.external_heads.target_layers_sorted
    H = trainer.external_heads.num_heads


    # Head importance JSON
    imp_dict = {}
    arr = trainer.head_importance.detach().cpu().tolist()

    for i, layer_idx in enumerate(layer_ids):
        for h in range(H):
            imp_dict[f"layer_{layer_idx}_head_{h}"] = arr[i][h]

    with open(os.path.join(save_dir, "head_importance.json"), "w") as f:
        json.dump(imp_dict, f, indent=2)

    # Zip the folder
    zip_path = f"{save_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(save_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                arcname = os.path.relpath(file_path, output_dir)
                zf.write(file_path, arcname)

    shutil.rmtree(save_dir)
    print(f"💾 PixArt-style checkpoint saved → {zip_path}")



def save_checkpoint(trainer, cfg, step):
    ckpt = os.path.join(cfg.output_dir, f"checkpoint_step_{step}")
    os.makedirs(ckpt, exist_ok=True)

    torch.save(
        trainer.external_heads.state_dict(),
        os.path.join(ckpt, "external_heads.pt"),
    )

    layer_ids = trainer.external_heads.target_layers_sorted
    H = trainer.external_heads.num_heads
    arr = trainer.head_importance.cpu().tolist()

    d = {}
    for i, l in enumerate(layer_ids):
        for h in range(H):
            d[f"layer_{l}_head_{h}"] = arr[i][h]

    with open(os.path.join(ckpt, "head_importance.json"), "w") as f:
        json.dump(d, f, indent=2)

    print(f"Saved checkpoint at step {step} → {ckpt}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = TrainConfig()
    cfg.num_epochs = 30
    print(f"Forcing num_epochs to: {cfg.num_epochs}")
    train(cfg)
