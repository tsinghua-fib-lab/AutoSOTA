#!/usr/bin/env python3
"""
Training External Heads for Concept Learning in PixArt-Alpha
"""

import os
import random
import zipfile
import shutil
import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from diffusers import PixArtAlphaPipeline, DDPMScheduler
from utils_data import get_dataloader


# ============================================================
# Helper: get attention module
# ============================================================
def _get_attn(block, attn_type: str):
    if attn_type not in ("attn1", "attn2"):
        raise ValueError(f"attn_type must be 'attn1' or 'attn2', got {attn_type}")
    return getattr(block, attn_type)


# ============================================================
# Probe per-layer token length
# ============================================================
class _RecordNProcessor:
    def __init__(self, original_processor, layer_idx, sink_dict):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.sink_dict = sink_dict

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        N = hidden_states.shape[1]
        self.sink_dict.setdefault(self.layer_idx, N)
        return self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


def probe_layer_token_lengths(pipe, target_layers, attn_type, tokenizer, device, resolution):
    print("\n🔎 Probing per-layer token lengths (one dry forward)...")
    orig_procs = {}
    sink = {}
    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        attn_mod = _get_attn(block, attn_type)
        orig = attn_mod.get_processor()
        orig_procs[l] = orig
        attn_mod.set_processor(_RecordNProcessor(orig, l, sink))

    B = 1
    H = W = resolution
    with torch.no_grad():
        dummy_img = torch.zeros(B, 3, H, W, device=device, dtype=torch.float32)
        latents = pipe.vae.encode(dummy_img).latent_dist.sample() * pipe.vae.config.scaling_factor

        ids = tokenizer(
            ["a"],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = ids.input_ids.to(device)
        attn_mask = ids.attention_mask.to(device)
        txt = pipe.text_encoder(input_ids, attention_mask=attn_mask, return_dict=False)[0]

        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        added_cond_kwargs = {
            "resolution": torch.as_tensor([H, W], dtype=torch.float32, device=device).repeat(B, 1),
            "aspect_ratio": torch.as_tensor([float(H) / float(W)], dtype=torch.float32, device=device).repeat(B, 1),
        }

        try:
            _ = pipe.transformer(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=txt,
                encoder_attention_mask=attn_mask,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=True,
            )
        except Exception as e:
            print(f"⚠️  Warning during probe: {e}")

    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        attn_mod = _get_attn(block, attn_type)
        attn_mod.set_processor(orig_procs[l])

    print(f"✓ Probed {len(sink)} layers")
    for l in sorted(sink.keys()):
        print(f"  Layer {l}: N={sink[l]} tokens")
    return sink


# ============================================================
# Config
# ============================================================
class TrainingConfig:
    model_id = "PixArt-alpha/PixArt-XL-2-1024-MS"

    target_layers: List[int] = list(range(11, 28))  # 17 layers
    num_heads_per_layer: int = 16

    attn_type = "attn2"  # MCA

    train_data_dir = "/datasets/./"
    resolution = 1024
    center_crop = True
    random_flip = True

    num_train_epochs = 30
    max_train_steps = 1_000_000
    train_batch_size = 4
    learning_rate = 1e-4
    weight_decay = 0.01
    gradient_accumulation_steps = 1

    dataloader_num_workers = 0
    select = "random"

    output_dir = "./output_train_model"
    log_every_steps = 10
    log_every_epochs = 10
    save_checkpoint_every = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


cfg = TrainingConfig()


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(external_heads_module, epoch: int, output_dir: str):
    ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(
        external_heads_module.state_dict(),
        os.path.join(ckpt_dir, "external_heads_full.pt"),
    )

    for layer_idx in external_heads_module.target_layers:
        layer_dir = os.path.join(ckpt_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        for head_idx in range(external_heads_module.num_heads):
            key = f"layer_{layer_idx}_head_{head_idx}"
            head_params = external_heads_module.external_heads[key]
            torch.save(head_params, os.path.join(layer_dir, f"head_{head_idx}.pt"))

    zip_path = f"{ckpt_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(ckpt_dir):
            for f in files:
                fp = os.path.join(root, f)
                arc = os.path.relpath(fp, output_dir)
                zf.write(fp, arc)

    print(f"💾 Checkpoint saved: {zip_path}")
    shutil.rmtree(ckpt_dir)


# ============================================================
# External heads
# ============================================================
class ExternalHeads(nn.Module):
    def __init__(self, target_layers: List[int], num_heads: int, per_layer_seq: dict, head_dim: int):
        super().__init__()
        self.target_layers = sorted(target_layers)
        self.num_heads = num_heads
        self.per_layer_seq = per_layer_seq
        self.head_dim = head_dim

        self.external_heads = nn.ParameterDict()
        for l in self.target_layers:
            S_l = self.per_layer_seq[l]
            for h in range(num_heads):
                key = f"layer_{l}_head_{h}"
                self.external_heads[key] = nn.Parameter(torch.zeros(S_l, head_dim))

        total = sum(self.external_heads[k].numel() for k in self.external_heads)
        print(f"✓ Initialized {len(self.external_heads)} external heads (total params: {total:,})")
        for l in self.target_layers:
            print(f"  Layer {l}: S_l={self.per_layer_seq[l]} tokens")

    def get_external_head(self, layer_idx: int, head_idx: int, batch_size: int):
        if layer_idx not in self.target_layers:
            S_any = next(iter(self.per_layer_seq.values()))
            device = next(iter(self.external_heads.values())).device
            return torch.zeros(batch_size, S_any, self.head_dim, device=device)
        key = f"layer_{layer_idx}_head_{head_idx}"
        return self.external_heads[key].unsqueeze(0).expand(batch_size, -1, -1)

    def get_all_heads_for_layer(self, layer_idx: int, batch_size: int):
        heads = [self.get_external_head(layer_idx, h, batch_size) for h in range(self.num_heads)]
        return torch.stack(heads, dim=0).permute(1, 0, 2, 3)


# ============================================================
# Custom attention processor
# ============================================================
class ExternalHeadProcessor:
    def __init__(self, original_processor, layer_idx, attn_module, external_heads_module, num_heads, head_dim):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.attn_module = attn_module
        self.external_heads_module = external_heads_module
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.first_call = True
        self.active = True  # runtime toggle

    def _apply_to_out_bias_free(self, attn, delta_concat):
        if not hasattr(attn, "to_out") or attn.to_out is None:
            return delta_concat
        to_out = attn.to_out

        if isinstance(to_out, nn.ModuleList):
            y = delta_concat
            for m in to_out:
                if isinstance(m, nn.Linear):
                    y = F.linear(y, m.weight, bias=None)
                else:
                    y = m(y)
            return y

        if isinstance(to_out, nn.Sequential):
            y = delta_concat
            for m in to_out:
                if isinstance(m, nn.Linear):
                    y = F.linear(y, m.weight, bias=None)
                else:
                    y = m(y)
            return y

        if isinstance(to_out, nn.Linear):
            return F.linear(delta_concat, to_out.weight, bias=None)

        return to_out(delta_concat)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        orig_output = self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        if not self.active:
            return orig_output

        B, N, C = orig_output.shape
        external_heads = self.external_heads_module.get_all_heads_for_layer(self.layer_idx, B)
        H = external_heads.shape[1]
        S = external_heads.shape[2]
        d_h = external_heads.shape[3]

        assert S == N, (
            f"Token mismatch in layer {self.layer_idx}: S={S}, N={N}. "
            "Fix per-layer sequence lengths."
        )

        if self.first_call:
            print(f"  Layer {self.layer_idx}: B={B}, N={N}, H={H}, d_h={d_h}")
            self.first_call = False

        delta_concat = external_heads.permute(0, 2, 1, 3).reshape(B, N, H * d_h)
        delta = self._apply_to_out_bias_free(attn, delta_concat)
        assert delta.shape == orig_output.shape
        return orig_output + delta


def setup_external_heads_processors(pipe, external_heads_module, target_layers, attn_type):
    print("\n🔧 Setting up external head processors...")
    installed = 0
    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        attn_mod = _get_attn(block, attn_type)
        num_heads = attn_mod.heads

        if hasattr(attn_mod.to_q, "out_features"):
            inner_dim = attn_mod.to_q.out_features
        elif hasattr(attn_mod.to_q, "weight"):
            inner_dim = attn_mod.to_q.weight.shape[0]
        else:
            raise ValueError(f"Cannot determine inner_dim in layer {l}")

        head_dim = inner_dim // num_heads
        original_processor = attn_mod.get_processor()

        attn_mod.set_processor(
            ExternalHeadProcessor(
                original_processor=original_processor,
                layer_idx=l,
                attn_module=attn_mod,
                external_heads_module=external_heads_module,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        )
        installed += 1

    print(f"✓ Installed {installed} custom processors")
    print(f"  Target layers: {target_layers}")
    print(f"  Attention type: {attn_type}")


# ============================================================
# Training wrapper
# ============================================================
class DiffusionTrainingModel(nn.Module):
    def __init__(self, pipe, external_heads_module, noise_scheduler):
        super().__init__()
        self.pipe = pipe
        self.external_heads_module = external_heads_module
        self.noise_scheduler = noise_scheduler
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.transformer = pipe.transformer

        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.vae.eval()
        self.text_encoder.eval()
        self.transformer.eval()

        print("✓ Froze backbone; training only external heads")

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.text_encoder.eval()
        self.transformer.eval()
        return self

    def use_external_heads(self, enable: bool):
        for l in self.external_heads_module.target_layers:
            block = self.pipe.transformer.transformer_blocks[l]
            attn_mod = _get_attn(block, cfg.attn_type)
            processor = attn_mod.get_processor()
            if hasattr(processor, "active"):
                processor.active = enable

    def forward_with_prompt_comparison(self, pixel_values, person_ids, concept_ids, person_mask, concept_mask):
        B = pixel_values.shape[0]
        device = pixel_values.device

        # encode image
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # conds
        height, width = pixel_values.shape[2], pixel_values.shape[3]
        added_cond_kwargs = {
            "resolution": torch.as_tensor([height, width], dtype=torch.float32, device=device).repeat(B, 1),
            "aspect_ratio": torch.as_tensor([float(height) / float(width)], dtype=torch.float32, device=device).repeat(B, 1),
        }

        # (1) heads OFF, concept prompt
        with torch.no_grad():
            te_concept = self.text_encoder(concept_ids, attention_mask=concept_mask, return_dict=False)[0]
        self.use_external_heads(False)
        with torch.no_grad():
            model_ref = self.transformer(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=te_concept,
                encoder_attention_mask=concept_mask,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        # (2) heads ON, person prompt
        with torch.no_grad():
            te_person = self.text_encoder(person_ids, attention_mask=person_mask, return_dict=False)[0]
        self.use_external_heads(True)
        model_pred = self.transformer(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=te_person,
            encoder_attention_mask=person_mask,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        C_lat = latents.shape[1]
        if model_ref.shape[1] == 2 * C_lat:
            model_ref = model_ref.chunk(2, dim=1)[0]
        if model_pred.shape[1] == 2 * C_lat:
            model_pred = model_pred.chunk(2, dim=1)[0]

        return model_pred, model_ref


# ============================================================
# Main training
# ============================================================
def train():
    print("\n" + "=" * 80)
    print("EXTERNAL HEADS TRAINING (NO INTERPOLATION) + PROMPT DELTA + HEAD IMPORTANCE")
    print("=" * 80)

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"📦 Loading PixArt model: {cfg.model_id}")
    pipe = PixArtAlphaPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(cfg.device)
    print(f"✓ Model loaded on {cfg.device}")

    # base tokenizer from pipeline
    base_tokenizer = pipe.tokenizer

    # WRAPPER: make tokenizer look like old one for utils_data.py
    def pixart_tokenize_for_dataset(texts):
        # texts: list[str]
        out = base_tokenizer(
            texts,
            max_length=base_tokenizer.model_max_length,
            padding=False,
            truncation=True,
        )
        # out["input_ids"] is a list[list[int]]
        return out["input_ids"]

    per_layer_seq = probe_layer_token_lengths(
        pipe, cfg.target_layers, cfg.attn_type, base_tokenizer, cfg.device, cfg.resolution
    )

    # attention dims
    first_layer = cfg.target_layers[0]
    blk = pipe.transformer.transformer_blocks[first_layer]
    attn_mod = _get_attn(blk, cfg.attn_type)
    num_heads = attn_mod.heads
    if hasattr(attn_mod.to_q, "out_features"):
        inner_dim = attn_mod.to_q.out_features
    else:
        inner_dim = attn_mod.to_q.weight.shape[0]
    head_dim = inner_dim // num_heads

    external_heads_module = ExternalHeads(
        target_layers=cfg.target_layers,
        num_heads=num_heads,
        per_layer_seq=per_layer_seq,
        head_dim=head_dim,
    ).to(cfg.device)

    setup_external_heads_processors(pipe, external_heads_module, cfg.target_layers, cfg.attn_type)

    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")
    model = DiffusionTrainingModel(pipe, external_heads_module, noise_scheduler).to(cfg.device)
    model.train()

    # transforms
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.resolution, cfg.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(cfg.resolution) if cfg.center_crop else transforms.RandomCrop(cfg.resolution),
        transforms.RandomHorizontalFlip() if cfg.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # collate: dataset returns (img, token_ids_from_dataset)
    def collate_fn(examples):
        pixel_values = torch.stack([ex[0] for ex in examples]).contiguous().float()
        input_ids = [ex[1] for ex in examples]
        padded = base_tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        input_conditions = torch.zeros(len(examples), 1)
        return {
            "pixel_values": pixel_values,
            "input_ids": padded.input_ids,
            "attention_mask": padded.attention_mask,
            "input_conditions": input_conditions,
        }

    # RESTORED CONTRACT: pass wrapper, not HF tokenizer directly
    train_dataloader = get_dataloader(
        cfg.train_data_dir,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        transform=train_tfms,
        tokenizer=pixart_tokenize_for_dataset,   # <-- key fix
        collate_fn=collate_fn,
        num_workers=cfg.dataloader_num_workers,
        max_concept_length=100,
        select=cfg.select,
    )

    print(f"✓ Data loader ready ({len(train_dataloader)} batches)")

    trainable_params = list(external_heads_module.parameters())
    optimizer = optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # head-importance EMA
    head_importance = {name: 0.0 for name in external_heads_module.external_heads.keys()}
    ema_decay = 0.9

    print("\n" + "=" * 80)
    print("🎯 STARTING TRAINING")
    print("=" * 80)

    global_step = 0
    for epoch in range(cfg.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        accum_counter = 0

        for step, batch in progress:
            if global_step >= cfg.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(cfg.device, dtype=torch.float32)

            B = pixel_values.shape[0]
            person_prompts = ["a photo of a person"] * B
            tokens_person = base_tokenizer(person_prompts, padding="max_length", truncation=True, return_tensors="pt")
            person_ids = tokens_person.input_ids.to(cfg.device)
            person_mask = tokens_person.attention_mask.to(cfg.device)

            # concept prompt
            concept_prompts = ["a photo of a woman"] * B
            tokens_concept = base_tokenizer(concept_prompts, padding="max_length", truncation=True, return_tensors="pt")
            concept_ids = tokens_concept.input_ids.to(cfg.device)
            concept_mask = tokens_concept.attention_mask.to(cfg.device)

            model_pred, model_ref = model.forward_with_prompt_comparison(
                pixel_values,
                person_ids=person_ids,
                concept_ids=concept_ids,
                person_mask=person_mask,
                concept_mask=concept_mask,
            )

            loss = F.mse_loss(model_pred.float(), model_ref.float(), reduction="mean")
            loss = loss / cfg.gradient_accumulation_steps

            if torch.isnan(loss):
                print(f"\n⚠️ NaN loss at global_step={global_step}, dataloader step={step}")
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                continue

            loss.backward()
            accum_counter += 1

            # update head-importance from grads
            for name, param in external_heads_module.external_heads.items():
                if param.grad is None:
                    continue
                g_norm = param.grad.detach().norm(p=2).item()
                head_importance[name] = ema_decay * head_importance[name] + (1 - ema_decay) * g_norm

            reached_accum = (accum_counter == cfg.gradient_accumulation_steps)
            last_batch = (step == len(train_dataloader) - 1)
            will_exceed_max = (global_step + 1 > cfg.max_train_steps)

            if reached_accum or last_batch or will_exceed_max:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                epoch_loss += loss.item() * cfg.gradient_accumulation_steps
                global_step += 1

                if global_step % cfg.log_every_steps == 0:
                    print(f"\n📊 Epoch {epoch}, Global step {global_step}, Loss: {loss.item() * cfg.gradient_accumulation_steps:.6f}")

            progress.set_postfix({
                "loss": f"{loss.item() * cfg.gradient_accumulation_steps:.6f}",
                "gstep": global_step
            })

        if epoch % cfg.log_every_epochs == 0:
            denom = max(1, len(train_dataloader))
            avg_loss = epoch_loss / denom
            print(f"\n✅ Epoch {epoch} complete. Avg loss: {avg_loss:.6f}")

        if (epoch + 1) % cfg.save_checkpoint_every == 0:
            save_checkpoint(external_heads_module, epoch + 1, cfg.output_dir)
            print("🔄 Continuing training...")

    # save final
    print("\n" + "=" * 80)
    print("💾 SAVING FINAL EXTERNAL HEADS + IMPORTANCE")
    print("=" * 80)
    save_dir = os.path.join(cfg.output_dir, "external_heads_final")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(external_heads_module.state_dict(), os.path.join(save_dir, "external_heads_full.pt"))
    for layer_idx in external_heads_module.target_layers:
        ld = os.path.join(save_dir, f"layer_{layer_idx}")
        os.makedirs(ld, exist_ok=True)
        for h in range(external_heads_module.num_heads):
            key = f"layer_{layer_idx}_head_{h}"
            torch.save(external_heads_module.external_heads[key], os.path.join(ld, f"head_{h}.pt"))

    with open(os.path.join(save_dir, "head_importance.json"), "w") as f:
        json.dump(head_importance, f, indent=2)

    print("\n🔥 Most relevant heads per layer (top 3):")
    per_layer = {}
    for name, score in head_importance.items():
        parts = name.split("_")
        l = int(parts[1])
        per_layer.setdefault(l, [])
        per_layer[l].append((name, score))
    for l, items in sorted(per_layer.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[:3]
        print(f"Layer {l}:")
        for n, s in top:
            print(f"  {n}: {s:.6f}")

    print(f"✓ Final external heads and importance saved to: {save_dir}")
    return external_heads_module, model


if __name__ == "__main__":
    external_heads, model = train()
