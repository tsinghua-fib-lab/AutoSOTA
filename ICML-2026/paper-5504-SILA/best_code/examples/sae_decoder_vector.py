#!/usr/bin/env python3
"""
End-to-end example: SAE decoder vector interpretation with SelfIE.

This script demonstrates how to use a trained SelfIE adapter to generate
natural language descriptions of SAE (Sparse Autoencoder) latent features:

1. Download a trained SelfIE adapter from HuggingFace Hub
2. Load a pretrained SAE using sae-lens (Goodfire SAE for Llama-3.1-8B layer 19)
3. Extract decoder vectors for specific latent indices
4. Pass through the trained adapter to create soft token embeddings
5. Inject into the SelfIE template and generate descriptions

Each SAE latent captures some learned feature of the language model's internal
representations. SelfIE lets the model describe these features in its own words.

Requirements:
    pip install selfie-adapters[sae]

Usage:
    python sae_decoder_vector.py
    python sae_decoder_vector.py --latents 100 200 300 400 500
    python sae_decoder_vector.py --num-random 10
"""

import argparse
import random

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfie_adapters import load_adapter

# ── Defaults ──────────────────────────────────────────────────────────────────

HF_REPO = "keenanpepper/selfie-adapters-llama-3.1-8b-instruct"
ADAPTER_FILENAME = "goodfire-sae-scalar-affine.safetensors"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Goodfire SAE for Llama-3.1-8B-Instruct layer 19
SAE_RELEASE = "goodfire-llama-3.1-8b-instruct"
SAE_ID = "layer_19"

# SelfIE prompt template
SELFIE_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    'What is the meaning of "<|reserved_special_token_0|>"?'
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    'The meaning of "<|reserved_special_token_0|>" is "'
)
RESERVED_TOKEN = "<|reserved_special_token_0|>"


def download_adapter() -> str:
    """Download the SelfIE adapter from HuggingFace Hub."""
    print("Downloading adapter from HuggingFace...")
    adapter_path = hf_hub_download(repo_id=HF_REPO, filename=ADAPTER_FILENAME)
    print(f"  Adapter: {adapter_path}")
    return adapter_path


def load_sae(device: str):
    """Load the Goodfire SAE using sae-lens."""
    try:
        from sae_lens import SAE
    except ImportError:
        raise ImportError(
            "sae-lens is required for this example. Install it with:\n"
            "  pip install selfie-adapters[sae]"
        )

    print(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    print(f"  SAE dimensions: {sae.cfg.d_in} -> {sae.cfg.d_sae}")
    return sae


def load_model(device: str) -> tuple:
    """Load the base language model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def generate_description(
    model,
    tokenizer,
    soft_token: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.5,
    device: str = "cuda",
) -> str:
    """Inject a soft token into the SelfIE template and generate a single description."""
    template_tokens = tokenizer(SELFIE_TEMPLATE, return_tensors="pt", add_special_tokens=False).to(device)

    reserved_token_id = tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
    inject_positions = [
        i for i, tid in enumerate(template_tokens.input_ids[0])
        if tid.item() == reserved_token_id
    ]

    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeddings = embed_layer(template_tokens.input_ids)

    soft_token_cast = soft_token.to(dtype=embeddings.dtype, device=embeddings.device)
    for pos in inject_positions:
        embeddings[:, pos, :] = soft_token_cast

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    label = text.rsplit('"', 1)[0] if '"' in text else text
    return label


def main():
    parser = argparse.ArgumentParser(description="SelfIE SAE decoder vector interpretation")
    parser.add_argument("--latents", type=int, nargs="+", default=None,
                        help="Specific latent indices to interpret")
    parser.add_argument("--num-random", type=int, default=5,
                        help="Number of random latents to interpret (used if --latents not given)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for latent selection")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Step 1: Download adapter
    adapter_path = download_adapter()

    # Step 2: Load SAE
    sae = load_sae(args.device)
    num_latents = sae.cfg.d_sae

    # Filter out dead latents (zero decoder norm) — these never activated during
    # SAE training and produce meaningless descriptions.
    decoder_norms = sae.W_dec.data.norm(dim=-1)
    alive_mask = decoder_norms > 0
    alive_indices = alive_mask.nonzero(as_tuple=True)[0].tolist()
    num_alive = len(alive_indices)
    num_dead = num_latents - num_alive
    print(f"\nSAE latent stats: {num_alive} alive, {num_dead} dead (zero decoder norm)")

    # Pick latent indices
    if args.latents is not None:
        latent_indices = args.latents
        # Warn if any user-specified latents are dead
        dead_requested = [i for i in latent_indices if not alive_mask[i]]
        if dead_requested:
            print(f"  Warning: latents {dead_requested} have zero decoder norm (dead latents)")
    else:
        random.seed(args.seed)
        latent_indices = sorted(random.sample(alive_indices, min(args.num_random, num_alive)))
    print(f"Latent indices to interpret: {latent_indices}")

    # Step 3: Get decoder vectors for the selected latents
    # The decoder weight matrix W_dec has shape (num_latents, d_model).
    # Each row is the decoder vector for one latent feature.
    decoder_vectors = sae.W_dec.data[latent_indices]  # (num_selected, d_model)
    print(f"Decoder vectors shape: {decoder_vectors.shape}")

    # Step 4: Load base model
    model, tokenizer = load_model(args.device)

    # Step 5: Load adapter
    print("\nLoading adapter...")
    adapter = load_adapter(adapter_path, device=args.device)

    # Step 6: Transform and generate for each latent
    print(f"\nGenerating descriptions for {len(latent_indices)} latents...\n")
    print("=" * 70)
    print(f"{'Latent':>8}  {'Decoder norm':>12}  Description")
    print("-" * 70)

    for i, latent_idx in enumerate(latent_indices):
        decoder_vec = decoder_vectors[i]
        dec_norm = decoder_vec.float().norm().item()

        # Transform through adapter to get soft token
        soft_token = adapter.transform(decoder_vec.to(args.device))

        # Generate description
        description = generate_description(model, tokenizer, soft_token, device=args.device)

        print(f"{latent_idx:>8}  {dec_norm:>12.4f}  {description}")

    print("=" * 70)
    print(f"\nSAE: {SAE_RELEASE} / {SAE_ID}")
    print(f"Adapter: {ADAPTER_FILENAME}")


if __name__ == "__main__":
    main()
