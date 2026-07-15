#!/usr/bin/env python3
"""
End-to-end example: Contrastive topic vector interpretation with SelfIE.

This script demonstrates the full pipeline for interpreting what a language model
"thinks about" when processing a given prompt:

1. Download a trained SelfIE adapter and mean vector from HuggingFace Hub
2. Load the base language model (Llama-3.1-8B-Instruct)
3. Extract the residual stream activation at layer 19 for a user prompt
4. Subtract the mean vector to create a contrastive topic vector
5. Pass through the trained adapter to create a soft token embedding
6. Inject the soft token into the SelfIE template and generate descriptions

The "contrastive" part means we subtract the average activation across ~50k topics,
so the vector captures what's *distinctive* about this particular prompt rather than
what's common to all prompts.

Requirements:
    pip install selfie-adapters

Usage:
    python contrastive_topic_vector.py
    python contrastive_topic_vector.py --prompt "Tell me about the Pythagorean theorem."
    python contrastive_topic_vector.py --adapter wikipedia-scalar-affine.safetensors
"""

import argparse

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfie_adapters import load_adapter

# ── Defaults ──────────────────────────────────────────────────────────────────

HF_REPO = "keenanpepper/selfie-adapters-llama-3.1-8b-instruct"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER_IDX = 19

# SelfIE prompt template — the reserved token is where the soft token gets injected
SELFIE_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    'What is the meaning of "<|reserved_special_token_0|>"?'
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    'The meaning of "<|reserved_special_token_0|>" is "'
)
RESERVED_TOKEN = "<|reserved_special_token_0|>"


def download_from_hf(adapter_filename: str) -> tuple[str, str]:
    """Download adapter and mean vectors from HuggingFace Hub."""
    print("Downloading adapter and mean vectors from HuggingFace...")
    adapter_path = hf_hub_download(repo_id=HF_REPO, filename=adapter_filename)
    mean_vectors_path = hf_hub_download(repo_id=HF_REPO, filename="mean-vectors.safetensors")
    print(f"  Adapter: {adapter_path}")
    print(f"  Mean vectors: {mean_vectors_path}")
    return adapter_path, mean_vectors_path


def load_model(device: str) -> tuple:
    """Load the base language model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Hidden size: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def extract_hidden_state(model, tokenizer, prompt: str, layer_idx: int, device: str) -> torch.Tensor:
    """
    Extract the residual stream vector at the last token position for a given layer.

    The prompt is formatted as a chat conversation with add_generation_prompt=True,
    so the last token is at the boundary right before the assistant would start responding.
    """
    # Format as a single-turn chat conversation
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )

    # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
    hidden_state = outputs.hidden_states[layer_idx + 1][0, -1, :]
    return hidden_state


def make_contrastive(hidden_state: torch.Tensor, mean_vectors_path: str, layer_idx: int) -> torch.Tensor:
    """Subtract the mean vector to create a contrastive topic vector."""
    mean_vectors = safetensors_load_file(mean_vectors_path)
    mean_vec = mean_vectors[f"layer_{layer_idx}"]
    contrastive = hidden_state.float().cpu() - mean_vec
    return contrastive


def generate_descriptions(
    model,
    tokenizer,
    soft_token: torch.Tensor,
    num_generations: int = 5,
    max_new_tokens: int = 50,
    temperature: float = 0.5,
    device: str = "cuda",
) -> list[str]:
    """Inject a soft token into the SelfIE template and generate descriptions."""
    # Tokenize the template
    template_tokens = tokenizer(SELFIE_TEMPLATE, return_tensors="pt", add_special_tokens=False).to(device)

    # Find positions of the reserved token
    reserved_token_id = tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
    inject_positions = [
        i for i, tid in enumerate(template_tokens.input_ids[0])
        if tid.item() == reserved_token_id
    ]

    # Get template embeddings and expand for multiple generations
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        template_embeds = embed_layer(template_tokens.input_ids)
    embeddings = template_embeds.repeat(num_generations, 1, 1)

    # Replace reserved token positions with the soft token
    soft_token_cast = soft_token.to(dtype=embeddings.dtype, device=embeddings.device)
    for pos in inject_positions:
        embeddings[:, pos, :] = soft_token_cast

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract labels (text before the closing quote)
    descriptions = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()
        label = text.rsplit('"', 1)[0] if '"' in text else text
        descriptions.append(label)

    return descriptions


def main():
    parser = argparse.ArgumentParser(description="SelfIE contrastive topic vector interpretation")
    parser.add_argument("--prompt", type=str,
                        default="Tell me about the first humans to walk on the Moon.",
                        help="Prompt to interpret")
    parser.add_argument("--adapter", type=str, default="wikipedia-full-rank.safetensors",
                        help="Adapter filename (must be a wikipedia-* adapter from the HF repo)")
    parser.add_argument("--num-generations", type=int, default=5,
                        help="Number of descriptions to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f'Prompt: "{args.prompt}"')
    print(f"Adapter: {args.adapter}")
    print()

    # Step 1: Download from HuggingFace
    adapter_path, mean_vectors_path = download_from_hf(args.adapter)

    # Step 2: Load base model
    model, tokenizer = load_model(args.device)

    # Step 3: Extract hidden state at layer 19
    print(f"\nExtracting layer {LAYER_IDX} hidden state...")
    hidden_state = extract_hidden_state(model, tokenizer, args.prompt, LAYER_IDX, args.device)
    print(f"  Hidden state norm: {hidden_state.float().norm().item():.2f}")

    # Step 4: Subtract mean to get contrastive vector
    print("Creating contrastive vector...")
    contrastive_vec = make_contrastive(hidden_state, mean_vectors_path, LAYER_IDX)
    print(f"  Contrastive vector norm: {contrastive_vec.norm().item():.2f}")

    # Step 5: Load adapter and transform to soft token
    print(f"\nLoading adapter...")
    adapter = load_adapter(adapter_path, device=args.device)
    soft_token = adapter.transform(contrastive_vec.to(args.device))
    print(f"  Soft token norm: {soft_token.float().norm().item():.2f}")

    # Step 6: Generate descriptions
    print(f"\nGenerating {args.num_generations} descriptions...\n")
    descriptions = generate_descriptions(
        model, tokenizer, soft_token,
        num_generations=args.num_generations,
        device=args.device,
    )

    # Print results
    print("=" * 60)
    print(f'Prompt: "{args.prompt}"')
    print(f"Adapter: {args.adapter}")
    print()
    print("SelfIE descriptions of the contrastive topic vector:")
    print("-" * 60)
    for i, desc in enumerate(descriptions, 1):
        print(f"  {i}. {desc}")
    print("-" * 60)


if __name__ == "__main__":
    main()
