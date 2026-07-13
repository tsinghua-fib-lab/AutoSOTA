#!/usr/bin/env python3
"""
SAE Top-Activating Sequence Collector
======================================
Collects the top-activating token sequences for each feature in a sparse
autoencoder (SAE) trained on intermediate representations of a LLaMA language
model. Output is a JSON file whose structure mirrors that of the ProtoT
prototype-analysis pipeline, enabling direct comparison.

Output format
-------------
{
    "layer_0": {
        "proto_0": [
            {
                "rank": int,
                "avg_activation": float,
                "sum_activation": float,
                "perplexity": float,
                "word_count": int,
                "sentence_text": str,
                "words": [{"word": str, "activation": float, "position": int}, ...],
                "original_tokens": [{"token": str, "activation": float, "position": int}, ...]
            },
            ...
        ],
        "proto_1": [...],
        ...
    },
    ...
}

Design notes
------------
- Uses the same 4 k validation sequences as the ProtoT script for a fair
  comparison.
- Hooks ``post_attention_layernorm`` (the residual stream entering each
  transformer block), matching the hook point used during SAE training.
- Supports three feature-selection strategies (``--feature_selection``):
    * ``top_frequency``  – features with the highest mean activation (default).
    * ``top_variance``   – features with the highest activation variance.
    * ``random``         – uniform random sample (useful for sensitivity tests).
- Runs multiple random seeds and writes one JSON file per seed so that
  downstream analyses can average over seeds.
"""

import os
import json
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoConfig
from tokenizers import Tokenizer

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from data_utils import NPZDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect top-activating sequences for SAE features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str,
        default="insert/your/llama_checkpoint_directory",
        help="Path to the HuggingFace LLaMA checkpoint directory.",
    )
    parser.add_argument(
        "--sae_root", type=str,
        default="insert/your/sae_checkpoints_root_directory",
        help="Root directory containing per-layer SAE checkpoints "
             "(expected layout: <sae_root>/layer_L/seed_S/trainer_0/ae.pt).",
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="tok/fineweb_bpe_16000.json",
        help="Path to the BPE tokenizer JSON file.",
    )
    parser.add_argument(
        "--val_path", type=str,
        default="data/FineWeb/val.npz",
        help="Path to the FineWeb validation NPZ dataset.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="insert/your/output_directory",
        help="Directory where per-seed JSON output files are written.",
    )
    parser.add_argument(
        "--dev_size", type=int, default=4000,
        help="Number of validation sequences to process. "
             "Should match the ProtoT setting (4 000) for fair comparison.",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=32,
        help="Maximum token length per sequence (sequences are truncated). "
             "Should match the ProtoT setting (32).",
    )
    parser.add_argument(
        "--top_k_sentences", type=int, default=10,
        help="Number of top-activating sequences to retain per SAE feature.",
    )
    parser.add_argument(
        "--n_features_to_score", type=int, default=32,
        help="Number of SAE features to score per layer. "
             "Should match the ProtoT prototype count (R=32).",
    )
    parser.add_argument(
        "--feature_selection", type=str,
        default="top_frequency",
        choices=["top_variance", "top_frequency", "random"],
        help=(
            "Strategy for selecting which SAE features to score:\n"
            "  top_variance  – features with the highest activation variance;\n"
            "  top_frequency – features with the highest mean activation;\n"
            "  random        – random sample (for sensitivity analysis)."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds identifying SAE checkpoints to evaluate.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=12,
        help="Number of transformer layers in the model.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="PyTorch device string (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--min_tokens", type=int, default=5,
        help="Minimum number of tokens a sequence must contain to be included.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BpeTokenizer:
    """Thin wrapper around a HuggingFace ``tokenizers`` BPE tokenizer.

    Provides the ``id_to_token`` and ``clean_token`` helpers used throughout
    the pipeline. The ``</w>`` suffix convention for word boundaries is
    assumed to match the training tokenizer.
    """

    def __init__(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size: int = self.tokenizer.get_vocab_size()
        self.pad_id: int = 0

    def id_to_token(self, token_id: int) -> str:
        tok = self.tokenizer.id_to_token(token_id)
        return tok if tok is not None else "<unk>"

    def clean_token(self, tok: str) -> str:
        """Replace zero-width and non-breaking Unicode spaces with a regular space."""
        return re.sub(r'[\u00A0\u200B\u200C\u200D\uFEFF]', ' ', tok)


# ---------------------------------------------------------------------------
# Subword → word aggregation
# ---------------------------------------------------------------------------

def aggregate_subwords_to_words(
    tokens: List[str],
    activations: List[float],
    positions: List[int],
) -> Tuple[List[str], List[float], List[int]]:
    """Merge BPE subword tokens into whole words and average their activations.

    The BPE tokenizer encodes word boundaries with the ``</w>`` suffix. This
    function walks the token list and accumulates subwords until it encounters
    a word-final token, then records the completed word together with the mean
    activation across its constituent subwords.

    Parameters
    ----------
    tokens:
        Decoded token strings (may contain the ``</w>`` suffix).
    activations:
        Scalar SAE feature activation for each token position.
    positions:
        Integer token positions corresponding to ``tokens``.

    Returns
    -------
    words, word_activations, word_positions:
        Parallel lists of aggregated words, their mean activations, and the
        position of the first subword in each word.
    """
    if not tokens:
        return [], [], []

    words: List[str] = []
    word_activations: List[float] = []
    word_positions: List[int] = []

    current_word = ""
    current_acts: List[float] = []
    current_pos: List[int] = []

    for token, activation, position in zip(tokens, activations, positions):
        # Skip special tokens enclosed in angle brackets (e.g. <pad>, <unk>).
        if token.startswith('<') and token.endswith('>'):
            continue

        clean_token = token.replace('</w>', '')
        is_word_end = token.endswith('</w>')

        if not current_word:
            current_word = clean_token
            current_acts = [activation]
            current_pos = [position]
        else:
            current_word += clean_token
            current_acts.append(activation)
            current_pos.append(position)

        if is_word_end:
            if current_word.strip():
                words.append(current_word.strip())
                word_activations.append(float(np.mean(current_acts)))
                word_positions.append(current_pos[0])
            current_word, current_acts, current_pos = "", [], []

    # Flush any trailing subwords that had no word-final marker.
    if current_word and current_acts:
        words.append(current_word.strip())
        word_activations.append(float(np.mean(current_acts)))
        word_positions.append(current_pos[0])

    return words, word_activations, word_positions


def reconstruct_sentence(words: List[str]) -> str:
    """Join words with spaces and remove spaces before common punctuation marks."""
    if not words:
        return ""
    sentence = " ".join(words)
    sentence = re.sub(r'\s+([.,!?;:])', r'\1', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_sequences(
    val_path: str,
    dev_size: int,
    max_seq_len: int,
    min_tokens: int,
) -> List[List[int]]:
    """Load token sequences from the FineWeb validation NPZ file.

    Sequences shorter than ``min_tokens`` are discarded. Each sequence is
    truncated to ``max_seq_len`` tokens (the final token is consumed by the
    dataset's next-token label convention and is therefore dropped).

    Parameters
    ----------
    val_path:
        Path to the ``.npz`` dataset file.
    dev_size:
        Maximum number of datapoints to load.
    max_seq_len:
        Desired sequence length; the NPZ dataset is initialised with
        ``seq_length = max_seq_len + 1`` so that the label token can be
        stripped.
    min_tokens:
        Sequences with fewer tokens than this are skipped.

    Returns
    -------
    List of integer token-ID sequences.
    """
    print(f"Loading validation sequences from {val_path} ...")
    ds = NPZDataset(val_path, seq_length=max_seq_len + 1,
                    max_num_datapoints=dev_size)
    sequences: List[List[int]] = []
    for i in range(len(ds)):
        # The dataset yields (seq_len + 1,) tensors; drop the last token
        # (which serves as the next-token label).
        tokens = ds[i][:-1].tolist()
        if len(tokens) >= min_tokens:
            sequences.append(tokens)
    print(f"  Loaded {len(sequences)} sequences.")
    return sequences


# ---------------------------------------------------------------------------
# Model and SAE loading
# ---------------------------------------------------------------------------

def load_llama(
    model_path: str,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, AutoConfig]:
    """Load a LLaMA model in bfloat16 precision.

    Parameters
    ----------
    model_path:
        Path to a HuggingFace model directory.
    device:
        PyTorch device on which the model is placed.

    Returns
    -------
    model:
        The loaded, eval-mode LLaMA model.
    config:
        The corresponding HuggingFace configuration object.
    """
    print(f"Loading LLaMA from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    print(f"  Loaded: {config.num_hidden_layers} layers, "
          f"hidden_size={config.hidden_size}.")
    return model, config


def load_sae(
    sae_root: str,
    layer: int,
    seed: int,
    device: torch.device,
) -> Optional[AutoEncoderTopK]:
    """Load an SAE checkpoint for the given layer and seed.

    Two checkpoint layouts are supported:

    1. ``<sae_root>/layer_<L>/seed_<S>/trainer_0/ae.pt``  (primary)
    2. ``<sae_root>/layer_<L>/seed_<S>/sae_weights.pth``  (fallback)

    Parameters
    ----------
    sae_root:
        Root directory containing per-layer, per-seed subdirectories.
    layer:
        Transformer layer index.
    seed:
        Random seed index.
    device:
        PyTorch device on which the SAE is placed.

    Returns
    -------
    An ``AutoEncoderTopK`` instance, or ``None`` if no checkpoint is found.
    """
    # Primary checkpoint path.
    primary = os.path.join(sae_root, f"layer_{layer}", f"seed_{seed}",
                           "trainer_0", "ae.pt")
    if os.path.exists(primary):
        sae = AutoEncoderTopK.from_pretrained(primary, device=str(device))
        sae.eval()
        return sae.to(device)

    # Fallback: raw state-dict file.
    fallback = os.path.join(sae_root, f"layer_{layer}", f"seed_{seed}",
                            "sae_weights.pth")
    if not os.path.exists(fallback):
        print(f"  WARNING: no SAE checkpoint found for layer {layer}, seed {seed}.")
        return None

    sd = torch.load(fallback, map_location=device, weights_only=False)
    encoder_weight = sd.get("encoder.weight", sd.get("W_enc", None))
    if encoder_weight is None:
        print(f"  WARNING: cannot locate encoder weight in {fallback}.")
        return None

    activation_dim = encoder_weight.shape[1]
    dict_size = encoder_weight.shape[0]
    sae = AutoEncoderTopK(activation_dim, dict_size, k=32)
    sae.load_state_dict(sd)
    sae.eval()
    return sae.to(device)


# ---------------------------------------------------------------------------
# SAE encoding helper
# ---------------------------------------------------------------------------

def sae_encode_activations(
    sae: AutoEncoderTopK,
    x: torch.Tensor,
) -> torch.Tensor:
    """Encode a batch of hidden states and return dense feature activations.

    ``AutoEncoderTopK.encode`` may return either a dense tensor or a tuple of
    (top_activations, top_indices[, extras]). This helper normalises all
    cases into a single dense float32 tensor of shape ``[batch, dict_size]``.

    Parameters
    ----------
    sae:
        The SAE whose encoder is used.
    x:
        Input tensor of shape ``[batch, hidden_size]``.

    Returns
    -------
    Dense activation tensor of shape ``[batch, dict_size]``.
    """
    with torch.no_grad():
        out = sae.encode(x)

    if isinstance(out, torch.Tensor):
        return out.float()

    # Sparse (top-k) output: reconstruct a dense tensor.
    top_acts, top_idx = out[0], out[1]
    batch = top_acts.shape[0]
    dict_size = sae.encoder.weight.shape[0]
    dense = torch.zeros(batch, dict_size, device=x.device, dtype=torch.float32)
    dense.scatter_(1, top_idx.long(), top_acts.float())
    return dense


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_sae_activations(
    model: AutoModelForCausalLM,
    sae: AutoEncoderTopK,
    sequences: List[List[int]],
    layer_idx: int,
    tokenizer: BpeTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[List[Dict], np.ndarray]:
    """Run all validation sequences through LLaMA and the SAE at one layer.

    A forward hook is registered on the layer normalisation that precedes the
    MLP sub-layer (``post_attention_layernorm`` if present, otherwise
    ``post_feedforward_layernorm``). This is the same residual-stream position
    used during SAE training.

    For each sequence the function records:

    * The decoded token strings.
    * The sequence-level perplexity under the LLaMA language model.
    * The dense SAE feature activations at every token position.

    Parameters
    ----------
    model:
        The LLaMA language model.
    sae:
        The SAE for this layer.
    sequences:
        List of integer token-ID sequences.
    layer_idx:
        Index of the transformer layer whose residual stream is captured.
    tokenizer:
        BPE tokenizer used to map token IDs back to strings.
    device:
        PyTorch device for computation.
    batch_size:
        Number of sequences processed per forward pass.

    Returns
    -------
    records:
        One dict per input sequence containing ``token_ids``,
        ``decoded_tokens``, ``perplexity``, and ``sae_acts``
        (a ``[seq_len, dict_size]`` float32 tensor).
    acts_matrix:
        Concatenation of all per-token SAE activations as a
        ``[total_tokens, dict_size]`` numpy array, used for feature selection.
    """
    # Identify the hook point: prefer post_attention_layernorm (matches SAE
    # training), fall back to post_feedforward_layernorm.
    block = model.model.layers[layer_idx]
    hook_attr = None
    for candidate in ["post_feedforward_layernorm", "post_attention_layernorm"]:
        if hasattr(block, candidate):
            hook_attr = candidate
            break
    if hook_attr is None:
        raise RuntimeError(
            f"Cannot find a suitable hook attribute on layer {layer_idx}."
        )

    capture: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        # Capture the *input* to the layer norm, i.e. the residual stream.
        capture["residual"] = inp[0].detach()

    handle = getattr(block, hook_attr).register_forward_hook(_hook)

    all_records: List[Dict] = []
    all_acts_flat: List[torch.Tensor] = []
    pad_id = tokenizer.pad_id

    for batch_start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[batch_start: batch_start + batch_size]
        B = len(batch_seqs)
        max_len = max(len(s) for s in batch_seqs)

        # Build padded input tensors.
        input_ids = torch.full((B, max_len), pad_id,
                               dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_len),
                                     dtype=torch.long, device=device)
        for i, s in enumerate(batch_seqs):
            L = len(s)
            input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, :L] = 1

        capture.clear()
        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
        except Exception as exc:
            print(f"  Forward pass error (batch {batch_start}): {exc}")
            continue

        if "residual" not in capture:
            continue

        residual = capture["residual"].float()  # [B, S, H]

        for i, seq in enumerate(batch_seqs):
            L = len(seq)

            # Compute sequence perplexity from next-token log-likelihoods.
            if L <= 1:
                ppl = float('inf')
            else:
                tgt = input_ids[i, 1:L].unsqueeze(0)
                pred = logits[i, :L - 1, :].unsqueeze(0)
                loss = F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=pad_id,
                )
                ppl = float(np.exp(loss.item()))

            decoded = [tokenizer.clean_token(tokenizer.id_to_token(tid))
                       for tid in seq]

            seq_residual = residual[i, :L, :]
            seq_acts = sae_encode_activations(sae, seq_residual)

            all_acts_flat.append(seq_acts.cpu())
            all_records.append({
                "token_ids": seq,
                "decoded_tokens": decoded,
                "perplexity": ppl,
                "sae_acts": seq_acts.cpu(),
            })

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Processed {batch_start + B}/{len(sequences)} sequences.")

    handle.remove()

    acts_matrix = (
        torch.cat(all_acts_flat, dim=0).numpy()
        if all_acts_flat
        else np.zeros((0, 1))
    )
    return all_records, acts_matrix


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    acts_matrix: np.ndarray,
    n_features: int,
    method: str,
    dead_indices: set,
    seed: int = 0,
) -> List[int]:
    """Select a subset of SAE features to score for a given layer.

    Parameters
    ----------
    acts_matrix:
        Array of shape ``[total_tokens, dict_size]`` containing per-token
        SAE activations across the full validation set.
    n_features:
        Number of features to select.
    method:
        Selection strategy. One of:

        * ``"top_frequency"``  – features ranked by mean activation.
        * ``"top_variance"``   – features ranked by activation variance.
        * ``"random"``         – uniform random sample.
    dead_indices:
        Set of feature indices that are considered dead (never activate) and
        should be excluded from selection.
    seed:
        Random seed used only when ``method="random"``.

    Returns
    -------
    List of selected feature indices (length ``<= n_features``).
    """
    dict_size = acts_matrix.shape[1]
    live_indices = [i for i in range(dict_size) if i not in dead_indices]

    if method == "top_variance":
        scores = acts_matrix[:, live_indices].var(axis=0)
    elif method == "top_frequency":
        scores = acts_matrix[:, live_indices].mean(axis=0)
    elif method == "random":
        rng = np.random.RandomState(seed)
        return list(
            rng.choice(live_indices,
                       size=min(n_features, len(live_indices)),
                       replace=False)
        )
    else:
        raise ValueError(f"Unknown feature_selection method: '{method}'.")

    sorted_idx = np.argsort(scores)[::-1]
    return [live_indices[i] for i in sorted_idx[:n_features]]


# ---------------------------------------------------------------------------
# Building output records
# ---------------------------------------------------------------------------

def build_top_records(
    records: List[Dict],
    feature_idx: int,
    top_k: int,
) -> List[Dict]:
    """Find the top-k sequences most strongly activating a given SAE feature.

    Sequences are ranked by their mean feature activation (averaged over all
    token positions). The returned list is formatted to match the ProtoT JSON
    schema so that the two analyses can be compared directly.

    Parameters
    ----------
    records:
        Output of :func:`collect_sae_activations` — one dict per sequence.
    feature_idx:
        Index of the SAE feature to score.
    top_k:
        Number of top sequences to return.

    Returns
    -------
    List of at most ``top_k`` dicts, each containing ``rank``,
    ``avg_activation``, ``sum_activation``, ``perplexity``, ``word_count``,
    ``sentence_text``, ``words``, and ``original_tokens``.
    """
    scored = []
    for rec in records:
        acts = rec["sae_acts"][:, feature_idx].numpy()
        scored.append((float(np.mean(acts)), float(np.sum(acts)), rec, acts))

    scored.sort(key=lambda x: x[0], reverse=True)

    output_records = []
    for rank, (avg_act, sum_act, rec, acts) in enumerate(scored[:top_k]):
        tokens = rec["decoded_tokens"]
        positions = list(range(len(tokens)))
        activations = acts.tolist()

        words, word_acts, word_pos = aggregate_subwords_to_words(
            tokens, activations, positions
        )
        sentence_text = reconstruct_sentence(words)

        output_records.append({
            "rank": rank + 1,
            "avg_activation": avg_act,
            "sum_activation": sum_act,
            "perplexity": rec["perplexity"],
            "word_count": len(words),
            "sentence_text": sentence_text,
            "words": [
                {"word": w, "activation": float(a), "position": int(p)}
                for w, a, p in zip(words, word_acts, word_pos)
            ],
            "original_tokens": [
                {"token": tok, "activation": float(act), "position": int(pos)}
                for tok, act, pos in zip(tokens, activations, positions)
            ],
        })

    return output_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    separator = "=" * 60
    print(separator)
    print("SAE INTERPRETABILITY — TOP-ACTIVATING SEQUENCES")
    print(separator)
    print(f"  Validation sequences : {args.dev_size}")
    print(f"  Max sequence length  : {args.max_seq_len}")
    print(f"  Features per layer   : {args.n_features_to_score}")
    print(f"  Feature selection    : {args.feature_selection}")
    print(f"  Seeds                : {args.seeds}")
    print(f"  Top-k per feature    : {args.top_k_sentences}")
    print(separator)

    tokenizer = BpeTokenizer(args.tokenizer_path)
    sequences = load_val_sequences(
        args.val_path, args.dev_size, args.max_seq_len, args.min_tokens,
    )
    model, _ = load_llama(args.model_path, device)

    for seed in args.seeds:
        print(f"\n{separator}")
        print(f"  SEED {seed}")
        print(separator)

        output: Dict[str, Dict[str, List[Dict]]] = {}

        for layer in range(args.num_layers):
            print(f"\n  --- Layer {layer}, Seed {seed} ---")

            sae = load_sae(args.sae_root, layer, seed, device)
            if sae is None:
                continue

            dict_size = sae.encoder.weight.shape[0]
            print(f"  SAE dictionary size: {dict_size}")

            print(f"  Collecting activations for {len(sequences)} sequences ...")
            records, acts_matrix = collect_sae_activations(
                model=model,
                sae=sae,
                sequences=sequences,
                layer_idx=layer,
                tokenizer=tokenizer,
                device=device,
            )
            print(f"  Collected {len(records)} records; "
                  f"activation matrix shape: {acts_matrix.shape}.")

            # Dead-feature detection is delegated to an external stats file;
            # if none is present the set is empty and all features are eligible.
            dead_features: set = set()

            selected_features = select_features(
                acts_matrix=acts_matrix,
                n_features=args.n_features_to_score,
                method=args.feature_selection,
                dead_indices=dead_features,
                seed=seed,
            )
            print(f"  Selected {len(selected_features)} features "
                  f"via '{args.feature_selection}'.")

            layer_key = f"layer_{layer}"
            output[layer_key] = {}

            for proto_rank, feat_idx in enumerate(selected_features):
                top_recs = build_top_records(
                    records=records,
                    feature_idx=feat_idx,
                    top_k=args.top_k_sentences,
                )
                output[layer_key][f"proto_{proto_rank}"] = top_recs

            # Free GPU memory before moving to the next layer.
            del sae
            torch.cuda.empty_cache()

        out_path = os.path.join(args.output_dir, f"sae_analysis_seed_{seed}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

    # Summary.
    print(f"\n{separator}")
    print("Output files:")
    for seed in args.seeds:
        p = os.path.join(args.output_dir, f"sae_analysis_seed_{seed}.json")
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1e6
            print(f"  {p}  ({size_mb:.1f} MB)")
    print(separator)


if __name__ == "__main__":
    main()
