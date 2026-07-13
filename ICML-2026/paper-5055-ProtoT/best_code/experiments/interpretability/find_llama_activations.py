#!/usr/bin/env python3
"""
LLaMA Attention Head Interpretability via Value Norms
======================================================
For each attention head in a LLaMA model, this script computes the L2 norm
of the per-head value vectors (‖V_head‖) at every token position.  A high
value norm at a given position indicates that the head has encoded a strong
representation for that token — i.e. it is "ready to write" a large update to
the residual stream if it attends to that position.

The script processes a held-out validation corpus, identifies the sentences
that most strongly activate each (layer, head) pair, and computes a suite of
summary statistics characterising the selectivity and sparsity of each head.

Outputs
-------
<OUTPUT_DIR>/llama_head_visualization.html
    Browser-viewable heatmap: for every (layer, head) the top-ranked
    sentences are shown with words colour-coded by their value norm.

<OUTPUT_DIR>/llama_head_features.json
    The same data in structured JSON, keyed by layer then head.

<METRICS_DIR>/llama_head_metrics.json
    Per-(layer, head) statistics: Gini coefficient, Shannon entropy,
    L1 sparsity ratio, activation density, mean L1, and mutual
    information between token identity and binned value norm.

Usage
-----
Edit the PATH CONFIGURATION and ANALYSIS SETTINGS sections below to match
your environment, then run::

    python llama_head_interpretability.py
"""

import os
import json
import re
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from sklearn.metrics import mutual_info_score

from data_utils import NPZDataset  # project-local data loader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 4257

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# PATH CONFIGURATION  ← edit these before running
# ---------------------------------------------------------------------------

# Directory containing args.json and model_state_dict.pth.
MODEL_DIR: str = "/path/to/llama_model"

# Individual file paths derived from MODEL_DIR (override if your layout differs).
ARGS_PATH: str       = os.path.join(MODEL_DIR, "args.json")
STATE_DICT_PATH: str = os.path.join(MODEL_DIR, "model_state_dict.pth")
TOKENIZER_PATH: str  = "tok/fineweb_bpe_16000.json"

# Validation corpus in NPZ format (token IDs, shape [n_sequences, seq_len+1]).
FINEWEB_VAL: str = "data/FineWeb/val.npz"

# Where visualisation outputs are written.
OUTPUT_DIR: str  = "./llama_head_analysis"

# Where per-head metric JSON is written.
METRICS_DIR: str = "./head_metrics"


# ---------------------------------------------------------------------------
# ANALYSIS SETTINGS
# ---------------------------------------------------------------------------

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Number of validation sequences to process.
DEV_SIZE: int = 4000

# Sequences are truncated to this many tokens before the forward pass.
MAX_SEQ_LEN: int = 64

# Sequences shorter than this are discarded.
MIN_TOKENS: int = 5

# Number of sequences processed per forward pass.
BATCH_SIZE: int = 16

# Number of top-activating sentences to retain per (layer, head).
TOP_SENTENCES: int = 10

# Maximum number of words rendered per sentence in the HTML output.
MAX_WORDS_RENDER: int = 30

# Set to an integer to restrict analysis to the last N layers; None = all layers.
MAX_LAYERS: Optional[int] = None

# Set to an integer to restrict analysis to the first N heads; None = all heads.
MAX_HEADS: Optional[int] = None


# ---------------------------------------------------------------------------
# Sparsity / selectivity metrics
# ---------------------------------------------------------------------------

def gini(x: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of non-negative values.

    A Gini coefficient of 0 indicates perfect uniformity (all values equal);
    1 indicates perfect concentration (all mass on a single element).  Used
    here as a measure of head selectivity across token positions.

    Parameters
    ----------
    x:
        1-D array of scalar activations.

    Returns
    -------
    Gini coefficient in [0, 1].
    """
    x = np.abs(x)
    if np.all(x == 0):
        return 0.0
    sorted_x = np.sort(x)
    n = len(sorted_x)
    ranks = np.arange(1, n + 1)
    return float(
        (2.0 * np.sum(ranks * sorted_x) / (n * np.sum(sorted_x))) - (n + 1) / n
    )


def shannon_entropy(values: np.ndarray) -> float:
    """Shannon entropy of the normalised absolute-activation distribution.

    Lower entropy indicates a more peaked (selective) distribution.

    Parameters
    ----------
    values:
        1-D array of scalar activations (need not be non-negative).

    Returns
    -------
    Entropy in nats.
    """
    absvals = np.abs(values)
    total = absvals.sum()
    if total == 0:
        return 0.0
    p = absvals / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def l1_sparsity_ratio_per_sample(samples: List[np.ndarray]) -> float:
    """Mean per-sample ratio of peak activation to mean activation.

    Formally: E_x[ max_p |a(x, p)| / mean_p |a(x, p)| ].

    A higher ratio indicates that the head fires sharply on a small number
    of positions within each sequence.  Averaging over samples makes this
    invariant to overall activation scale.

    Parameters
    ----------
    samples:
        List of 1-D arrays, one per sequence, each containing the value
        norms at every token position in that sequence.

    Returns
    -------
    Mean sparsity ratio across sequences.
    """
    ratios = []
    for acts in samples:
        absvals = np.abs(acts)
        mean_val = absvals.mean()
        if mean_val < 1e-12:
            continue
        ratios.append(float(absvals.max() / mean_val))
    return float(np.mean(ratios)) if ratios else 0.0


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class BpeWrapper:
    """Thin wrapper around a HuggingFace ``tokenizers`` BPE tokenizer.

    Exposes the two helpers needed by the analysis pipeline: converting a
    token ID to its string form and decoding a sequence of IDs to text.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tok = tokenizer
        self.vocab_size: int = tokenizer.get_vocab_size()

    def id_to_token(self, idx: int) -> str:
        try:
            t = self.tok.id_to_token(idx)
            return t if t is not None else "<unk>"
        except Exception:
            return "<unk>"

    def decode_ids(self, ids: List[int]) -> str:
        return self.tok.decode(ids)


def clean_bpe_token(tok: str) -> str:
    """Replace zero-width and non-breaking Unicode spaces with a regular space.

    The ``</w>`` word-boundary marker is preserved so that downstream
    aggregation can correctly detect word boundaries.
    """
    if not isinstance(tok, str):
        return "<unk>"
    return re.sub(r"[\u00A0\u200B\u200C\u200D\uFEFF]", " ", tok)


# ---------------------------------------------------------------------------
# Subword → word aggregation
# ---------------------------------------------------------------------------

def aggregate_tokens_to_words(
    tokens: List[str],
    values_per_metric: Dict[str, List[float]],
    positions: List[int],
) -> Tuple[List[str], Dict[str, List[float]], List[int]]:
    """Merge BPE subword tokens into whole words and average their metric values.

    Tokens produced by a BPE tokenizer with the ``</w>`` convention are
    accumulated until a word-final token (one ending in ``</w>``) is reached,
    at which point the completed word is recorded together with the mean of
    each metric across its constituent subwords.

    Parameters
    ----------
    tokens:
        Decoded token strings for one sequence (may contain ``</w>``).
    values_per_metric:
        Mapping from metric name to a list of per-token scalar values.
        All lists must have the same length as ``tokens``.
    positions:
        Integer token positions (0-indexed) corresponding to ``tokens``.

    Returns
    -------
    words:
        Reconstructed whole-word strings.
    word_values:
        Mapping from metric name to a list of per-word mean values.
    word_positions:
        Position of the first subword in each reconstructed word.
    """
    if not tokens:
        return [], {k: [] for k in values_per_metric}, []

    words: List[str] = []
    word_positions: List[int] = []
    word_values: Dict[str, List[float]] = {k: [] for k in values_per_metric}

    cur_word = ""
    cur_positions: List[int] = []
    cur_vals: Dict[str, List[float]] = {k: [] for k in values_per_metric}

    for tok, pos in zip(tokens, positions):
        tok = clean_bpe_token(tok)
        is_word_end = tok.endswith("</w>")
        base = tok.replace("</w>", "")

        if not cur_word:
            cur_word = base
            cur_positions = [pos]
            for k in values_per_metric:
                cur_vals[k] = [values_per_metric[k][pos]]
        else:
            cur_word += base
            cur_positions.append(pos)
            for k in values_per_metric:
                cur_vals[k].append(values_per_metric[k][pos])

        if is_word_end:
            if cur_word.strip():
                words.append(cur_word.strip())
                word_positions.append(cur_positions[0])
                for k in values_per_metric:
                    word_values[k].append(float(np.mean(cur_vals[k])))
            cur_word = ""
            cur_positions = []
            cur_vals = {k: [] for k in values_per_metric}

    # Flush any trailing subwords that had no word-final marker.
    if cur_word and any(len(v) > 0 for v in cur_vals.values()):
        words.append(cur_word.strip())
        word_positions.append(cur_positions[0])
        for k in values_per_metric:
            word_values[k].append(float(np.mean(cur_vals[k])))

    return words, word_values, word_positions


def reconstruct_sentence(words: List[str]) -> str:
    """Join words with spaces and remove spaces before common punctuation."""
    if not words:
        return ""
    s = " ".join(words)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_args(args_path: str) -> Dict[str, Any]:
    with open(args_path, "r") as fh:
        return json.load(fh)


def _build_llama_config(args_json: Dict[str, Any]) -> LlamaConfig:
    """Construct a ``LlamaConfig`` from the training ``args.json``.

    The intermediate FFN size is rounded down to the nearest multiple of 16
    to match the rounding applied during model training.
    """
    dim        = args_json["BOTTLENECK"]
    n_layers   = args_json["LAYERS"]
    n_heads    = args_json["HEADS"]
    vocab_size = args_json["VOCAB_SIZE"]
    max_seq    = args_json["SEQ_LEN"]
    ffn_ratio  = args_json.get("TF_FFN_RATIO", 2.7)

    intermediate_size = (int(ffn_ratio * dim) // 16) * 16

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,   # no grouped-query attention
        intermediate_size=intermediate_size,
        max_position_embeddings=max_seq,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=False,
    )
    # Force eager (non-fused) attention so that intermediate tensors are
    # accessible to forward hooks.
    cfg.attn_implementation = "eager"
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    return cfg


def load_model_and_tokenizer() -> Tuple[LlamaForCausalLM, BpeWrapper, int, int, int]:
    """Load the LLaMA model and BPE tokenizer from disk.

    The state dict is cleaned to remove prefixes introduced by
    ``torch.compile`` (``_orig_mod.``) and a custom HF adapter (``hf.``),
    and to rename ``*.lin.weight`` keys to ``*.weight`` as expected by
    the HuggingFace LLaMA implementation.

    Returns
    -------
    model:
        Eval-mode LLaMA model on ``DEVICE``.
    tok_wrap:
        BPE tokenizer wrapper.
    hidden_dim:
        Model hidden dimension.
    num_layers:
        Number of transformer layers.
    num_heads:
        Number of attention heads per layer.
    """
    print(f"Loading model configuration from {ARGS_PATH} ...")
    args_json = _load_args(ARGS_PATH)
    cfg = _build_llama_config(args_json)

    print("Instantiating LLaMA model (eager attention) ...")
    model = LlamaForCausalLM(cfg)

    print(f"Loading state dict from {STATE_DICT_PATH} ...")
    raw_state = torch.load(STATE_DICT_PATH, map_location="cpu")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in raw_state.items():
        k = k.removeprefix("_orig_mod.").removeprefix("hf.")
        k = k.replace(".lin.weight", ".weight")
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing key(s) in state dict — "
              "those parameters will be randomly initialised.")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected key(s) in state dict "
              "— they will be ignored.")

    model.to(DEVICE)
    model.eval()

    print(f"Loading BPE tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tok_wrap = BpeWrapper(tokenizer)

    hidden_dim = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    num_heads  = cfg.num_attention_heads
    print(f"  Loaded: {num_layers} layers, hidden_dim={hidden_dim}, "
          f"num_heads={num_heads}.")
    return model, tok_wrap, hidden_dim, num_layers, num_heads


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_validation_dataset() -> NPZDataset:
    """Load the FineWeb validation corpus.

    Each item in the dataset is a 1-D tensor of ``MAX_SEQ_LEN + 1`` token
    IDs; the trailing token is the next-token label and is stripped before
    processing.
    """
    print(f"Loading validation dataset from {FINEWEB_VAL} ...")
    ds = NPZDataset(
        FINEWEB_VAL,
        seq_length=MAX_SEQ_LEN + 1,
        max_num_datapoints=DEV_SIZE,
    )
    print(f"  Dataset size: {len(ds)} sequences.")
    return ds


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_head_features_batch(
    model: LlamaForCausalLM,
    batch_token_ids: List[List[int]],
    tok_wrap: BpeWrapper,
    layers_to_use: List[int],
    heads_to_use: List[int],
    num_heads: int,
    metrics_acts: Dict[int, Dict[int, List[np.ndarray]]],
    metrics_tokens: Dict[int, Dict[int, List[int]]],
) -> List[Dict[str, Any]]:
    """Extract per-head value norms for a batch of sequences.

    A forward hook is registered on ``v_proj`` for each selected layer.
    The hook captures the raw value vectors before attention weighting,
    which are then split into per-head slices and L2-normed per token.

    The per-sample norm arrays and token IDs are accumulated in
    ``metrics_acts`` and ``metrics_tokens`` respectively for later
    statistical analysis.

    Parameters
    ----------
    model:
        The LLaMA model (eval mode, on ``DEVICE``).
    batch_token_ids:
        List of integer token-ID sequences for this batch.
    tok_wrap:
        BPE tokenizer wrapper used to recover token strings.
    layers_to_use:
        Layer indices for which hooks are registered.
    heads_to_use:
        Head indices to process within each hooked layer.
    num_heads:
        Total number of attention heads in the model.
    metrics_acts:
        Accumulator: ``metrics_acts[layer][head]`` is a list of 1-D numpy
        arrays, one per sequence, containing per-token value norms.
    metrics_tokens:
        Accumulator: ``metrics_tokens[layer][head]`` is a flat list of
        token IDs aligned with the concatenation of ``metrics_acts``.

    Returns
    -------
    List of record dicts, one per (sequence, layer, head) triple, each
    containing ``layer``, ``head``, ``avg_val_norm``, ``words``, and
    ``sentence``.
    """
    if not batch_token_ids:
        return []

    B = len(batch_token_ids)
    max_len = max(len(s) for s in batch_token_ids)

    input_ids = torch.full(
        (B, max_len), fill_value=0, dtype=torch.long, device=DEVICE,
    )
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=DEVICE)

    for i, ids in enumerate(batch_token_ids):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=DEVICE)
        # Token ID 0 is the pad token by FineWeb BPE convention.
        attention_mask[i, :L] = (input_ids[i, :L] != 0).long()

    # Register hooks to capture v_proj outputs for the selected layers.
    captures_v: Dict[int, torch.Tensor] = {}

    def make_v_hook(layer_idx: int):
        def _hook(module, inp, out):
            # out: [B, S, hidden_size] — raw value projections.
            captures_v[layer_idx] = out.detach()
        return _hook

    handles = []
    for layer_idx in layers_to_use:
        h = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(
            make_v_hook(layer_idx)
        )
        handles.append(h)

    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in handles:
        h.remove()

    hidden_size = model.config.hidden_size
    head_dim    = hidden_size // num_heads

    all_records: List[Dict[str, Any]] = []

    for b_idx in range(B):
        ids     = batch_token_ids[b_idx]
        seq_len = len(ids)
        tokens  = [tok_wrap.id_to_token(int(t)) for t in ids]
        positions = list(range(seq_len))

        for layer_idx in layers_to_use:
            if layer_idx not in captures_v:
                continue

            # Slice out this sequence, cast to float32, reshape to per-head.
            v_layer = captures_v[layer_idx][b_idx, :seq_len].float()  # [S, D]
            v_layer = v_layer.view(seq_len, num_heads, head_dim)       # [S, H, d]

            for head_idx in heads_to_use:
                v_head = v_layer[:, head_idx, :]                       # [S, d]
                val_norm = torch.norm(v_head, dim=-1).cpu().numpy()    # [S]

                # Accumulate for metric computation.
                metrics_acts[layer_idx][head_idx].append(val_norm.copy())
                metrics_tokens[layer_idx][head_idx].extend(ids)

                words, word_values, word_positions = aggregate_tokens_to_words(
                    tokens,
                    {"val_norm": val_norm.tolist()},
                    positions,
                )
                if not words:
                    continue

                all_records.append({
                    "layer":        layer_idx,
                    "head":         head_idx,
                    "avg_val_norm": float(np.mean(word_values["val_norm"])),
                    "words": [
                        {"word": w, "val_norm": float(vn), "position": int(p)}
                        for w, vn, p in zip(
                            words, word_values["val_norm"], word_positions
                        )
                    ],
                    "sentence": reconstruct_sentence(words),
                })

    return all_records


# ---------------------------------------------------------------------------
# HTML visualisation
# ---------------------------------------------------------------------------

def make_html(
    results: Dict[int, Dict[int, List[Dict[str, Any]]]],
    output_path: str,
) -> None:
    """Write a self-contained HTML heatmap visualisation to ``output_path``.

    For each (layer, head) pair the top-ranked sentences are rendered with
    each word coloured on a blue-to-red scale according to its normalised
    value norm.

    Parameters
    ----------
    results:
        Nested dict ``results[layer][head]`` → list of top sentence records.
    output_path:
        Filesystem path for the output HTML file.
    """
    page = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>LLaMA Head V-Norm Explorer</title>
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 40px;
    background: #fafafa;
    color: #333;
}
h1 { text-align: center; color: #2c3e50; }
h2 {
    margin-top: 40px;
    color: #34495e;
    border-bottom: 2px solid #3498db;
    padding-bottom: 4px;
}
h3 { margin-top: 24px; color: #2980b9; }
.head-box {
    background: #fff;
    padding: 16px;
    margin-top: 16px;
    border-radius: 10px;
    border: 1px solid #eee;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
}
.sentence-box {
    background: #fff;
    padding: 14px;
    margin-top: 14px;
    border-radius: 8px;
    border: 1px solid #f0f0f0;
}
.stats { font-size: 12px; color: #555; margin-bottom: 8px; }
.row-title {
    font-size: 12px; color: #666;
    margin-top: 6px; margin-bottom: 4px; font-weight: 600;
}
.word-row {
    font-size: 14px; line-height: 2.0;
    padding: 6px 8px; border-radius: 6px;
    background: #f7f7f7; margin-bottom: 4px;
}
.word {
    display: inline-block;
    padding: 4px 6px; margin: 2px;
    border-radius: 4px; font-size: 14px; cursor: default;
}
.full-sentence {
    margin-top: 10px; padding: 10px;
    border-radius: 6px; background: #f0f0f0; font-size: 14px;
}
</style>
</head>
<body>
<h1>LLaMA Attention Head Value-Norm Explorer</h1>
<p style="text-align:center;color:#666;">
  Word-level heatmap of <strong>per-head value norms</strong> (&#x2016;V<sub>head</sub>&#x2016;).<br>
  Blue = low norm &nbsp;&#x2192;&nbsp; Red = high norm. Top-ranked sentences per (layer, head).
</p>
"""

    for layer in sorted(results):
        page += f"<h2>Layer {layer}</h2>\n"
        for head in sorted(results[layer]):
            recs = results[layer][head]
            page += "<div class='head-box'>\n"
            page += f"<h3>Head {head}</h3>\n"

            for rank, r in enumerate(recs, 1):
                words = r["words"]
                if not words:
                    continue

                norms = [w["val_norm"] for w in words]
                lo, hi = min(norms), max(norms)
                rng = (hi - lo) or 1.0

                page += "<div class='sentence-box'>\n"
                page += (
                    f"<div class='stats'>Rank #{rank} &nbsp; "
                    f"avg_val_norm={r['avg_val_norm']:.4f}</div>\n"
                )
                page += "<div class='row-title'>Value norm (&#x2016;V<sub>head</sub>&#x2016;)</div>\n"
                page += "<div class='word-row'>\n"

                for w in words[:MAX_WORDS_RENDER]:
                    norm = (w["val_norm"] - lo) / rng
                    R = int(40 + 215 * norm)
                    G = int(60 + 40  * norm)
                    B = int(40 + 215 * (1 - norm))
                    color = f"rgb({R},{G},{B})"
                    page += (
                        f"<span class='word' style='background:{color};' "
                        f"title='val_norm={val_norm:.5f}'>"
                        f"{word_text}</span> "
                    )

                if len(words) > MAX_WORDS_RENDER:
                    extra = len(words) - MAX_WORDS_RENDER
                    page += (
                        f"<span style='color:#999;font-size:12px;'>"
                        f"... +{extra} more</span>"
                    )
                page += "</div>\n"
                page += (
                    f"<div class='full-sentence'>"
                    f"{html.escape(r['sentence'])}</div>\n"
                )
                page += "</div>\n"  # sentence-box

            page += "</div>\n"  # head-box

    page += "</body></html>\n"

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"  HTML saved to {output_path}")


# ---------------------------------------------------------------------------
# Per-head metric computation
# ---------------------------------------------------------------------------

def compute_head_metrics(
    metrics_acts: Dict[int, Dict[int, List[np.ndarray]]],
    metrics_tokens: Dict[int, Dict[int, List[int]]],
    layers_to_use: List[int],
    heads_to_use: List[int],
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Compute summary statistics for every (layer, head) pair.

    Metrics
    -------
    L0_zero_fraction
        Fraction of token positions where the value norm is exactly zero.
    activation_density
        Fraction of token positions where the value norm is positive.
    mean_L1_activation
        Mean absolute value norm across all token positions.
    gini
        Gini coefficient of the value-norm distribution (0 = uniform,
        1 = fully concentrated).
    entropy
        Shannon entropy of the normalised absolute-activation distribution.
    l1_sparsity_ratio
        Mean per-sequence ratio of peak to mean value norm.
    mutual_info_token
        Mutual information (in nats) between token identity and binned
        value norm — a measure of token-type selectivity.
    mutual_info_normalized
        Mutual information normalised by log(unique tokens), giving a
        value in [0, 1].

    Parameters
    ----------
    metrics_acts:
        ``metrics_acts[layer][head]`` → list of per-sequence value-norm arrays.
    metrics_tokens:
        ``metrics_tokens[layer][head]`` → flat list of token IDs aligned with
        the concatenation of ``metrics_acts[layer][head]``.
    layers_to_use, heads_to_use:
        Layer and head indices to process.

    Returns
    -------
    Nested dict ``results[layer][head]`` → metric dict.
    """
    results: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for layer in layers_to_use:
        results[layer] = {}
        print(f"\n  Layer {layer}")

        for head in heads_to_use:
            acts_list = metrics_acts[layer][head]
            toks_list = metrics_tokens[layer][head]
            if not acts_list or not toks_list:
                continue

            acts_flat = np.concatenate(acts_list)
            toks      = np.array(toks_list, dtype=np.int32)

            zero_frac = float(np.mean(acts_flat == 0))
            density   = float(np.mean(acts_flat > 0))
            l1        = float(np.mean(np.abs(acts_flat)))
            g         = gini(acts_flat)
            entropy   = shannon_entropy(acts_flat)
            sparsity  = l1_sparsity_ratio_per_sample(acts_list)

            # Discretise value norms into 20 bins for mutual information.
            bins     = np.histogram(acts_flat, bins=20)[1]
            act_bins = np.digitize(acts_flat, bins)
            mi       = float(mutual_info_score(toks, act_bins))
            n_uniq   = len(np.unique(toks))
            mi_norm  = float(mi / np.log(n_uniq)) if n_uniq > 1 else 0.0

            print(
                f"    Head {head:>2}: entropy={entropy:.4f}  gini={g:.4f}  "
                f"sparsity={sparsity:.2f}  density={density:.4f}  "
                f"MI_norm={mi_norm:.4f}  n={len(acts_list)}"
            )

            results[layer][head] = {
                "L0_zero_fraction":    zero_frac,
                "activation_density":  density,
                "mean_L1_activation":  l1,
                "gini":                g,
                "entropy":             entropy,
                "l1_sparsity_ratio":   sparsity,
                "mutual_info_token":   mi,
                "mutual_info_normalized": mi_norm,
                "num_samples":         int(len(acts_list)),
            }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    model, tok_wrap, hidden_dim, num_layers, num_heads = load_model_and_tokenizer()
    ds = load_validation_dataset()

    # Determine which layers and heads to analyse.
    all_layers = list(range(num_layers))
    layers_to_use = (
        all_layers if MAX_LAYERS is None
        else all_layers[max(0, num_layers - MAX_LAYERS):]
    )

    all_heads = list(range(num_heads))
    heads_to_use = (
        all_heads if MAX_HEADS is None
        else all_heads[:min(MAX_HEADS, num_heads)]
    )

    print(f"Analysing layers: {layers_to_use}")
    print(f"Analysing heads:  {heads_to_use}")

    # Accumulators for per-head metric computation.
    metrics_acts: Dict[int, Dict[int, List[np.ndarray]]] = {
        L: {H: [] for H in heads_to_use} for L in layers_to_use
    }
    metrics_tokens: Dict[int, Dict[int, List[int]]] = {
        L: {H: [] for H in heads_to_use} for L in layers_to_use
    }

    all_records: List[Dict[str, Any]] = []
    batch: List[List[int]] = []

    print("\nExtracting per-head value norms ...")
    for idx in tqdm(range(len(ds)), desc="Sequences"):
        token_ids = ds[idx].tolist()[:-1][:MAX_SEQ_LEN]
        if len(token_ids) < MIN_TOKENS:
            continue

        batch.append(token_ids)

        if len(batch) == BATCH_SIZE:
            all_records.extend(
                extract_head_features_batch(
                    model, batch, tok_wrap,
                    layers_to_use, heads_to_use, num_heads,
                    metrics_acts, metrics_tokens,
                )
            )
            batch = []

    if batch:
        all_records.extend(
            extract_head_features_batch(
                model, batch, tok_wrap,
                layers_to_use, heads_to_use, num_heads,
                metrics_acts, metrics_tokens,
            )
        )

    print(f"  Collected {len(all_records)} (layer, head, sequence) records.")

    # Group records and select top sentences per (layer, head).
    grouped: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for r in all_records:
        grouped.setdefault(r["layer"], {}).setdefault(r["head"], []).append(r)

    results: Dict[int, Dict[int, List[Dict[str, Any]]]] = {
        L: {
            H: sorted(grouped.get(L, {}).get(H, []),
                      key=lambda x: x["avg_val_norm"], reverse=True)[:TOP_SENTENCES]
            for H in heads_to_use
        }
        for L in layers_to_use
    }

    # Write outputs.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html_path = os.path.join(OUTPUT_DIR, "llama_head_visualization.html")
    json_path = os.path.join(OUTPUT_DIR, "llama_head_features.json")

    make_html(results, html_path)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"  Features JSON saved to {json_path}")

    # Compute and save per-head statistics.
    print("\nComputing per-head metrics ...")
    head_metrics = compute_head_metrics(
        metrics_acts, metrics_tokens, layers_to_use, heads_to_use,
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "llama_head_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(head_metrics, fh, indent=2)
    print(f"  Head metrics saved to {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    
