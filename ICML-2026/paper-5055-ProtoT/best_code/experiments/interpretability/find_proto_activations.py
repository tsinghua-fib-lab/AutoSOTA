import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import os
import re
from datetime import datetime
from pathlib import Path
import html
from typing import List, Dict, Any, Tuple
import torch.nn.functional as F
from prototype_attn import ProtoBroadcastLM
from data_utils import NPZDataset, create_causal_collate_fn
from torch.utils.data import DataLoader
import random

# --- CONFIGURATION ---
DEV_SIZE = 4000           # Number of sequences used for prototype activation analysis
MAX_SEQ_LEN = 32          # Sequences are truncated to this length
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K_SENTENCES = 10      # Top-k highest-activating sentences stored per prototype
MIN_TOKENS_PER_SENTENCE = 5
MAX_WORDS_TO_RENDER = 30  # Maximum words rendered per sentence in the HTML output
MODEL_ABL = 'original'    # Label appended to output directory / file names
SEED = 42
BATCH_SIZE = 32
AUTOCAST_DTYPE = torch.bfloat16

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- MODEL PATHS ---
LOAD_DIR = '/path/to/your/model_checkpoint_directory'  # <-- UPDATE THIS PATH

MODEL_STATE_DICT_PATH = os.path.join(LOAD_DIR, "model_state_dict.pth")
TOKENIZER_JSON_PATH   = os.path.join(LOAD_DIR, "tokenizer.json")
HALF_LIVES_JSON_PATH  = os.path.join(LOAD_DIR, "half_lives.json")
ARGS_CONFIG_PATH      = os.path.join(LOAD_DIR, "args.json")


def load_training_args(args_path):
    try:
        with open(args_path, 'r') as f:
            args = json.load(f)
        print(f"Loaded training args from {args_path}")
        return args
    except FileNotFoundError:
        print(f"Training args file not found at {args_path}, using default values")
        return {}
    except Exception as e:
        print(f"Error loading training args: {e}, using default values")
        return {}


@torch.no_grad()
def evaluate_model_perplexity(model, loader, pad_idx):
    """Evaluate cross-entropy loss and perplexity on a data loader."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in loader:
        torch.compiler.cudagraph_mark_step_begin()
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        pad_mask = (inputs == pad_idx)

        with torch.amp.autocast(device_type='cuda', dtype=AUTOCAST_DTYPE):
            logits = model(inputs, pad_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_idx,
            )
            total_loss += loss.item()
            num_batches += 1

    avg_loss   = total_loss / num_batches if num_batches > 0 else 0
    perplexity = np.exp(avg_loss) if avg_loss > 0 else float('inf')
    return avg_loss, perplexity


def load_fineweb_for_evaluation(seq_len, dev_size=None):
    """Load FineWeb data for perplexity evaluation."""
    print("Loading FineWeb dataset for evaluation...")

    dev_path = 'data/FineWeb/val.npz'
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"FineWeb dev set not found at {dev_path}")

    eval_dev_size = dev_size if dev_size is not None else 20000
    dev_ds = NPZDataset(dev_path, seq_length=seq_len + 1, max_num_datapoints=eval_dev_size)
    print(f"Loaded evaluation dataset: {len(dev_ds)} samples")
    return dev_ds


# --- BPE Tokenizer Loader ---
class BpeTokenizerLoader:
    @staticmethod
    def load_from_file(file_path: str):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(file_path)
        print(f"BPE Tokenizer loaded from {file_path}")

        class LoadedBpeTokenizer:
            def __init__(self, hf_tokenizer):
                self.tokenizer = hf_tokenizer
                self.vocab_size = hf_tokenizer.get_vocab_size()
                self.pad_id = 0

            def clean_token(self, tok: str) -> str:
                if not isinstance(tok, str):
                    return '<unk>'
                tok = re.sub(r'[ ​‌‍﻿]', ' ', tok)
                return tok

        return LoadedBpeTokenizer(tokenizer)


def aggregate_subwords_to_words(
    tokens: List[str],
    activations: List[float],
    positions: List[int],
) -> Tuple[List[str], List[float], List[int]]:
    """
    Aggregate BPE subword tokens into complete words with averaged activations.
    Uses the </w> end-of-word marker to detect word boundaries.
    """
    if not tokens:
        return [], [], []

    words, word_activations, word_positions = [], [], []
    current_word, current_activations, current_positions = "", [], []

    for token, activation, position in zip(tokens, activations, positions):
        # Skip special tokens (e.g. <pad>, <unk>)
        if token.startswith('<') and token.endswith('>'):
            continue

        clean_token = token.replace('</w>', '')
        is_word_end  = token.endswith('</w>')

        if not current_word:
            current_word        = clean_token
            current_activations = [activation]
            current_positions   = [position]
        else:
            current_word += clean_token
            current_activations.append(activation)
            current_positions.append(position)

        if is_word_end:
            if current_word.strip():
                words.append(current_word.strip())
                word_activations.append(np.mean(current_activations))
                word_positions.append(current_positions[0])
            current_word, current_activations, current_positions = "", [], []

    # Flush any trailing (non-terminated) subword
    if current_word and current_activations:
        words.append(current_word.strip())
        word_activations.append(np.mean(current_activations))
        word_positions.append(current_positions[0])

    return words, word_activations, word_positions


def reconstruct_sentence_from_words(words: List[str]) -> str:
    """Reconstruct a readable sentence from a list of word tokens."""
    if not words:
        return ""
    sentence = " ".join(words)
    sentence = re.sub(r'\s+([.,!?;:])', r'\1', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


class CustomForwardModel(nn.Module):
    """
    Thin wrapper around ProtoBroadcastLM that returns both logits and the
    per-layer routing matrices (Pi) needed for prototype activation analysis.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pad_mask=None):
        x = self.model.tok_emb(input_ids)
        x = self.model.drop(x)

        all_routing = []

        for layer_idx, block in enumerate(self.model.backbone.blocks):
            norm1_x = block.norm1(x)
            mixer_out, mixer_aux = block.mixer(
                norm1_x, pad_mask=pad_mask, return_aux=True, layer_id=layer_idx
            )

            routing_pi = None
            for key in mixer_aux:
                if 'routing_Pi' in key:
                    routing_pi = mixer_aux[key]
                    break

            if routing_pi is not None:
                all_routing.append((layer_idx, routing_pi.detach().cpu()))

            x = x + block.resid_drop1(mixer_out)
            x = x + block.resid_drop2(block.ffn(block.norm2(x)))

        x      = self.model.backbone.norm_out(x)
        logits = self.model.lm_head(x)

        return logits, all_routing


def calculate_perplexity(logits, targets, pad_mask=None):
    """Calculate per-sequence perplexity, ignoring padded positions."""
    valid_mask = ~pad_mask if pad_mask is not None else torch.ones_like(targets, dtype=torch.bool)

    logits_flat     = logits.view(-1, logits.size(-1))
    targets_flat    = targets.view(-1)
    valid_mask_flat = valid_mask.view(-1)

    ce_loss       = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    valid_ce_loss = ce_loss[valid_mask_flat]

    if len(valid_ce_loss) == 0:
        return float('inf')

    return np.exp(valid_ce_loss.mean().item())


class LightweightPrototypeDataset(Dataset):
    """
    Runs a forward pass over each sequence in token_ids_list, extracts the
    per-layer routing weights (Pi), and aggregates them to word level.
    The resulting sentence_records list can be sorted to find the
    highest-activating sentences for each (layer, prototype) pair.
    """

    def __init__(
        self,
        token_ids_list: List[List[int]],
        model,
        tokenizer,
        device='cuda',
        max_seq_len=512,
    ):
        self.token_ids_list = token_ids_list
        self.model          = CustomForwardModel(model)
        self.tokenizer      = tokenizer
        self.device         = device
        self.max_seq_len    = max_seq_len
        self.sentence_records = []
        self._process_texts()

    def _process_texts(self):
        global_sentence_id = 0
        successful_samples = 0

        for text_idx, tokens in enumerate(self.token_ids_list):
            if text_idx % 10 == 0:
                print(f"Processing {text_idx + 1}/{len(self.token_ids_list)} (successful: {successful_samples})")

            if len(tokens) < MIN_TOKENS_PER_SENTENCE:
                continue

            tokens = tokens[:self.max_seq_len]

            # Decode token IDs to string tokens
            decoded_tokens = []
            for token in tokens:
                try:
                    word = self.tokenizer.tokenizer.id_to_token(token)
                    word = "<unk>" if word is None else self.tokenizer.clean_token(word)
                    decoded_tokens.append(word)
                except Exception:
                    decoded_tokens.append("<unk>")

            # Forward pass to obtain logits and routing weights
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            pad_mask  = (input_ids == self.tokenizer.pad_id)

            with torch.no_grad():
                try:
                    logits, all_routing = self.model(input_ids, pad_mask=pad_mask)
                    if len(all_routing) == 0:
                        continue
                    successful_samples += 1

                    # Per-sequence perplexity
                    if len(tokens) > 1:
                        targets        = input_ids[:, 1:]
                        pred_logits    = logits[:, :-1, :]
                        target_pad_mask = pad_mask[:, 1:]
                        perplexity = calculate_perplexity(pred_logits, targets, target_pad_mask)
                    else:
                        perplexity = float('inf')

                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue

            # Build records from routing weights, aggregated to word level
            for layer_idx, Pi in all_routing:
                try:
                    if Pi.dim() == 4:
                        Pi_processed = Pi.squeeze(0).squeeze(1)  # [S, r]
                    elif Pi.dim() == 3:
                        Pi_processed = Pi.squeeze(0)             # [S, r]
                    else:
                        continue

                    Pi_processed   = Pi_processed.numpy()
                    seq_len        = min(Pi_processed.shape[0], len(decoded_tokens))
                    num_prototypes = Pi_processed.shape[1]

                    for proto_idx in range(num_prototypes):
                        subword_activations = []
                        subword_tokens      = []
                        subword_positions   = []

                        for pos in range(seq_len):
                            subword_activations.append(Pi_processed[pos, proto_idx])
                            subword_tokens.append(decoded_tokens[pos])
                            subword_positions.append(pos)

                        if len(subword_activations) < MIN_TOKENS_PER_SENTENCE:
                            continue

                        # Aggregate BPE subwords to whole words
                        word_tokens, word_activations, word_positions = aggregate_subwords_to_words(
                            subword_tokens, subword_activations, subword_positions
                        )

                        if not word_activations:
                            continue

                        avg_activation        = float(np.mean(word_activations))
                        sum_activation        = float(np.sum(word_activations))
                        reconstructed_sentence = reconstruct_sentence_from_words(word_tokens)

                        word_data = [
                            {'word': word, 'activation': float(act), 'position': pos}
                            for word, act, pos in zip(word_tokens, word_activations, word_positions)
                        ]

                        record = {
                            'layer_idx':         layer_idx,
                            'prototype_idx':     proto_idx,
                            'avg_activation':    avg_activation,
                            'sum_activation':    sum_activation,
                            'perplexity':        perplexity,
                            'sentence_text':     reconstructed_sentence,
                            'word_count':        len(word_activations),
                            'words':             word_data,
                            'global_sentence_id': global_sentence_id,
                            # Original subword-level data retained for reference
                            'original_tokens': [
                                {'token': tok, 'activation': float(act), 'position': pos}
                                for tok, act, pos in zip(
                                    subword_tokens, subword_activations, subword_positions
                                )
                            ],
                        }
                        self.sentence_records.append(record)

                    global_sentence_id += 1

                except Exception as e:
                    print(f"Error processing routing: {e}")
                    continue

        print(f"Finished: {successful_samples} successful samples, "
              f"{len(self.sentence_records)} records collected.")


def load_half_lives(half_lives_path: str) -> Dict[str, List[float]]:
    """
    Load half-life values from a JSON file.
    Returns an empty dict if the file is missing or unreadable so that the
    rest of the pipeline can continue without half-life data.
    """
    if not os.path.exists(half_lives_path):
        print(f"Warning: half-lives file not found at {half_lives_path} — skipping half-life data.")
        return {}
    try:
        with open(half_lives_path, 'r') as f:
            half_lives = json.load(f)
        print(f"Half-lives loaded from {half_lives_path}")
        return half_lives
    except Exception as e:
        print(f"Warning: could not load half-lives from {half_lives_path}: {e} — skipping half-life data.")
        return {}


def get_half_life_display(layer_idx: int, proto_idx: int, half_lives: Dict[str, List[float]]) -> str:
    """Return a formatted half-life string for the HTML badge, or '' if unavailable."""
    if not half_lives:
        return ""
    layer_key = f"L{layer_idx}"
    if layer_key in half_lives and proto_idx < len(half_lives[layer_key]):
        hl = half_lives[layer_key][proto_idx]
        if hl < 1:
            return f"half_life={hl:.3f}"
        elif hl < 100:
            return f"half_life={hl:.1f}"
        elif hl < 10000:
            return f"half_life={hl:.0f}"
        else:
            return f"half_life={hl:.0e}"
    return ""


def aggregate_and_sort_sentences(sentence_records: List[Dict], top_k: int = 5):
    """
    Group sentence_records by (layer, prototype) and return the top_k
    highest-activating sentences for each group.
    """
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for record in sentence_records:
        grouped[record['layer_idx']][record['prototype_idx']].append(record)

    top_sentences = defaultdict(dict)
    for layer in grouped:
        for proto in grouped[layer]:
            top_sentences[layer][proto] = sorted(
                grouped[layer][proto],
                key=lambda x: x['avg_activation'],
                reverse=True,
            )[:top_k]

    return top_sentences


def generate_safe_html(
    top_sentences: Dict,
    output_path: str,
    half_lives: Dict[str, List[float]] = None,
    overall_perplexity: float = None,
):
    """
    Write an interactive HTML file visualising the top-activating sentences
    per prototype.  Words are colour-coded from blue (low activation) to
    red (high activation).  Optional half-life badges and a perplexity
    summary panel are included when the corresponding data is available.
    """
    half_lives   = half_lives or {}
    has_half_lives = bool(half_lives)

    perplexity_section = ""
    if overall_perplexity is not None:
        perplexity_section = f"""
    <div class="overall-stats">
        <h2>Model Performance</h2>
        <p><strong>Dev Set Perplexity:</strong> <span class="highlight">{overall_perplexity:.5f}</span></p>
    </div>"""

    half_life_legend = (
        " | Half-life badges shown per prototype"
        if has_half_lives
        else " | Half-life data not available"
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Word-Level Prototype Activation Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #fafafa;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }}
        h3 {{
            color: #2980b9;
            margin: 25px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .half-life {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
        }}
        .sentence {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            border-left: 5px solid #3498db;
        }}
        .word {{
            padding: 6px 10px;
            margin: 3px 4px;
            border-radius: 6px;
            display: inline-block;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 16px;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
            white-space: nowrap;
            cursor: pointer;
        }}
        .word:hover {{
            transform: scale(1.08);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10;
            position: relative;
        }}
        .stats {{
            font-size: 0.95em;
            color: #7f8c8d;
            margin: 8px 0;
            font-weight: 500;
        }}
        .full-sentence {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 3px solid #28a745;
            font-size: 16px;
            line-height: 1.5;
            color: #2c3e50;
            font-style: italic;
        }}
        .layer {{
            margin: 40px 0;
        }}
        .prototype {{
            background: #f8f9fa;
            padding: 20px;
            margin: 25px 0;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        .highlight {{
            background-color: #fff9db;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .metric {{
            display: inline-block;
            margin-right: 15px;
        }}
        .overall-stats {{
            background: #e8f5e8;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            text-align: center;
        }}
        footer {{
            text-align: center;
            margin-top: 60px;
            color: #95a5a6;
            font-size: 0.9em;
        }}
        .word-viz {{
            font-size: 18px;
            line-height: 2.2;
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <h1>Word-Level Prototype Explorer</h1>
    {perplexity_section}
    <p style="text-align: center; color: #7f8c8d;">
        Data: <strong>FineWeb Dev Set</strong><br>
        Words colour-coded from <span style="color: #3498db;">blue (low activation)</span>
        to <span style="color: #e74c3c;">red (high activation)</span><br>
        <strong>Word-level aggregation{half_life_legend}</strong>
    </p>
"""

    for layer in sorted(top_sentences.keys()):
        html_content += f'<div class="layer"><h2>Layer {layer}</h2>'
        for proto in sorted(top_sentences[layer].keys()):
            half_life_display = get_half_life_display(layer, proto, half_lives)
            half_life_html = (
                f'<span class="half-life">{half_life_display}</span>'
                if half_life_display else ''
            )

            html_content += f'<div class="prototype"><h3>L{layer} Prototype {proto} {half_life_html}</h3>'
            for rank, record in enumerate(top_sentences[layer][proto]):
                word_acts = [w['activation'] for w in record['words']]
                if not word_acts:
                    continue

                min_act   = min(word_acts)
                max_act   = max(word_acts)
                range_act = max_act - min_act if max_act > min_act else 1.0

                words_html = ""
                for word_data in record['words'][:MAX_WORDS_TO_RENDER]:
                    act  = word_data['activation']
                    norm = (act - min_act) / range_act if range_act > 0 else 0.0
                    r    = int(220 * norm + 30)
                    b    = int(220 * (1 - norm) + 30)
                    g    = int(50 + 60 * norm)
                    color = f"rgb({r}, {g}, {b})"

                    display_word = html.escape(word_data["word"])
                    words_html += (
                        f'<span class="word" '
                        f'style="background-color: {color}; color: #000; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" '
                        f'title="Word: {display_word} | Activation: {act:.5f}">'
                        f'{display_word}</span>'
                    )

                if len(record['words']) > MAX_WORDS_TO_RENDER:
                    extra = len(record['words']) - MAX_WORDS_TO_RENDER
                    words_html += f' <span style="color: #999; font-weight: bold;">... ({extra} more words)</span>'

                ppl = record['perplexity']
                ppl_str = f"{ppl:.3f}" if ppl != float('inf') else "∞"

                html_content += f'''
                <div class="sentence">
                    <div class="stats">
                        Rank #{rank + 1} &nbsp;|&nbsp;
                        Avg: <span class="highlight">{record['avg_activation']:.5f}</span> &nbsp;|&nbsp;
                        Sum: <span class="highlight">{record['sum_activation']:.5f}</span> &nbsp;|&nbsp;
                        Perplexity: <span class="highlight">{ppl_str}</span> &nbsp;|&nbsp;
                        Words: {record['word_count']}
                    </div>
                    <div class="word-viz">{words_html}</div>
                    <div class="full-sentence">
                        <strong>Full sentence:</strong> "{record['sentence_text']}"
                    </div>
                </div>
                '''
            html_content += '</div>'
        html_content += '</div>'

    html_content += """
    <footer>Word-Level Prototype Activation Visualiser</footer>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML saved to: {output_path}")


def generate_json_output(top_sentences: Dict, output_path: str):
    """Serialise the top-sentence records to a structured JSON file."""
    serializable = {}
    for layer in sorted(top_sentences.keys()):
        serializable[f"layer_{layer}"] = {}
        for proto in sorted(top_sentences[layer].keys()):
            serializable[f"layer_{layer}"][f"proto_{proto}"] = [
                {
                    'rank':             rank + 1,
                    'avg_activation':   record['avg_activation'],
                    'sum_activation':   record['sum_activation'],
                    'perplexity':       record['perplexity'],
                    'word_count':       record['word_count'],
                    'sentence_text':    record['sentence_text'],
                    'words':            record['words'],
                    'original_tokens':  record['original_tokens'],
                }
                for rank, record in enumerate(top_sentences[layer][proto])
            ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {output_path}")


def load_fineweb_dev_set(dev_size: int = 500):
    """Load tokenised sequences from the FineWeb validation NPZ."""
    print("Loading FineWeb dev dataset...")
    dev_path = 'data/FineWeb/val.npz'
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"FineWeb dev set not found at {dev_path}")

    dev_ds = NPZDataset(dev_path, seq_length=MAX_SEQ_LEN + 1, max_num_datapoints=dev_size)

    token_ids_list = []
    for i in range(len(dev_ds)):
        tokens = dev_ds[i][:-1].tolist()
        if len(tokens) >= MIN_TOKENS_PER_SENTENCE:
            token_ids_list.append(tokens)

    print(f"Filtered to {len(token_ids_list)} sequences (≥{MIN_TOKENS_PER_SENTENCE} tokens)")
    return token_ids_list


def load_model_and_tokenizer(
    checkpoint_path: str,
    args_config_path: str,
    tokenizer_json_path: str,
    device: str = 'cuda',
):
    """Instantiate ProtoBroadcastLM and load its weights from a checkpoint."""
    with open(args_config_path, 'r') as f:
        args_config = json.load(f)

    ffn_inner = int(args_config['TF_FFN_RATIO'] * args_config['BOTTLENECK'])
    ffn_inner = (ffn_inner // 16) * 16  # Keep divisible by 16 (matches training convention)

    model_config = {
        'vocab_size':    args_config['VOCAB_SIZE'],
        'dim':           args_config['BOTTLENECK'],
        'depth':         args_config['LAYERS'],
        'r':             args_config['R'],
        'max_seq_len':   args_config['SEQ_LEN'],
        'ffn_inner_size': ffn_inner,
        'tie_weights':   args_config['TIE_HEAD'],
        'pad_id':        0,
    }

    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_json_path}")

    tokenizer = BpeTokenizerLoader.load_from_file(tokenizer_json_path)

    print(f"Model vocab: {model_config['vocab_size']}, Tokenizer vocab: {tokenizer.vocab_size}")
    assert model_config['vocab_size'] == tokenizer.vocab_size, "Vocab size mismatch!"

    model = ProtoBroadcastLM(
        vocab_size    = model_config['vocab_size'],
        dim           = model_config['dim'],
        depth         = model_config['depth'],
        r             = model_config['r'],
        max_seq_len   = model_config['max_seq_len'],
        ffn_inner_size = model_config['ffn_inner_size'],
        dropout       = 0.0,
        pad_id        = model_config.get('pad_id', tokenizer.pad_id),
        tie_weights   = model_config.get('tie_weights', True),
    ).to(device)

    # Strip torch.compile prefix if present
    raw_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in raw_state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def main():
    print("Starting word-level prototype analysis on FineWeb...")

    training_args = load_training_args(ARGS_CONFIG_PATH)

    half_lives = load_half_lives(HALF_LIVES_JSON_PATH)
    if half_lives:
        print(f"Half-life data loaded for {len(half_lives)} layers.")
    else:
        print("Running without half-life data.")

    model, tokenizer = load_model_and_tokenizer(
        MODEL_STATE_DICT_PATH,
        ARGS_CONFIG_PATH,
        TOKENIZER_JSON_PATH,
        DEVICE,
    )

    # Evaluate overall perplexity on the dev set
    print("Evaluating model perplexity on dev set...")
    eval_seq_len = training_args.get('SEQ_LEN', 512) if training_args else 512
    eval_dataset = load_fineweb_for_evaluation(eval_seq_len, dev_size=20000)
    causal_collate_fn = create_causal_collate_fn(tokenizer.pad_id, eval_seq_len)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=causal_collate_fn,
        shuffle=False,
        num_workers=0,
    )
    val_loss, overall_perplexity = evaluate_model_perplexity(model, eval_loader, tokenizer.pad_id)
    print(f"Dev set: loss={val_loss:.4f}, perplexity={overall_perplexity:.5f}")

    # Collect prototype activation records
    dev_token_ids = load_fineweb_dev_set(DEV_SIZE)
    print("Creating word-level activation dataset...")
    dataset = LightweightPrototypeDataset(dev_token_ids, model, tokenizer, DEVICE, MAX_SEQ_LEN)

    if len(dataset.sentence_records) == 0:
        print("No records collected. Try lowering MIN_TOKENS_PER_SENTENCE or check model output.")
        return

    print("Aggregating top sentences by word-level activations...")
    top_sentences = aggregate_and_sort_sentences(dataset.sentence_records, TOP_K_SENTENCES)

    output_dir = f"prototype_analysis_word_level_{MODEL_ABL}"
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(output_dir, f"prototype_visualization_word_level_{MODEL_ABL}.html")
    json_path = os.path.join(output_dir, f"prototype_analysis_word_level_{MODEL_ABL}.json")

    generate_safe_html(top_sentences, html_path, half_lives, overall_perplexity)
    generate_json_output(top_sentences, json_path)

    print(f"\nDone! Results in '{output_dir}'")
    print(f"   Perplexity : {overall_perplexity:.5f}")
    print(f"   HTML       : {html_path}")
    print(f"   JSON       : {json_path}")

    # Print a brief example for quick sanity-check
    if 0 in top_sentences and 0 in top_sentences[0]:
        ex = top_sentences[0][0][0]
        print(f"\nExample — Layer 0, Prototype 0, Rank 1")
        print(f"   Avg activation : {ex['avg_activation']:.5f}")
        print(f"   Perplexity     : {ex['perplexity']:.3f}")
        print(f"   Sentence       : \"{ex['sentence_text'][:100]}...\"")
        if half_lives:
            hl = get_half_life_display(0, 0, half_lives)
            if hl:
                print(f"   {hl}")


if __name__ == "__main__":
    main()
