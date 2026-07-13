

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from glue_data_utils import GLUEClassificationHead, GLUE_TASKS
from llama_baseline import create_llama31_from_args
from prototype_attn import ProtoBroadcastLM
from mamba import create_mamba_from_args
from deltanet import create_deltanet_from_args

class GLUEModelWrapper(nn.Module):

    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int, task_name: str):
        super().__init__()
        self.base_model = base_model
        self.task_name = task_name.lower()
        self.num_labels = num_labels
        self.classification_head = GLUEClassificationHead(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden_states = self._get_hidden_states(input_ids)
        pooled = hidden_states.mean(dim=1)
        logits = self.classification_head(pooled)

        loss = None
        if labels is not None:
            if GLUE_TASKS[self.task_name]["is_regression"]:
                loss_fct = nn.MSELoss()
                labels = labels.to(dtype=logits.dtype, device=logits.device)
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, "hf"):
            outputs = self.base_model.hf.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1]
        if hasattr(self.base_model, "mamba"):
            x_emb = self.base_model.embedding(input_ids)
            x_emb = self.base_model.dropout(x_emb)
            x_mamba = self.base_model.mamba(x_emb)
            return self.base_model.norm_f(x_mamba)
        if isinstance(self.base_model, ProtoBroadcastLM):
            return self.base_model(input_ids, return_logits=False)
        return self.base_model(input_ids)

def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _load_model_args(model_path: Path) -> SimpleNamespace:
    cfg = _load_json(model_path / "args.json")
    return SimpleNamespace(**cfg)

def _load_checkpoint(base_model: nn.Module, checkpoint_path: Path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        sanitized = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                sanitized[k[10:]] = v
            else:
                sanitized[k] = v
        state_dict = sanitized
    base_model.load_state_dict(state_dict, strict=False)

def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))

def build_glue_model(
    model_type: str,
    model_path: Path,
    tokenizer_path: Path,
    task_name: str,
) -> Tuple[GLUEModelWrapper, Tokenizer, int]:

    tokenizer = load_tokenizer(tokenizer_path)
    model_args = _load_model_args(model_path)

    if model_type == "PrototypeAttn":
        base_model = ProtoBroadcastLM(
            vocab_size=model_args.VOCAB_SIZE,
            dim=model_args.BOTTLENECK,
            depth=model_args.LAYERS,
            r=model_args.R,
            max_seq_len=model_args.SEQ_LEN,
            ffn_inner_size=1376,
            dropout=0.0,
            w_entropy=getattr(model_args, "W_ENTROPY", 0.0),
            w_balance=getattr(model_args, "W_BALANCE", 0.0),
        )
    elif model_type in {"mamba", "mamba1"}:
        pad_token_id = tokenizer.token_to_id("<pad>") or 0
        base_model = create_mamba_from_args(model_args, pad_idx=pad_token_id, dropout=0.0)
    elif model_type == "deltanet":
        class SimpleTokenizerForDeltaNet:
            def __init__(self, bpe_tokenizer: Tokenizer):
                self.tokenizer = bpe_tokenizer
                end_token = bpe_tokenizer.token_to_id("<|endoftext|>") or 0
                self.specials = {
                    "<eos>": end_token,
                    "<bos>": end_token,
                    "<sos>": end_token,
                }

            def encode(self, text: str):
                return self.tokenizer.encode(text)

        pad_token_id = tokenizer.token_to_id("<pad>") or 0
        tok_adapter = SimpleTokenizerForDeltaNet(tokenizer)
        base_model = create_deltanet_from_args(
            model_args,
            PAD_IDX=pad_token_id,
            TF_FFN_SIZE=1376,
            tok=tok_adapter,
        )
    elif model_type == "llama":
        class SimpleTokenizerForLlama:
            def __init__(self, tok: Tokenizer):
                self.bpe_tokenizer = tok
                self.specials = {"<pad>": 0, "<sos>": 1, "<bos>": 1, "<eos>": 2}

        simple_tok = SimpleTokenizerForLlama(tokenizer)
        base_model = create_llama31_from_args(model_args, simple_tok, 1376)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    _load_checkpoint(base_model, model_path / "model_state_dict.pth")

    hidden_size = model_args.BOTTLENECK
    num_labels = GLUE_TASKS[task_name]["num_labels"]
    model = GLUEModelWrapper(base_model, hidden_size, num_labels, task_name)

    if hasattr(base_model, "hf") and hasattr(base_model.hf, "model"):
        base_dtype = next(base_model.hf.model.parameters()).dtype
        model.classification_head = model.classification_head.to(dtype=base_dtype)

    return model, tokenizer, hidden_size
