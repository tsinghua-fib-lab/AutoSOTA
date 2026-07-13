# robustness/models.py
import os, json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from prototype_attn import ProtoBroadcastLM
from llama_baseline import create_llama31_from_args
from mamba import MambaConfig, MambaLMAdapter
from deltanet import create_deltanet_from_args   # DeltaNet support
from data_utils import NPZDataset, create_causal_collate_fn


# ----------------------------
# Spec & Base
# ----------------------------
@dataclass
class ModelSpec:
    name: str
    kind: str        # "protoattn", "llama", "mamba", "deltanet"
    path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseCausalLM:
    def encode(self, text: str): raise NotImplementedError
    def next_token_probs(self, text: str): raise NotImplementedError
    def forward_for_ppl(self, ids: torch.Tensor): raise NotImplementedError


# ----------------------------
# Tokenizer (FineWeb BPE)
# ----------------------------
def _load_tokenizer(tok_path="tok/fineweb_bpe_16000.json"):
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tok_path))
    # ensure specials
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "<eos>"})
    if tok.bos_token is None:
        tok.add_special_tokens({"bos_token": "<s>"})
    def encode(text): return tok.encode(text, add_special_tokens=False)
    return encode, tok.vocab_size, tok


# ----------------------------
# Perplexity helpers (from run_clm.py)
# ----------------------------
@torch.no_grad()
def evaluate_model_ppl(model, npz_path="data/FineWeb/val.npz", max_num=20000, seq_len=512, batch_size=16, pad_id=0, device="cuda"):
    """
    Exact reproduction of run_clm.py evaluation loop (large-dev only, 20k tokens).
    """
    ds = NPZDataset(npz_path, seq_length=seq_len+1, max_num_datapoints=max_num)
    collate_fn = create_causal_collate_fn(pad_id, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)

    total_loss = 0.0
    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pad_mask = (inputs == pad_id)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.model(inputs, pad_mask)  # call underlying LM
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_id
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return float(np.exp(avg_loss))


def _maybe_load_logged_ppl(spec: ModelSpec):
    txt_path = os.path.join(spec.path, "final_val_ppl.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r") as f:
                val = f.read().strip()
                print(f"📄 Loaded PPL for {spec.name} from final_val_ppl.txt: {val}")
                return float(val)
        except Exception as e:
            print(f"Could not parse {txt_path}: {e}")
    return None


# ----------------------------
# Helper: match run_clm.py FFN computation
# ----------------------------
def _ffn_inner_size(cfg):
    raw = int(cfg["BOTTLENECK"] * cfg["TF_FFN_RATIO"])
    return (raw // 16) * 16


# ----------------------------
# Loader
# ----------------------------
def load_model(spec: ModelSpec, compute_full_ppl=False, force_recompute=False) -> BaseCausalLM:
    with open(f"{spec.path}/args.json") as f:
        cfg = json.load(f)

    TF_FFN_SIZE = _ffn_inner_size(cfg)

    # --- model selection ---
    if spec.kind == "protoattn":
        model = ProtoBroadcastLM(
            vocab_size=cfg["VOCAB_SIZE"],
            dim=cfg["BOTTLENECK"],
            depth=cfg["LAYERS"],
            r=cfg["R"],
            max_seq_len=cfg["SEQ_LEN"],
            ffn_inner_size=TF_FFN_SIZE,
            dropout=0.1,
            pad_id=0,
            tie_weights=cfg["TIE_HEAD"],
            w_entropy=cfg.get("W_ENTROPY", 0.0),
            w_balance=cfg.get("W_BALANCE", 0.0),
        ).to(spec.device)

    elif spec.kind == "llama":
        class DummyArgs:
            VOCAB_SIZE = cfg["VOCAB_SIZE"]
            BOTTLENECK = cfg["BOTTLENECK"]
            LAYERS = cfg["LAYERS"]
            HEADS = cfg["HEADS"]
            SEQ_LEN = cfg["SEQ_LEN"]
            TIE_HEAD = cfg["TIE_HEAD"]
            DEVICE = spec.device

        class TokStub:
            specials = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sos>": 1}

        model = create_llama31_from_args(DummyArgs, TokStub(), TF_FFN_SIZE=TF_FFN_SIZE)

    elif spec.kind == "mamba":
        mamba_cfg = MambaConfig(
            d_model=cfg["BOTTLENECK"],
            n_layers=cfg["LAYERS"],
            d_state=cfg.get("d_state", 16),
            d_conv=cfg.get("d_conv", 4),
            expand_factor=cfg.get("expand_factor", 2),
        )
        model = MambaLMAdapter(
            config=mamba_cfg,
            vocab_size=cfg["VOCAB_SIZE"],
            pad_token_id=0,
            dropout=0.0
        ).to(spec.device)

    elif spec.kind == "deltanet":
        class DummyArgs:
            VOCAB_SIZE = cfg["VOCAB_SIZE"]
            BOTTLENECK = cfg["BOTTLENECK"]
            LAYERS = cfg["LAYERS"]
            HEADS = cfg["HEADS"]
            SEQ_LEN = cfg["SEQ_LEN"]
            TIE_HEAD = cfg["TIE_HEAD"]
            DEVICE = spec.device

        class TokStub:
            specials = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sos>": 1}

        model = create_deltanet_from_args(
            args=DummyArgs,
            PAD_IDX=0,
            TF_FFN_SIZE=TF_FFN_SIZE,
            tok=TokStub(),
        )

    else:
        raise ValueError(f"Unknown kind: {spec.kind}")

    # --- load weights ---
    raw_sd = torch.load(f"{spec.path}/model_state_dict.pth", map_location="cpu")
    clean_sd = {k.replace("_orig_mod.", "").replace("hf.", ""): v for k, v in raw_sd.items()}

    if spec.kind in ["llama", "deltanet"] and hasattr(model, "hf"):
        model.hf.load_state_dict(clean_sd, strict=False)
    else:
        model.load_state_dict(clean_sd, strict=False)

    if "lm_head.weight" in clean_sd:
        with torch.no_grad():
            if hasattr(model, "lm_head"):
                model.lm_head.weight.copy_(clean_sd["lm_head.weight"])
            elif hasattr(model, "hf"):
                model.hf.lm_head.weight.copy_(clean_sd["lm_head.weight"])

    model.eval()

    # --- tokenizer ---
    encode, _, tok = _load_tokenizer()
    pad_id = tok.pad_token_id

    # --- wrapper ---
    class _M(BaseCausalLM):
        def __init__(self):
            self.pad_id = pad_id
            self.name = spec.name
            self.device = spec.device
            self.model = model
            self.encode_fn = encode

            # always load saved PPL
            saved_ppl = _maybe_load_logged_ppl(spec)
            recomputed_ppl = None

            # optionally recompute
            if compute_full_ppl:
                recomputed_ppl = evaluate_model_ppl(
                    self, max_num=20000, seq_len=cfg["SEQ_LEN"],
                    pad_id=self.pad_id, device=self.device
                )

            # decide which one to "use"
            if recomputed_ppl is not None and force_recompute:
                self.ppl = recomputed_ppl
            elif saved_ppl is not None:
                self.ppl = saved_ppl
            elif recomputed_ppl is not None:
                self.ppl = recomputed_ppl
            else:
                self.ppl = 999.0

            # always expose both
            self.saved_ppl = saved_ppl
            self.recomputed_ppl = recomputed_ppl

            # logging
            if saved_ppl is not None and recomputed_ppl is not None:
                print(f"{self.name}: saved={saved_ppl:.4f}, recomputed={recomputed_ppl:.4f}, diff={abs(saved_ppl - recomputed_ppl):.4f}")
            elif recomputed_ppl is not None:
                print(f"{self.name}: recomputed={recomputed_ppl:.4f}")
            elif saved_ppl is not None:
                print(f"{self.name}: saved={saved_ppl:.4f}")

        def encode(self, text): return self.encode_fn(text)

        @torch.no_grad()
        def next_token_probs(self, text):
            ids = torch.tensor([self.encode(text)], device=self.device)
            if spec.kind in ["protoattn", "mamba"]:
                out = self.model(ids)
                logits = out
            else:  # llama, deltanet → HF-style
                out = self.model.hf(input_ids=ids)
                logits = out.logits
            probs = torch.softmax(logits[0, -1], dim=-1)
            return probs.to(dtype=torch.float32)  # ensure float32 for numpy metrics

        @torch.no_grad()
        def forward_for_ppl(self, ids: torch.Tensor):
            if spec.kind in ["protoattn", "mamba"]:
                out = self.model(ids)
                logits = out
            else:  # llama, deltanet
                out = self.model.hf(input_ids=ids)
                logits = out.logits
            return logits.to(dtype=torch.float32)  # ensure float32 for loss/metrics

    return _M()
