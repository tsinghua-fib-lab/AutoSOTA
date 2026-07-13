import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import HFDecoderAdapter, EmbeddingWithDropout, LinearWithDropout
# DeltaNet imports for FLA integration
DELTANET_IMPORT_ERROR = None
try:
    from fla.models.delta_net import DeltaNetConfig
    from transformers import AutoModelForCausalLM
    DELTANET_AVAILABLE = True
except Exception as exc:
    DELTANET_IMPORT_ERROR = exc
    print(f"⚠️  FLA DeltaNet not available: {exc}")
    DELTANET_AVAILABLE = False


class DeltaMLPWithIntermediateDropout(nn.Module):
    """
    Adds dropout inside the MLP:
      - SwiGLU layout:  drop(gate_proj(x).act * up_proj(x)) -> down_proj
      - FFN layout:     drop(act(fc1(x))) -> fc2
    It mirrors the original submodules to preserve parameter names in the state_dict.
    """
    def __init__(self, mlp: nn.Module, p_intermediate: float):
        super().__init__()
        self.inter_drop = nn.Dropout(p_intermediate)

        # Try common layouts in order of likelihood
        if all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
            # LLaMA-style SwiGLU
            self._layout = "swi"
            self.gate_proj = mlp.gate_proj
            self.up_proj   = mlp.up_proj
            self.down_proj = mlp.down_proj
            self.act_fn = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_fn", None) or F.silu

        elif all(hasattr(mlp, a) for a in ("fc1", "fc2")):
            # Plain FFN
            self._layout = "ffn"
            self.fc1 = mlp.fc1
            self.fc2 = mlp.fc2
            self.act_fn = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_fn", None) or F.silu

        elif all(hasattr(mlp, a) for a in ("in_proj", "out_proj")):
            # Some implementations expose in/out proj with an activation
            self._layout = "proj"
            self.in_proj  = mlp.in_proj
            self.out_proj = mlp.out_proj
            self.act_fn = getattr(mlp, "act_fn", None) or getattr(mlp, "activation_fn", None) or F.silu

        else:
            # Unknown layout: keep the original as a submodule and just forward (no intermediate dropout)
            self._layout = "unknown"
            self.inner = mlp
            print("[DeltaMLPWithIntermediateDropout] Unknown MLP layout; "
                  "intermediate dropout will be skipped for these blocks.")

    def forward(self, x=None, *args, **kwargs):
        # HF sometimes passes 'hidden_states=' instead of positional
        if x is None:
            x = kwargs.pop("hidden_states", None)
        if x is None:
            raise TypeError("DeltaMLPWithIntermediateDropout.forward expected 'x' or 'hidden_states'")

        # Ignore generation-time extras unused by this wrapper.
        kwargs.pop("cache_position", None)
        kwargs.pop("position_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("layer_idx", None)
        kwargs.pop("residual", None)

        if self._layout == "swi":
            gated = self.act_fn(self.gate_proj(x))
            up    = self.up_proj(x)
            inter = self.inter_drop(gated * up)
            return self.down_proj(inter)

        elif self._layout == "ffn":
            inter = self.inter_drop(self.act_fn(self.fc1(x)))
            return self.fc2(inter)

        elif self._layout == "proj":
            inter = self.inter_drop(self.act_fn(self.in_proj(x)))
            return self.out_proj(inter)

        else:
            # Preserve original behavior/signature for unknown layouts
            return self.inner(x, *args, **kwargs)


def create_deltanet_from_args(args, PAD_IDX, TF_FFN_SIZE, tok):
    """
    Build a DeltaNet model from args using FLA (Flash Linear Attention) library.
    Uses HF AutoModelForCausalLM with DeltaNetConfig for compatibility.
    Note: DeltaNet is not compatible with torch.compile() due to CUDA graphs issues.
    """
    if not DELTANET_AVAILABLE:
        raise ImportError("DeltaNet requires a working flash-linear-attention installation") from DELTANET_IMPORT_ERROR
    
    base_drop = float(args.DROPOUT)

    config = DeltaNetConfig(
        vocab_size=args.VOCAB_SIZE,
        hidden_size=args.BOTTLENECK,           # Match the dimension used by other models
        num_hidden_layers=args.LAYERS,         # Same number of layers
        max_position_embeddings=args.SEQ_LEN,  # Context length
        pad_token_id=PAD_IDX,
        bos_token_id=(tok.specials['<sos>'] if '<sos>' in tok.specials else tok.specials['<bos>']),
        eos_token_id=tok.specials['<eos>'],
        tie_word_embeddings=args.TIE_HEAD,
        # DeltaNet-specific parameters
        intermediate_size=TF_FFN_SIZE,  # FFN size
        norm_eps=1e-6,              # RMSNorm epsilon
        use_cache=False,  # Disable for training
        num_heads=args.HEADS,          # Attention heads
    )
    
    # Create HF model using AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_config(config)

    # Inject dropout:
    hf_model = inject_deltanet_dropout(
        hf_model,
        emb_p=base_drop,
        mlp_intermediate_p=base_drop,   # <--- inside-FFN
        mlp_output_p=base_drop,         # residual-branch after down_proj
        attn_output_resid_p=base_drop,  # residual-branch after o_proj
    )
    
    # DeltaNet requires bfloat16 - convert model to bfloat16
    if args.DEVICE == 'cuda':
        hf_model = hf_model.to(dtype=torch.bfloat16)
    
    # Wrap in HFDecoderAdapter for compatibility with training loop
    adapter = HFDecoderAdapter(hf_model, pad_token_id=PAD_IDX).to(args.DEVICE)

    # Mark this model as incompatible with torch.compile
    adapter._disable_compile = True
    
    return adapter


def inject_deltanet_dropout(
    model,
    emb_p: float = 0.0,
    mlp_intermediate_p: float = 0.0,  # <--- inside-FFN dropout
    mlp_output_p: float = 0.0,        # residual-branch dropout after MLP down_proj
    attn_output_resid_p: float = 0.0  # residual-branch dropout after attention o_proj
):
    base = getattr(model, "model", model)

    # 1) Embedding dropout (handle common names)
    if emb_p > 0:
        for emb_name in ("embed_tokens", "embeddings", "wte", "tok_embeddings"):
            if hasattr(base, emb_name):
                emb_mod = getattr(base, emb_name)
                if not isinstance(emb_mod, EmbeddingWithDropout) and isinstance(emb_mod, nn.Embedding):
                    setattr(base, emb_name, EmbeddingWithDropout(emb_mod, emb_p))
                break

    # Helper to wrap a linear with dropout if present
    def _wrap_linear(module, attr, p):
        if p <= 0 or not hasattr(module, attr):
            return False
        lin = getattr(module, attr)
        if isinstance(lin, LinearWithDropout):
            return False
        if isinstance(lin, nn.Linear):
            setattr(module, attr, LinearWithDropout(lin, p))
            return True
        return False

    # 2) Find layers
    layers = getattr(base, "layers", None)
    if layers is None:
        # heuristic fallback: collect blocks that look like (attn|mixer) + (mlp|ffn)
        candidates = []
        for m in base.modules():
            has_attn = any(hasattr(m, nm) for nm in ("self_attn", "attn", "mixer"))
            has_mlp  = any(hasattr(m, nm) for nm in ("mlp", "ffn"))
            if has_attn and has_mlp:
                candidates.append(m)
        # de-dup
        seen, layers = set(), []
        for m in candidates:
            if id(m) not in seen:
                layers.append(m); seen.add(id(m))

    if not layers:
        raise AttributeError("Could not find DeltaNet-style layers; expected base.layers or blocks with attn+mlp")

    wrapped_attn = wrapped_mlp_out = wrapped_mlp_mid = 0

    for layer in layers:
        # -- Attention output (residual-branch) dropout
        attn = next((getattr(layer, n) for n in ("self_attn", "attn", "mixer") if hasattr(layer, n)), None)
        if attn is not None:
            for proj_name in ("o_proj", "out_proj", "proj_out", "linear_out"):
                if _wrap_linear(attn, proj_name, attn_output_resid_p):
                    wrapped_attn += 1
                    break

        # -- MLP / FFN module
        mlp_name = next((n for n in ("mlp", "ffn") if hasattr(layer, n)), None)
        if mlp_name is None:
            continue
        mlp = getattr(layer, mlp_name)

        # First: residual-branch dropout after the final MLP projection
        for proj_name in ("down_proj", "out_proj", "fc2", "proj"):
            if _wrap_linear(mlp, proj_name, mlp_output_p):
                wrapped_mlp_out += 1
                break

        # Then: inside-FFN (intermediate) dropout
        if mlp_intermediate_p > 0 and not isinstance(mlp, DeltaMLPWithIntermediateDropout):
            wrapped = DeltaMLPWithIntermediateDropout(mlp, mlp_intermediate_p)
            setattr(layer, mlp_name, wrapped)
            wrapped_mlp_mid += 1

    print(f"[inject_deltanet_dropout] attn-out wrapped: {wrapped_attn} layers | "
          f"mlp-out wrapped: {wrapped_mlp_out} layers | "
          f"mlp-intermediate wrapped: {wrapped_mlp_mid} layers")

    return model
