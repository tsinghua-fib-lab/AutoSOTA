import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from utils import is_flash_attn_2_available, HFDecoderAdapter, EmbeddingWithDropout, LinearWithDropout


def create_llama31_from_args(args, tok, TF_FFN_SIZE):
    """
    Build a Llama-3.1-style config from command-line arguments.
    Mapping:
      hidden_size      <- BOTTLENECK
      num_hidden_layers<- LAYERS
      num_attention_heads <- HEADS
      intermediate_size<- TF_FFN_SIZE
      max_position_embeddings <- SEQ_LEN
      vocab_size       <- VOCAB_SIZE
      pad/bos/eos ids  <- tokenizer specials
    """
    base_drop = float(args.DROPOUT)
    attn_p = float(args.ATTN_DROPOUT)

    cfg = LlamaConfig(
        vocab_size=args.VOCAB_SIZE,
        hidden_size=args.BOTTLENECK,                 # model width
        num_hidden_layers=args.LAYERS,
        num_attention_heads=args.HEADS,
        # LLaMA often uses GQA to reduce compute; set num_key_value_heads=HEADS below to disable it.
        num_key_value_heads=args.HEADS,              # set to HEADS to disable GQA (group-query-attention); setting to 1 simplifies to MQA (multi-query attention)
        intermediate_size=TF_FFN_SIZE,          # FFN width
        max_position_embeddings=args.SEQ_LEN,
        attention_dropout=attn_p,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=False,
        pad_token_id=tok.specials['<pad>'],
        bos_token_id=(tok.specials['<sos>'] if '<sos>' in tok.specials else tok.specials['<bos>']),
        eos_token_id=tok.specials['<eos>'],
        tie_word_embeddings=args.TIE_HEAD,  # Head tying is a regularizer for small datasets.
    )

    # Prefer FlashAttention-2 if available; otherwise SDPA fast path
    try:
        cfg.attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except Exception:
        cfg.attn_implementation = "sdpa"

    hf = LlamaForCausalLM(cfg)
    hf = inject_llama_dropout(
        hf,
        emb_p=base_drop,                 # embedding dropout
        mlp_intermediate_p=base_drop,    # inside SwiGLU
        mlp_output_p=base_drop,          # residual branch after MLP
        attn_output_resid_p=base_drop,   # residual branch after attention o_proj
    )
    emb = hf.get_input_embeddings()           # may be wrapped by EmbeddingWithDropout
    head = hf.get_output_embeddings()         # lm_head
    print("tie_word_embeddings:", hf.config.tie_word_embeddings)
    print("is same object:", emb.weight is head.weight)
    print("same storage:", emb.weight.data_ptr() == head.weight.data_ptr())
    # HF modules initialize in float32; autocast handles BF16 during training.
    return HFDecoderAdapter(hf, pad_token_id=tok.specials['<pad>']).to(args.DEVICE)


class LlamaMLPWithDropout(nn.Module):
    """
    Wraps HF LlamaMLP to add:
      - intermediate dropout (on gated product before down_proj)
      - output dropout (after down_proj, i.e., residual-branch dropout)
    """
    def __init__(self, mlp: nn.Module, p_intermediate: float, p_output: float):
        super().__init__()
        # Reuse original submodules/activation to keep weights
        self.gate_proj = mlp.gate_proj
        self.up_proj   = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn    = mlp.act_fn
        self.inter_drop = nn.Dropout(p_intermediate)
        self.out_drop   = nn.Dropout(p_output)
    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x))
        up    = self.up_proj(x)
        inter = gated * up                     # SwiGLU
        inter = self.inter_drop(inter)         # <--- inside SwiGLU
        out   = self.down_proj(inter)
        out   = self.out_drop(out)             # <--- residual-branch dropout
        return out

def inject_llama_dropout(
    model,
    emb_p: float = 0.0,
    mlp_intermediate_p: float = 0.0,
    mlp_output_p: float = 0.0,
    attn_output_resid_p: float = 0.0,
):
    """
    Works with transformers' LlamaForCausalLM / LlamaModel structure:
      model.model.embed_tokens -> embedding
      model.model.layers[*].self_attn.o_proj -> attention output projection
      model.model.layers[*].mlp -> MLP block
    """
    # 1) Embedding dropout
    base = getattr(model, "model", model)
    if hasattr(base, "embed_tokens") and emb_p > 0:
        if not isinstance(base.embed_tokens, EmbeddingWithDropout):
            base.embed_tokens = EmbeddingWithDropout(base.embed_tokens, emb_p)

    # 2) Per-layer injections
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError("Could not find LLaMA decoder layers at model.model.layers")

    for layer in layers:
        # 2a) Residual-branch dropout after attention’s output projection
        if attn_output_resid_p > 0:
            attn = layer.self_attn
            if not isinstance(attn.o_proj, LinearWithDropout):
                attn.o_proj = LinearWithDropout(attn.o_proj, attn_output_resid_p)

        # 2b) MLP: inside SwiGLU + residual-branch dropout
        mlp = layer.mlp
        if (mlp_intermediate_p > 0) or (mlp_output_p > 0):
            if not isinstance(mlp, LlamaMLPWithDropout):
                layer.mlp = LlamaMLPWithDropout(mlp, mlp_intermediate_p, mlp_output_p)

    return model
