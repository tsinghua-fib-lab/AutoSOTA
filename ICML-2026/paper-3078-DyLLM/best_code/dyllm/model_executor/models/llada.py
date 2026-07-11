import torch
from torch import nn
import torch.distributed as dist
from collections.abc import Iterable

from dyllm.configs import LLaDAConfig
from dyllm.utils.context import get_context
from dyllm.utils.weight_loader import AutoWeightsLoader
from dyllm.model_executor.layers.activations import SiluAndMul
from dyllm.model_executor.layers.attention import Attention
from dyllm.model_executor.layers.layernorm import RMSNorm
from dyllm.model_executor.layers.linear import (
    QKParallelLinear,
    KVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
from dyllm.model_executor.layers.rotary_embedding import get_rope
from dyllm.model_executor.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from dyllm.model_executor.layers.mlp_cache_manage import MLPcache
from dyllm.engine.cache_manager import CacheManager
from dyllm.utils.metadata import get_metadata
from dyllm.utils.util import gather_rows_2D, scatter_update_2D


class LLaDAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()
        self.cache_update = MLPcache(hidden_size)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        x = self.cache_update(x)
        return x


class LLaDAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: tuple | None = None,
        threshold: float = 0.99,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)

        self.kv_proj = KVParallelLinear(hidden_size, self.head_dim, self.total_num_kv_heads, bias=qkv_bias)

        # self.qk_proj = QKParallelLinear(
        #     hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias
        # )

        # self.v_proj = ColumnParallelLinear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)

        # self.qkv_proj = QKVParallelLinear(
        #     hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias
        # )

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False)

        self.k_cache = CacheManager(self.num_kv_heads * self.head_dim)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, threshold)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        metadata = get_metadata()

        q = self.q_proj(hidden_states)
        if ctx.is_full:
            kv = self.kv_proj(hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            kv = self.kv_proj(hidden_states[ctx.idx_salient_row])
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            if ctx.idx_salient_row_k is not None:
                k_temp = torch.zeros(
                    ctx.total_seqlen, self.num_kv_heads * self.head_dim, dtype=k.dtype, device=k.device
                )
                k_temp[ctx.idx_salient_row] = k
            else:
                k_temp = torch.zeros(
                    ctx.total_seqlen_k, self.num_kv_heads * self.head_dim, dtype=k.dtype, device=k.device
                )
                k_temp[ctx.idx_salient_row] = k
            k = k_temp

        def split_last(x, H, D):
            *prefix, _ = x.shape
            return x.view(*prefix, H, D)

        q = split_last(q, self.num_heads, self.head_dim)
        k = split_last(k, self.num_heads, self.head_dim)
        v = split_last(v, self.num_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        if ctx.is_full:
            self.k_cache.reset_full(k.flatten(-2, -1), metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)
            o = self.attn(q, k, v)
        else:
            if ctx.idx_salient_row_k is not None:
                self.k_cache.scatter_update(
                    metadata.running_seqs_tensor, ctx.idx_salient_row_k, k[ctx.idx_salient_row].flatten(-2, -1)
                )
            else:
                self.k_cache.scatter_update(
                    metadata.running_seqs_tensor, ctx.idx_salient_row, k[ctx.idx_salient_row].flatten(-2, -1)
                )
            o = self.attn(
                q, self.k_cache.get_seqs(metadata.running_seqs_tensor).view(-1, self.num_kv_heads, self.head_dim), v
            )
        output = self.o_proj(o.flatten(-2, -1))
        self.k_cache.finish(metadata.finished_seqs)
        return output


class LLaDADecoderLayer(nn.Module):
    def __init__(self, config: LLaDAConfig, threshold: float) -> None:
        super().__init__()
        self.self_attn = LLaDAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.n_kv_heads,
            max_position=config.max_sequence_length,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            threshold=threshold,
        )

        self.mlp = LLaDAMLP(hidden_size=config.hidden_size, intermediate_size=config.mlp_hidden_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        ctx = get_context()
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if not ctx.is_full:
            hidden_states = gather_rows_2D(hidden_states, ctx.idx_salient_row)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LLaDAModel(nn.Module):
    def __init__(self, config: LLaDAConfig, threshold: float):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLaDADecoderLayer(config, threshold) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LLaDAForDLM(nn.Module):
    packed_modules_mapping = {
        "kv_proj": ["k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: LLaDAConfig, threshold: float):
        super().__init__()
        self.config = config
        self.model = LLaDAModel(config, threshold)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # if config.tie_word_embeddings:
        #     self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def normalize_weight_name(self, name: str):
        name = name.replace("transformer.blocks", "layers")
        name = name.replace("transformer.", "")
        name = name.replace("wte", "embed_tokens")
        name = name.replace("ln_f", "norm")
        name = name.replace("model.ff_out", "lm_head")  # never change the order

        if "q_proj" in name or "k_proj" in name or "v_proj" in name or "attn_out" in name:
            name = name.replace("attn_out", "o_proj")
            parts = name.split(".")
            parts.insert(-2, "self_attn")
            name = ".".join(parts)
        if "ff_out" in name or "ff_proj" in name or "up_proj" in name:
            name = name.replace("ff_out", "down_proj")
            name = name.replace("ff_proj", "gate_proj")
            parts = name.split(".")
            parts.insert(-2, "mlp")
            name = ".".join(parts)

        name = name.replace("attn_norm", "input_layernorm")
        name = name.replace("ff_norm", "post_attention_layernorm")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if getattr(self.config, "weight_tying", False) else None),
        )

        def _gen():
            for name, w in weights:
                yield name, w

        return loader.load_weights(_gen())

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
