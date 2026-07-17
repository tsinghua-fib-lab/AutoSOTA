from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast


# ----------------- Config -----------------
class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        hidden_act="silu",
        bos_token_id=151643,
        eos_token_id=151643,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        use_cache=True,
        max_window_layers=24,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        use_sliding_window=False,
        use_mrope=False,
        sliding_window=32768,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.max_window_layers = max_window_layers
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.use_mrope = use_mrope
        self.sliding_window = sliding_window

        # Store torch dtype as torch.dtype object
        # Accept string like "bfloat16" or actual torch dtype
        if isinstance(torch_dtype, str):
            self.torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(torch_dtype.lower(), torch.float32)
        else:
            self.torch_dtype = torch_dtype


AutoConfig.register("qwen2", Qwen2Config)


# ----------------- RMSNorm -----------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS normalization
        rms = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        x_normed = x / (rms + self.eps)
        return x_normed * self.weight


# ----------------- Attention -----------------
class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.head_dim * self.num_key_value_heads  # e.g. 64 * 2 = 128

        # Query projection shape: hidden_size -> kv_dim * num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.kv_dim * self.num_heads, bias=False)
        # Key and Value combined projection: hidden_size -> kv_dim * num_heads * 2
        self.kv_proj = nn.Linear(self.hidden_size, self.kv_dim * self.num_heads * 2, bias=False)
        self.out_proj = nn.Linear(self.kv_dim * self.num_heads, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
        self.rope_theta = config.rope_theta

    def _apply_rope(self, x, seq_len, device):
        # Rotary positional embeddings (RoPE) applied to last dimension of x
        dim = x.size(-1)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Apply rotation
        x_out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x_out = x_out.flatten(-2)
        return x_out

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        bsz, seq_len, _ = hidden_states.size()
        device = hidden_states.device

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.kv_dim)
        kv = self.kv_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.kv_dim * 2)
        k, v = kv.split(self.kv_dim, dim=-1)

        # Apply RoPE to query and key
        q = self._apply_rope(q, seq_len, device)
        k = self._apply_rope(k, seq_len, device)

        # Rearrange for attention computation
        q = q.permute(0, 2, 1, 3)  # [batch, heads, seq_len, kv_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Concatenate past key/values if available (for caching during generation)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.kv_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # [bsz, heads, seq_len, kv_dim]

        # Rearrange output back to [bsz, seq_len, hidden_size]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.num_heads * self.kv_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, present_key_value


# ----------------- MLP -----------------
class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


# ----------------- Transformer Block -----------------
class Qwen2Block(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        def _attn_forward(x):
            x_norm = self.norm1(x)
            attn_out, present = self.attn(x_norm, attention_mask, past_key_value, use_cache)
            return attn_out, present

        # Use gradient checkpointing to save memory during training
        if self.training and torch.is_grad_enabled():
            attn_out, present = checkpoint(_attn_forward, x)
        else:
            attn_out, present = _attn_forward(x)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present


# ----------------- Model -----------------
class Qwen2ForCausalLM(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "qwen2"

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False  # default off
        self.use_cache = config.use_cache

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.tie_weights()

        self.to(dtype=config.torch_dtype)

    def tie_weights(self):
        # Tie embeddings and lm_head weights
        self.lm_head.weight = self.embed_tokens.weight

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,  # add labels here
    ):
        if input_ids is None:
            raise ValueError("You must provide input_ids")

        bsz, seq_len = input_ids.size()
        device = input_ids.device

        if use_cache is None:
            use_cache = self.use_cache

        hidden_states = self.embed_tokens(input_ids).to(dtype=self.config.torch_dtype)

        present_key_values = ()
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return layer(*inputs, attention_mask=attention_mask, use_cache=use_cache)

                hidden_states, present = checkpoint(custom_forward, hidden_states, past)
            else:
                hidden_states, present = layer(hidden_states, attention_mask=attention_mask, past_key_value=past, use_cache=use_cache)

            if use_cache:
                present_key_values = present_key_values + (present,)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        if not return_dict:
            outputs = (logits,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if use_cache:
                outputs = outputs + (present_key_values,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


# ----------------------------
# Register config and model for Auto classes
# ----------------------------

AutoConfig.register("qwen2", Qwen2Config)
AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM)
