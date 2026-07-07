"""Looped GPT-2 model matching paper architecture: 2 prelude + 4 recurrent + 2 coda blocks.
Effective depth = 2 + 4*tau + 2 = 132 at tau=32.  Total ~135.1M params."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class InjectionAdapter(nn.Module):
    """Projects input embeddings for injection into loop steps."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(self.proj(x))


class LoopedGPT2(nn.Module):
    """Looped GPT-2 with prelude-recurrent-coda architecture."""

    def __init__(
        self,
        vocab_size=50304,
        d_model=768,
        n_head=12,
        n_prelude=2,
        n_recurrent=4,
        n_coda=2,
        tau=32,
        seq_len=128,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.tau = tau
        self.seq_len = seq_len

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)

        self.prelude = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout) for _ in range(n_prelude)
        ])
        self.ln_prelude_out = nn.LayerNorm(d_model)

        self.injection = InjectionAdapter(d_model)

        self.recurrent = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout) for _ in range(n_recurrent)
        ])
        self.ln_recurrent_out = nn.LayerNorm(d_model)

        self.coda = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout) for _ in range(n_coda)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Untied LM head (paper model has ~135M, not 96M with tying)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        assert T <= self.seq_len

        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device).unsqueeze(0)

        tok_emb = self.wte(tokens)
        pos_emb = self.wpe(pos)
        h = tok_emb + pos_emb


        for block in self.prelude:
            h = block(h)
        h = self.ln_prelude_out(h)

        for _ in range(self.tau):
            inj_signal = self.injection(tok_emb + pos_emb)
            h = h + inj_signal
            for block in self.recurrent:
                h = block(h)
        h = self.ln_recurrent_out(h)

        for block in self.coda:
            h = block(h)
        h = self.ln_f(h)

        logits = self.lm_head(h)
        return logits

    @property
    def loop_body(self):
        return self.recurrent


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    body_total = 0
    for name, param in model.named_parameters():
        if "recurrent" in name or "injection" in name:
            body_total += param.numel()
    return {"total": total, "trainable": trainable, "body": body_total}


if __name__ == "__main__":
    model = LoopedGPT2(tau=32, seq_len=128)
    counts = count_params(model)
    t = counts["total"]
    tr = counts["trainable"]
    b = counts["body"]
    print("Total params: %s (%.1fM)" % (t, t / 1e6))
    print("Trainable:   %s (%.1fM)" % (tr, tr / 1e6))
    print("Body (recurrent+injection): %s (%.1fM)" % (b, b / 1e6))

    x = torch.randint(0, 50304, (2, 128))
    with torch.no_grad():
        y = model(x)
    print("Input: %s, Output: %s" % (x.shape, y.shape))
    print("Recurrent blocks: %d" % len(model.recurrent))
    print("Body submodules for SDI targeting:")
    for name, _ in model.named_children():
        if "recurrent" in name or "injection" in name:
            print("  - " + name)
