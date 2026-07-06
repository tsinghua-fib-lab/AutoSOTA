# probes/prev_attn.py

import numpy as np
import torch
import torch.distributed as dist
from typing import Optional
from .base import ProbeBase
from .registry import register_probe
from model.hooks import PatternCache, setup_pattern_hooks


@register_probe("prev_attn")
class PrevAttnProbe(ProbeBase):
    """
    Previous-token attention probe.
    Probe function: f = sum over sequences of sum_t A[t, t-1],
    where A is the attention pattern of the target head.
    Matches the original find_sample_for_prev.py implementation.
    """

    def build_dataloader(self, cfg: dict) -> None:
        return None

    @staticmethod
    def _generate_random_sequences(
        num_seqs: int,
        seq_len: int,
        vocab_size: int,
        avoid_ids: Optional[set],
        seed: int
    ) -> torch.Tensor:
        """
        Generate random token sequences, excluding special tokens.
        Uses numpy RNG for reproducibility across ranks.
        Returns LongTensor [num_seqs, seq_len].
        """
        rng     = np.random.default_rng(seed)
        allowed = np.array(
            [i for i in range(vocab_size) if (avoid_ids is None or i not in avoid_ids)],
            dtype=np.int64
        )
        if allowed.size == 0:
            raise ValueError("Allowed vocab after exclusions is empty.")
        out = rng.choice(allowed, size=(num_seqs, seq_len), replace=True)
        return torch.from_numpy(out.astype(np.int64))

    def compute_grad(self, model, cfg: dict, mode: str) -> tuple:
        pcfg  = cfg["probe"]
        layer = cfg["target"]["layer"]
        head  = cfg["target"]["head"]
        seq_len     = pcfg["seq_len"]
        num_samples = pcfg["num_samples"]
        seed        = pcfg.get("seed", 42)

        device = next(model.parameters()).device
        inner  = model.module if hasattr(model, "module") else model

        rank  = dist.get_rank()       if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        seed_rank = seed + rank

        avoid = set()
        if hasattr(inner, "tokenizer"):
            tok = inner.tokenizer
            if getattr(tok, "pad_token_id", None) is not None:
                avoid.add(int(tok.pad_token_id))
            if getattr(tok, "eos_token_id", None) is not None:
                avoid.add(int(tok.eos_token_id))

        vocab   = inner.cfg.d_vocab
        local_n = int(np.ceil(num_samples / world))
        synth   = self._generate_random_sequences(local_n, seq_len, int(vocab), avoid, seed_rank)

        attn      = inner.blocks[layer].attn
        pat_cache = PatternCache()
        hooks     = setup_pattern_hooks(inner, layer, pat_cache)

        if mode == "qk":
            v_acc = torch.zeros(inner.cfg.d_model, 2 * inner.cfg.d_head, device=device, dtype=torch.float32)

            try:
                for b in range(synth.shape[0]):
                    pat_cache.clear()
                    tokens = synth[b:b + 1].to(device, non_blocking=True)

                    with torch.enable_grad():
                        _ = model(tokens)

                    if pat_cache.pattern is None:
                        raise RuntimeError("hook_pattern not captured.")

                    head_pat = pat_cache.pattern[0, head]  # [seq_len, seq_len]
                    f = torch.diagonal(head_pat, offset=-1).sum()

                    grads = torch.autograd.grad(
                        outputs=f,
                        inputs=[attn.W_Q, attn.W_K],
                        retain_graph=False, create_graph=False, allow_unused=False
                    )
                    gq = grads[0][head].float()  # [d_model, d_head]
                    gk = grads[1][head].float()
                    v_acc.add_(torch.cat([gq, gk], dim=-1))

            finally:
                try:
                    inner.reset_hooks(hooks)
                except Exception:
                    try:
                        inner.remove_all_hook_fns()
                    except Exception:
                        pass

            if dist.is_initialized():
                dist.all_reduce(v_acc, op=dist.ReduceOp.SUM)

            return (v_acc / num_samples,)

        else:  # qkvo
            v_acc_qk = torch.zeros(inner.cfg.d_model, 2 * inner.cfg.d_head, device=device, dtype=torch.float32)
            v_acc_v  = torch.zeros(inner.cfg.d_model, inner.cfg.d_head,     device=device, dtype=torch.float32)
            v_acc_o  = torch.zeros(inner.cfg.d_head,  inner.cfg.d_model,    device=device, dtype=torch.float32)

            try:
                for b in range(synth.shape[0]):
                    pat_cache.clear()
                    tokens = synth[b:b + 1].to(device, non_blocking=True)

                    with torch.enable_grad():
                        _ = model(tokens)

                    if pat_cache.pattern is None:
                        raise RuntimeError("hook_pattern not captured.")

                    head_pat = pat_cache.pattern[0, head]
                    f = torch.diagonal(head_pat, offset=-1).sum()

                    params = [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]
                    grads = torch.autograd.grad(
                        outputs=f,
                        inputs=params,
                        retain_graph=False, create_graph=False, allow_unused=True
                    )
                    grads = [g if g is not None else torch.zeros_like(p)
                            for g, p in zip(grads, params)]

                    gq = grads[0][head].float()
                    gk = grads[1][head].float()
                    gv = grads[2][head].float()
                    go = grads[3][head].float()


            finally:
                try:
                    inner.reset_hooks(hooks)
                except Exception:
                    try:
                        inner.remove_all_hook_fns()
                    except Exception:
                        pass

            if dist.is_initialized():
                dist.all_reduce(v_acc_qk, op=dist.ReduceOp.SUM)
                dist.all_reduce(v_acc_v,  op=dist.ReduceOp.SUM)
                dist.all_reduce(v_acc_o,  op=dist.ReduceOp.SUM)

            return (v_acc_qk / num_samples, v_acc_v / num_samples, v_acc_o / num_samples)


