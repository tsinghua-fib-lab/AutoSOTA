# probes/copy_target.py

import torch
from torch.utils.data import DataLoader
from .base import ProbeBase
from .registry import register_probe
import torch.distributed as dist


def _find_match_p(tokens: torch.Tensor, t: int, induction_match: str, match_choice: str) -> int:
    """
    Find the previous occurrence of the key token before position t,
    and return the token immediately following that occurrence as the induction target.

    Args:
        tokens:          1D token tensor
        t:               current position
        induction_match: "previous" (key = tokens[t-1]) or "current" (key = tokens[t])
        match_choice:    "last" or "first"
    Returns:
        target token id, or -1 if no valid match found
    """
    if induction_match == "previous":
        if t == 0:
            return -1
        key  = int(tokens[t - 1].item())
        left = tokens[:t - 1]
    else:  # "current"
        key  = int(tokens[t].item())
        left = tokens[:t]

    pos = (left == key).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return -1

    match_pos = int(pos[-1].item() if match_choice == "last" else pos[0].item())
    next_pos  = match_pos + 1
    if next_pos >= len(tokens):
        return -1
    return int(tokens[next_pos].item())


def _compute_loss(model, tokens: torch.Tensor, induction_match: str, match_choice: str, device) -> torch.Tensor:
    """
    Forward pass and compute sum of log-probs of induction match targets.
    """
    logits    = model(tokens)
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    seq       = tokens[0]
    loss      = torch.tensor(0.0, device=device, requires_grad=True)
    for t in range(1, seq.size(0)):
        target = _find_match_p(seq, t, induction_match, match_choice)
        if target != -1:
            loss = loss + log_probs[t - 1, target]
    return loss


@register_probe("copy_target_synthetic")
class CopyTargetSyntheticProbe(ProbeBase):
    """
    Induction/copy-target probe using synthetic repeated sequences.
    Matches the original find_sample_for_induction.py implementation.
    """

    def build_dataloader(self, cfg: dict) -> None:
        return None

    def compute_grad(self, model, cfg: dict, mode: str) -> tuple:
        pcfg  = cfg["probe"]
        layer = cfg["target"]["layer"]
        head  = cfg["target"]["head"]
        seq_len         = pcfg["seq_len"]
        num_samples     = pcfg["num_samples"]
        induction_match = pcfg["induction_match"]
        match_choice    = pcfg["match_choice"]

        device = next(model.parameters()).device
        inner  = model.module if hasattr(model, "module") else model
        attn   = inner.blocks[layer].attn

        if mode == "qk":
            v_acc = torch.zeros(inner.cfg.d_model, 2 * inner.cfg.d_head, device=device, dtype=torch.float32)

            for _ in range(num_samples):
                half   = seq_len // 2
                prefix = torch.randint(0, inner.cfg.d_vocab, (half,), device=device)
                tokens = torch.cat([prefix, prefix]).unsqueeze(0)

                with torch.enable_grad():
                    loss = _compute_loss(inner, tokens, induction_match, match_choice, device)

                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=[attn.W_Q, attn.W_K],
                    retain_graph=False, create_graph=False, allow_unused=False
                )
                gq = grads[0][head].float()
                gk = grads[1][head].float()
                v_acc.add_(torch.cat([gq, gk], dim=-1))
            

            if dist.is_initialized():
                dist.all_reduce(v_acc, op=dist.ReduceOp.SUM)
                v_acc /= dist.get_world_size()
            return (v_acc / num_samples,)

        else:  # qkvo
            v_acc_qk = torch.zeros(inner.cfg.d_model, 2 * inner.cfg.d_head, device=device, dtype=torch.float32)
            v_acc_v  = torch.zeros(inner.cfg.d_model, inner.cfg.d_head,     device=device, dtype=torch.float32)
            v_acc_o  = torch.zeros(inner.cfg.d_head,  inner.cfg.d_model,    device=device, dtype=torch.float32)

            for _ in range(num_samples):
                half   = seq_len // 2
                prefix = torch.randint(0, inner.cfg.d_vocab, (half,), device=device)
                tokens = torch.cat([prefix, prefix]).unsqueeze(0)

                with torch.enable_grad():
                    loss = _compute_loss(inner, tokens, induction_match, match_choice, device)

                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=[attn.W_Q, attn.W_K, attn.W_V, attn.W_O],
                    retain_graph=False, create_graph=False, allow_unused=False
                )
                gq = grads[0][head].float()
                gk = grads[1][head].float()
                gv = grads[2][head].float()
                go = grads[3][head].float()
                v_acc_qk.add_(torch.cat([gq, gk], dim=-1))
                v_acc_v.add_(gv)
                v_acc_o.add_(go)

            if dist.is_initialized():
                dist.all_reduce(v_acc_qk, op=dist.ReduceOp.SUM)
                dist.all_reduce(v_acc_v,  op=dist.ReduceOp.SUM)
                dist.all_reduce(v_acc_o,  op=dist.ReduceOp.SUM)
                v_acc_qk /= dist.get_world_size()
                v_acc_v  /= dist.get_world_size()
                v_acc_o  /= dist.get_world_size()
            return (v_acc_qk / num_samples, v_acc_v / num_samples, v_acc_o / num_samples)


@register_probe("copy_target_dataset")
class CopyTargetDatasetProbe(ProbeBase):
    """
    Induction/copy-target probe using real dataset sequences.
    """

    def build_dataloader(self, cfg: dict) -> DataLoader:
        from data.loader import build_dataloaders   
        _, dl_influence = build_dataloaders(cfg)
        return dl_influence

    def compute_grad(self, model, cfg: dict, mode: str) -> tuple:
        pcfg  = cfg["probe"]
        layer = cfg["target"]["layer"]
        head  = cfg["target"]["head"]
        induction_match = pcfg["induction_match"]
        match_choice    = pcfg["match_choice"]

        device = next(model.parameters()).device
        inner  = model.module if hasattr(model, "module") else model
        attn   = inner.blocks[layer].attn
        dl     = self.build_dataloader(cfg)

        if mode == "qk":
            v_acc = torch.zeros(inner.cfg.d_model, 2 * inner.cfg.d_head, device=device, dtype=torch.float32)
            count = 0

            for batch, _ in dl:
                batch = batch.to(device)
                for i in range(batch.size(0)):
                    tokens = batch[i].unsqueeze(0)

                    with torch.enable_grad():
                        loss = _compute_loss(inner, tokens, induction_match, match_choice, device)

                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=[attn.W_Q, attn.W_K],
                        retain_graph=False, create_graph=False, allow_unused=False
                    )
                    gq = grads[0][head].float()
                    gk = grads[1][head].float()
                    v_acc.add_(torch.cat([gq, gk], dim=-1))
                    count += 1

            if dist.is_initialized():
                dist.all_reduce(v_acc, op=dist.ReduceOp.SUM)
                v_acc /= dist.get_world_size()
            return (v_acc / count,)

        else:  # qkvo
            v_acc_qk = torch.zeros(inner.cfg.d_model, 2 * inner.cfg.d_head, device=device, dtype=torch.float32)
            v_acc_v  = torch.zeros(inner.cfg.d_model, inner.cfg.d_head,     device=device, dtype=torch.float32)
            v_acc_o  = torch.zeros(inner.cfg.d_head,  inner.cfg.d_model,    device=device, dtype=torch.float32)
            count = 0

            for batch, _ in dl:
                batch = batch.to(device)
                for i in range(batch.size(0)):
                    tokens = batch[i].unsqueeze(0)

                    with torch.enable_grad():
                        loss = _compute_loss(inner, tokens, induction_match, match_choice, device)

                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=[attn.W_Q, attn.W_K, attn.W_V, attn.W_O],
                        retain_graph=False, create_graph=False, allow_unused=False
                    )
                    gq = grads[0][head].float()
                    gk = grads[1][head].float()
                    gv = grads[2][head].float()
                    go = grads[3][head].float()
                    v_acc_qk.add_(torch.cat([gq, gk], dim=-1))
                    v_acc_v.add_(gv)
                    v_acc_o.add_(go)
                    count += 1

            if dist.is_initialized():
                dist.all_reduce(v_acc_qk, op=dist.ReduceOp.SUM)
                dist.all_reduce(v_acc_v,  op=dist.ReduceOp.SUM)
                dist.all_reduce(v_acc_o,  op=dist.ReduceOp.SUM)
                v_acc_qk /= dist.get_world_size()
                v_acc_v  /= dist.get_world_size()
                v_acc_o  /= dist.get_world_size()
            return (v_acc_qk / count, v_acc_v / count, v_acc_o / count)

