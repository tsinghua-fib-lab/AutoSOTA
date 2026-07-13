#!/usr/bin/env python3
"""
Train Sparse Autoencoders on the post-attention residual stream of LLaMA.

Key design choices:
  - Hooks post-attention residual stream gives contextualized representations comparable to ProtoT.
  - Padding tokens excluded from SAE training via attention_mask filtering.
  - Dead-feature tracking + reconstruction loss logged every log_steps.
  - Per-(layer, seed) metrics saved to metrics.jsonl + summary.json.
  - Training timestamps and duration recorded in summary.json.
  - Overall run_report.json written at the end.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from data_utils import NPZDataset


###############################################################################
# CONFIG / ARGS
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train sparse autoencoders on the post-attention residual stream "
            "of LLaMA. One SAE per layer, streamed on-the-fly."
        )
    )

    parser.add_argument("--device", type=str, default="cuda:0")

    # --- Data ---
    parser.add_argument(
        "--fineweb", type=str,
        default="data/FineWeb/train.npz",
        help="Path to training NPZ of token IDs.",
    )
    parser.add_argument(
        "--target_tokens", type=int, default=100_000_000,
        help="Approximate number of tokens to stream per (layer, seed). Default: 100M.",
    )
    parser.add_argument(
        "--seq_length", type=int, default=256,
        help="Sequence length used during LLM forward passes.",
    )

    # --- Model ---
    parser.add_argument("--model_path", type=str, default="./llama_model/hf_export")

    # --- SAE hyperparams ---
    parser.add_argument(
        "--dict_mult", type=int, default=8,
        help="SAE dictionary size = hidden_size * dict_mult. Default: 8.",
    )
    parser.add_argument(
        "--k_sparse", type=int, default=32,
        help="TopK sparsity: number of active features per token. Default: 32.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # --- Seeds ---
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds for SAE training. Default: 0 1 2.",
    )

    # --- Batch sizes ---
    parser.add_argument("--llm_batch_size", type=int, default=16,
                        help="Sequences per LLM forward pass.")
    parser.add_argument("--sae_batch_size", type=int, default=2048,
                        help="Tokens per SAE training step.")

    # --- Layer range ---
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=None,
                        help="Inclusive. Defaults to last layer.")

    # --- Output ---
    parser.add_argument(
        "--save_root", type=str, default="./sae_outputs/residual",
        help="Root directory. Structure: save_root/layer_L/seed_S/",
    )
    parser.add_argument("--log_steps", type=int, default=500)

    return parser.parse_args()


###############################################################################
# DATASET
###############################################################################

class TokenSequenceDataset(IterableDataset):
    """
    Iterable over fixed-length token sequences from an NPZ file.
    Cycles indefinitely so the SAE trainer can run for as many steps
    as needed regardless of dataset size.
    """

    def __init__(self, npz_path: str, seq_length: int, min_tokens: int = 5):
        super().__init__()
        self.npz_path = npz_path
        self.seq_length = seq_length
        self.min_tokens = min_tokens

    def __iter__(self):
        while True:
            ds = NPZDataset(self.npz_path, seq_length=self.seq_length)
            for sample in ds:
                ids = sample.tolist()
                if len(ids) < self.min_tokens:
                    continue
                yield torch.tensor(ids[: self.seq_length], dtype=torch.long)


###############################################################################
# RESIDUAL STREAM ACTIVATION GENERATOR
###############################################################################

@torch.no_grad()
def residual_stream_iterator(
    model: AutoModelForCausalLM,
    dataset: IterableDataset,
    layer_idx: int,
    device: torch.device,
    llm_batch_size: int,
    sae_batch_size: int,
    target_tokens: int,
) -> Iterator[torch.Tensor]:
    """
    Yields [sae_batch_size, hidden_size] float32 tensors of the
    post-attention residual stream at `layer_idx`.

    Padding positions are excluded via attention_mask before tokens
    are added to the buffer, so the SAE never sees pad activations.

    Hook target: input of the FFN layernorm (= post-attention residual).
    Tries 'post_feedforward_layernorm' then 'post_attention_layernorm',
    falls back to block.mlp with a warning.
    """
    config = model.config
    hidden_size = config.hidden_size

    tokens_yielded = 0
    token_buffer: list[torch.Tensor] = []
    iterator = iter(dataset)
    capture: dict = {}

    def hook_fn(module, inp, out):
        # inp[0]: [B, S, hidden_size] — input to the post-attention layernorm
        capture["residual"] = inp[0].detach()

    block = model.model.layers[layer_idx]
    ffn_norm = None
    for attr in ["post_feedforward_layernorm", "post_attention_layernorm"]:
        if hasattr(block, attr):
            ffn_norm = getattr(block, attr)
            break
    if ffn_norm is None:
        ffn_norm = block.mlp
        print(f"  [Layer {layer_idx}] WARNING: using MLP hook as fallback.")

    handle = ffn_norm.register_forward_hook(hook_fn)

    try:
        while tokens_yielded < target_tokens:
            # Collect one LLM batch
            seqs = []
            while len(seqs) < llm_batch_size:
                seqs.append(next(iterator))

            max_len = max(s.size(0) for s in seqs)
            B = len(seqs)
            pad_id = getattr(config, "pad_token_id", None) or 0

            input_ids = torch.full(
                (B, max_len), fill_value=pad_id, dtype=torch.long, device=device,
            )
            attention_mask = torch.zeros(
                (B, max_len), dtype=torch.long, device=device,
            )
            for i, s in enumerate(seqs):
                L = s.size(0)
                input_ids[i, :L] = s.to(device)
                attention_mask[i, :L] = 1

            capture.clear()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            if "residual" not in capture:
                continue

            res = capture["residual"]       # [B, S, H]
            mask = attention_mask.bool()    # [B, S]

            # Exclude padding positions — flatten and keep only real tokens
            flat_res  = res.reshape(-1, hidden_size)    # [B*S, H]
            flat_mask = mask.reshape(-1)                # [B*S]
            flat = flat_res[flat_mask].to(dtype=torch.float32)  # [real_tokens, H]

            token_buffer.append(flat.cpu())
            tokens_yielded += flat.size(0)

            # Yield full SAE-sized chunks
            combined = torch.cat(token_buffer, dim=0)
            n = combined.size(0)
            start = 0
            while start + sae_batch_size <= n:
                yield combined[start: start + sae_batch_size].to(device)
                start += sae_batch_size
            token_buffer = [combined[start:]] if start < n else []

    finally:
        handle.remove()

    # Yield any remaining tokens
    if token_buffer:
        leftover = torch.cat(token_buffer, dim=0)
        n = leftover.size(0)
        for start in range(0, n, sae_batch_size):
            chunk = leftover[start: start + sae_batch_size]
            if chunk.size(0) > 0:
                yield chunk.to(device)


###############################################################################
# METRICS TRACKER
###############################################################################

class SAEMetricsTracker:
    """
    Tracks SAE training metrics and writes them to <save_dir>/metrics.jsonl.

    Metrics logged every log_steps steps (windowed average):
      step, recon_loss, relative_recon_loss, l0, n_dead, frac_dead, tokens_seen

    relative_recon_loss = recon_loss / input_variance  (scale-free, cross-layer comparable)
    l0 = mean active features per token (should be ~= k for TopK)
    frac_dead = fraction of features that have never fired since run start

    File is flushed after every window so you can `tail -f metrics.jsonl` live.
    """

    def __init__(self, dict_size: int, device: torch.device, save_dir: str, log_steps: int):
        self.dict_size = dict_size
        self.device = device
        self.log_steps = log_steps
        self.metrics_path = os.path.join(save_dir, "metrics.jsonl")

        self._recon_loss_sum = 0.0
        self._input_var_sum  = 0.0
        self._l0_sum         = 0.0
        self._steps_in_window = 0

        self.ever_fired = torch.zeros(dict_size, dtype=torch.bool, device=device)
        self.tokens_seen = 0

        self._fh = open(self.metrics_path, "w")

    def update(
        self,
        step: int,
        batch: torch.Tensor,        # [B, H]
        recon: torch.Tensor,        # [B, H]
        top_indices: torch.Tensor,  # [B, k] or flat [B*k]
    ):
        B = batch.size(0)
        self.tokens_seen += B

        recon_loss = (batch - recon).pow(2).mean().item()
        input_var  = batch.var(dim=0).mean().item() + 1e-8

        if top_indices.dim() == 2:
            l0 = float(top_indices.size(1))
        else:
            l0 = top_indices.numel() / B

        self.ever_fired[top_indices.reshape(-1)] = True

        self._recon_loss_sum  += recon_loss
        self._input_var_sum   += input_var
        self._l0_sum          += l0
        self._steps_in_window += 1

        if step > 0 and step % self.log_steps == 0:
            self._flush(step)

    def _flush(self, step: int):
        n = self._steps_in_window
        if n == 0:
            return

        avg_recon = self._recon_loss_sum / n
        avg_var   = self._input_var_sum  / n
        avg_l0    = self._l0_sum         / n
        n_dead    = int((~self.ever_fired).sum().item())
        frac_dead = n_dead / self.dict_size

        record = {
            "step":                step,
            "recon_loss":          round(avg_recon, 6),
            "relative_recon_loss": round(avg_recon / avg_var, 6),
            "l0":                  round(avg_l0, 2),
            "n_dead":              n_dead,
            "frac_dead":           round(frac_dead, 4),
            "tokens_seen":         self.tokens_seen,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

        print(
            f"    step={step:>7,} | "
            f"recon={avg_recon:.4f} | "
            f"rel_recon={avg_recon / avg_var:.4f} | "
            f"L0={avg_l0:.1f} | "
            f"dead={frac_dead:.1%} ({n_dead}/{self.dict_size}) | "
            f"tokens={self.tokens_seen:,}"
        )

        self._recon_loss_sum  = 0.0
        self._input_var_sum   = 0.0
        self._l0_sum          = 0.0
        self._steps_in_window = 0

    def flush_final(self, step: int):
        if self._steps_in_window > 0:
            self._flush(step)

    def close(self):
        self._fh.close()

    @property
    def n_dead(self) -> int:
        return int((~self.ever_fired).sum().item())

    @property
    def frac_dead(self) -> float:
        return self.n_dead / self.dict_size


###############################################################################
# INSTRUMENTED ITERATOR
###############################################################################

class _InstrumentedIterator:
    """
    Wraps the activation iterator and intercepts each batch to compute
    metrics against the current SAE state.

    The SAE reference is injected via set_sae() before training starts.
    """

    def __init__(self, inner: Iterator[torch.Tensor], tracker: SAEMetricsTracker, steps: int):
        self._inner   = inner
        self._tracker = tracker
        self._steps   = steps
        self._step    = 0
        self._sae     = None

    def set_sae(self, sae):
        self._sae = sae

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self._step >= self._steps:
            raise StopIteration

        batch = next(self._inner)

        if self._sae is not None:
            with torch.no_grad():
                encode_out = self._sae.encode(batch)

                # Handle different return signatures across library versions
                if isinstance(encode_out, torch.Tensor):
                    acts    = encode_out
                    top_idx = acts.nonzero(as_tuple=False)[:, 1]
                    recon   = self._sae.decode(acts)
                elif len(encode_out) == 2:
                    acts, top_idx = encode_out
                    recon = self._sae.decode(acts)
                elif len(encode_out) == 3:
                    acts, top_idx, _ = encode_out
                    recon = self._sae.decode(acts)
                else:
                    acts    = encode_out[0]
                    top_idx = encode_out[-1]
                    recon   = self._sae.decode(acts)

            self._tracker.update(
                step=self._step,
                batch=batch,
                recon=recon,
                top_indices=top_idx,
            )

        self._step += 1
        return batch


###############################################################################
# PATCHED TRAINER — captures SAE instance at construction time
###############################################################################

def _make_capturing_trainer_class(instrumented: _InstrumentedIterator):
    """
    Returns a subclass of TopKTrainer whose __init__ calls super().__init__
    normally, then immediately injects self.ae into the instrumented iterator.

    This works regardless of what keyword arguments TopKTrainer accepts,
    because we forward **kwargs unchanged and only add the injection step.
    """

    class CapturingTopKTrainer(TopKTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # self.ae is set by TopKTrainer.__init__ — inject it now so
            # the instrumented iterator has a live reference from step 0.
            instrumented.set_sae(self.ae)

    return CapturingTopKTrainer


###############################################################################
# TRAINING WRAPPER
###############################################################################

def train_sae_with_metrics(
    data_iter: Iterator[torch.Tensor],
    trainer_cfg: dict,
    steps: int,
    save_dir: str,
    log_steps: int,
    dict_size: int,
    device: torch.device,
) -> SAEMetricsTracker:
    """
    Calls trainSAE with a patched trainer class that captures the SAE
    instance the moment TopKTrainer constructs it, injecting it into the
    instrumented iterator before the first training step runs.
    """
    tracker = SAEMetricsTracker(
        dict_size=dict_size,
        device=device,
        save_dir=save_dir,
        log_steps=log_steps,
    )

    instrumented = _InstrumentedIterator(
        inner=data_iter,
        tracker=tracker,
        steps=steps,
    )

    # Swap in a patched trainer class that will inject self.ae on construction.
    # All other config keys are forwarded unchanged to TopKTrainer.__init__.
    patched_cfg = {
        **trainer_cfg,
        "trainer": _make_capturing_trainer_class(instrumented),
    }

    print(f"  Starting trainSAE ({steps:,} steps)...")

    trainSAE(
        data=instrumented,
        trainer_configs=[patched_cfg],
        steps=steps,
        save_dir=save_dir,
        log_steps=log_steps,
        use_wandb=False,
    )

    tracker.flush_final(step=steps)
    tracker.close()

    return tracker


###############################################################################
# TIMESTAMP / DURATION HELPERS
###############################################################################

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


###############################################################################
# MAIN
###############################################################################

def main():
    args = parse_args()
    device = torch.device(args.device)

    run_start_ts   = now_iso()
    run_start_wall = time.monotonic()

    print("=" * 60)
    print("SAE TRAINING — POST-ATTENTION RESIDUAL STREAM")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Dataset:        {args.fineweb}")
    print(f"  Target tokens:  {args.target_tokens:,} per (layer, seed)")
    print(f"  Seq length:     {args.seq_length}")
    print(f"  Dict mult:      {args.dict_mult}x")
    print(f"  TopK (k):       {args.k_sparse}")
    print(f"  LR:             {args.lr}")
    print(f"  Seeds:          {args.seeds}")
    print(f"  LLM batch:      {args.llm_batch_size}")
    print(f"  SAE batch:      {args.sae_batch_size}")
    print(f"  Save root:      {args.save_root}")
    print(f"  Run started:    {run_start_ts}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading LLaMA model...")
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device},
    )
    model.eval()

    hidden_size = config.hidden_size
    num_layers  = config.num_hidden_layers

    print(f"  Loaded: {num_layers} layers, hidden_size={hidden_size}")
    print(f"  SAE dict size per layer: {hidden_size * args.dict_mult}")

    # Confirm which layer-norm attribute will be used as the hook target
    block0 = model.model.layers[0]
    hook_attr = next(
        (a for a in ["post_feedforward_layernorm", "post_attention_layernorm"] if hasattr(block0, a)),
        None,
    )
    if hook_attr:
        print(f"  Hook target: block.{hook_attr}  (post-attention residual stream)")
    else:
        print("  WARNING: Neither 'post_feedforward_layernorm' nor 'post_attention_layernorm' found "
              "— will fall back to block.mlp as hook target.")
    print()

    layer_start = args.layer_start
    layer_end   = (num_layers - 1) if args.layer_end is None else min(args.layer_end, num_layers - 1)

    steps_per_run = args.target_tokens // args.sae_batch_size
    print(f"  Steps per (layer, seed): {steps_per_run:,}  "
          f"({args.target_tokens:,} tokens / {args.sae_batch_size} batch)\n")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Loading token dataset (streaming)...")
    token_dataset = TokenSequenceDataset(
        npz_path=args.fineweb,
        seq_length=args.seq_length,
        min_tokens=5,
    )
    print("  Dataset ready (infinite cycling iterator).\n")

    all_summaries: list[dict] = []

    # ------------------------------------------------------------------
    # Main loop: layer × seed
    # ------------------------------------------------------------------
    for layer in range(layer_start, layer_end + 1):
        print(f"\n{'=' * 60}")
        print(f"  LAYER {layer}")
        print(f"{'=' * 60}")

        for seed in args.seeds:
            print(f"\n  --- Layer {layer}, Seed {seed} ---")

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            save_dir = os.path.join(args.save_root, f"layer_{layer}", f"seed_{seed}")
            os.makedirs(save_dir, exist_ok=True)

            ts_start   = now_iso()
            wall_start = time.monotonic()
            print(f"  Start time: {ts_start}")

            dict_size = hidden_size * args.dict_mult

            data_iter = residual_stream_iterator(
                model=model,
                dataset=token_dataset,
                layer_idx=layer,
                device=device,
                llm_batch_size=args.llm_batch_size,
                sae_batch_size=args.sae_batch_size,
                target_tokens=args.target_tokens,
            )

            trainer_cfg = {
                "trainer":        TopKTrainer,
                "dict_class":     AutoEncoderTopK,
                "activation_dim": hidden_size,
                "dict_size":      dict_size,
                "k":              args.k_sparse,
                "lr":             args.lr,
                "warmup_steps":   args.warmup_steps,
                "device":         args.device,
                "steps":          steps_per_run,
                "layer":          layer,
                "lm_name":        args.model_path,
                "seed":           seed,
            }

            tracker = train_sae_with_metrics(
                data_iter=data_iter,
                trainer_cfg=trainer_cfg,
                steps=steps_per_run,
                save_dir=save_dir,
                log_steps=args.log_steps,
                dict_size=dict_size,
                device=device,
            )

            wall_end     = time.monotonic()
            ts_end       = now_iso()
            duration_sec = wall_end - wall_start

            print(f"  End time:   {ts_end}")
            print(f"  Duration:   {format_duration(duration_sec)}")

            # Save weights
            ckpt_path = os.path.join(save_dir, "trainer_0", "ae.pt")
            weights_path = None
            if os.path.exists(ckpt_path):
                sae = AutoEncoderTopK.from_pretrained(ckpt_path, device=args.device)
                weights_path = os.path.join(save_dir, "sae_weights.pth")
                torch.save(sae.state_dict(), weights_path)
                print(f"  Saved SAE weights → {weights_path}")
            else:
                print(f"  WARNING: checkpoint not found at {ckpt_path}")

            # Save summary.json
            summary = {
                "layer":              layer,
                "seed":               seed,
                "ts_start":           ts_start,
                "ts_end":             ts_end,
                "duration_seconds":   round(duration_sec, 1),
                "duration_human":     format_duration(duration_sec),
                "target_tokens":      args.target_tokens,
                "steps":              steps_per_run,
                "dict_size":          dict_size,
                "dict_mult":          args.dict_mult,
                "k_sparse":           args.k_sparse,
                "lr":                 args.lr,
                "warmup_steps":       args.warmup_steps,
                "sae_batch_size":     args.sae_batch_size,
                "llm_batch_size":     args.llm_batch_size,
                "seq_length":         args.seq_length,
                "n_dead_features":    tracker.n_dead,
                "frac_dead_features": round(tracker.frac_dead, 4),
                "tokens_processed":   tracker.tokens_seen,
                "metrics_path":       tracker.metrics_path,
                "weights_path":       weights_path,
            }

            summary_path = os.path.join(save_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"  Dead features: {tracker.n_dead}/{dict_size} ({tracker.frac_dead:.1%})")
            print(f"  Summary saved → {summary_path}")
            print(f"  Metrics log  → {tracker.metrics_path}")

            all_summaries.append(summary)

    # ------------------------------------------------------------------
    # Overall run report
    # ------------------------------------------------------------------
    run_end_ts       = now_iso()
    run_duration_sec = time.monotonic() - run_start_wall

    run_report = {
        "ts_start":         run_start_ts,
        "ts_end":           run_end_ts,
        "duration_seconds": round(run_duration_sec, 1),
        "duration_human":   format_duration(run_duration_sec),
        "model_path":       args.model_path,
        "layers_trained":   list(range(layer_start, layer_end + 1)),
        "seeds":            args.seeds,
        "per_run":          all_summaries,
    }

    run_report_path = os.path.join(args.save_root, "run_report.json")
    os.makedirs(args.save_root, exist_ok=True)
    with open(run_report_path, "w") as f:
        json.dump(run_report, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL LAYERS × SEEDS COMPLETED")
    print(f"  Total duration: {format_duration(run_duration_sec)}")
    print(f"  Run report    → {run_report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
