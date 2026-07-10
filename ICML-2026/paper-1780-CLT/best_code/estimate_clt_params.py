"""
estimate_clt_params.py

Analytically estimates parameter counts for PLT, CLT, and Block CLT models
without requiring PyTorch or ESM to be installed.

Usage examples:
  # Per-Layer Transcoder (PLT), ESM2 12-layer (480-dim)
  python estimate_clt_params.py --model plt --num-layers 12 --d-model 480 --d-hidden 4800

  # Regular CLT, ESM2 6-layer (320-dim)
  python estimate_clt_params.py --model clt --num-layers 6 --d-model 320 --d-hidden 3200

  # Regular CLT, ESM2 33-layer (1280-dim)
  python estimate_clt_params.py --model clt --num-layers 33 --d-model 1280 --d-hidden 16000

  # Block CLT, ESM2 12-layer (480-dim), block_size=6
  python estimate_clt_params.py --model block --num-layers 12 --d-model 480 --d-hidden 4800 --block-size 6

  # Compare all three for the same config
  python estimate_clt_params.py --model all --num-layers 12 --d-model 480 --d-hidden 4800 --block-size 6
"""

import argparse


def fmt(n: int) -> str:
    """Human-readable number with commas and an M/B suffix."""
    if n >= 1_000_000_000:
        return f"{n:>15,}  ({n/1e9:.3f}B)"
    elif n >= 1_000_000:
        return f"{n:>15,}  ({n/1e6:.3f}M)"
    else:
        return f"{n:>15,}"


def count_plt(L: int, H: int, D: int) -> dict:
    """
    Parameter counts for PerLayerTranscoder (PLT).

    Each layer is fully independent: one encoder and one decoder per layer.

    Components:
      encoders : L × Linear(H, D)  →  L × (H*D + D)   [weight + bias]
      decoders : L matrices, each (D, H)
      b_enc    : L × D   (added on top of Linear bias in forward pass)
      b_pre    : L × H

    Total = LH + 2LD + LDH + LDH
    """
    enc_weight   = L * H * D
    b_enc        = L * D
    b_pre        = L * H
    num_decoders = L
    decoders     = num_decoders * D * H

    total = enc_weight + b_enc + b_pre + decoders
    return {
        "enc_weight":   enc_weight,
        "b_enc":        b_enc,
        "b_pre":        b_pre,
        "num_decoders": num_decoders,
        "decoders":     decoders,
        "total":        total,
    }


def count_clt(L: int, H: int, D: int) -> dict:
    """
    Parameter counts for CrossLayerTranscoder (regular CLT).

    Components:
      encoders : L × Linear(H, D)  →  L × (H*D + D)   [weight + bias]
      b_enc    : L × D   (added on top of Linear bias in forward pass)
      b_pre    : L × H
      decoders : L*(L+1)/2 matrices, each (D, H)   [full lower-triangular]

    Total = LH + 2LD + LDH + (L(L+1)/2)(DH)
    """
    enc_weight   = L * H * D
    b_enc        = L * D
    b_pre        = L * H
    num_decoders = L * (L + 1) // 2
    decoders     = num_decoders * D * H

    total = enc_weight + b_enc + b_pre + decoders
    return {
        "enc_weight":   enc_weight,
        "b_enc":        b_enc,
        "b_pre":        b_pre,
        "num_decoders": num_decoders,
        "decoders":     decoders,
        "total":        total,
    }


def count_block_clt(L: int, H: int, D: int, B: int) -> dict:
    """
    Parameter counts for BlockCrossLayerTranscoder.

    Encoders / biases are identical to the regular CLT.

    Decoders use a windowed structure:
      For target layer tgt (0-indexed):
        src ∈ [max(0, tgt - B + 1), tgt]   →  min(tgt+1, B) source layers

      Total decoder matrices:
        sum_{tgt=0}^{L-1} min(tgt+1, B)
        = B*(B+1)/2  +  (L-B)*B          [when L >= B]

    Each decoder matrix has shape (D, H).
    """
    enc_weight = L * H * D
    b_enc      = L * D
    b_pre      = L * H

    if L <= B:
        num_decoders = L * (L + 1) // 2
    else:
        num_decoders = B * (B + 1) // 2 + (L - B) * B

    decoders = num_decoders * D * H

    total = enc_weight + b_enc + b_pre + decoders
    return {
        "enc_weight":   enc_weight,
        "b_enc":        b_enc,
        "b_pre":        b_pre,
        "num_decoders": num_decoders,
        "decoders":     decoders,
        "total":        total,
    }


def print_breakdown(name: str, L: int, H: int, D: int, counts: dict, B: int = None):
    header = f"{'='*60}"
    print(header)
    print(f"  {name}")
    print(f"  num_layers={L},  d_model={H},  d_hidden={D}" +
          (f",  block_size={B}" if B is not None else ""))
    print(header)
    print(f"  {'Component':<22} {'Params':>20}")
    print(f"  {'-'*44}")
    print(f"  {'Encoder weights':<22} {fmt(counts['enc_weight'])}")
    print(f"  {'b_enc':<22} {fmt(counts['b_enc'])}")
    print(f"  {'b_pre':<22} {fmt(counts['b_pre'])}")
    print(f"  {'Decoders':<22} {fmt(counts['decoders'])}")
    print(f"    ({counts['num_decoders']} matrices × D×H = "
          f"{counts['num_decoders']} × {D}×{H})")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<22} {fmt(counts['total'])}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate parameter counts for PLT / CLT / Block CLT models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", choices=["plt", "clt", "block", "all"], default="all",
                        help="Which model(s) to estimate (default: all)")
    parser.add_argument("--num-layers", type=int, required=True,
                        help="Number of transformer layers (L)")
    parser.add_argument("--d-model", type=int, required=True,
                        help="Embedding dimension of the backbone (H)")
    parser.add_argument("--d-hidden", type=int, required=True,
                        help="CLT/PLT latent dimension (D)")
    parser.add_argument("--block-size", type=int, default=6,
                        help="Window size for Block CLT (default: 6). "
                             "Ignored for --model plt/clt.")
    args = parser.parse_args()

    L, H, D, B = args.num_layers, args.d_model, args.d_hidden, args.block_size

    if args.model in ("plt", "all"):
        counts = count_plt(L, H, D)
        print_breakdown("PerLayerTranscoder (PLT)", L, H, D, counts)

    if args.model in ("clt", "all"):
        counts = count_clt(L, H, D)
        print_breakdown("CrossLayerTranscoder (CLT)", L, H, D, counts)

    if args.model in ("block", "all"):
        counts = count_block_clt(L, H, D, B)
        print_breakdown("BlockCrossLayerTranscoder (Block CLT)", L, H, D, counts, B=B)

    if args.model == "all":
        plt_p   = count_plt(L, H, D)["total"]
        clt_p   = count_clt(L, H, D)["total"]
        block_p = count_block_clt(L, H, D, B)["total"]

        print(f"  {'Model':<30} {'Total params':>22}  {'vs PLT':>10}")
        print(f"  {'-'*66}")
        print(f"  {'PLT':<30} {fmt(plt_p).strip():>22}  {'—':>10}")
        print(f"  {'CLT':<30} {fmt(clt_p).strip():>22}  {clt_p/plt_p:>9.1f}x")
        print(f"  {'Block CLT':<30} {fmt(block_p).strip():>22}  {block_p/plt_p:>9.1f}x")
        print()
        print(f"  Block CLT vs CLT: saves {fmt(clt_p - block_p).strip()} "
              f"({100*(clt_p-block_p)/clt_p:.1f}% reduction)")
        print(f"  Decoder count:  PLT={count_plt(L,H,D)['num_decoders']}  "
              f"CLT={count_clt(L,H,D)['num_decoders']}  "
              f"Block CLT={count_block_clt(L,H,D,B)['num_decoders']}")
        print()


if __name__ == "__main__":
    main()
