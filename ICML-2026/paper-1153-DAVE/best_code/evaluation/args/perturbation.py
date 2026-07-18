import argparse


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "Evaluate Pixel Perturbation curves on ImageNet1K dataset."
    )

    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--label-mode", type=str, default="firstcol", choices=["argmax", "firstcol"])

    # testing
    p.add_argument("--max-samples", type=int, default=-1)

    # models
    p.add_argument("--model-cfg-path", type=str, required=True)

    # pixel perturbation params
    p.add_argument("--direction", type=str, default="most", choices=["most", "least"])
    p.add_argument("--num-pixels", type=float, default=0.25, help="Fraction of pixels to perturb (0..1].")
    p.add_argument("--sample-points", type=int, default=128, help="Approx number of points along the curve.")
    p.add_argument("--mask-baseline", type=float, default=0.0,
                   help="Constant baseline value in the *model input space*. "
                        "For normalized inputs, 0.0 corresponds to ImageNet mean. "
                        "For BCos constant masking, default 0.0 is treated as 0.5 (neutral) internally.")
    p.add_argument("--mask-batch-size", type=int, default=32,
                   help="Batch size for evaluating many masked variants per image.")

    # masking mode
    p.add_argument("--mask-mode", type=str, default="constant", choices=["constant", "blur"],
                   help="Masking strategy: constant baseline or blurred input baseline.")
    p.add_argument("--blur-kernel", type=int, default=31,
                   help="Gaussian blur kernel size (odd). Used when --mask-mode blur.")
    p.add_argument("--blur-sigma", type=float, default=7.0,
                   help="Gaussian blur sigma. Used when --mask-mode blur.")

    # sample filtering (correct + confident)
    p.add_argument("--only-correct", action="store_true",
                   help="If set, only evaluate samples the model classifies correctly and above --min-conf.")
    p.add_argument("--min-conf", type=float, default=0.90,
                   help="Confidence threshold (0..1]. Only used if --only-correct is set.")
    p.add_argument("--conf-on", type=str, default="pred", choices=["pred", "target"],
                   help="Confidence definition: 'pred' uses max prob; 'target' uses target-class prob. "
                        "Note: with --only-correct and argmax correctness, they are identical.")

    # saving
    p.add_argument("--save-curves", action="store_true",
                   help="Save per-sample curves as .npy (can be large).")
    p.add_argument("--save-every", type=int, default=1)

    # DAVE
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--post-proc", action="store_true")
    
    # device/seed/logging
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0xDEADBEEF)
    p.add_argument("--log-every", type=int, default=10)

    return p
    