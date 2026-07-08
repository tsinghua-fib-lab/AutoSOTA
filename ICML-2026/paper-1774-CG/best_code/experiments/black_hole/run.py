"""Clean entry point for the black-hole imaging experiment (paper §6.2).

Reconstructs black-hole images from radio-interferometry measurements with
Calibrated Bayesian Guidance, in two flavours:

    # fast: one-step mean-flow renoise sampler (needs a mean-flow checkpoint)
    python run.py black-hole --posterior meanflow --mf-ckpt <ckpt> --reinf-k 128

    # slow: DDPM inner-loop sampler (the original; only needs the diffusion prior)
    python run.py black-hole --posterior ddpm --reinf-k 512 --num-steps 100

It reuses the vendored InverseBench engine (forward operator, scheduler, dataset,
evaluator, diffusion prior) and the shared ``calibrated_guidance`` estimator, and
logs to the unified W&B project. Mirrors the orchestration of the original
``third_party/InverseBench/main.py`` but with a clean, flag-driven interface.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from experiments.common import seed_everything
from experiments.common.wandb_logging import WandbRun
from experiments.black_hole import _engine
from experiments.black_hole.cbg import run_cbg_inference

EXPERIMENT = "black_hole"
_DEFAULT_PRIOR = os.path.join(_engine.INVERSEBENCH_ROOT, "checkpoints", "blackhole-50k.pt")
_DEFAULT_DATA = os.path.join(_engine.INVERSEBENCH_ROOT, "data", "blackhole")

# VP scheduler used by the CBG estimator (from configs/algorithm/dps.yaml).
_SCHEDULER_BASE = {"schedule": "vp", "timestep": "vp", "scaling": "vp"}


def _load_prior(prior_ckpt: str, device: str):
    """Load the pretrained VP diffusion prior (mirrors main.py's try/except)."""
    import pickle

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from utils.helper import open_url

    try:
        with open_url(prior_ckpt, "rb") as f:
            net = pickle.load(f)["ema"].to(device)
    except Exception:
        pretrain = OmegaConf.load(
            os.path.join(_engine.INVERSEBENCH_ROOT, "configs", "pretrain", "blackhole.yaml")
        )
        net = instantiate(pretrain.model)
        ckpt = torch.load(prior_ckpt, map_location=device)
        net.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["net"])
        net = net.to(device)
    net.eval()
    return net


def _build_forward_op_and_data(args, device: str):
    """Instantiate the forward operator, test set and evaluator from the vendored
    problem config, overriding only the paths / id-list from the CLI."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.load(
        os.path.join(_engine.INVERSEBENCH_ROOT, "configs", "problem", "blackhole.yaml")
    )
    with open_dict(cfg):
        cfg.model.root = args.measure_root or os.path.join(args.data_dir, "measure")
        cfg.data.root = args.test_root or os.path.join(args.data_dir, "test")
        cfg.data.id_list = args.id_list

    forward_op = instantiate(cfg.model, device=device)
    testset = instantiate(cfg.data)
    evaluator = instantiate(cfg.evaluator, forward_op=forward_op)
    return forward_op, testset, evaluator


def run(args) -> int:
    seed_everything(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available. Pass --device cpu for a (slow) CPU run.")

    forward_op, testset, evaluator = _build_forward_op_and_data(args, device)
    net = _load_prior(args.prior_ckpt, device)

    mf_net = None
    if args.posterior == "meanflow":
        if not args.mf_ckpt:
            raise SystemExit("--posterior meanflow requires --mf-ckpt (mean-flow checkpoint).")
        from algo.meanflow_posterior import load_meanflow_net

        mf_net = load_meanflow_net(
            args.mf_ckpt, device,
            easy_meanflow_root=args.mf_root or _engine.EASY_MEANFLOW_ROOT,
        )

    scheduler_config = {**_SCHEDULER_BASE, "num_steps": args.num_steps}
    os.makedirs(args.output_dir, exist_ok=True)

    n_total = len(testset) if args.limit is None else min(args.limit, len(testset))
    config = {
        "posterior": args.posterior,
        "estimator": "reparam" if args.reparam else "reinforce",
        "reinf_k": args.reinf_k,
        "num_steps": args.num_steps,
        "inner_loop_steps": args.inner_loop_steps,
        "guidance_scale": args.guidance_scale,
        "use_clamp": args.clamp,
        "stop_at_sigma": args.stop_at_sigma,
        "last_hop": not args.no_last_hop,
        "samples_per_image": args.samples_per_image,
        "n_images": n_total,
        "seed": args.seed,
        "prior_ckpt": os.path.basename(args.prior_ckpt),
        "mf_ckpt": os.path.basename(args.mf_ckpt) if args.mf_ckpt else None,
    }

    run_name = args.run_name or (
        f"bh_{args.posterior}_{'reparam' if args.reparam else 'reinforce'}"
        f"_k{args.reinf_k}_s{args.num_steps}"
    )

    with WandbRun(EXPERIMENT, config, run_name=run_name) as wb:
        for i in range(n_total):
            data = testset[i]
            data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
            data_id = testset.id_list[i]
            observation = forward_op(data)
            target = data["target"]

            for sample_idx in range(args.samples_per_image):
                t0 = time.time()
                recon = run_cbg_inference(
                    net=net,
                    forward_op=forward_op,
                    observation=observation,
                    posterior=args.posterior,
                    scheduler_config=scheduler_config,
                    reinf_k=args.reinf_k,
                    inner_loop_steps=args.inner_loop_steps,
                    mf_net=mf_net,
                    guidance_scale=args.guidance_scale,
                    reparam=args.reparam,
                    use_clamp=args.clamp,
                    stop_at_sigma=args.stop_at_sigma,
                    last_hop=not args.no_last_hop,
                    output_path=os.path.join(args.output_dir, f"steps_{data_id}_{sample_idx}"),
                    device=device,
                )
                elapsed = time.time() - t0

                result = {
                    "observation": observation,
                    "generated_sample": recon,
                    "recon": forward_op.unnormalize(recon).cpu(),
                    "target": forward_op.unnormalize(target).cpu(),
                }
                torch.save(
                    result,
                    os.path.join(args.output_dir, f"result_{data_id}_sample_{sample_idx}.pt"),
                )

                metrics = evaluator(
                    pred=result["recon"], target=result["target"], observation=result["observation"]
                )
                wb.log({**{f"image/{k}": v for k, v in metrics.items()},
                        "image/seconds": elapsed,
                        "image/index": i, "image/id": data_id})
                print(f"[{i + 1}/{n_total}] id={data_id} sample={sample_idx} "
                      f"psnr={metrics['psnr']:.2f} cp_chi2={metrics['cp_chi2']:.3f} "
                      f"camp_chi2={metrics['camp_chi2']:.3f} ({elapsed:.1f}s)")

        final = evaluator.compute()
        wb.log({f"final/{k}": v for k, v in final.items()})
        print("Aggregate:", {k: round(float(v), 4) for k, v in final.items()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py black-hole", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--posterior", choices=["meanflow", "ddpm"], default="meanflow",
                   help="p(x0|xt) sampler: fast mean-flow renoise, or slow DDPM inner loop.")
    p.add_argument("--estimator", choices=["reinforce", "reparam"], default="reinforce",
                   help="gradient-free (reinforce) or gradient-based (reparam) CBG.")
    p.add_argument("--reinf-k", type=int, default=128, help="K candidate samples per step.")
    p.add_argument("--num-steps", type=int, default=100, help="outer SDE steps.")
    p.add_argument("--inner-loop-steps", type=int, default=25, help="DDPM inner-loop steps (ddpm only).")
    p.add_argument("--guidance-scale", type=float, default=0.003, help="likelihood temperature.")
    p.add_argument("--id-list", default="0-99", help="dataset ids, e.g. '0-99' or '0,5,10'.")
    p.add_argument("--limit", type=int, default=None, help="process only the first N ids (smoke tests).")
    p.add_argument("--samples-per-image", type=int, default=1)
    p.add_argument("--data-dir", default=_DEFAULT_DATA, help="black-hole data dir (with measure/ and test/).")
    p.add_argument("--measure-root", default=None, help="override forward-op measurement root.")
    p.add_argument("--test-root", default=None, help="override test-set root.")
    p.add_argument("--prior-ckpt", default=_DEFAULT_PRIOR, help="VP diffusion prior checkpoint.")
    p.add_argument("--mf-ckpt", default=os.environ.get("BH_MF_CKPT"), help="mean-flow checkpoint (meanflow).")
    p.add_argument("--mf-root", default=None, help="easy_meanflow root for checkpoint unpickling.")
    p.add_argument("--output-dir", default="results/black_hole")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    # Paper Table-2 config: no clamp, stop the SDE at sigma=0.05, one last hop.
    p.add_argument("--clamp", action="store_true",
                   help="clamp the guided mean to [-1,1] each step (paper: off).")
    p.add_argument("--stop-at-sigma", type=float, default=0.05,
                   help="truncate the SDE at the first sigma <= this (paper: 0.05; set <=0 to disable).")
    p.add_argument("--no-last-hop", action="store_true",
                   help="disable the final deterministic guided hop (paper: on).")
    p.add_argument("--device", default="cuda")
    return p


def cli(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.reparam = args.estimator == "reparam"
    if args.stop_at_sigma is not None and args.stop_at_sigma <= 0:
        args.stop_at_sigma = None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(cli())
