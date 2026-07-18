#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint

from diffusion_strings.adapters.from_bioemu import (
    bioemu_vector_fields_from_model,
    prepare_bioemu_embedding_cache,
)
from diffusion_strings.dynamics_se3 import b_transport


CHIGNOLIN_SEQUENCE = "GYDPETGTWG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Chignolin by transporting BioEmu prior states with the b field only."
    )
    parser.add_argument("--sequence", default=CHIGNOLIN_SEQUENCE)
    parser.add_argument("--model-name", default="bioemu-v1.1")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("outputs/chignolin_b_only_samples.npz"))
    parser.add_argument("--cache-embeds-dir", type=Path, default=None)
    parser.add_argument("--single-embeds-file", type=Path, default=None)
    parser.add_argument("--pair-embeds-file", type=Path, default=None)
    parser.add_argument("--cache-so3-dir", type=Path, default=None)
    parser.add_argument("--msa-file", type=Path, default=None)
    parser.add_argument("--msa-host-url", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--embedding-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait when generating missing BioEmu embeddings in a subprocess.",
    )
    parser.add_argument(
        "--allow-embedding-generation",
        action="store_true",
        help=(
            "Allow BioEmu to generate missing AlphaFold/ColabFold embeddings. "
            "Leave this off in environments where that path crashes with XLA/protobuf errors."
        ),
    )
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tail_text(value: str | bytes | None, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")[-max_chars:]
    return value[-max_chars:]


def generate_embeddings_in_subprocess(args: argparse.Namespace) -> None:
    code = (
        "from diffusion_strings.adapters.from_bioemu import "
        "prepare_bioemu_embedding_cache; "
        "prepare_bioemu_embedding_cache("
        f"sequence={args.sequence!r}, "
        f"cache_embeds_dir={str(args.cache_embeds_dir) if args.cache_embeds_dir is not None else None!r}, "
        f"single_embeds_file={str(args.single_embeds_file) if args.single_embeds_file is not None else None!r}, "
        f"pair_embeds_file={str(args.pair_embeds_file) if args.pair_embeds_file is not None else None!r}, "
        f"msa_file={str(args.msa_file) if args.msa_file is not None else None!r}, "
        f"msa_host_url={args.msa_host_url!r}, "
        "allow_embedding_generation=True)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=args.embedding_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            "BioEmu embedding generation did not finish before "
            f"--embedding-timeout={args.embedding_timeout}. The main sampling "
            "process was kept alive, but embeddings were not created.\n\n"
            "Pass precomputed embeddings with --single-embeds-file and "
            "--pair-embeds-file, pass an --msa-file, or retry in an environment "
            "where BioEmu's AlphaFold/ColabFold embedding path can run "
            "reliably.\n\n"
            f"stdout tail:\n{tail_text(exc.stdout, 1000)}\n"
            f"stderr tail:\n{tail_text(exc.stderr, 4000)}\n"
        ) from exc
    if result.returncode != 0:
        raise SystemExit(
            "BioEmu embedding generation failed in a subprocess, so the main "
            "sampling process did not enter the TensorFlow/XLA crash path. If "
            "stderr mentions 'xla/xla_data.proto', this is the protobuf "
            "descriptor collision from BioEmu's embedding-generation stack, not "
            "from the diffusion-strings adapter itself.\n\n"
            f"return code: {result.returncode}\n"
            f"stdout tail:\n{tail_text(result.stdout, 1000)}\n"
            f"stderr tail:\n{tail_text(result.stderr, 4000)}\n"
        )


def prepare_embeddings(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.allow_embedding_generation:
        generate_embeddings_in_subprocess(args)

    try:
        return prepare_bioemu_embedding_cache(
            sequence=args.sequence,
            cache_embeds_dir=args.cache_embeds_dir,
            single_embeds_file=args.single_embeds_file,
            pair_embeds_file=args.pair_embeds_file,
            msa_file=args.msa_file,
            msa_host_url=args.msa_host_url,
            allow_embedding_generation=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_embeds_dir is not None:
        args.cache_embeds_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_so3_dir is not None:
        args.cache_so3_dir.mkdir(parents=True, exist_ok=True)
    single_embeds_file, pair_embeds_file = prepare_embeddings(args)
    print(f"BioEmu single embeddings: {single_embeds_file}")
    print(f"BioEmu pair embeddings: {pair_embeds_file}")

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=args.model_name,
        ckpt_path=None,
        model_config_path=None,
    )
    score_model = load_model(ckpt_path, model_config_path).to(device)
    score_model.eval()
    sdes = load_sdes(
        model_config_path=model_config_path,
        cache_so3_dir=args.cache_so3_dir,
    )
    for sde in sdes.values():
        if isinstance(sde, torch.nn.Module):
            sde.to(device)

    b_field, score_field = bioemu_vector_fields_from_model(
        score_model=score_model,
        sequence=args.sequence,
        sdes=sdes,
        cache_embeds_dir=single_embeds_file.parent,
        single_embeds_file=single_embeds_file,
        pair_embeds_file=pair_embeds_file,
        cache_so3_dir=args.cache_so3_dir,
        msa_file=args.msa_file,
        msa_host_url=args.msa_host_url,
        device=device,
    )
    print("BioEmu adapter fields created: b_field and score_field")
    print("Sampling uses b_field only.")
    _ = score_field

    n_residues = len(args.sequence)
    pos0 = sdes["pos"].prior_sampling((args.n_samples, n_residues, 3), device=device)
    rotations0 = sdes["node_orientations"].prior_sampling(
        (args.n_samples, n_residues, 3, 3),
        device=device,
    )

    times = torch.linspace(0.0, 1.0, args.n_steps + 1, device=device, dtype=pos0.dtype)
    with torch.no_grad():
        pos, rotations = b_transport(pos0, rotations0, b_field, times)

    np.savez(
        args.output,
        sequence=args.sequence,
        model_name=args.model_name,
        checkpoint=str(ckpt_path),
        model_config=str(model_config_path),
        times=times.detach().cpu().numpy(),
        initial_pos=pos0.detach().cpu().numpy(),
        initial_node_orientations=rotations0.detach().cpu().numpy(),
        final_pos=pos.detach().cpu().numpy(),
        final_node_orientations=rotations.detach().cpu().numpy(),
    )

    print(f"device: {device}")
    print(f"initial_pos: {tuple(pos0.shape)}")
    print(f"final_pos: {tuple(pos.shape)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
