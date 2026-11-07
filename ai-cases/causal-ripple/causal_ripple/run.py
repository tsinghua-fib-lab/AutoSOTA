"""Generate code from a public-prompt JSONL file using a named paper profile."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .baselines import generate_remdm_conf
from .cascade import visible_cascade
from .config import BENCHMARK_PROFILES, MODEL_ID, MODEL_REVISION, PROFILES, get_config
from .io import CostCounter, benchmark_diagnostics, clean_tokens, file_digest, read_jsonl
from .sampling import generate

SAMPLERS = {
    "ripple": "ripple",
    "core": "core",
    "topk": "topk_margin",
    "confidence": "low_confidence",
    "random": "random",
    "margin-remask": "margin_remask",
    "remdm-conf": "remdm_conf",
}
PRIVATE_FIELDS = {
    "test",
    "tests",
    "code_context",
    "canonical_solution",
    "reference_code",
    "solutions",
    "base_input",
    "plus_input",
    "input_output",
    "passed",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARK_PROFILES, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=list(SAMPLERS) + ["visible-cascade"], default="ripple")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--length", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    profile = args.profile or (
        "ds1000-cascade"
        if args.method == "visible-cascade" and args.benchmark == "ds1000"
        else BENCHMARK_PROFILES[args.benchmark]
    )
    config = get_config(profile, args.method)
    length = args.length or (256 if args.benchmark == "ds1000" else 512)
    rows = read_jsonl(args.prompts)
    expected = {row["task_id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Prompt file contains duplicate task IDs")
    for row in rows:
        if PRIVATE_FIELDS & row.keys():
            raise ValueError(
                "Generation inputs must contain public prompts, not evaluation payloads"
            )
        if row["prompt_sha256"] != hashlib.sha256(row["prompt"].encode()).hexdigest():
            raise ValueError(f"Prompt hash mismatch: {row['task_id']}")
    protocol = dict(
        benchmark=args.benchmark,
        method=args.method,
        profile=profile,
        model=args.model,
        model_revision=args.revision,
        seed=args.seed,
        ordinary_steps=args.steps,
        length=length,
        temperature=0.0,
        sampler=asdict(config),
        public_sha256=file_digest(args.prompts),
        source_sha256={p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    )
    if args.method == "remdm-conf":
        protocol["revision_baseline"] = dict(
            adaptation="CoRe Appendix B.1 fixed-schedule LLaDA plug-in",
            selection="one sample without replacement with weights exp(-cached probability)",
            cache="probability at last unmasking",
            rng="sha256(seed:task_id), first 8 bytes big-endian modulo 2**63",
            extra_transfer_slots=1,
            active_fraction=[0.25, 0.75],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.output.with_suffix(".manifest.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != protocol:
            raise ValueError("Resume configuration differs from the original output manifest")
    else:
        if args.output.exists():
            raise ValueError("Existing output lacks a reproducibility manifest")
        sidecar.write_text(json.dumps(protocol, indent=2) + "\n")
    existing = read_jsonl(args.output) if args.output.exists() else []
    seen = {row["task_id"] for row in existing}
    if len(seen) != len(existing) or not seen <= expected.keys():
        raise ValueError("Existing output has duplicated or unknown task IDs")
    if any(row["prompt_sha256"] != expected[row["task_id"]]["prompt_sha256"] for row in existing):
        raise ValueError("Existing output does not match the public prompts")
    selected = rows[: args.limit] if args.limit else rows
    pending = [row for row in selected if row["task_id"] not in seen]
    print(
        f"benchmark={args.benchmark} method={args.method} profile={profile} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    if args.method == "random" and existing:
        raise ValueError(
            "Random filling uses a trajectory-wide RNG stream; restart with a new output file"
        )
    from transformers import AutoModel, AutoTokenizer

    set_seed(args.seed)
    loader = dict(
        trust_remote_code=True, revision=args.revision, local_files_only=args.local_files_only
    )
    model = (
        AutoModel.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **loader
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **loader)
    counter = CostCounter(model)
    cuda = torch.device(args.device).type == "cuda"
    try:
        with args.output.open("a", encoding="utf-8", buffering=1) as stream, torch.inference_mode():
            for row in pending:
                if args.method == "remdm-conf":
                    task_seed = int.from_bytes(
                        hashlib.sha256(f"{args.seed}:{row['task_id']}".encode()).digest()[:8],
                        "big",
                    ) % (2**63)
                    set_seed(task_seed)
                ids = tokenizer(row["prompt"])["input_ids"]
                counter.reset()
                start = time.perf_counter()
                if "prompt_tokens" in row and row["prompt_tokens"] != len(ids):
                    raise ValueError("Tokenizer differs from the tokenizer used to prepare prompts")
                if len(ids) + length > 4096:
                    record = dict(
                        task_id=row["task_id"],
                        prompt_sha256=row["prompt_sha256"],
                        method=args.method,
                        generation_status="context_limit",
                        completion="",
                        program="",
                        generated_token_ids=[],
                        stats={},
                        cost=dict(counter.values),
                        elapsed_seconds=0.0,
                    )
                else:
                    prompt = torch.tensor([ids], device=args.device)
                    if cuda:
                        torch.cuda.reset_peak_memory_stats()
                    kwargs = dict(
                        model=model,
                        prompt=prompt,
                        steps=args.steps,
                        gen_length=length,
                        block_length=length,
                        temperature=0.0,
                        mask_id=126336,
                        tokenizer=tokenizer,
                        config=config,
                    )
                    if args.method == "visible-cascade":
                        certificate, visible = benchmark_diagnostics(row, tokenizer)
                        output, stats, _ = visible_cascade(
                            certificate=certificate, visible_score=visible, **kwargs
                        )
                    elif args.method == "remdm-conf":
                        output, stats, _ = generate_remdm_conf(**kwargs)
                    else:
                        output, stats, _ = generate(remasking=SAMPLERS[args.method], **kwargs)
                    if cuda:
                        torch.cuda.synchronize()
                    completion, program, cleaning = clean_tokens(
                        output[0], len(ids), tokenizer, row
                    )
                    generated = output[0, len(ids) :].detach().cpu().tolist()
                    record = dict(
                        task_id=row["task_id"],
                        prompt_sha256=row["prompt_sha256"],
                        method=args.method,
                        generation_status="generated",
                        completion=completion,
                        program=program,
                        cleaning=cleaning,
                        generated_token_ids=generated,
                        stats=stats,
                        cost=dict(counter.values),
                        elapsed_seconds=time.perf_counter() - start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                    )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(row["task_id"])
                if len(seen) % 10 == 0 or len(seen) == len(selected):
                    print(f"progress={len(seen)}/{len(selected)}", flush=True)
    finally:
        counter.close()


if __name__ == "__main__":
    main()
