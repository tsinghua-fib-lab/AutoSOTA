#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


MODELS = {
    "bert-base": {"n_layer": 12, "n_embd": 768, "intermediate": 0, "gate": "gelu"},
    "bert-large": {"n_layer": 24, "n_embd": 1024, "intermediate": 0, "gate": "gelu"},
    "gpt2": {"n_layer": 12, "n_embd": 768, "intermediate": 0, "gate": "gelu"},
    "llama7b": {"n_layer": 32, "n_embd": 4096, "intermediate": 11008, "gate": "silu"},
    "llama3-8b": {"n_layer": 32, "n_embd": 4096, "intermediate": 14336, "gate": "silu"},
}

MODEL_ALIASES = {
    "llama8b": "llama3-8b",
    "llama-8b": "llama3-8b",
    "llama3.1-8b": "llama3-8b",
    "llama-3.1-8b": "llama3-8b",
}


def run(cmd, cwd=None, env=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def parse_fusefss_json(output):
    last = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last = line
    if not last:
        raise ValueError(f"Failed to find JSON in FuseFSS output:\n{output}")
    return json.loads(last)


def run_fusefss_bench(fusefss_bin, model, seq, intervals, degree, helpers, iters, mask_aware, mask_val, env=None):
    cmd = [
        str(fusefss_bin),
        "--model",
        model,
        "--seq",
        str(seq),
        "--intervals",
        str(intervals),
        "--degree",
        str(degree),
        "--helpers",
        str(helpers),
        "--iters",
        str(iters),
        "--json",
    ]
    if mask_aware:
        cmd.extend(["--mask-aware", "--mask", str(mask_val)])
    output = run(cmd, env=env)
    return parse_fusefss_json(output)


def parse_sigma_log(text, gate):
    keygen_time = re.search(r"Keygen time=(\d+)", text)
    key_size = re.search(r"Key size=(\d+)", text)
    eval_time = re.search(rf"{gate.capitalize()} time=(\d+)", text)
    eval_comm = re.search(r"Eval comm bytes=(\d+)", text)
    if keygen_time and key_size and eval_time and eval_comm:
        return {
            "keygen_us": int(keygen_time.group(1)),
            "key_bytes": int(key_size.group(1)),
            "eval_us": int(eval_time.group(1)),
            "eval_comm_bytes": int(eval_comm.group(1)),
        }
    raise ValueError(f"Failed to parse Sigma output for {gate}:\n{text}")


def run_sigma_gate(sigma_bin, gate, n_elems, addr, env, gpu0, gpu1):
    with tempfile_dir() as tmpdir:
        p0_log = Path(tmpdir) / "p0.log"
        p1_log = Path(tmpdir) / "p1.log"
        env0 = env.copy()
        env1 = env.copy()
        if gpu0:
            env0["CUDA_VISIBLE_DEVICES"] = str(gpu0)
        if gpu1:
            env1["CUDA_VISIBLE_DEVICES"] = str(gpu1)
        with p0_log.open("w") as f0:
            p0 = subprocess.Popen([sigma_bin, "0", addr, str(n_elems)], env=env0, stdout=f0, stderr=f0)
        time.sleep(0.8)
        with p1_log.open("w") as f1:
            p1 = subprocess.Popen([sigma_bin, "1", addr, str(n_elems)], env=env1, stdout=f1, stderr=f1)
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            log_text = p0_log.read_text() + "\n" + p1_log.read_text()
            raise RuntimeError(f"Sigma {gate} failed: rc0={rc0} rc1={rc1}\n{log_text}")
        text = p0_log.read_text()
    return parse_sigma_log(text, gate)


def tempfile_dir():
    import tempfile

    return tempfile.TemporaryDirectory()


def estimate_fusefss_eval_bytes(gate, gate_elems, intervals):
    if gate == "silu":
        per_elem = 26.75
    else:
        per_elem = 26.5 if intervals >= 512 else 26.25
    return int(round(gate_elems * per_elem + 4))


def estimate_fusefss_key_bytes(gate, gate_elems, intervals):
    if gate == "silu":
        per_elem = 124.125
    else:
        per_elem = 108.125 if intervals >= 512 else 92.125
    return int(round(gate_elems * per_elem + 28))


def median(values):
    return statistics.median(values) if values else 0.0


def write_csv(path, rows):
    if not path or not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare FuseFSS vs Sigma activation microbenchmarks with communication estimates. "
            "This is a synthetic gate-level check; use repro_eval_latest.py for paper-style e2e runs."
        )
    )
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--models", type=str, default="bert-base,bert-large,gpt2,llama7b")
    parser.add_argument("--helpers", type=int, default=2)
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--intervals-gelu", type=int, default=256)
    parser.add_argument("--intervals-silu", type=int, default=1024)
    parser.add_argument("--mask-aware", action="store_true")
    parser.add_argument("--mask", type=int, default=0)
    parser.add_argument("--addr", type=str, default="127.0.0.1")
    parser.add_argument("--fusefss-bin", type=str, default=str(root_dir / "build" / "bench_fusefss_model"))
    parser.add_argument("--sigma-gelu", type=str, default=str(root_dir / "build" / "gpu_mpc_upstream" / "gelu"))
    parser.add_argument("--sigma-silu", type=str, default=str(root_dir / "build" / "gpu_mpc_upstream" / "silu"))
    parser.add_argument("--sigma-keybuf-mb", type=int, default=4096)
    parser.add_argument("--sigma-mempool-mb", type=int, default=4096)
    parser.add_argument("--sigma-verify", action="store_true")
    parser.add_argument("--sigma-gpu0", type=str, default=os.environ.get("SIGMA_GPU0", "0"))
    parser.add_argument("--sigma-gpu1", type=str, default=os.environ.get("SIGMA_GPU1", "1"))
    parser.add_argument("--fusefss-gpu", type=str,
                        default=os.environ.get("FUSEFSS_GPU", os.environ.get("SUF_GPU", "0")))
    parser.add_argument("--single-gpu", action="store_true",
                        help="Run both Sigma parties and the FuseFSS bench on CUDA_VISIBLE_DEVICES=0.")
    parser.add_argument("--fusefss-sigma-generic", "--sigma-bridge-generic",
                        dest="fusefss_sigma_generic", action="store_true",
                        help="Run the Sigma pair with FUSEFSS_SIGMA_GENERIC=1; the standalone FuseFSS bench is unchanged.")
    parser.add_argument("--fusefss-sigma-generic-strict", "--sigma-bridge-generic-strict",
                        dest="fusefss_sigma_generic_strict", action="store_true",
                        help="Run the Sigma pair with FUSEFSS_SIGMA_GENERIC_STRICT=1; the standalone FuseFSS bench is unchanged.")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--csv-out", type=str, default="")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup < 0 or args.warmup >= args.runs:
        raise ValueError("--warmup must be non-negative and smaller than --runs")
    if args.single_gpu:
        args.sigma_gpu0 = "0"
        args.sigma_gpu1 = "0"
        args.fusefss_gpu = "0"

    model_list = [MODEL_ALIASES.get(m.strip().lower(), m.strip().lower())
                  for m in args.models.split(",") if m.strip()]
    for name in model_list:
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")

    fusefss_bin = Path(args.fusefss_bin)
    if not fusefss_bin.exists():
        raise FileNotFoundError(f"FuseFSS bench not found: {fusefss_bin}")

    sigma_env = os.environ.copy()
    sigma_env.pop("FUSEFSS_SIGMA_GENERIC", None)
    sigma_env.pop("FUSEFSS_SIGMA_GENERIC_STRICT", None)
    sigma_env.pop("SUF_SIGMA_GENERIC", None)
    sigma_env.pop("SUF_SIGMA_GENERIC_STRICT", None)
    sigma_env["SIGMA_KEYBUF_MB"] = str(args.sigma_keybuf_mb)
    sigma_env["SIGMA_MEMPOOL_MB"] = str(args.sigma_mempool_mb)
    if not args.sigma_verify:
        sigma_env["SIGMA_SKIP_VERIFY"] = "1"
    fusefss_env = os.environ.copy()
    fusefss_env.pop("FUSEFSS_SIGMA_GENERIC", None)
    fusefss_env.pop("FUSEFSS_SIGMA_GENERIC_STRICT", None)
    fusefss_env.pop("SUF_SIGMA_GENERIC", None)
    fusefss_env.pop("SUF_SIGMA_GENERIC_STRICT", None)
    fusefss_env["CUDA_VISIBLE_DEVICES"] = str(args.fusefss_gpu)
    if args.fusefss_sigma_generic_strict:
        sigma_env["FUSEFSS_SIGMA_GENERIC_STRICT"] = "1"
    elif args.fusefss_sigma_generic:
        sigma_env["FUSEFSS_SIGMA_GENERIC"] = "1"

    results = []
    for model in model_list:
        spec = MODELS[model]
        gate = spec["gate"]
        if gate == "gelu":
            gate_elems = args.seq * 4 * spec["n_embd"]
            intervals = args.intervals_gelu
            sigma_bin = args.sigma_gelu
        else:
            gate_elems = args.seq * spec["intermediate"]
            intervals = args.intervals_silu
            sigma_bin = args.sigma_silu
        gate_count = spec["n_layer"]

        fusefss_runs = []
        sigma_runs = []
        for run_idx in range(args.runs):
            fusefss = run_fusefss_bench(
                fusefss_bin,
                model,
                args.seq,
                intervals,
                args.degree,
                args.helpers,
                args.iters,
                args.mask_aware,
                args.mask,
                fusefss_env,
            )
            sigma = run_sigma_gate(
                sigma_bin,
                gate,
                gate_elems,
                args.addr,
                sigma_env,
                args.sigma_gpu0,
                args.sigma_gpu1,
            )
            if run_idx >= args.warmup:
                fusefss_runs.append(fusefss)
                sigma_runs.append(sigma)

        fusefss_key_ms = median([r["per_gate_key_ms"] for r in fusefss_runs])
        fusefss_eval_ms = median([r["per_gate_eval_ms"] for r in fusefss_runs])
        sigma_key_ms = median([r["keygen_us"] / 1000.0 for r in sigma_runs])
        sigma_eval_ms = median([r["eval_us"] / 1000.0 for r in sigma_runs])

        sigma_key_bytes = sigma_runs[0]["key_bytes"]
        sigma_eval_bytes = sigma_runs[0]["eval_comm_bytes"]
        fusefss_key_bytes = estimate_fusefss_key_bytes(gate, gate_elems, intervals)
        fusefss_eval_bytes = estimate_fusefss_eval_bytes(gate, gate_elems, intervals)

        results.append({
            "model": model,
            "gate": gate,
            "seq": args.seq,
            "gate_elems": gate_elems,
            "gate_count": gate_count,
            "fusefss_key_ms": fusefss_key_ms,
            "sigma_key_ms": sigma_key_ms,
            "fusefss_eval_ms": fusefss_eval_ms,
            "sigma_eval_ms": sigma_eval_ms,
            "speedup": (sigma_eval_ms / fusefss_eval_ms) if fusefss_eval_ms else 0.0,
            "fusefss_key_bytes": fusefss_key_bytes,
            "sigma_key_bytes": sigma_key_bytes,
            "fusefss_eval_bytes": fusefss_eval_bytes,
            "sigma_eval_bytes": sigma_eval_bytes,
            "fusefss_key_bytes_measured": False,
            "fusefss_eval_bytes_measured": False,
            "fusefss_bytes_source": "estimated_from_gate_count",
            "runs_used": len(fusefss_runs),
            "warmup_discarded": args.warmup,
            "mask_model": "representative-single-mask" if args.mask_aware else "unmasked-synthetic-input",
            "scope": "synthetic_gate_microbenchmark",
        })

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({
                "note": (
                    "Gate-level synthetic benchmark. It does not prove the paper's per-element fresh-mask "
                    "two-server security semantics; run scripts/repro_eval_latest.py for end-to-end Sigma bridge checks."
                ),
                "config": vars(args),
                "results": results,
            }, f, indent=2)
    write_csv(args.csv_out, results)

    print("| Model / Gate | Sigma keygen (ms) | FuseFSS keygen (ms) | Sigma eval (ms) | FuseFSS eval (ms) | Eval speedup | Sigma key (bytes) | FuseFSS key est. (bytes) | Sigma eval (bytes) | FuseFSS eval est. (bytes) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        label = f"{r['model']} {r['gate'].upper()}"
        print(
            f"| {label} | {r['sigma_key_ms']:.3f} | {r['fusefss_key_ms']:.3f} | "
            f"{r['sigma_eval_ms']:.3f} | {r['fusefss_eval_ms']:.3f} | {r['speedup']:.2f}x | "
            f"{int(r['sigma_key_bytes'])} | {int(r['fusefss_key_bytes'])} | "
            f"{int(r['sigma_eval_bytes'])} | {int(r['fusefss_eval_bytes'])} |"
        )


if __name__ == "__main__":
    main()
