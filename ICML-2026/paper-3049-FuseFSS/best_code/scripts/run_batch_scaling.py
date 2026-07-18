#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROUND_COUNTS = {
    ("bert-base", 128): (1128, 1116),
    ("gpt2", 128): (1128, 1116),
}


def run_cmd(cmd, cwd, env, log_path):
    with open(log_path, "w") as f:
        return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=f)


def ensure_output_dirs(run_dir):
    (run_dir / "output" / "P0" / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "output" / "P1" / "models").mkdir(parents=True, exist_ok=True)


def output_dir(run_dir, model, seq, batch):
    suffix = f"{model}-{seq}"
    if batch > 1:
        suffix += f"-b{batch}"
    return run_dir / "output" / "P0" / "models" / suffix


def clean_dir(path):
    if path.exists():
        shutil.rmtree(path)


def parse_dealer(path):
    text = path.read_text()
    total_us = int(re.search(r"Total time=(\d+) us", text).group(1))
    key_bytes = int(re.search(r"Key size=(\d+) B", text).group(1))
    batch = int(re.search(r"Batch=(\d+)", text).group(1)) if "Batch=" in text else 1
    return {"keygen_us": total_us, "key_bytes": key_bytes, "batch": batch}


def parse_evaluator(path):
    text = path.read_text()
    total_us = int(re.search(r"Total time=(\d+) us", text).group(1))
    comm_us = int(re.search(r"Comm time=(\d+) us", text).group(1))
    total_comm_bytes = int(re.search(r"Total Comm=(\d+) B", text).group(1))
    batch = int(re.search(r"Batch=(\d+)", text).group(1)) if "Batch=" in text else 1
    return {
        "total_us": total_us,
        "comm_us": comm_us,
        "total_comm_bytes": total_comm_bytes,
        "batch": batch,
    }


def bytes_to_gb(bytes_val):
    # Match Sigma's benchmark output convention: "GB" is printed as
    # bytes / 1024^3. Raw byte counts are retained in JSON/CSV outputs.
    return bytes_val / (1024.0 ** 3)


def project_times(total_us, comm_us, comm_bytes, rounds, bandwidth, latency_s):
    comp_time = (total_us - comm_us) / 1e6
    # Sigma logs Total Comm as bytesSent()+bytesReceived(), so do not double it.
    return comp_time + comm_bytes / bandwidth + rounds * latency_s


def run_pair(run_dir, bin_path, model, seq, batch, threads, addr, env_base, gpu0, gpu1, tag, run_idx):
    ensure_output_dirs(run_dir)
    out_dir0 = output_dir(run_dir, model, seq, batch)
    out_dir1 = run_dir / "output" / "P1" / "models" / out_dir0.name
    clean_dir(out_dir0)
    clean_dir(out_dir1)

    env0 = env_base.copy()
    env1 = env_base.copy()
    env0["SIGMA_BATCH"] = str(batch)
    env1["SIGMA_BATCH"] = str(batch)
    if gpu0 is not None:
        env0["CUDA_VISIBLE_DEVICES"] = str(gpu0)
    if gpu1 is not None:
        env1["CUDA_VISIBLE_DEVICES"] = str(gpu1)

    cmd0 = [str(bin_path), model, str(seq), "0", addr, str(threads)]
    cmd1 = [str(bin_path), model, str(seq), "1", addr, str(threads)]

    suffix = f"{model}_{seq}_b{batch}_r{run_idx}"
    log0 = Path(f"/tmp/{tag}_{suffix}_p0.log")
    log1 = Path(f"/tmp/{tag}_{suffix}_p1.log")

    p0 = run_cmd(cmd0, run_dir, env0, log0)
    time.sleep(1.0)
    p1 = run_cmd(cmd1, run_dir, env1, log1)
    rc0 = p0.wait()
    rc1 = p1.wait()
    if rc0 != 0 or rc1 != 0:
        raise RuntimeError(
            f"{tag} failed for {model}-{seq}-b{batch} (rc0={rc0} rc1={rc1}). Logs: {log0}, {log1}"
        )
    return out_dir0, out_dir1


def collect_party_result(out_dir):
    dealer = parse_dealer(out_dir / "dealer.txt")
    evaluator = parse_evaluator(out_dir / "evaluator.txt")
    return {
        **dealer,
        **evaluator,
    }


def collect_result(out_dir0, out_dir1):
    p0 = collect_party_result(out_dir0)
    p1 = collect_party_result(out_dir1)
    out = {}
    for key in set(p0) | set(p1):
        v0 = p0.get(key)
        v1 = p1.get(key)
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            out[key] = max(v0, v1)
            out[f"p0_{key}"] = v0
            out[f"p1_{key}"] = v1
        else:
            out[key] = v0 if v0 is not None else v1
    out["pair_statistic"] = "max_party"
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Run FuseFSS/Sigma batch scaling experiments.")
    ap.add_argument("--models", type=str, default="bert-base,gpt2")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batches", type=str, default="1,2,4,8")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--addr", type=str, default="127.0.0.1")
    ap.add_argument("--gpu0", type=str, default=os.environ.get("SIGMA_GPU0", "0"))
    ap.add_argument("--gpu1", type=str, default=os.environ.get("SIGMA_GPU1", "1"))
    ap.add_argument("--sigma-bin", type=str, default=str(root / "build" / "gpu_mpc_upstream" / "sigma"))
    ap.add_argument("--fusefss-bin", type=str, default=str(root / "build" / "gpu_mpc_vendor" / "sigma"))
    ap.add_argument("--sigma-run-dir", type=str, default=str(root / "ezpc_upstream" / "GPU-MPC" / "experiments" / "sigma"))
    ap.add_argument("--fusefss-run-dir", type=str, default=str(root / "third_party" / "EzPC_vendor" / "GPU-MPC" / "experiments" / "sigma"))
    ap.add_argument("--mode", type=str, choices=["serial", "internal"], default="serial",
                    help="serial: run batch times with SIGMA_BATCH=1; internal: run once with SIGMA_BATCH=batch")
    ap.add_argument("--pinned-keybuf", type=str, choices=["auto", "on", "off"], default="auto",
                    help="Control SIGMA_PINNED_KEYBUF; auto leaves default.")
    ap.add_argument("--out", type=str, default=str(root / "batch_scaling_results.json"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batches = [int(b.strip()) for b in args.batches.split(",") if b.strip()]

    sigma_bin = Path(args.sigma_bin)
    fusefss_bin = Path(args.fusefss_bin)
    if not sigma_bin.exists():
        raise FileNotFoundError(f"Sigma binary not found: {sigma_bin}")
    if not fusefss_bin.exists():
        raise FileNotFoundError(f"FuseFSS binary not found: {fusefss_bin}")

    sigma_dir = Path(args.sigma_run_dir)
    fusefss_dir = Path(args.fusefss_run_dir)

    base_env = os.environ.copy()
    base_env["SIGMA_MEMPOOL_DISABLE"] = "1"
    base_env["SIGMA_PINNED_COMM_BUFS"] = os.environ.get("SIGMA_PINNED_COMM_BUFS", "0")
    base_env["SIGMA_PINNED_KEYBUF"] = os.environ.get("SIGMA_PINNED_KEYBUF", "0")
    if args.pinned_keybuf == "auto":
        # Ensure we don't inherit a global override; let Sigma defaults decide.
        base_env.pop("SIGMA_PINNED_KEYBUF", None)
    elif args.pinned_keybuf == "on":
        base_env["SIGMA_PINNED_KEYBUF"] = "1"
    elif args.pinned_keybuf == "off":
        base_env["SIGMA_PINNED_KEYBUF"] = "0"
    base_env["OMP_NUM_THREADS"] = str(args.threads)

    fusefss_env = base_env.copy()
    fusefss_env["FUSEFSS_SOFTMAX"] = "1"
    fusefss_env["FUSEFSS_LAYERNORM"] = "1"
    fusefss_env["FUSEFSS_ACTIVATION"] = "1"
    fusefss_env["FUSEFSS_NEXP_BITS"] = "10"
    fusefss_env["FUSEFSS_INV_BITS"] = "10"
    fusefss_env["FUSEFSS_RSQRT_BITS"] = "9"

    results = []
    for model in models:
        for batch in batches:
            for variant, run_dir, bin_path, env, tag in [
                ("sigma", sigma_dir, sigma_bin, base_env, "sigma_batch"),
                ("fusefss", fusefss_dir, fusefss_bin, fusefss_env, "fusefss_batch"),
            ]:
                repeat = batch if (args.mode == "serial" and batch > 1) else 1
                sig_batch = 1 if (args.mode == "serial" and batch > 1) else batch
                total_us = 0
                comm_us = 0
                total_comm_bytes = 0
                keygen_us = 0
                key_bytes = 0
                out_dir0 = None
                out_dir1 = None
                for r in range(repeat):
                    out_dir0, out_dir1 = run_pair(
                        run_dir,
                        bin_path,
                        model,
                        args.seq,
                        sig_batch,
                        args.threads,
                        args.addr,
                        env,
                        args.gpu0,
                        args.gpu1,
                        tag,
                        r,
                    )
                    stats = collect_result(out_dir0, out_dir1)
                    total_us += stats["total_us"]
                    comm_us += stats["comm_us"]
                    total_comm_bytes += stats["total_comm_bytes"]
                    keygen_us += stats["keygen_us"]
                    key_bytes += stats["key_bytes"]
                batch_seen = batch

                total_s = total_us / 1e6
                per_inf_ms = (total_us / batch_seen) / 1000.0
                tokens_per_s = (batch_seen * args.seq) / total_s if total_s else 0.0
                keygen_s = keygen_us / 1e6

                lan_bw = 1e9
                wan_bw = 400e6
                lan_lat = 0.0005
                wan_lat = 0.004
                rounds = None
                if (model, args.seq) in ROUND_COUNTS:
                    rounds = ROUND_COUNTS[(model, args.seq)][0 if variant == "sigma" else 1]
                    rounds *= batch_seen
                lan = project_times(total_us, comm_us, total_comm_bytes, rounds or 0, lan_bw, lan_lat)
                wan = project_times(total_us, comm_us, total_comm_bytes, rounds or 0, wan_bw, wan_lat)

                results.append({
                    "model": model,
                    "seq": args.seq,
                    "batch": batch_seen,
                    "variant": variant,
                    "mode": args.mode,
                    "online_ms_total": total_us / 1000.0,
                    "online_ms_per_inf": per_inf_ms,
                    "tokens_per_s": tokens_per_s,
                    "comm_gb": bytes_to_gb(total_comm_bytes),
                    "comm_gb_per_inf": bytes_to_gb(total_comm_bytes / batch_seen),
                    "keygen_s_total": keygen_s,
                    "keygen_s_per_inf": keygen_s / batch_seen,
                    "key_gb": bytes_to_gb(key_bytes),
                    "key_gb_per_inf": bytes_to_gb(key_bytes / batch_seen),
                    "lan_s": lan,
                    "wan_s": wan,
                    "rounds": rounds,
                    "logs": {
                        "p0_dealer": str(out_dir0 / "dealer.txt"),
                        "p0_evaluator": str(out_dir0 / "evaluator.txt"),
                        "p1_dealer": str(out_dir1 / "dealer.txt"),
                        "p1_evaluator": str(out_dir1 / "evaluator.txt"),
                    },
                })

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
