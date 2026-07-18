#!/usr/bin/env python3
import argparse
import csv
import datetime as _datetime
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path


MODEL_DISPLAY = {
    "bert-tiny": "BERT-tiny",
    "bert-base": "BERT-base",
    "bert-large": "BERT-large",
    "gpt2": "GPT-2",
    "gpt-neo": "GPT-Neo",
    "llama7b": "LLaMA-7B",
    "llama13b": "LLaMA-13B",
    "llama3-8b": "LLaMA-3.1-8B",
}

MODEL_ALIASES = {
    "llama-7b": "llama7b",
    "llama2-7b": "llama7b",
    "llama8b": "llama3-8b",
    "llama-8b": "llama3-8b",
    "llama3-8b": "llama3-8b",
    "llama3.1-8b": "llama3-8b",
    "llama-3.1-8b": "llama3-8b",
    "llama-3-8b": "llama3-8b",
}

PAPER_MAIN_TASKS = [
    ("bert-tiny", 128),
    ("bert-base", 128),
    ("bert-large", 128),
    ("gpt2", 128),
    ("gpt-neo", 128),
]

LLAMA_TARGETS = {
    ("llama7b", 16): {"min_speedup": 1.09, "max_fusefss_key_gb": 68.95},
    ("llama7b", 32): {"min_speedup": 1.17, "max_fusefss_key_gb": 90.17},
    ("llama7b", 64): {"min_speedup": 1.22, "max_fusefss_key_gb": 134.19},
    ("llama3-8b", 16): {"min_speedup": 1.05, "max_fusefss_key_gb": 83.09},
    ("llama3-8b", 32): {"min_speedup": 1.11, "max_fusefss_key_gb": 108.74},
    ("llama3-8b", 64): {"min_speedup": 1.17, "max_fusefss_key_gb": 160.04},
}

LLAMA_CLOSE_TOLERANCE = {
    "speedup_relative": 0.05,
    "key_relative": 0.02,
}

PAPER_MAIN_TARGET = {
    "min_speedup": 1.24,
    "min_comm_reduction": 0.09,
    "min_keygen_speedup": 1.14,
    # Table 1's GPT-Neo row is 81.805 -> 65.729 GB, i.e. 19.65% and
    # reported in the paper text as the rounded 20% lower end.
    "min_key_reduction": 0.195,
}

ROUND_COUNTS = {
    ("bert-tiny", 128): (188, 186),
    ("bert-base", 32): (1080, 1068),
    ("bert-base", 64): (1104, 1092),
    ("bert-base", 128): (1128, 1116),
    ("bert-large", 128): (2256, 2232),
    ("gpt2", 64): (1104, 1092),
    ("gpt2", 128): (1128, 1116),
    ("gpt2", 256): (1152, 1140),
    ("gpt-neo", 64): (2208, 2184),
    ("gpt-neo", 128): (2256, 2232),
}

DEFAULT_TASKS = PAPER_MAIN_TASKS + list(LLAMA_TARGETS.keys())


def canonical_model_name(model):
    m = model.strip().lower()
    return MODEL_ALIASES.get(m, m)


def display_model(model):
    return MODEL_DISPLAY.get(model, model)


def is_llama_model(model):
    return model.startswith("llama")


def run_cmd(cmd, cwd, env, log_path):
    with open(log_path, "w") as f:
        return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=f)


def affinity_command(cmd, cpu_affinity):
    if not cpu_affinity:
        return cmd
    taskset = shutil.which("taskset")
    if not taskset:
        return cmd
    return [taskset, "-c", cpu_affinity] + cmd


def auto_cpu_affinities():
    cpu_count = os.cpu_count() or 0
    if cpu_count >= 384:
        # Common two-socket RTX PRO 6000 Blackwell validation hosts expose
        # GPU0 near NUMA node0 and GPU1 near NUMA node1.
        return "0-95,192-287", "96-191,288-383"
    if cpu_count >= 8:
        mid = cpu_count // 2
        return f"0-{mid - 1}", f"{mid}-{cpu_count - 1}"
    return "", ""


def ensure_output_dirs(run_dir):
    (run_dir / "output" / "P0" / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "output" / "P1" / "models").mkdir(parents=True, exist_ok=True)


def wait_pair(p0, p1, timeout_s, tag, model, seq, log0, log1):
    try:
        rc0 = p0.wait(timeout=timeout_s if timeout_s > 0 else None)
        rc1 = p1.wait(timeout=timeout_s if timeout_s > 0 else None)
    except subprocess.TimeoutExpired as exc:
        p0.kill()
        p1.kill()
        p0.wait()
        p1.wait()
        raise TimeoutError(
            f"{tag} timed out for {model}-{seq} after {timeout_s}s. Logs: {log0}, {log1}"
        ) from exc
    return rc0, rc1


def run_sigma_pair(run_dir, bin_path, model, seq, threads, addr, env_base, gpu0, gpu1, tag, log_dir, timeout_s,
                   cpu0_affinity="", cpu1_affinity=""):
    ensure_output_dirs(run_dir)
    out_dir0 = run_dir / "output" / "P0" / "models" / f"{model}-{seq}"
    out_dir1 = run_dir / "output" / "P1" / "models" / f"{model}-{seq}"
    if out_dir0.exists():
        shutil.rmtree(out_dir0)
    if out_dir1.exists():
        shutil.rmtree(out_dir1)

    env0 = env_base.copy()
    env1 = env_base.copy()
    if gpu0 is not None:
        env0["CUDA_VISIBLE_DEVICES"] = str(gpu0)
    if gpu1 is not None:
        env1["CUDA_VISIBLE_DEVICES"] = str(gpu1)

    cmd0 = affinity_command([str(bin_path), model, str(seq), "0", addr, str(threads)], cpu0_affinity)
    cmd1 = affinity_command([str(bin_path), model, str(seq), "1", addr, str(threads)], cpu1_affinity)

    log_dir.mkdir(parents=True, exist_ok=True)
    log0 = log_dir / "p0.log"
    log1 = log_dir / "p1.log"

    wall_start = time.monotonic()
    p0 = run_cmd(cmd0, run_dir, env0, log0)
    time.sleep(1.0)
    p1 = run_cmd(cmd1, run_dir, env1, log1)
    rc0, rc1 = wait_pair(p0, p1, timeout_s, tag, model, seq, log0, log1)
    wall_s = time.monotonic() - wall_start
    if rc0 != 0 or rc1 != 0:
        raise RuntimeError(
            f"{tag} failed for {model}-{seq} (rc0={rc0} rc1={rc1}). Logs: {log0}, {log1}"
        )
    return out_dir0, out_dir1, wall_s


def parse_dealer(path):
    text = path.read_text()
    total_us = int(re.search(r"Total time=(\d+) us", text).group(1))
    key_bytes = int(re.search(r"Key size=(\d+) B", text).group(1))
    return {"keygen_us": total_us, "key_bytes": key_bytes}


def parse_evaluator(path):
    text = path.read_text()
    total_us = int(re.search(r"Total time=(\d+) us", text).group(1))
    comm_us = int(re.search(r"Comm time=(\d+) us", text).group(1))
    total_comm_bytes = int(re.search(r"Total Comm=(\d+) B", text).group(1))
    out = {
        "total_us": total_us,
        "comm_us": comm_us,
        "total_comm_bytes": total_comm_bytes,
    }
    for label, key in [
        ("Per-inference time", "per_inference_us"),
        ("Transfer time", "transfer_us"),
        ("MHA time", "mha_us"),
        ("Matmul time", "matmul_us"),
        ("Truncate time", "truncate_us"),
        ("Gelu time", "gelu_us"),
        ("Softmax time", "softmax_us"),
        ("Layernorm time", "layernorm_us"),
    ]:
        m = re.search(rf"{re.escape(label)}=(\d+) us", text)
        if m:
            out[key] = int(m.group(1))
    for label, key in [
        ("Gelu Comm", "gelu_comm"),
        ("Softmax Comm", "softmax_comm"),
        ("Layernorm Comm", "layernorm_comm"),
    ]:
        m = re.search(rf"{re.escape(label)}=(\d+) B", text)
        if m:
            bytes_val = int(m.group(1))
            out[f"{key}_bytes"] = bytes_val
            out[f"{key}_gb"] = bytes_to_gb(bytes_val)
    return out


def bytes_to_gb(bytes_val):
    # Sigma's benchmark binary prints "GB" as bytes / 1024^3.  Use the same
    # convention here so reproduced tables compare directly with the paper
    # artifact tables and the raw Sigma logs.  Raw byte counts are preserved in
    # results.json for independent unit conversions.
    return bytes_val / (1024.0 ** 3)


def project_times(total_us, comm_us, comm_bytes, rounds, bandwidth, latency_s):
    comp_time = (total_us - comm_us) / 1e6
    # Sigma logs Total Comm as bytesSent()+bytesReceived(), i.e. an aggregate
    # bidirectional byte count. Do not multiply by two again here.
    return comp_time + comm_bytes / bandwidth + rounds * latency_s


def collect_party_result(out_dir):
    dealer = parse_dealer(out_dir / "dealer.txt")
    evaluator = parse_evaluator(out_dir / "evaluator.txt")
    return {
        **dealer,
        **evaluator,
        "online_ms": evaluator["total_us"] / 1000.0,
        "comm_gb": bytes_to_gb(evaluator["total_comm_bytes"]),
        "keygen_s": dealer["keygen_us"] / 1e6,
        "key_gb": bytes_to_gb(dealer["key_bytes"]),
    }


def collect_pair_result(out_dir0, out_dir1, wall_s=None):
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
    out["pair_wall_s"] = wall_s if wall_s is not None else 0.0
    out["pair_statistic"] = "max_party"
    out["online_ms"] = out["total_us"] / 1000.0
    out["comm_gb"] = bytes_to_gb(out["total_comm_bytes"])
    out["keygen_s"] = out["keygen_us"] / 1e6
    out["key_gb"] = bytes_to_gb(out["key_bytes"])
    return out


def copy_raw_outputs(out_dir, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dealer.txt", "evaluator.txt"):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, raw_dir / name)


def copy_pair_raw_outputs(out_dir0, out_dir1, raw_dir):
    copy_raw_outputs(out_dir0, raw_dir / "P0")
    copy_raw_outputs(out_dir1, raw_dir / "P1")


def parse_tasks(task_text):
    tasks = []
    for item in task_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            model, seq_text = item.split(":", 1)
        elif "-" in item:
            model, seq_text = item.rsplit("-", 1)
        else:
            raise ValueError(f"Task must be model:seq, got {item!r}")
        model = canonical_model_name(model)
        if model not in MODEL_DISPLAY:
            raise ValueError(f"Unknown model {model!r}. Known models: {', '.join(sorted(MODEL_DISPLAY))}")
        tasks.append((model, int(seq_text)))
    if not tasks:
        raise ValueError("At least one task is required")
    return tasks


def median_result(run_results):
    if not run_results:
        raise ValueError("No non-warmup runs available for median")
    keys = run_results[0].keys()
    out = {}
    for key in keys:
        vals = [r[key] for r in run_results]
        out[key] = statistics.median(vals) if isinstance(vals[0], (int, float)) else vals[0]
        if isinstance(vals[0], (int, float)):
            out[f"{key}_min"] = min(vals)
            out[f"{key}_max"] = max(vals)
    out["runs_used"] = len(run_results)
    return out


def choose_existing_path(requested, fallbacks):
    path = Path(requested)
    if path.exists():
        return path
    for fallback in fallbacks:
        fb = Path(fallback)
        if fb.exists():
            return fb
    return path


def safe_cmd_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # best-effort metadata only
        return f"unavailable: {exc}"


def sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return "unavailable"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def collect_metadata(args, sigma_bin, fusefss_bin):
    fusefss_env_keys = [
        "FUSEFSS_SOFTMAX",
        "FUSEFSS_LAYERNORM",
        "FUSEFSS_ACTIVATION",
        "FUSEFSS_NEXP_BITS",
        "FUSEFSS_INV_BITS",
        "FUSEFSS_RSQRT_BITS",
        "FUSEFSS_SIGMA_GENERIC",
        "FUSEFSS_SIGMA_GENERIC_STRICT",
        "FUSEFSS_SIGMA_VECTOR_LUT_LEGACY",
        "SIGMA_PINNED_COMM_BUFS",
        "SIGMA_PINNED_KEYBUF",
        "SIGMA_COMM_BUF_MB",
        "SIGMA_COMM_BUF_GB",
        "SIGMA_LLAMA_PORT_BASE",
    ]
    return {
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "hostname": safe_cmd_output(["hostname"]),
        "uname": safe_cmd_output(["uname", "-a"]),
        "nvidia_smi": safe_cmd_output([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]),
        "nvcc": safe_cmd_output(["nvcc", "--version"]),
        "git_commit": safe_cmd_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": safe_cmd_output(["git", "status", "--short"]),
        "gpu_arch": os.environ.get("GPU_ARCH", ""),
        "binaries": {
            "sigma": {"path": str(sigma_bin), "sha256": sha256_file(sigma_bin)},
            "fusefss": {"path": str(fusefss_bin), "sha256": sha256_file(fusefss_bin)},
        },
        "fusefss_env": {k: os.environ.get(k, "") for k in fusefss_env_keys},
        "paper_protocol_note": (
            "Paper measurements use two non-colluding parties on two RTX PRO 6000 Blackwell GPUs; "
            "the --single-gpu mode runs both local parties on one visible GPU only for low-cost ratio checks."
        ),
        "run_config": {
            "tasks": args.tasks,
            "runs": args.runs,
            "warmup": args.warmup,
            "single_gpu": args.single_gpu,
            "gpu0": args.gpu0,
            "gpu1": args.gpu1,
            "auto_cpu_affinity": args.auto_cpu_affinity,
            "cpu0_affinity": args.cpu0_affinity,
            "cpu1_affinity": args.cpu1_affinity,
            "threads": args.threads,
            "timeout_s": args.timeout_s,
            "fusefss_sigma_generic": args.fusefss_sigma_generic,
            "fusefss_sigma_generic_strict": args.fusefss_sigma_generic_strict,
            "fusefss_sigma_vector_lut_legacy": os.environ.get(
                "FUSEFSS_SIGMA_VECTOR_LUT_LEGACY",
                os.environ.get("SUF_SIGMA_VECTOR_LUT_LEGACY", ""),
            ),
            "llama_close_tolerance": LLAMA_CLOSE_TOLERANCE,
        },
    }


def fmt_float(x, places=2):
    if x is None:
        return "-"
    return f"{x:.{places}f}"


def fmt_speedup(a, b):
    if b == 0:
        return "inf"
    return f"{a / b:.2f}x"


def displayed_key_gb(value):
    return round(value + 1e-12, 2)


def llama_target_status(speedup, key_gb, target):
    display_key = displayed_key_gb(key_gb)
    speed_ok = speedup >= target["min_speedup"]
    key_ok = display_key <= target["max_fusefss_key_gb"]
    if speed_ok and key_ok:
        return "PASS"
    close_speed_ok = speedup >= target["min_speedup"] * (1.0 - LLAMA_CLOSE_TOLERANCE["speedup_relative"])
    close_key_ok = display_key <= target["max_fusefss_key_gb"] * (1.0 + LLAMA_CLOSE_TOLERANCE["key_relative"])
    return "PASS_CLOSE" if close_speed_ok and close_key_ok else "FAIL"


def make_tables(results):
    lan_bw = 1e9
    wan_bw = 400e6
    lan_lat = 0.0005
    wan_lat = 0.004

    def lan_wan(model, seq, variant):
        if (model, seq) not in ROUND_COUNTS:
            return None, None, None
        rounds = ROUND_COUNTS[(model, seq)][0 if variant == "sigma" else 1]
        r = results[(model, seq, variant)]
        lan = project_times(r["total_us"], r["comm_us"], r["total_comm_bytes"], rounds, lan_bw, lan_lat)
        wan = project_times(r["total_us"], r["comm_us"], r["total_comm_bytes"], rounds, wan_bw, wan_lat)
        return rounds, lan, wan

    # Section 3 main table (seq=128)
    base_models = [model for model, seq in PAPER_MAIN_TASKS if seq == 128]
    base_rows = []
    for model in base_models:
        seq = 128
        if (model, seq, "sigma") not in results or (model, seq, "fusefss") not in results:
            continue
        sigma = results[(model, seq, "sigma")]
        fusefss = results[(model, seq, "fusefss")]
        s_rounds, s_lan, s_wan = lan_wan(model, seq, "sigma")
        u_rounds, u_lan, u_wan = lan_wan(model, seq, "fusefss")
        base_rows.append({
            "model": f"{display_model(model)}-{seq}",
            "sigma_ms": sigma["online_ms"],
            "fusefss_ms": fusefss["online_ms"],
            "speedup": fmt_speedup(sigma["online_ms"], fusefss["online_ms"]),
            "sigma_comm": sigma["comm_gb"],
            "fusefss_comm": fusefss["comm_gb"],
            "sigma_rounds": s_rounds,
            "fusefss_rounds": u_rounds,
            "sigma_lan": s_lan,
            "fusefss_lan": u_lan,
            "sigma_wan": s_wan,
            "fusefss_wan": u_wan,
        })

    # Keygen/key size table
    key_rows = []
    for model in base_models:
        seq = 128
        if (model, seq, "sigma") not in results or (model, seq, "fusefss") not in results:
            continue
        sigma = results[(model, seq, "sigma")]
        fusefss = results[(model, seq, "fusefss")]
        key_rows.append({
            "model": f"{display_model(model)}-{seq}",
            "sigma_keygen": sigma["keygen_s"],
            "fusefss_keygen": fusefss["keygen_s"],
            "sigma_key": sigma["key_gb"],
            "fusefss_key": fusefss["key_gb"],
        })

    # Additional seq (GPT-2 / GPT-Neo)
    extra_rows = []
    for model, seqs in [("gpt2", [64, 128, 256]), ("gpt-neo", [64, 128])]:
        for seq in seqs:
            if (model, seq, "sigma") not in results or (model, seq, "fusefss") not in results:
                continue
            sigma = results[(model, seq, "sigma")]
            fusefss = results[(model, seq, "fusefss")]
            s_rounds, s_lan, s_wan = lan_wan(model, seq, "sigma")
            u_rounds, u_lan, u_wan = lan_wan(model, seq, "fusefss")
            extra_rows.append({
                "model": display_model(model),
                "seq": seq,
                "sigma_ms": sigma["online_ms"],
                "fusefss_ms": fusefss["online_ms"],
                "speedup": fmt_speedup(sigma["online_ms"], fusefss["online_ms"]),
                "sigma_comm": sigma["comm_gb"],
                "fusefss_comm": fusefss["comm_gb"],
                "sigma_rounds": s_rounds,
                "fusefss_rounds": u_rounds,
                "sigma_lan": s_lan,
                "fusefss_lan": u_lan,
                "sigma_wan": s_wan,
                "fusefss_wan": u_wan,
            })

    # Scaling (BERT-base)
    scale_rows = []
    for seq in [32, 64, 128]:
        if ("bert-base", seq, "sigma") not in results or ("bert-base", seq, "fusefss") not in results:
            continue
        sigma = results[("bert-base", seq, "sigma")]
        fusefss = results[("bert-base", seq, "fusefss")]
        s_rounds, s_lan, s_wan = lan_wan("bert-base", seq, "sigma")
        u_rounds, u_lan, u_wan = lan_wan("bert-base", seq, "fusefss")
        scale_rows.append({
            "seq": seq,
            "sigma_ms": sigma["online_ms"],
            "fusefss_ms": fusefss["online_ms"],
            "speedup": fmt_speedup(sigma["online_ms"], fusefss["online_ms"]),
            "sigma_comm": sigma["comm_gb"],
            "fusefss_comm": fusefss["comm_gb"],
            "sigma_rounds": s_rounds,
            "fusefss_rounds": u_rounds,
            "sigma_lan": s_lan,
            "fusefss_lan": u_lan,
            "sigma_wan": s_wan,
            "fusefss_wan": u_wan,
        })

    llama_rows = []
    for (model, seq), target in LLAMA_TARGETS.items():
        if (model, seq, "sigma") not in results or (model, seq, "fusefss") not in results:
            continue
        sigma = results[(model, seq, "sigma")]
        fusefss = results[(model, seq, "fusefss")]
        speedup = sigma["online_ms"] / fusefss["online_ms"] if fusefss["online_ms"] else 0.0
        llama_rows.append({
            "model": display_model(model),
            "seq": seq,
            "sigma_ms": sigma["online_ms"],
            "fusefss_ms": fusefss["online_ms"],
            "speedup": speedup,
            "min_speedup": target["min_speedup"],
            "sigma_key": sigma["key_gb"],
            "fusefss_key": fusefss["key_gb"],
            "max_fusefss_key": target["max_fusefss_key_gb"],
            "target_status": llama_target_status(speedup, fusefss["key_gb"], target),
        })

    return base_rows, key_rows, extra_rows, scale_rows, llama_rows


def make_pair_rows(results, tasks):
    rows = []
    for model, seq in tasks:
        sigma = results.get((model, seq, "sigma"))
        fusefss = results.get((model, seq, "fusefss"))
        if not sigma or not fusefss:
            continue
        speedup = sigma["online_ms"] / fusefss["online_ms"] if fusefss["online_ms"] else 0.0
        comm_reduction = 1.0 - (fusefss["comm_gb"] / sigma["comm_gb"]) if sigma["comm_gb"] else 0.0
        keygen_speedup = sigma["keygen_s"] / fusefss["keygen_s"] if fusefss["keygen_s"] else 0.0
        key_reduction = 1.0 - (fusefss["key_gb"] / sigma["key_gb"]) if sigma["key_gb"] else 0.0
        target = LLAMA_TARGETS.get((model, seq))
        if target:
            target_status = llama_target_status(speedup, fusefss["key_gb"], target)
            min_speedup = target["min_speedup"]
            max_fusefss_key_gb = target["max_fusefss_key_gb"]
        elif (model, seq) in PAPER_MAIN_TASKS:
            target_status = (
                "PASS"
                if speedup >= PAPER_MAIN_TARGET["min_speedup"]
                and comm_reduction >= PAPER_MAIN_TARGET["min_comm_reduction"]
                and keygen_speedup >= PAPER_MAIN_TARGET["min_keygen_speedup"]
                and key_reduction >= PAPER_MAIN_TARGET["min_key_reduction"]
                else "FAIL"
            )
            min_speedup = PAPER_MAIN_TARGET["min_speedup"]
            max_fusefss_key_gb = None
        else:
            target_status = "N/A"
            min_speedup = None
            max_fusefss_key_gb = None
        rows.append({
            "model": display_model(model),
            "model_id": model,
            "seq": seq,
            "sigma_online_ms": sigma["online_ms"],
            "fusefss_online_ms": fusefss["online_ms"],
            "speedup": speedup,
            "sigma_comm_gb": sigma["comm_gb"],
            "fusefss_comm_gb": fusefss["comm_gb"],
            "comm_reduction": comm_reduction,
            "sigma_keygen_s": sigma["keygen_s"],
            "fusefss_keygen_s": fusefss["keygen_s"],
            "keygen_speedup": keygen_speedup,
            "sigma_key_gb": sigma["key_gb"],
            "fusefss_key_gb": fusefss["key_gb"],
            "key_reduction": key_reduction,
            "min_speedup_target": min_speedup,
            "max_fusefss_key_gb_target": max_fusefss_key_gb,
            "acceptance_status": target_status,
        })
    return rows


def acceptance_summary(pair_rows):
    checked = [r for r in pair_rows if r["acceptance_status"] in ("PASS", "PASS_CLOSE", "FAIL")]
    failed = [r for r in checked if r["acceptance_status"] == "FAIL"]
    return {
        "checked": len(checked),
        "passed": len(checked) - len(failed),
        "close": sum(1 for r in checked if r["acceptance_status"] == "PASS_CLOSE"),
        "failed": len(failed),
        "status": "PASS" if not failed else "FAIL",
        "failures": failed,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    default_tasks = ",".join(f"{model}:{seq}" for model, seq in DEFAULT_TASKS)
    default_results_dir = root / "results" / (
        "paper_eval_" + _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(description="Reproduce FuseFSS/Sigma evaluation tables.")
    parser.add_argument("--sigma-bin", type=str, default=str(root / "build" / "gpu_mpc_upstream" / "sigma"))
    parser.add_argument("--fusefss-bin", type=str, default=str(root / "build" / "gpu_mpc_vendor" / "sigma"))
    parser.add_argument("--sigma-run-dir", type=str, default=str(root / "ezpc_upstream" / "GPU-MPC" / "experiments" / "sigma"))
    parser.add_argument("--fusefss-run-dir", type=str, default=str(root / "third_party" / "EzPC_vendor" / "GPU-MPC" / "experiments" / "sigma"))
    parser.add_argument("--tasks", type=str, default=default_tasks,
                        help="Comma-separated model:seq list, e.g. bert-tiny:128,gpt2:128")
    parser.add_argument("--runs", type=int, default=5,
                        help="Total repetitions per variant. Paper uses 5.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Discard this many leading repetitions. Paper discards the first run.")
    parser.add_argument("--results-dir", type=str, default=str(default_results_dir),
                        help="Directory for raw logs, JSON, and CSV summaries.")
    parser.add_argument("--timeout-s", type=int, default=7200,
                        help="Timeout per two-party Sigma run. Set 0 to disable.")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--addr", type=str, default="127.0.0.1")
    parser.add_argument("--gpu0", type=str, default="0")
    parser.add_argument("--gpu1", type=str, default="1")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Run both parties on CUDA_VISIBLE_DEVICES=0 for low-cost checks.")
    parser.add_argument("--auto-cpu-affinity", action="store_true",
                        help="Bind the two local parties to disjoint CPU sets when taskset is available.")
    parser.add_argument("--cpu0-affinity", type=str, default="",
                        help="CPU list passed to taskset for party 0, e.g. 0-95,192-287.")
    parser.add_argument("--cpu1-affinity", type=str, default="",
                        help="CPU list passed to taskset for party 1, e.g. 96-191,288-383.")
    parser.add_argument("--fusefss-sigma-generic", action="store_true",
                        help="Run the FuseFSS binary with FUSEFSS_SIGMA_GENERIC=1.")
    parser.add_argument("--fusefss-sigma-generic-strict", action="store_true",
                        help="Run the FuseFSS binary with FUSEFSS_SIGMA_GENERIC_STRICT=1.")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--csv-out", type=str, default="")
    parser.add_argument("--assert-targets", action="store_true",
                        help="Exit nonzero if paper-main or Llama acceptance targets are missed.")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not args.no_run and args.warmup >= args.runs:
        raise ValueError("--warmup must be smaller than --runs")

    if args.single_gpu:
        args.gpu0 = "0"
        args.gpu1 = "0"
    if args.auto_cpu_affinity and not args.single_gpu:
        auto0, auto1 = auto_cpu_affinities()
        args.cpu0_affinity = args.cpu0_affinity or auto0
        args.cpu1_affinity = args.cpu1_affinity or auto1

    sigma_bin = choose_existing_path(
        args.sigma_bin,
        [root / "ezpc_upstream" / "GPU-MPC" / "experiments" / "sigma" / "sigma"],
    )
    fusefss_bin = choose_existing_path(
        args.fusefss_bin,
        [root / "third_party" / "EzPC_vendor" / "GPU-MPC" / "experiments" / "sigma" / "sigma"],
    )
    sigma_dir = Path(args.sigma_run_dir)
    fusefss_dir = Path(args.fusefss_run_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_root = results_dir / "raw"

    tasks = parse_tasks(args.tasks)
    if not args.no_run:
        for name, path in [("Sigma baseline", sigma_bin), ("FuseFSS", fusefss_bin)]:
            if not path.exists():
                raise FileNotFoundError(f"{name} binary not found: {path}")

    results = {}
    per_run = {}
    sigma_env = os.environ.copy()
    sigma_env.pop("FUSEFSS_SIGMA_GENERIC", None)
    sigma_env.pop("FUSEFSS_SIGMA_GENERIC_STRICT", None)
    sigma_env.pop("SUF_SIGMA_GENERIC", None)
    sigma_env.pop("SUF_SIGMA_GENERIC_STRICT", None)
    sigma_env["SIGMA_MEMPOOL_DISABLE"] = os.environ.get("SIGMA_MEMPOOL_DISABLE", "1")
    sigma_env["SIGMA_PINNED_COMM_BUFS"] = os.environ.get("SIGMA_PINNED_COMM_BUFS", "0")
    sigma_env["SIGMA_PINNED_KEYBUF"] = os.environ.get("SIGMA_PINNED_KEYBUF", "0")
    sigma_env["OMP_NUM_THREADS"] = str(args.threads)

    fusefss_env = os.environ.copy()
    fusefss_env.pop("FUSEFSS_SIGMA_GENERIC", None)
    fusefss_env.pop("FUSEFSS_SIGMA_GENERIC_STRICT", None)
    fusefss_env.pop("SUF_SIGMA_GENERIC", None)
    fusefss_env.pop("SUF_SIGMA_GENERIC_STRICT", None)
    fusefss_env["SIGMA_MEMPOOL_DISABLE"] = os.environ.get("SIGMA_MEMPOOL_DISABLE", "1")
    fusefss_env["SIGMA_PINNED_COMM_BUFS"] = os.environ.get("SIGMA_PINNED_COMM_BUFS", "0")
    fusefss_env["SIGMA_PINNED_KEYBUF"] = os.environ.get("SIGMA_PINNED_KEYBUF", "0")
    fusefss_env["OMP_NUM_THREADS"] = str(args.threads)
    fusefss_env["FUSEFSS_SOFTMAX"] = "1"
    fusefss_env["FUSEFSS_LAYERNORM"] = "1"
    fusefss_env["FUSEFSS_ACTIVATION"] = "1"
    fusefss_env["FUSEFSS_NEXP_BITS"] = "10"
    fusefss_env["FUSEFSS_INV_BITS"] = "10"
    fusefss_env["FUSEFSS_RSQRT_BITS"] = "9"
    if args.fusefss_sigma_generic_strict:
        fusefss_env["FUSEFSS_SIGMA_GENERIC_STRICT"] = "1"
    elif args.fusefss_sigma_generic:
        fusefss_env["FUSEFSS_SIGMA_GENERIC"] = "1"

    pair_run_counter = 0
    for model, seq in tasks:
        for variant, run_dir, bin_path, env, tag in [
            ("sigma", sigma_dir, sigma_bin, sigma_env, "sigma_base"),
            ("fusefss", fusefss_dir, fusefss_bin, fusefss_env, "fusefss"),
        ]:
            run_results = []
            if args.no_run:
                out_dir0 = run_dir / "output" / "P0" / "models" / f"{model}-{seq}"
                out_dir1 = run_dir / "output" / "P1" / "models" / f"{model}-{seq}"
                if not out_dir0.exists() or not out_dir1.exists():
                    raise FileNotFoundError(f"Missing output pair: {out_dir0}, {out_dir1}")
                res = collect_pair_result(out_dir0, out_dir1)
                res["run_idx"] = 0
                res["warmup"] = False
                run_results.append(res)
            else:
                for run_idx in range(args.runs):
                    run_tag = f"{tag}_r{run_idx:02d}"
                    raw_dir = raw_root / f"{model}_{seq}" / variant / f"run_{run_idx:02d}"
                    print(f"[{variant}] {model}-{seq} run {run_idx + 1}/{args.runs}")
                    run_env = env.copy()
                    llama_port_base = None
                    if is_llama_model(model):
                        llama_port_base = 43000 + pair_run_counter * 10
                        pair_run_counter += 1
                        run_env["SIGMA_LLAMA_PORT_BASE"] = str(llama_port_base)
                    out_dir0, out_dir1, wall_s = run_sigma_pair(
                        run_dir,
                        bin_path,
                        model,
                        seq,
                        args.threads,
                        args.addr,
                        run_env,
                        args.gpu0,
                        args.gpu1,
                        run_tag,
                        raw_dir,
                        args.timeout_s,
                        args.cpu0_affinity,
                        args.cpu1_affinity,
                    )
                    if not out_dir0.exists() or not out_dir1.exists():
                        raise FileNotFoundError(f"Missing output pair: {out_dir0}, {out_dir1}")
                    copy_pair_raw_outputs(out_dir0, out_dir1, raw_dir)
                    res = collect_pair_result(out_dir0, out_dir1, wall_s)
                    res["run_idx"] = run_idx
                    res["warmup"] = run_idx < args.warmup
                    if llama_port_base is not None:
                        res["llama_port_base"] = llama_port_base
                    run_results.append(res)

            usable = [r for r in run_results if not r["warmup"]]
            results[(model, seq, variant)] = median_result(usable)
            per_run[f"{model}:{seq}:{variant}"] = run_results

    base_rows, key_rows, extra_rows, scale_rows, llama_rows = make_tables(results)
    pair_rows = make_pair_rows(results, tasks)
    acceptance = acceptance_summary(pair_rows)

    json_path = Path(args.json_out) if args.json_out else results_dir / "results.json"
    csv_path = Path(args.csv_out) if args.csv_out else results_dir / "summary.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(
            {
                "metadata": collect_metadata(args, sigma_bin, fusefss_bin),
                "median_results": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in results.items()},
                "per_run": per_run,
                "summary": pair_rows,
                "acceptance": acceptance,
                "base": base_rows,
                "key": key_rows,
                "extra": extra_rows,
                "scale": scale_rows,
                "llama": llama_rows,
            },
            f,
            indent=2,
        )
    write_csv(csv_path, pair_rows)

    def print_table(headers, rows, keys, fmts):
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            cells = []
            for k, fmt in zip(keys, fmts):
                val = row[k]
                if fmt:
                    cells.append(fmt(val))
                else:
                    cells.append(str(val))
            print("| " + " | ".join(cells) + " |")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    if args.single_gpu:
        print("Note: --single-gpu uses one physical GPU for both parties; paper hardware uses two GPUs.")

    print("\n## Requested Task Summary\n")
    print_table(
        ["Model", "Seq", "Sigma online (ms)", "FuseFSS online (ms)", "Speedup",
         "Sigma comm (GB)", "FuseFSS comm (GB)", "Keygen speedup", "Key reduction", "Target"],
        pair_rows,
        ["model", "seq", "sigma_online_ms", "fusefss_online_ms", "speedup",
         "sigma_comm_gb", "fusefss_comm_gb", "keygen_speedup", "key_reduction", "acceptance_status"],
        [None, None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
         lambda v: f"{v:.2f}x", lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3),
         lambda v: f"{v:.2f}x", lambda v: f"{100.0 * v:.1f}%", None],
    )

    if base_rows:
        print("\n## FuseFSS vs Sigma (end-to-end, seq=128)\n")
        print_table(
            ["Model", "Sigma online (ms)", "FuseFSS online (ms)", "Speedup", "Sigma comm (GB)", "FuseFSS comm (GB)",
             "Sigma rounds", "FuseFSS rounds", "Sigma LAN (s)", "FuseFSS LAN (s)", "Sigma WAN (s)", "FuseFSS WAN (s)"],
            base_rows,
            ["model", "sigma_ms", "fusefss_ms", "speedup", "sigma_comm", "fusefss_comm",
             "sigma_rounds", "fusefss_rounds", "sigma_lan", "fusefss_lan", "sigma_wan", "fusefss_wan"],
            [None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2), None,
             lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3),
             None, None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
             lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2)],
        )

    if key_rows:
        print("\n### Keygen and key size\n")
        print_table(
            ["Model", "Sigma keygen (s)", "FuseFSS keygen (s)", "Sigma key (GB)", "FuseFSS key (GB)"],
            key_rows,
            ["model", "sigma_keygen", "fusefss_keygen", "sigma_key", "fusefss_key"],
            [None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
             lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3)],
        )

    if extra_rows:
        print("\n### Additional sequence points (GPT-2 / GPT-Neo)\n")
        print_table(
            ["Model", "Seq", "Sigma time (ms)", "FuseFSS time (ms)", "Speedup", "Sigma comm (GB)", "FuseFSS comm (GB)",
             "Sigma rounds", "FuseFSS rounds", "Sigma LAN (s)", "FuseFSS LAN (s)", "Sigma WAN (s)", "FuseFSS WAN (s)"],
            extra_rows,
            ["model", "seq", "sigma_ms", "fusefss_ms", "speedup", "sigma_comm", "fusefss_comm",
             "sigma_rounds", "fusefss_rounds", "sigma_lan", "fusefss_lan", "sigma_wan", "fusefss_wan"],
            [None, None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2), None,
             lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3), None, None,
             lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
             lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2)],
        )

    if llama_rows:
        print("\n### Llama extension targets\n")
        print_table(
            ["Model", "Seq", "Sigma (ms)", "FuseFSS (ms)", "Speedup", "Min speedup",
             "Sigma key (GB)", "FuseFSS key (GB)", "Max FuseFSS key (GB)", "Target"],
            llama_rows,
            ["model", "seq", "sigma_ms", "fusefss_ms", "speedup", "min_speedup",
             "sigma_key", "fusefss_key", "max_fusefss_key", "target_status"],
            [None, None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
             lambda v: f"{v:.2f}x", lambda v: f"{v:.2f}x",
             lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
             lambda v: fmt_float(v, 2), None],
        )

    if scale_rows:
        print("\n### Scaling (BERT-base seq sweep)\n")
        print_table(
            ["Seq", "Sigma time (ms)", "FuseFSS time (ms)", "Speedup", "Sigma comm (GB)", "FuseFSS comm (GB)",
             "Sigma rounds", "FuseFSS rounds", "Sigma LAN (s)", "FuseFSS LAN (s)", "Sigma WAN (s)", "FuseFSS WAN (s)"],
            scale_rows,
            ["seq", "sigma_ms", "fusefss_ms", "speedup", "sigma_comm", "fusefss_comm",
             "sigma_rounds", "fusefss_rounds", "sigma_lan", "fusefss_lan", "sigma_wan", "fusefss_wan"],
            [None, lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2), None,
             lambda v: fmt_float(v, 3), lambda v: fmt_float(v, 3), None, None,
             lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2),
             lambda v: fmt_float(v, 2), lambda v: fmt_float(v, 2)],
        )

    close_note = f", {acceptance['close']} close" if acceptance.get("close") else ""
    print(f"\nAcceptance: {acceptance['status']} "
          f"({acceptance['passed']}/{acceptance['checked']} checked targets passed{close_note})")
    if args.assert_targets and acceptance["failed"]:
        failed = ", ".join(f"{r['model']}:{r['seq']}" for r in acceptance["failures"])
        raise SystemExit(f"Acceptance targets failed: {failed}")


if __name__ == "__main__":
    main()
