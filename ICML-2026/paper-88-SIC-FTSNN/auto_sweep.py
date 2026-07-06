#!/usr/bin/env python3
"""
Parallel fault-ratio sweep for SNN scripts with options (baseline/ecoc/soft/routing/frag).
- Spawns multiple Python processes concurrently **within the same fault_ratio** group.
- Processes fault_ratio groups **sequentially** to reduce peak GPU memory usage.
- Supports depth handling:
  * VGG: --vgg_depths 7,11,15  (or --vgg_depth 11 for single)
  * ResNet: --resnet_depths 18,34 (or --resnet_depth 18 for single)
  * MLP (simple_snn.py): no depth needed.
- Records per-run stdout/stderr and appends a single CSV with accuracy.
"""

import argparse
import os
import re
import json
import csv
import time
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

RE_ACC = re.compile(r"(?:Test\s*Set\s*)?Classification?\s*Accuracy\s*[:=]\s*([0-9.]+)%", re.IGNORECASE)
RE_CT  = re.compile(r"Total\s+correctly\s+classified.*?:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)

BOOL_KEYS = {"ECOC", "Soft", "Routing", "Astrocyte", "Falvolt", "LIFA", "Frag", "Proposed"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, required=True, help="Training script: vgg_snn.py / simple_snn.py / resnet_snn.py")
    ap.add_argument("--python", type=str, default="python3")
    ap.add_argument("--results_dir", type=str, default="results_auto_sweep")
    ap.add_argument("--devices", type=str, default="0", help="Comma-separated device ids, or 'cpu'")
    ap.add_argument("--max_procs", type=int, default=2, help="Max parallel processes **within a single fault_ratio group**")
    ap.add_argument("--save_logs", choices=["on", "off"], default="off",
                    help="Save per-run stdout/stderr/cmd/args to disk (default: off). Set off to disable.")
    ap.add_argument("--runs_dir", type=str, default=None,
                    help="Root directory for per-run logs (default: <results_dir>_runs). Used only when save_logs=on.")

    # core sweep controls
    ap.add_argument("--fault_type", type=str, default="stuck", choices=["stuck","random","connectivity"])
    ap.add_argument("--fault_ratio_start", type=float, required=True)
    ap.add_argument("--fault_ratio_stop", type=float, required=True)
    ap.add_argument("--fault_ratio_step", type=float, required=True)
    ap.add_argument("--options", type=str, required=True,
                    help="Comma-separated list among: baseline,ecoc,soft,routing,frag")

    # model-agnostic basics
    ap.add_argument("--num_steps", type=int, default=2)
    ap.add_argument("--epochs", type=int, dest="num_epochs", default=50)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--data_path", type=str, default=None)

    # depth handling (optional)
    ap.add_argument("--vgg_depth", type=int, default=None, help="Single depth for VGG (7/11/15)")
    ap.add_argument("--vgg_depths", type=str, default=None, help="Comma-separated depths for VGG, e.g., '7,11,15'")
    ap.add_argument("--resnet_depth", type=int, default=None, help="Single depth for ResNet (18/34)")
    ap.add_argument("--resnet_depths", type=str, default=None, help="Comma-separated depths for ResNet, e.g., '18,34'")

    # misc
    ap.add_argument("--fault_dist", type=str, default="sporadic")
    ap.add_argument("--fault_start_epoch", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.001)

    return ap.parse_args()

def frange(start, stop, step):
    vals = []
    cur = start
    eps = step * 1e-6
    while cur <= stop + eps:
        vals.append(round(cur, 10))
        cur += step
    return vals

def csv_init(csv_path):
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "script","option","fault_ratio","fault_type",
                "num_steps","depth","device","start_time","duration_sec",
                "returncode","test_acc_percent","correct","total",
                "stdout_path","stderr_path","cmd"
            ])

def options_to_flags(opt_name):
    base = {"ECOC": False, "Soft": False, "Routing": False, "Astrocyte":False, "Falvolt": False, "LIFA":False, "Frag": False}
    opt = opt_name.lower()
    if opt == "baseline":
        return base
    elif opt == "ecoc":
        d = dict(base); d["ECOC"]=True; return d
    elif opt == "soft":
        d = dict(base); d["Soft"]=True; return d
    elif opt == "routing":
        d = dict(base); d["Routing"]=True; return d
    elif opt == "astrocyte":
        d = dict(base); d["Astrocyte"] = True; return d
    elif opt == "falvolt":
        d = dict(base); d["Falvolt"] = True; return d
    elif opt == "lifa":
        d = dict(base); d["LIFA"] = True; return d
    elif opt == "frag":
        d = dict(base); d["Frag"]=True; return d
    elif opt == "proposed":
        d = dict(base); d["Frag"] = True; return d
    else:
        raise ValueError(f"Unknown option '{opt_name}'")

def to_cli(k, v):
    if k in BOOL_KEYS:
        sv = str(v).strip().lower()
        if sv in {"1", "true", "yes", "y", "on"} or v is True:
            return [f"--{k}", "True"]
        else:
            return []
    return [f"--{k}", str(v)]

def device_list(arg):
    arg = arg.strip()
    if arg.lower() == "cpu":
        return ["cpu"]
    return [x.strip() for x in arg.split(",") if x.strip()!=""]

def env_for_device(device_id):
    env = dict(os.environ)
    if device_id == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return env


def parse_metrics(stdout_text):
    acc = None; corr = None; tot = None
    for ln in stdout_text.splitlines():
        m = RE_ACC.search(ln)
        if m:
            try:
                acc = float(m.group(1))
            except:
                pass
        m2 = RE_CT.search(ln)
        if m2:
            try:
                corr = int(m2.group(1)); tot = int(m2.group(2))
            except:
                pass

    if acc is None and (corr is not None) and (tot is not None) and tot > 0:
        acc = round(100.0 * corr / tot, 4)

    return acc, corr, tot

def percent_col(r):
    # format like '0%', '10%', '5%' even for non 0.1 steps
    val = int(round(float(r)*100))
    return f"{val}%"

def write_pivot_csv(results_root, script, depth_str, fault_type, num_steps, ratios, opts, table):
    # table: dict option -> dict ratio -> acc
    stem = Path(script).stem
    depth_tag = (depth_str if depth_str else 'NA')
    out_name = f"{stem}-d{depth_tag}-{fault_type}-ts{num_steps}.csv"
    out_path = Path(results_root) / out_name
    # header
    cols = ['option'] + [percent_col(r) for r in ratios]
    lines = []
    lines.append(','.join(cols))
    for opt in opts:
        row = [opt]
        for r in ratios:
            acc = table.get(opt, {}).get(r, '')
            if isinstance(acc, (int, float)):
                row.append(f"{acc}")
            else:
                row.append("")
        lines.append(','.join(map(str, row)))
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"[CSV] Wrote pivot CSV -> {out_path}")

def launch(job, save_logs="on"):
    env = env_for_device(job["device"])
    start = time.time()
    job["start_time"] = start
    job["start_iso"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if save_logs == "on":
        out = open(job["run_dir"] / "stdout.txt", "w", encoding="utf-8")
        err = open(job["run_dir"] / "stderr.txt", "w", encoding="utf-8")
        p = subprocess.Popen(job["cli"], stdout=out, stderr=err, env=env, text=True)
        job["proc"] = p
        job["stdout_file"] = out
        job["stderr_file"] = err
    else:
        p = subprocess.Popen(job["cli"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        job["proc"] = p
        job["stdout_file"] = None
        job["stderr_file"] = None

    print(f"[LAUNCH] pid={p.pid} dev={job['device']} opt={job['meta']['option']} depth={job['meta']['depth']} fr={job['meta']['fault_ratio']} -> {job['run_dir']}")
    return job


def run_group(pending, max_procs, results_dir, save_logs="on"):
    results = []  # accumulate results per job: dict with option, ratio, depth, acc, meta
    running = []
    # fill initial slots
    while pending and len(running) < max_procs:
        running.append(launch(pending.pop(0), save_logs=save_logs))

    # monitor loop
    while running or pending:
        time.sleep(1.0)
        for j in list(running):
            rc = j["proc"].poll()
            if rc is None:
                continue

            # finished
            if j.get("stdout_file"):
                try:
                    j["stdout_file"].flush();
                    j["stdout_file"].close()
                except Exception:
                    pass
            if j.get("stderr_file"):
                try:
                    j["stderr_file"].flush();
                    j["stderr_file"].close()
                except Exception:
                    pass

            end = time.time()
            dur = round(end - j["start_time"], 2)

            # parse metrics
            if save_logs == "on":
                std_path = j["run_dir"] / "stdout.txt"
                txt = ""
                if std_path.exists():
                    with open(std_path, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                acc, corr, tot = parse_metrics(txt)
                stdout_path = str(std_path)
                stderr_path = str(j["run_dir"] / "stderr.txt")
            else:
                out_txt, err_txt = j["proc"].communicate()
                acc, corr, tot = parse_metrics(out_txt or "")
                stdout_path = ""
                stderr_path = ""

            # save compact result record
            results.append({
                "script": j["meta"]["script"],
                "option": j["meta"]["option"],
                "fault_ratio": j["meta"]["fault_ratio"],
                "fault_type": j["meta"]["fault_type"],
                "num_steps": j["meta"]["num_steps"],
                "depth": j["meta"]["depth"],
                "device": j["device"],
                "start": j["start_iso"],
                "duration_sec": dur,
                "returncode": rc,
                "acc": acc,
                "stdout": stdout_path,
                "stderr": stderr_path
            })
            print(
                f"[DONE] pid={j['proc'].pid} rc={rc} opt={j['meta']['option']} depth={j['meta']['depth']} fr={j['meta']['fault_ratio']} acc={acc}")

            running.remove(j)
            if pending:
                running.append(launch(pending.pop(0), save_logs=save_logs))
    return results


def build_depth_list(script, vgg_depths_str, vgg_depth_single, res_depths_str, res_depth_single):
    script = script.lower()
    if script.endswith("vgg_snn.py"):
        if vgg_depths_str:
            return [int(x) for x in vgg_depths_str.split(",") if x.strip()!=""]
        elif vgg_depth_single is not None:
            return [int(vgg_depth_single)]
        else:
            # default VGG depth if none provided
            return [11]
    elif script.endswith("resnet_snn.py"):
        if res_depths_str:
            return [int(x) for x in res_depths_str.split(",") if x.strip()!=""]
        elif res_depth_single is not None:
            return [int(res_depth_single)]
        else:
            return [18]
    else:
        # simple_snn.py or others: no depth concept
        return [None]

def main():

    args = parse_args()
    results_root = Path(args.results_dir); results_root.mkdir(parents=True, exist_ok=True)

    runs_root = None
    if args.save_logs == "on":
        runs_root = Path(args.runs_dir).resolve() if args.runs_dir else Path(args.results_dir + "_runs").resolve()
        runs_root.mkdir(parents=True, exist_ok=True)

    results_root = Path(args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    devices = device_list(args.devices)
    ratios = frange(args.fault_ratio_start, args.fault_ratio_stop, args.fault_ratio_step)
    opts = [o.strip().lower() for o in args.options.split(",") if o.strip()!=""]

    depth_list = build_depth_list(args.script, args.vgg_depths, args.vgg_depth, args.resnet_depths, args.resnet_depth)

    device_idx = 0

    # ===== process by ratio groups sequentially =====
    # Prepare per-depth aggregation: depth_str -> { option -> { ratio -> acc } }
    from collections import defaultdict
    depth_tables = {}  # depth_str -> table
    for r in ratios:
        print(f"\n====== Starting group: fault_ratio={r} ======")
        group_jobs = []
        for depth in depth_list:
            for opt in opts:
                flags = options_to_flags(opt)

                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                depth_tag = (f"d{depth}" if depth is not None else "dNA")
                run_id = f"{Path(args.script).stem}-{depth_tag}-{opt}-fr{str(r).replace('.','_')}-{ts}"

                if args.save_logs == "on":
                    run_dir = (runs_root / run_id) if runs_root else (results_root / run_id)
                    run_dir.mkdir(parents=True, exist_ok=True)
                else:
                    run_dir = None

                run_args = {
                    "num_steps": args.num_steps,
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "fault_type": args.fault_type,
                    "fault_dist": args.fault_dist,
                    "fault_start_epoch": args.fault_start_epoch,
                    "learning_rate": args.learning_rate,
                    "Fault": (False if r==0.0 else True),
                    "fault_ratio": r
                }
                if args.data_path:
                    run_args["data_path"] = args.data_path

                depth_str = ""
                if args.script.endswith("vgg_snn.py") and depth is not None:
                    run_args["vgg_depth"] = depth
                    depth_str = str(depth)
                elif args.script.endswith("resnet_snn.py") and depth is not None:
                    run_args["resnet_depth"] = depth
                    depth_str = str(depth)
                else:
                    depth_str = ""  # MLP or unspecified

                run_args.update(flags)

                cli = [args.python, args.script]
                for k,v in run_args.items():
                    cli += to_cli(k,v)

                if args.save_logs == "on":
                    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
                        json.dump(run_args, f, indent=2)
                    with open(run_dir / "cmd.txt", "w", encoding="utf-8") as f:
                        f.write(" ".join(shlex.quote(x) for x in cli) + "\n")

                device = devices[device_idx % len(devices)]

                if device != "cpu":
                    cli += ["--gpu_num", str(int(device))]

                device_idx += 1

                group_jobs.append({
                    "cli": cli,
                    "run_dir": run_dir,
                    "device": device,
                    "meta": {
                        "script": args.script,
                        "option": opt,
                        "fault_ratio": r,
                        "fault_type": args.fault_type,
                        "num_steps": args.num_steps,
                        "depth": depth_str
                    }
                })

        records = run_group(pending=list(group_jobs), max_procs=args.max_procs, results_dir=args.results_dir,
                            save_logs=args.save_logs)

        # aggregate
        for rec in records:
            dkey = rec["depth"] if rec["depth"] != "" else "NA"
            if dkey not in depth_tables:
                depth_tables[dkey] = defaultdict(dict)
            depth_tables[dkey][rec["option"]][rec["fault_ratio"]] = rec["acc"]
        print(f"====== Finished group: fault_ratio={r} ======")

    # After all ratios done, write one CSV per depth
    for dkey, table in depth_tables.items():
        write_pivot_csv(results_root, args.script, dkey, args.fault_type, args.num_steps, ratios, opts, table)

if __name__ == "__main__":
    main()
