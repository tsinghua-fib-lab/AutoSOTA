#!/usr/bin/env python3
"""Run TimeGuard defense for all 3 models in parallel on 2 GPUs."""
import subprocess, sys, os, time, json, re, yaml, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_model(model, gpu_id, args, log_dir, ts):
    """Run a single model on specified GPU."""
    config_path = "./configs/timeguard/PEMS03_backtime_FEDformer_1212/%s/TimeGuard.yaml" % model
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    cfg = config_data["Defense"]
    # Map CLI arg names to config key names
    arg_to_cfg = {
        "t2": "t_2", "tb": "t_b", "t1": "t_1",
        "knn": "k_nn", "knn_max": "k_nn_max",
        "alpha": "alpha", "beta": "beta", "lr": "learning_rate"
    }
    for arg_name, cfg_key in arg_to_cfg.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            cfg[cfg_key] = val

    tmp_config = "/tmp/timeguard_%s_%s_gpu%d.yaml" % (model, ts, gpu_id)
    with open(tmp_config, "w") as f:
        yaml.dump(config_data, f)

    log_path = "%s/%s_%s_gpu%d.log" % (log_dir, ts, model, gpu_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("[%s] Starting on GPU %d (log: %s)" % (model, gpu_id, log_path))
    
    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            result = subprocess.run(
                ["python3", "defense_timeguard.py", "--defense_config_path", tmp_config],
                stdout=logf, stderr=subprocess.STDOUT,
                text=True, cwd="/repo", env=env,
                timeout=args.timeout
            )
        elapsed = time.time() - t0

        with open(log_path) as f:
            stdout = f.read()

        match_cln = re.search(r"cln_mae:\s*([0-9.]+)", stdout)
        match_atk = re.search(r"atk_mae:\s*([0-9.]+)", stdout)

        if match_cln and match_atk:
            cln = float(match_cln.group(1))
            atk = float(match_atk.group(1))
            print("[%s] SUCCESS: MAEc=%.4f, MAEp=%.4f (%.1fs)" % (model, cln, atk, elapsed))
            return {"model": model, "cln_mae": cln, "atk_mae": atk, "elapsed": elapsed, "status": "success"}
        else:
            print("[%s] PARSE FAILURE" % model)
            return {"model": model, "elapsed": elapsed, "status": "parse_failure"}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print("[%s] TIMEOUT after %.1fs" % (model, elapsed))
        return {"model": model, "elapsed": elapsed, "status": "timeout"}
    except Exception as e:
        elapsed = time.time() - t0
        print("[%s] ERROR: %s" % (model, str(e)))
        return {"model": model, "elapsed": elapsed, "status": "error", "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2", type=int, default=15)
    parser.add_argument("--tb", type=int, default=5)
    parser.add_argument("--t1", type=int, default=5)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--knn-max", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--log-dir", default="/repo/eval_logs")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-model timeout (s)")
    args = parser.parse_args()

    os.chdir("/repo")
    os.makedirs(args.log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Schedule: FEDformer on GPU0, SimpleTM on GPU1 (parallel)
    # TimesNet on GPU1 after SimpleTM (GPU0 stays with FEDformer since FEDformer is slowest)
    models_gpu = [
        ("FEDformer", 0),
        ("SimpleTM", 1),
    ]

    print("=" * 60)
    print("Parallel TimeGuard Evaluation")
    print("Config: t_b=%d t_1=%d t_2=%d k_nn=%d k_nn_max=%d" % (
        args.tb, args.t1, args.t2, args.knn, args.knn_max))
    print("Per-model timeout: %ds" % args.timeout)
    print("=" * 60)

    results = {}
    
    # Phase 1+2: Run FEDformer (GPU0) and SimpleTM (GPU1) in parallel.
    # When SimpleTM finishes, immediately start TimesNet on GPU1.
    import threading
    timesnet_started = threading.Event()
    
    def run_and_maybe_chain(model, gpu, chain_model=None, chain_gpu=None):
        result = run_model(model, gpu, args, args.log_dir, ts)
        results[model] = result
        if chain_model is not None:
            print("[PIPELINE] %s done, starting %s on GPU%d" % (model, chain_model, chain_gpu))
            chain_result = run_model(chain_model, chain_gpu, args, args.log_dir, ts)
            results[chain_model] = chain_result
            timesnet_started.set()
        return result
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_and_maybe_chain, "FEDformer", 0)
        f2 = executor.submit(run_and_maybe_chain, "SimpleTM", 1, "TimesNet", 1)
        f1.result()
        f2.result()

    # Phase 3: Compute aggregates
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    models = ["FEDformer", "SimpleTM", "TimesNet"]
    cln_list = []
    atk_list = []

    for m in models:
        r = results.get(m, {})
        if r.get("status") == "success":
            print("  %s: MAEc=%.4f, MAEp=%.4f (%.1fs)" % (m, r["cln_mae"], r["atk_mae"], r["elapsed"]))
            cln_list.append(r["cln_mae"])
            atk_list.append(r["atk_mae"])
        else:
            print("  %s: %s" % (m, r.get("status", "unknown")))

    if len(cln_list) == 3:
        avg_cln = sum(cln_list) / 3
        avg_atk = sum(atk_list) / 3

        maec_undef = 17.607
        maep_undef = 14.201
        rho_maep = max(0, 1 - maep_undef / avg_atk)
        rho_maec = max(0, 1 - maec_undef / avg_cln)
        fder = (rho_maep - rho_maec + 1) / 2

        print("\nAverage MAEc: %.4f" % avg_cln)
        print("Average MAEp: %.4f" % avg_atk)
        print("FDER: %.4f" % fder)

        summary = {"avg_cln_mae": avg_cln, "avg_atk_mae": avg_atk, "fder": fder,
                   "individual_cln": cln_list, "individual_atk": atk_list, "results": results}
        print("\nJSON_SUMMARY: %s" % json.dumps(summary))
        return summary
    else:
        print("\nWARNING: Only %d/3 models succeeded" % len(cln_list))
        return {"results": results, "status": "partial"}

if __name__ == "__main__":
    result = main()
