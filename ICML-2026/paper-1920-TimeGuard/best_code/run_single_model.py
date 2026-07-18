#!/usr/bin/env python3
"""Run a single TimeGuard defense model and output parseable results."""
import subprocess, sys, os, time, json, re, yaml, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["FEDformer", "SimpleTM", "TimesNet"])
    parser.add_argument("--t2", type=int, default=45)
    parser.add_argument("--tb", type=int, default=10)
    parser.add_argument("--t1", type=int, default=10)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--knn-max", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr2", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--log-dir", default="/repo/eval_logs")
    parser.add_argument("--timeout", type=int, default=5400, help="Per-model timeout in seconds")
    args = parser.parse_args()

    os.chdir("/repo")
    os.makedirs(args.log_dir, exist_ok=True)
    
    config_path = "./configs/timeguard/PEMS03_backtime_FEDformer_1212/%s/TimeGuard.yaml" % args.model
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    cfg = config_data["Defense"]
    cfg["t_2"] = args.t2
    cfg["t_b"] = args.tb
    cfg["t_1"] = args.t1
    cfg["k_nn"] = args.knn
    cfg["k_nn_max"] = args.knn_max
    cfg["learning_rate"] = args.lr
    if args.lr2 is not None:
        cfg["learning_rate_phase_2"] = args.lr2
    cfg["alpha"] = args.alpha
    cfg["beta"] = args.beta

    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_config = "/tmp/timeguard_%s_%s.yaml" % (args.model, ts)
    with open(tmp_config, "w") as f:
        yaml.dump(config_data, f)

    log_path = "%s/%s_%s.log" % (args.log_dir, ts, args.model)
    print("Running %s (log: %s)" % (args.model, log_path))
    print("Config: t_b=%d t_1=%d t_2=%d k_nn=%d k_nn_max=%d lr=%.1e alpha=%.2f beta=%.2f" % (
        args.tb, args.t1, args.t2, args.knn, args.knn_max, args.lr, args.alpha, args.beta))
    sys.stdout.flush()

    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            result = subprocess.run(
                ["python3", "defense_timeguard.py", "--defense_config_path", tmp_config],
                stdout=logf, stderr=subprocess.STDOUT,
                text=True, cwd="/repo", timeout=args.timeout
            )
        elapsed = time.time() - t0

        with open(log_path) as f:
            stdout = f.read()

        match_cln = re.search(r"cln_mae:\s*([0-9.]+)", stdout)
        match_atk = re.search(r"atk_mae:\s*([0-9.]+)", stdout)

        if match_cln and match_atk:
            cln = float(match_cln.group(1))
            atk = float(match_atk.group(1))
            result_json = {"model": args.model, "cln_mae": cln, "atk_mae": atk, "elapsed": elapsed, "status": "success"}
            print("RESULT: %s" % json.dumps(result_json))
        else:
            result_json = {"model": args.model, "elapsed": elapsed, "status": "parse_failure"}
            print("RESULT: %s" % json.dumps(result_json))
            print("STDERR: Parse failure - log tail: %s" % stdout[-300:])

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result_json = {"model": args.model, "elapsed": elapsed, "status": "timeout"}
        print("RESULT: %s" % json.dumps(result_json))
    except Exception as e:
        elapsed = time.time() - t0
        result_json = {"model": args.model, "elapsed": elapsed, "status": "error", "error": str(e)}
        print("RESULT: %s" % json.dumps(result_json))

if __name__ == "__main__":
    main()
