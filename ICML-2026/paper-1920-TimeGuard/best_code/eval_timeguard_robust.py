#!/usr/bin/env python3
"""TimeGuard evaluation script — robust version with per-model logging and timeout handling.
Averaged over FEDformer, SimpleTM, TimesNet per paper Section 3.1.
"""
import subprocess
import sys
import re
import os
import time
import json
import argparse

os.chdir("/repo")

def parse_results(stdout):
    """Extract cln_mae and atk_mae from defense output."""
    match_cln = re.search(r"cln_mae:\s*([0-9.]+)", stdout)
    match_atk = re.search(r"atk_mae:\s*([0-9.]+)", stdout)
    if match_cln and match_atk:
        return float(match_cln.group(1)), float(match_atk.group(1))
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Run single model only")
    parser.add_argument("--t2", type=int, default=None, help="Override t_2 epochs")
    parser.add_argument("--tb", type=int, default=None, help="Override t_b epochs")
    parser.add_argument("--t1", type=int, default=None, help="Override t_1 epochs")
    parser.add_argument("--knn", type=int, default=None, help="Override k_nn")
    parser.add_argument("--knn-max", type=int, default=None, help="Override k_nn_max")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--lr2", type=float, default=None, help="Override learning_rate_phase_2")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha")
    parser.add_argument("--beta", type=float, default=None, help="Override beta")
    parser.add_argument("--log-dir", type=str, default="/repo/eval_logs", help="Log directory")
    args = parser.parse_args()

    MODELS = ["FEDformer", "SimpleTM", "TimesNet"] if args.model is None else [args.model]
    CONFIG_BASE = "./configs/timeguard/PEMS03_backtime_FEDformer_1212"

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results_cln = []
    results_atk = []
    model_results = {}

    print("=" * 60)
    print("TimeGuard Defense Evaluation (Robust)")
    print("Models: %s" % ", ".join(MODELS))
    if args.t2: print("t_2 override: %d" % args.t2)
    if args.tb: print("t_b override: %d" % args.tb)
    print("=" * 60)

    for model_idx, model in enumerate(MODELS):
        print("\n--- Running TimeGuard for %s (%d/%d) ---" % (model, model_idx+1, len(MODELS)))
        
        config_path = "%s/%s/TimeGuard.yaml" % (CONFIG_BASE, model)
        log_path = "%s/%s_%s.log" % (args.log_dir, timestamp, model)
        
        # Build command with optional overrides
        cmd = ["python3", "defense_timeguard.py", "--defense_config_path", config_path]
        
        # Use config override approach: modify YAML before running
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        modified = False
        if args.t2 is not None:
            config_data["Defense"]["t_2"] = args.t2
            modified = True
        if args.tb is not None:
            config_data["Defense"]["t_b"] = args.tb
            modified = True
        if args.t1 is not None:
            config_data["Defense"]["t_1"] = args.t1
            modified = True
        if args.knn is not None:
            config_data["Defense"]["k_nn"] = args.knn
            modified = True
        if args.knn_max is not None:
            config_data["Defense"]["k_nn_max"] = args.knn_max
            modified = True
        if args.lr is not None:
            config_data["Defense"]["learning_rate"] = args.lr
            modified = True
        if args.lr2 is not None:
            config_data["Defense"]["learning_rate_phase_2"] = args.lr2
            modified = True
        if args.alpha is not None:
            config_data["Defense"]["alpha"] = args.alpha
            modified = True
        if args.beta is not None:
            config_data["Defense"]["beta"] = args.beta
            modified = True
        
        if modified:
            tmp_config = "/tmp/timeguard_%s_%s.yaml" % (model, timestamp)
            with open(tmp_config, "w") as f:
                yaml.dump(config_data, f)
            cmd = ["python3", "defense_timeguard.py", "--defense_config_path", tmp_config]
        
        print("  Command: %s" % " ".join(cmd))
        print("  Log: %s" % log_path)
        sys.stdout.flush()
        
        t0 = time.time()
        try:
            with open(log_path, "w") as logf:
                result = subprocess.run(
                    cmd,
                    stdout=logf, stderr=subprocess.STDOUT,
                    text=True, cwd="/repo"
                )
            elapsed = time.time() - t0
            
            # Read back the log
            with open(log_path) as f:
                stdout = f.read()
            
            cln, atk = parse_results(stdout)
            if cln is not None and atk is not None:
                results_cln.append(cln)
                results_atk.append(atk)
                model_results[model] = {"cln_mae": cln, "atk_mae": atk, "elapsed": elapsed}
                print("  %s: MAEc=%.6f, MAEp=%.6f (%.1fs)" % (model, cln, atk, elapsed))
            else:
                print("  ERROR: Could not parse results for %s" % model)
                print("  Log tail: %s" % stdout[-300:])
                model_results[model] = {"cln_mae": None, "atk_mae": None, "elapsed": elapsed, "error": "parse_failure"}
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print("  ERROR: %s timed out after %.1fs" % (model, elapsed))
            model_results[model] = {"cln_mae": None, "atk_mae": None, "elapsed": elapsed, "error": "timeout"}
        except Exception as e:
            elapsed = time.time() - t0
            print("  ERROR: %s failed: %s" % (model, str(e)))
            model_results[model] = {"cln_mae": None, "atk_mae": None, "elapsed": elapsed, "error": str(e)}

    # Print aggregated results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for model in MODELS:
        r = model_results.get(model, {})
        if r.get("cln_mae") is not None:
            print("  %s: MAEc=%.3f, MAEp=%.3f (%.1fs)" % (model, r["cln_mae"], r["atk_mae"], r["elapsed"]))
        else:
            print("  %s: FAILED (%s)" % (model, r.get("error", "unknown")))

    if len(results_cln) == len(MODELS):
        # Compute averages
        avg_cln = sum(results_cln) / len(results_cln)
        avg_atk = sum(results_atk) / len(results_atk)

        # Compute FDER using undefended baselines from paper Table 3
        maec_undef = 17.607
        maep_undef = 14.201

        rho_maep = max(0, 1 - maep_undef / avg_atk)
        rho_maec = max(0, 1 - maec_undef / avg_cln)
        fder = (rho_maep - rho_maec + 1) / 2

        print("\nModels averaged: %s" % ", ".join(MODELS))
        print("Individual MAEc: %s" % ", ".join("%.3f" % c for c in results_cln))
        print("Individual MAEp: %s" % ", ".join("%.3f" % a for a in results_atk))
        print("Average MAEc: %.3f" % avg_cln)
        print("Average MAEp: %.3f" % avg_atk)
        print("FDER: %.3f" % fder)
        print("\nPaper reported:    MAEc=18.048, MAEp=39.303, FDER=0.808")
        print("Our result:        MAEc=%.3f, MAEp=%.3f, FDER=%.3f" % (avg_cln, avg_atk, fder))
        
        # Print JSON summary for easy parsing
        summary = {
            "avg_cln_mae": avg_cln,
            "avg_atk_mae": avg_atk,
            "fder": fder,
            "individual_cln": results_cln,
            "individual_atk": results_atk,
            "model_results": model_results
        }
        print("\nJSON_SUMMARY: %s" % json.dumps(summary))
    else:
        print("\nWARNING: Only %d/%d models completed successfully" % (len(results_cln), len(MODELS)))
        print("Cannot compute reliable averages.")

    print("\n=== Evaluation Complete ===")
    return 0 if len(results_cln) == len(MODELS) else 1

if __name__ == "__main__":
    sys.exit(main())
