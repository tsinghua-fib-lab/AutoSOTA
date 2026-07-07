"""Extract best MSE/MAE from QuITE training logs."""
import sys, os, re, glob

def extract_best(log_path):
    if not os.path.exists(log_path):
        return None
    best_mse = float("inf")
    best_mae = float("inf")
    best_epoch = -1
    with open(log_path) as f:
        for line in f:
            m = re.search(r"MAPE: (\d+), ([\d.]+), [\d.]+, [\d.]+, ([\d.]+)", line)
            if m:
                epoch = int(m.group(1))
                mse = float(m.group(2))
                mae = float(m.group(3))
                if mse < best_mse:
                    best_mse = mse
                    best_mae = mae
                    best_epoch = epoch
    if best_epoch >= 0:
        return {"mse": best_mse, "mae": best_mae, "epoch": best_epoch}
    return None

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="logs/physionet_patchmixer_quite_*seed.log")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    
    results = {}
    for f in sorted(glob.glob(args.pattern)):
        seed_match = re.search(r"(\d+)seed", f)
        seed = seed_match.group(1) if seed_match else "?"
        r = extract_best(f)
        if r:
            results[seed] = r
            print("Seed %s: epoch=%d MSE=%.5f MAE=%.5f" % (seed, r["epoch"], r["mse"], r["mae"]))
    
    if len(results) > 0:
        avg_mse = sum(r["mse"] for r in results.values()) / len(results)
        avg_mae = sum(r["mae"] for r in results.values()) / len(results)
        print("Average (%d seeds): MSE=%.5f MAE=%.5f" % (len(results), avg_mse, avg_mae))
