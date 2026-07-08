import json, glob

out = {}

for path in sorted(glob.glob("results_new/showcase/eig/eig_seed*_logs.json")):
    name = path.split("/")[-1].replace("_logs.json", "")
    with open(path) as f:
        out[name] = json.load(f)

# 🚀 Write in COMPACT mode (no indent, minimal separators)
with open("results_new/showcase/eig/all_eig_runs.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

print("Wrote all_eig_runs.json with", len(out), "runs.")

