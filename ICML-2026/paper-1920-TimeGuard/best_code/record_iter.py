#!/usr/bin/env python3
"""Parse parallel eval output and record score via record_score.sh"""
import sys, os, json, re, subprocess

def main():
    if len(sys.argv) < 5:
        print("Usage: record_iter.py ITER IDEA_ID TITLE EXTRA_ARGS_STR OUTPUT_FILE")
        sys.exit(1)

    iter_num = sys.argv[1]
    idea_id = sys.argv[2]
    title = sys.argv[3]
    extra_args = sys.argv[4]
    output_file = sys.argv[5]

    # Read the output
    with open(output_file) as f:
        text = f.read()

    # Find JSON_SUMMARY
    match = re.search(r'JSON_SUMMARY:\s*(\{.*\})', text, re.DOTALL)
    if not match:
        print("FAILED: No JSON_SUMMARY found in output")
        # Record as failed
        subprocess.run([
            "/tools/record_score.sh",
            "--scores", "/autosota_artifacts/paper-1920/sota/scores.jsonl",
            "--iter", iter_num,
            "--idea-id", idea_id,
            "--title", title,
            "--status", "failed",
            "--primary", "0.0",
            "--metrics", "{}",
            "--notes", "No JSON_SUMMARY in output. Extra args: %s" % extra_args
        ])
        sys.exit(1)

    try:
        summary = json.loads(match.group(1))
    except json.JSONDecodeError:
        print("FAILED: JSON parse error")
        subprocess.run([
            "/tools/record_score.sh",
            "--scores", "/autosota_artifacts/paper-1920/sota/scores.jsonl",
            "--iter", iter_num,
            "--idea-id", idea_id,
            "--title", title,
            "--status", "failed",
            "--primary", "0.0",
            "--metrics", "{}",
            "--notes", "JSON parse error in output. Extra args: %s" % extra_args
        ])
        sys.exit(1)

    if "avg_cln_mae" not in summary:
        print("FAILED: Incomplete results")
        subprocess.run([
            "/tools/record_score.sh",
            "--scores", "/autosota_artifacts/paper-1920/sota/scores.jsonl",
            "--iter", iter_num,
            "--idea-id", idea_id,
            "--title", title,
            "--status", "failed",
            "--primary", "0.0",
            "--metrics", "{}",
            "--notes", "Incomplete results. Extra args: %s" % extra_args
        ])
        sys.exit(1)

    avg_cln = summary["avg_cln_mae"]
    avg_atk = summary["avg_atk_mae"]
    fder = summary["fder"]
    ind_cln = summary.get("individual_cln", [])
    ind_atk = summary.get("individual_atk", [])

    metrics = json.dumps({"MAEc": avg_cln, "MAEp": avg_atk, "FDER": fder})
    notes = "Patches: CODE-001/002/004 + ALGO-001 (cosine annealing). Args: %s. Individual: " % extra_args
    if len(ind_cln) == 3:
        notes += "F=%.3f/%.3f, S=%.3f/%.3f, T=%.3f/%.3f" % (
            ind_cln[0], ind_atk[0], ind_cln[1], ind_atk[1], ind_cln[2], ind_atk[2])

    print("MAEc=%.4f, MAEp=%.4f, FDER=%.4f" % (avg_cln, avg_atk, fder))

    result = subprocess.run([
        "/tools/record_score.sh",
        "--scores", "/autosota_artifacts/paper-1920/sota/scores.jsonl",
        "--iter", iter_num,
        "--idea-id", idea_id,
        "--title", title,
        "--status", "success",
        "--primary", str(avg_cln),
        "--metrics", metrics,
        "--notes", notes
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)

if __name__ == "__main__":
    main()
