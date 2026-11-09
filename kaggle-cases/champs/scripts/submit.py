#!/usr/bin/env python3
"""Validate a concrete submission, submit once, and preserve the API result."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from metrics import validate_submission

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--reference-test", type=Path, default=ROOT / "data/raw/test.csv")
    parser.add_argument("--message", default="CHAMPS reproducible inference; no test labels used")
    args = parser.parse_args()
    submission = args.file.resolve()
    validate_submission(pd.read_csv(submission), pd.read_csv(args.reference_test))
    env = os.environ.copy()
    if args.config_dir:
        env["KAGGLE_CONFIG_DIR"] = str(args.config_dir.resolve())
    modern_python = ROOT / "envs/kaggle-cli/bin/python"
    cli = ([str(modern_python), str(ROOT / "scripts/kaggle_cli.py")]
           if modern_python.exists() else [str(ROOT / "envs/champs/bin/kaggle")])
    command = cli + ["competitions", "submit", "champs-scalar-coupling",
               "-f", str(submission), "-m", args.message]
    run = subprocess.run(command, env=env, capture_output=True, text=True)
    text = run.stdout + run.stderr
    # Legacy Kaggle CLI may exit 0 on an HTTP 403; success must come from the API response.
    success = run.returncode == 0 and "Successfully submitted" in text
    record = {"file": str(submission.relative_to(ROOT)), "reference_test": str(args.reference_test.resolve().relative_to(ROOT)), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
              "process_returncode": run.returncode, "accepted_by_kaggle": success, "api_response": text}
    out = ROOT / "artifacts/submission_attempts"
    out.mkdir(exist_ok=True)
    latest = out / (submission.stem + ".json")
    history = out / "history"
    history.mkdir(exist_ok=True)
    if latest.exists():
        previous = json.loads(latest.read_text())
        stamp = previous.get("utc", "previous").replace(":", "").replace("+", "_")
        archived = history / (submission.stem + "_" + stamp + ".json")
        if not archived.exists():
            archived.write_text(json.dumps(previous, indent=2))
    stamp = record["utc"].replace(":", "").replace("+", "_")
    (history / (submission.stem + "_" + stamp + ".json")).write_text(json.dumps(record, indent=2))
    latest.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if success else 1)
