#!/usr/bin/env python3
"""Read accepted Kaggle scores and preserve immutable API snapshots."""
import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    command = [str(ROOT / "envs/kaggle-cli/bin/python"), str(ROOT / "scripts/kaggle_cli.py"),
               "competitions", "submissions", "champs-scalar-coupling", "--format", "json", "--page-size", "100"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    raw = json.loads(result.stdout) if result.stdout.strip() != "No submissions found" else []
    normalized = []
    for row in raw:
        record = dict(row)
        for field in ("publicScore", "privateScore"):
            record[field] = float(record[field]) if record.get(field) not in (None, "", "None") else None
        attempt = ROOT / "artifacts/submission_attempts" / (Path(row["fileName"]).stem + ".json")
        if attempt.exists():
            record["local_file_sha256"] = json.loads(attempt.read_text()).get("sha256")
        normalized.append(record)
    complete = [r for r in normalized if r["status"].endswith(".COMPLETE") and r["privateScore"] is not None]
    control = next((r for r in complete if r["fileName"] == "champion_reproduced.csv"), None)
    best = min(complete, key=lambda r: r["privateScore"]) if complete else None
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    errors_path = ROOT / "artifacts/kaggle_scoring/submission_errors.json"
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else []
    backend_rows_verified = any("Submission must have 2505542 rows" in e.get("error_description", "") for e in errors)
    snapshot = {"utc": now, "competition": "champs-scalar-coupling", "submissions": normalized,
                "champion_control": control, "best_private": best,
                "historic_private_winner_score": -3.23968,
                "beats_control_on_private": bool(best and control and best["privateScore"] < control["privateScore"]),
                "private_score_below_historic_winner": bool(best and best["privateScore"] < -3.23968),
                "new_official_champion_claimed": False,
                "scoring_backend_2505542_rows_verified": backend_rows_verified,
                "scoring_error_records": errors,
                "data_version_caveat": "Current downloadable test has 2505190 rows, while the scoring backend explicitly requires 2505542 rows, matching the organizer archive. These API results are late-submission scores; the historical official leaderboard is unchanged."}
    out = ROOT / "artifacts/kaggle_scoring"
    out.mkdir(exist_ok=True)
    (out / (now.replace(":", "").replace("+", "_") + ".json")).write_text(json.dumps(snapshot, indent=2))
    (out / "latest.json").write_text(json.dumps(snapshot, indent=2))
    status_file = ROOT / "artifacts/project_status.json"
    status = json.loads(status_file.read_text())
    status["new_kaggle_scores_available"] = bool(complete)
    status["beats_winner_verified"] = bool(snapshot["beats_control_on_private"] and snapshot["private_score_below_historic_winner"])
    status["winner_comparison_scope"] = "API late private score below the reproduced winner control and below the historical winner's numeric score; no new official rank or medal claimed."
    status["private_score_below_historic_winner"] = snapshot["private_score_below_historic_winner"]
    status["best_api_private_score"] = best["privateScore"] if best else None
    status["best_api_file"] = "submissions/" + best["fileName"] if best else None
    status["actual_score_records"] = "artifacts/kaggle_scoring/latest.json"
    status["verification_caveat"] = snapshot["data_version_caveat"]
    for record in status.get("submissions", []):
        score = next((r for r in complete if r["fileName"] == Path(record["file"]).name), None)
        if score:
            record["kaggle_score"] = score["privateScore"]
            record["kaggle_public_score"] = score["publicScore"]
            record["submission_ref"] = score["ref"]
    status_file.write_text(json.dumps(status, indent=2))
    print(json.dumps(snapshot, indent=2), flush=True)


if __name__ == "__main__":
    main()
