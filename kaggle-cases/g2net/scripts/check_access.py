"""Probe Kaggle through its configured SDK without printing credentials."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from kaggle import api

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "g2net-gravitational-wave-detection"


def main():
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "competition": COMPETITION}
    try:
        api.competition_download_file(COMPETITION, "sample_submission.csv", path=str(ROOT / "data/raw"), quiet=True)
        result["download_access"] = True
    except requests.HTTPError as exc:
        result["download_access"] = False
        result["http_status"] = exc.response.status_code
        try:
            error = exc.response.json().get("error", {})
            result["message"] = error.get("message", "")
            result["reasons"] = [d.get("reason") for d in error.get("details", []) if d.get("reason")]
        except ValueError:
            result["message"] = "Server returned a non-JSON error; response omitted."
    (ROOT / "reports/access_status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["download_access"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
