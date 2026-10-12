#!/usr/bin/env bash
# record_score.sh — called after each optimization iteration (runs on the host, inside repo_path).
#
# Usage:
#   bash /path/to/record_score.sh \
#       --repo    /absolute/path/to/repo \
#       --scores  /path/to/scores.jsonl \
#       --iter    <N> \
#       --idea-id <IDEA-XXX|baseline> \
#       --title   "<idea title>" \
#       --status  <success|failed> \
#       --primary <float> \
#       --metrics '<json object, e.g. {"metric_a": 1.23, "metric_b": 4.56}>' \
#       --notes   "<optional notes>"
#
# What it does:
#   1. git add -A && git commit (captures the current working-tree state as a real commit)
#   2. Reads the real commit hash via git rev-parse HEAD
#   3. On success AND improvement: moves the _best tag to this commit
#   4. Appends one JSON line to scores.jsonl with the real hash
#
# Exit codes: 0 = OK, 1 = argument error

set -euo pipefail

# ── parse args ────────────────────────────────────────────────────────────────
REPO_ROOT=""
SCORES=""
ITER=""
IDEA_ID=""
TITLE=""
STATUS=""
PRIMARY=""
METRICS="{}"
NOTES=""
IS_BEST=""   # "true" / "false" / "" (auto-detect, only used when status=success)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO_ROOT="$2"; shift 2 ;;
    --scores)   SCORES="$2";   shift 2 ;;
    --iter)     ITER="$2";     shift 2 ;;
    --idea-id)  IDEA_ID="$2";  shift 2 ;;
    --title)    TITLE="$2";    shift 2 ;;
    --status)   STATUS="$2";   shift 2 ;;
    --primary)  PRIMARY="$2";  shift 2 ;;
    --metrics)  METRICS="$2";  shift 2 ;;
    --notes)    NOTES="$2";    shift 2 ;;
    --is-best)  IS_BEST="$2";  shift 2 ;;
    *) echo "[record_score] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

for var in REPO_ROOT SCORES ITER IDEA_ID TITLE STATUS PRIMARY; do
  if [[ -z "${!var}" ]]; then
    echo "[record_score] Missing required arg: --${var,,}" >&2
    exit 1
  fi
done

# ── derive run_dir from scores path (scores → results → run_dir) ────────────
RUN_DIR="$(cd "$(dirname "$(dirname "$SCORES")")" && pwd)"

# ── protected-paths SHA256 verification ──────────────────────────────────────
# Reads `protected_paths: [list of repo-relative paths]` from the run's
# effective_config.yaml. Snapshots SHA256 on the baseline iteration; on every
# later iteration verifies the hashes match. Mismatch → exit 9 (PROTOCOL
# VIOLATION), the iteration is NOT committed/recorded so it cannot move _best.
#
# Rationale: prompt-level red-lines alone don't stop models from "improving"
# numbers by silently editing the eval script. Cutting the reward signal
# (no row in scores.jsonl, no _best update) is the only thing that works.
EFFECTIVE_CFG="$RUN_DIR/logs/effective_config.yaml"
PROTECTED_HASH_FILE="$REPO_ROOT/.autosota_protected_hashes.json"

if [[ -f "$EFFECTIVE_CFG" ]]; then
  set +e
  python3 - "$EFFECTIVE_CFG" "$REPO_ROOT" "$PROTECTED_HASH_FILE" "$ITER" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

import yaml

cfg_path, repo_str, hash_file, iter_str = sys.argv[1:5]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

protected = cfg.get("protected_paths") or []
if not protected:
    sys.exit(0)

repo = Path(repo_str)


def compute_hashes(rel_paths):
    out = {}
    for rel in rel_paths:
        full = (repo / rel).resolve()
        if full.is_file():
            out[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
        elif full.is_dir():
            for sub in sorted(full.rglob("*")):
                if sub.is_file():
                    key = str(sub.relative_to(repo))
                    out[key] = hashlib.sha256(sub.read_bytes()).hexdigest()
        else:
            out[rel] = "<missing>"
    return out


cur = compute_hashes(protected)
hash_path = Path(hash_file)
is_baseline = iter_str in ("0", "baseline")

if is_baseline or not hash_path.exists():
    hash_path.write_text(json.dumps(cur, indent=2, sort_keys=True))
    print(f"[record_score] protected snapshot stored: {len(cur)} file(s) → {hash_path}")
    sys.exit(0)

baseline = json.loads(hash_path.read_text())
diffs = sorted(k for k in set(cur) | set(baseline) if cur.get(k) != baseline.get(k))

if not diffs:
    print(f"[record_score] protected files OK ({len(cur)} checked)")
    sys.exit(0)

print("[record_score] ❌ PROTOCOL VIOLATION — protected file(s) modified:")
for k in diffs:
    was = (baseline.get(k) or "<absent>")[:12]
    now = (cur.get(k) or "<deleted>")[:12]
    print(f"    - {k}   ({was} → {now})")
print("[record_score] This iteration is REJECTED. The score will not be")
print("              committed or appended to scores.jsonl, so _best stays put.")
print("              Roll back the protected file(s) and re-run the evaluation:")
print("                cd <repo> && git checkout -- <file>")
print("              If the change was intentional (e.g. fixing a bug in the eval")
print("              harness itself), update protected_paths in config.yaml and")
print("              delete .autosota_protected_hashes.json to re-snapshot.")
sys.exit(9)
PYEOF
  PROTECT_RC=$?
  set -e
  if [[ "$PROTECT_RC" -ne 0 ]]; then
    exit "$PROTECT_RC"
  fi
fi

# ── idempotency: if this iter+idea already recorded, skip the recording phase
# and jump straight to the pause check. Allows safe re-invocation when Claude's
# bash tool kills the previous call mid-pause (interactive mode).
ALREADY_RECORDED=false
if [[ -f "$SCORES" ]] && [[ "$ITER" != "final" ]]; then
  if python3 - "$SCORES" "$ITER" "$IDEA_ID" <<'PYEOF' >/dev/null 2>&1
import json, sys
scores, want_iter, want_idea = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    want_iter_norm = int(want_iter)
except ValueError:
    want_iter_norm = want_iter
for line in open(scores):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("iter") == want_iter_norm and r.get("idea_id") == want_idea:
        sys.exit(0)
sys.exit(1)
PYEOF
  then
    ALREADY_RECORDED=true
    echo "[record_score] iter=${ITER} idea=${IDEA_ID} already in scores.jsonl — skipping commit/append (idempotent re-run)."
  fi
fi

# ── git commit (capture current state) ───────────────────────────────────────
cd "$REPO_ROOT"

# Ensure git is configured
git config user.name  "optimizer" 2>/dev/null || true
git config user.email "opt@local" 2>/dev/null || true

if [[ "$ALREADY_RECORDED" == "false" ]]; then
  # Stage all changes and commit (--allow-empty in case nothing changed)
  git add -A
  COMMIT_MSG="iter-${ITER}: ${TITLE} [${STATUS}]"
  git commit -q -m "$COMMIT_MSG" --allow-empty
  COMMIT_HASH=$(git rev-parse HEAD)
  echo "[record_score] git commit: ${COMMIT_HASH:0:10} — $COMMIT_MSG"
else
  COMMIT_HASH=$(git rev-parse HEAD)
fi

# ── update _best tag ─────────────────────────────────────────────────────────
# Prefer caller-supplied --is-best flag (prompt pseudocode already knows direction).
# Fallback: auto-tag on the very first successful record (baseline or first iter).
if [[ "$STATUS" == "success" ]]; then
  TAG_EXISTS=$(git rev-parse --verify _best >/dev/null 2>&1 && echo yes || echo no)

  if [[ "$TAG_EXISTS" == "no" ]]; then
    # No _best yet — always tag the first successful commit
    git tag -f _best "$COMMIT_HASH"
    echo "[record_score] _best tag created → ${COMMIT_HASH:0:10} (first success)"

  elif [[ "$IS_BEST" == "true" ]]; then
    git tag -f _best "$COMMIT_HASH"
    echo "[record_score] _best tag updated → ${COMMIT_HASH:0:10}"

  elif [[ -z "$IS_BEST" ]]; then
    echo "[record_score] WARNING: --is-best not provided; _best tag unchanged"
    echo "               Pass --is-best true|false so the tag tracks the real best commit."
  fi
  # IS_BEST == "false": do nothing, _best stays where it is
fi

# ── append to scores.jsonl ────────────────────────────────────────────────────
mkdir -p "$(dirname "$SCORES")"

# Escape notes for JSON
NOTES_ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$NOTES" 2>/dev/null \
  || echo "\"${NOTES//\"/\'}\"")
TITLE_ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$TITLE" 2>/dev/null \
  || echo "\"${TITLE//\"/\'}\"")

if [[ "$ALREADY_RECORDED" == "false" ]]; then
  python3 - <<PYEOF
import json, os
raw_iter = "$ITER"
try:
    iter_val = int(raw_iter)
except ValueError:
    iter_val = raw_iter  # keep as string for special values like "final"
entry = {
    "iter":           iter_val,
    "idea_id":        "$IDEA_ID",
    "idea_title":     "$TITLE",
    "metrics":        $METRICS,
    "primary_metric": float("$PRIMARY"),
    "commit":         "$COMMIT_HASH",
    "status":         "$STATUS",
    "notes":          "$NOTES",
}
scores_path = "$SCORES"
with open(scores_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"[record_score] Appended iter={entry['iter']} to {scores_path}")
PYEOF
fi

echo "[record_score] Done."

# ── interactive pause (after recording) ──────────────────────────────────────
# When run_dir/interactive.flag exists, block here until the user releases via
# `autosota continue` (writes continue.flag) or disables interactive mode via
# `autosota pause --off` (removes interactive.flag).
#
# This is intentionally embedded in record_score.sh because Claude is REQUIRED
# to call this script after every iteration — far more reliable than asking
# Claude to run a separate pause-check command (which it tends to skip).
#
# Skipped when iter == "final" (post-PHASE-4 marker; no further iterations).
INTERACTIVE_FLAG="$RUN_DIR/interactive.flag"
PAUSED_FLAG="$RUN_DIR/paused.flag"
CONTINUE_FLAG="$RUN_DIR/continue.flag"

# Read max_iterations from the run's effective config so we can skip pause
# after the final optimization iteration (otherwise the user has to `continue`
# just to enter PHASE 4, which is annoying).
MAX_ITER=""
EFFECTIVE_CFG="$RUN_DIR/logs/effective_config.yaml"
if [[ -f "$EFFECTIVE_CFG" ]]; then
  MAX_ITER=$(python3 -c "import yaml,sys
try:
    print(yaml.safe_load(open('$EFFECTIVE_CFG')).get('max_iterations',''))
except Exception:
    pass" 2>/dev/null || echo "")
fi

IS_LAST_ITER=false
if [[ -n "$MAX_ITER" ]] && [[ "$ITER" =~ ^[0-9]+$ ]] && [[ "$ITER" -ge "$MAX_ITER" ]]; then
  IS_LAST_ITER=true
fi

if [[ -f "$INTERACTIVE_FLAG" ]] && [[ "$ITER" != "final" ]] && [[ "$IS_LAST_ITER" == "false" ]]; then
  # Stale continue.flag from a previous pause? Clear it so we don't auto-release.
  rm -f "$CONTINUE_FLAG"
  printf 'iter=%s idea=%s status=%s primary=%s paused_at=%s\n' \
    "$ITER" "$IDEA_ID" "$STATUS" "$PRIMARY" "$(date -u +%FT%TZ)" > "$PAUSED_FLAG"

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "[Interactive] PAUSED after iter ${ITER} (status=${STATUS}, primary=${PRIMARY})"
  echo "[Interactive] Run dir : $RUN_DIR"
  echo "[Interactive] Release : autosota continue"
  echo "[Interactive] Steer   : autosota steer \"<message>\""
  echo "[Interactive] Disable : autosota pause --off"
  echo "════════════════════════════════════════════════════════════════"

  # Hard ceiling: 24 hours. If you really need longer, you have bigger issues.
  WAIT_END=$((SECONDS + 86400))
  while [[ $SECONDS -lt $WAIT_END ]]; do
    if [[ -f "$CONTINUE_FLAG" ]]; then
      rm -f "$CONTINUE_FLAG" "$PAUSED_FLAG"
      echo "[Interactive] RELEASED — proceeding to next iteration."
      exit 0
    fi
    if [[ ! -f "$INTERACTIVE_FLAG" ]]; then
      rm -f "$PAUSED_FLAG"
      echo "[Interactive] interactive.flag removed — disabling pause, proceeding."
      exit 0
    fi
    sleep 5
  done

  echo "[Interactive] WARNING: 24h pause window expired without 'autosota continue'."
  echo "[Interactive] Removing paused.flag and proceeding to next iteration."
  rm -f "$PAUSED_FLAG"
elif [[ -f "$INTERACTIVE_FLAG" ]] && [[ "$IS_LAST_ITER" == "true" ]]; then
  echo "[Interactive] iter=${ITER} is the final iteration (max=${MAX_ITER}) — skipping pause, proceeding to PHASE 4."
fi
