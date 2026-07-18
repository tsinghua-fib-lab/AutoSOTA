#!/usr/bin/env bash
set -euo pipefail

logs_dir="${1:-logs}"

if [[ ! -d "$logs_dir" ]]; then
  echo "logs dir not found: $logs_dir" >&2
  exit 1
fi

out_csv="statistics.csv"
header="file,total,up,down,avg_compute,node_fail,full_recovery"
for i in $(seq 1 26); do
  header+=",stack${i}"
done
header+=",recompute"
echo "$header" > "$out_csv"

count_fixed() {
  local pattern="$1"
  local file="$2"
  local c
  c=$(grep -cF "$pattern" "$file" || true)
  printf "%s" "$c"
}

for f in "$logs_dir"/*.log; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"

  # Pull summary line from the end for speed on large logs.
  summary_line="$(tail -n 5 "$f" | grep -m 1 "Summary:" || true)"
  [[ -n "$summary_line" ]] || continue

  total="$(printf "%s" "$summary_line" | grep -o "total=[^ ]*" | cut -d= -f2)"
  up="$(printf "%s" "$summary_line" | grep -o "up=[^ ]*" | cut -d= -f2)"
  down="$(printf "%s" "$summary_line" | grep -o "down=[^ ]*" | cut -d= -f2)"
  avg="$(printf "%s" "$summary_line" | grep -o "avg-compute=[^ ]*" | cut -d= -f2)"

  fwd_fail="$(grep -F "Computation: Fwd+Bwd-fail" "$f" | grep -v "pending fail" | wc -l | tr -d " " || true)"
  allreduce_fail="$(count_fixed "Allreduce-fail" "$f")"
  system_recovery_fail="$(count_fixed "system-recovery-fail" "$f")"
  checkpoint_fail="$(count_fixed "Checkpoint-fail" "$f")"
  full_recovery="$(count_fixed "Starting FULL system recovery" "$f")"
  recompute="$(count_fixed "Recompute" "$f")"
  node_fail=$((fwd_fail + allreduce_fail + system_recovery_fail + checkpoint_fail))

  line="$base,$total,$up,$down,$avg,$node_fail,$full_recovery"
  for i in $(seq 1 26); do
    count="$(count_fixed "Stack-depth ${i}/" "$f")"
    line+=",$count"
  done
  line+=",$recompute"
  echo "$line" >> "$out_csv"
done
