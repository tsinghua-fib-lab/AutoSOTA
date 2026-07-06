import subprocess
import time
import csv
import os
import re
import sys
from datetime import datetime


# =========================================
# =========================================

GPU_FILTER = None

INTERVAL = 0.1

OUTPUT_DIR = "./gpu_memory_results"

CSV_FIELDS = [
    "timestamp",
    "pid",
    "gpu_name",
    "gpu_uuid",
    "mem_used_mib",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_process(command):
    proc = subprocess.Popen(command)
    print(f"Started process PID={proc.pid}")
    print(f"Command: {' '.join(command)}")
    return proc


def query_gpu_usage(pid):
    """Return GPU memory usage for this PID on each GPU"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_name,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except subprocess.CalledProcessError:
        return []

    records = []
    for line in out.strip().splitlines():
        parts = re.split(r",\s*", line, maxsplit=3)
        if len(parts) != 4:
            continue

        p, name, uuid, mem = parts
        if int(p) != pid:
            continue

        if GPU_FILTER:
            if not any(k in name or k in uuid for k in GPU_FILTER):
                continue

        records.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "pid": pid,
                "gpu_name": name,
                "gpu_uuid": uuid,
                "mem_used_mib": int(mem),
            }
        )
    return records


def write_csv_header(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)


def append_csv(path, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([r[k] for k in CSV_FIELDS])


def analyze_max(csv_path):
    stats = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid = row["gpu_uuid"]
            mem = int(row["mem_used_mib"])
            if uuid not in stats or mem > stats[uuid]["mem"]:
                stats[uuid] = {
                    "mem": mem,
                    "time": row["timestamp"],
                    "name": row["gpu_name"],
                }
    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_gpu.py <script_to_run> [args...]")
        sys.exit(1)

    target_script = sys.argv[1]
    script_args = sys.argv[2:]
    command = ["python", target_script] + script_args

    ensure_dir(OUTPUT_DIR)

    proc = run_process(command)
    pid = proc.pid

    log_csv = os.path.join(OUTPUT_DIR, f"gpu_log_{pid}.csv")
    summary_txt = os.path.join(OUTPUT_DIR, f"gpu_summary_{pid}.txt")

    write_csv_header(log_csv)

    try:
        while proc.poll() is None:
            rows = query_gpu_usage(pid)
            if rows:
                append_csv(log_csv, rows)
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("Manually interrupted, terminating process")
        proc.terminate()

    print("Process ended, analyzing results...")

    stats = analyze_max(log_csv)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("GPU Maximum Memory Usage Statistics\n")
        f.write("====================================\n")
        f.write(f"Command: {' '.join(command)}\n\n")

        for uuid, s in stats.items():
            f.write(
                f"GPU: {s['name']} ({uuid})\n"
                f"  Max Memory: {s['mem']} MiB ({s['mem']/1024:.2f} GB)\n"
                f"  Time: {s['time']}\n\n"
            )

    print(f"CSV Log: {os.path.abspath(log_csv)}")
    print(f"Summary Results: {os.path.abspath(summary_txt)}")


if __name__ == "__main__":
    main()
