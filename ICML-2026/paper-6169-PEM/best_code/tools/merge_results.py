#!/usr/bin/env python3
"""
Merge multiple BBOB result directories (bbob_summary.csv + trace_index.csv + traces/)
into a single directory suitable for plotting and metrics.

This is useful for running algorithms in parallel into separate output folders.
"""

import argparse
import csv
import os
import shutil

from _project import BASE_DIR, repo_relpath

def read_csv_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = [row for row in reader]
    return header, rows


def write_csv(path: str, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_same_header(headers: list[list[str]], *, name: str) -> list[str]:
    unique = {tuple(h) for h in headers if h}
    if not unique:
        return []
    if len(unique) != 1:
        msg = "\n".join(f"- {name} header: {list(h)}" for h in sorted(unique))
        raise ValueError(f"Header mismatch for {name}:\n{msg}")
    return list(next(iter(unique)))


def copy_trace_file(src_trace_path: str, dst_traces_dir: str) -> str:
    os.makedirs(dst_traces_dir, exist_ok=True)
    filename = os.path.basename(src_trace_path)
    dst = os.path.join(dst_traces_dir, filename)
    if os.path.abspath(src_trace_path) != os.path.abspath(dst):
        shutil.copy2(src_trace_path, dst)
    return os.path.abspath(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory (will be created if missing).",
    )
    parser.add_argument(
        "--input-dirs",
        required=True,
        nargs="+",
        help="Input directories, each containing bbob_summary.csv and trace_index.csv (and optionally state_index.csv).",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_traces_dir = os.path.join(out_dir, "traces")

    summary_headers = []
    summary_rows_all: list[dict] = []

    trace_headers = []
    trace_rows_all: list[dict] = []

    state_headers = []
    state_rows_all: list[dict] = []

    for d in args.input_dirs:
        d_abs = os.path.abspath(d)
        summary_path = os.path.join(d_abs, "bbob_summary.csv")
        trace_index_path = os.path.join(d_abs, "trace_index.csv")
        state_index_path = os.path.join(d_abs, "state_index.csv")

        if not os.path.isfile(summary_path):
            raise FileNotFoundError(summary_path)
        if not os.path.isfile(trace_index_path):
            raise FileNotFoundError(trace_index_path)

        s_header, s_rows = read_csv_rows(summary_path)
        t_header, t_rows = read_csv_rows(trace_index_path)

        summary_headers.append(s_header)
        trace_headers.append(t_header)

        summary_rows_all.extend(s_rows)

        for row in t_rows:
            src_trace_file = row.get("trace_file", "")
            if src_trace_file and os.path.isfile(src_trace_file):
                row = dict(row)
                row["trace_file"] = copy_trace_file(src_trace_file, out_traces_dir)
                trace_rows_all.append(row)
            else:
                trace_rows_all.append(row)

        if os.path.isfile(state_index_path):
            st_header, st_rows = read_csv_rows(state_index_path)
            state_headers.append(st_header)
            for row in st_rows:
                src_state_file = row.get("state_file", "")
                if src_state_file and os.path.isfile(src_state_file):
                    row = dict(row)
                    row["state_file"] = copy_trace_file(src_state_file, out_traces_dir)
                    state_rows_all.append(row)
                else:
                    state_rows_all.append(row)

    summary_header = ensure_same_header(summary_headers, name="bbob_summary.csv")
    trace_header = ensure_same_header(trace_headers, name="trace_index.csv")

    write_csv(os.path.join(out_dir, "bbob_summary.csv"), summary_header, summary_rows_all)
    write_csv(os.path.join(out_dir, "trace_index.csv"), trace_header, trace_rows_all)

    if state_rows_all:
        state_header = ensure_same_header(state_headers, name="state_index.csv")
        if state_header:
            write_csv(os.path.join(out_dir, "state_index.csv"), state_header, state_rows_all)

    print("Merged bbob_summary rows:", len(summary_rows_all))
    print("Merged trace_index rows:", len(trace_rows_all))
    if state_rows_all:
        print("Merged state_index rows:", len(state_rows_all))
    print("Output:", repo_relpath(out_dir))


if __name__ == "__main__":
    main()
