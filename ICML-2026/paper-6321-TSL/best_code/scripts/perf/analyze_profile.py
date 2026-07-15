#!/usr/bin/env python3
"""Print top self / inclusive functions from a samply JSON capture.

Resolves hex addresses to symbol names using the syms.json sidecar that
samply writes next to the profile when `--unstable-presymbolicate` is on.

Usage:
    python3 scripts/perf/analyze_profile.py <profile.json> [thread_name_substring]
                                            [--top-self N] [--top-inclusive N]
                                            [--syms <profile.syms.json>]

Defaults to top 20 by self-time and top 15 by inclusive-time across all
matched threads. The sidecar path defaults to `<profile>.syms.json` (the
".json" suffix is replaced with ".syms.json").
"""
import argparse
import bisect
import collections
import json
import os
import sys


def load_syms(path):
    """Build {debug_name: (starts, ends, names)} from a samply syms sidecar."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        syms = json.load(f)
    string_table = syms["string_table"]
    mod_idx = {}
    for m in syms["data"]:
        st = sorted(m["symbol_table"], key=lambda s: s["rva"])
        starts = [s["rva"] for s in st]
        ends = [s["rva"] + s["size"] for s in st]
        names = [string_table[s["symbol"]] for s in st]
        mod_idx[m["debug_name"]] = (starts, ends, names)
    return mod_idx


def build_frame_labels(thread, libs, mod_idx):
    strings = thread.get("stringTable") or thread.get("stringArray")
    if strings is None:
        raise KeyError("thread has neither stringTable nor stringArray")
    ft = thread["frameTable"]
    fn = thread["funcTable"]
    rt = thread.get("resourceTable")
    fn_name = fn["name"]
    fr_func = ft["func"]
    fr_addr = ft.get("address", [None] * ft["length"])
    fn_resource = fn.get("resource", [-1] * fn["length"])

    resource_debug_name = []
    if rt is not None:
        res_lib = rt.get("lib", [])
        for i in range(rt["length"]):
            li = res_lib[i] if i < len(res_lib) else None
            if li is None or li < 0:
                resource_debug_name.append(None)
            else:
                resource_debug_name.append(libs[li].get("debugName"))

    def resolve(debug_name, addr):
        if not debug_name or mod_idx is None or debug_name not in mod_idx:
            return None
        starts, ends, names = mod_idx[debug_name]
        idx = bisect.bisect_right(starts, addr) - 1
        if 0 <= idx and addr < ends[idx]:
            return names[idx]
        return None

    labels = []
    for i in range(ft["length"]):
        fi = fr_func[i]
        ri = fn_resource[fi] if 0 <= fi < len(fn_resource) else -1
        dn = resource_debug_name[ri] if (rt is not None and 0 <= ri < len(resource_debug_name)) else None
        addr = fr_addr[i] if fr_addr else None
        name = resolve(dn, addr) if (addr is not None and addr >= 0) else None
        if name is None and 0 <= fi:
            sidx = fn_name[fi]
            if sidx is not None and sidx >= 0:
                name = strings[sidx]
        labels.append(name or "<unknown>")
    return labels


def aggregate(threads, libs, mod_idx):
    self_by = collections.Counter()
    inc_by = collections.Counter()
    total = 0
    for thread in threads:
        labels = build_frame_labels(thread, libs, mod_idx)
        st = thread["stackTable"]
        s_prefix = st["prefix"]
        s_frame = st["frame"]
        for stk in thread["samples"]["stack"]:
            if stk is None:
                continue
            total += 1
            cur = stk
            path = []
            while cur is not None and cur >= 0:
                path.append(labels[s_frame[cur]])
                cur = s_prefix[cur]
            if not path:
                continue
            self_by[path[0]] += 1
            for lbl in set(path):
                inc_by[lbl] += 1
    return total, self_by, inc_by


def print_table(title, counter, total, n):
    print(f"\n# {title}  (top {n})")
    print(f"  {'samples':>8}  {'pct':>6}  function")
    for lbl, count in counter.most_common(n):
        print(f"  {count:8d}  {100 * count / total:5.1f}%  {lbl}")


def default_syms_path(profile_path):
    if profile_path.endswith(".json"):
        return profile_path[: -len(".json")] + ".syms.json"
    return profile_path + ".syms.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("filter", nargs="?", default=None)
    parser.add_argument("--top-self", type=int, default=20)
    parser.add_argument("--top-inclusive", type=int, default=15)
    parser.add_argument("--syms", default=None)
    args = parser.parse_args()

    with open(args.path) as f:
        prof = json.load(f)
    threads = prof.get("threads", [])
    libs = prof.get("libs", [])
    if args.filter:
        threads = [t for t in threads if args.filter in t.get("name", "")]
    if not threads:
        print(f"no threads matched filter={args.filter!r}", file=sys.stderr)
        return 2

    syms_path = args.syms or default_syms_path(args.path)
    mod_idx = load_syms(syms_path)
    if mod_idx is None:
        print(f"note: syms sidecar not found at {syms_path}; "
              "addresses will appear as hex", file=sys.stderr)

    total, self_by, inc_by = aggregate(threads, libs, mod_idx)
    print(f"Profile: {args.path}")
    print(f"Threads matched: {len(threads)}  Total samples: {total}")
    if total == 0:
        print("no samples; nothing to report", file=sys.stderr)
        return 2
    print_table("Top by SELF time", self_by, total, args.top_self)
    print_table("Top by INCLUSIVE time", inc_by, total, args.top_inclusive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
