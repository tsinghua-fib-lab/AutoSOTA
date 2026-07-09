#!/usr/bin/env python3
"""
verify_partition.py

Usage:
    python verify_partition.py <leaf_file> <children_file> <n> [--reversed]

Formats (binary; little-endian):
  leaf_file: records (any order):
      int32 ID
      double low0, double high0, ..., double low_{n-1}, double high_{n-1}
      int32 split_ID, int32 lemma_ID (auxiliary data)

  children_file: N records (either in ID order 1..N or reversed order N..1 if --reversed):
      int32 child_left, int32 child_right
    (Use 0 to denote no child / leaf)

Notes:
 - The program keeps a disk-backed recon file of N fixed-size records (2n doubles per record).
 - A small presence bitmap (N bits) marks which recon slots are filled.
 - All numeric checks use native floats (==, arithmetic).
 - This script assumes input doubles were produced by the splitting procedure (dyadic values).
"""

import os
import struct
import mmap
import math
import tempfile

INT32_FMT = '<i'
INT32_SIZE = 4
DOUBLE_SIZE = 8

def get_N_from_children(children_path):
    sz = os.path.getsize(children_path)
    if sz % (2 * INT32_SIZE) != 0:
        raise RuntimeError("children file size not multiple of 8 bytes")
    return sz // (2 * INT32_SIZE)

def make_presence_files(N):
    # presence bytes = ceil(N/8)
    nb = (N + 7) // 8
    tmp = tempfile.NamedTemporaryFile(prefix="presence_", delete=False)
    tmp_name = tmp.name
    tmp.close()
    # allocate file
    with open(tmp_name, 'wb') as f:
        f.truncate(nb)
    f = open(tmp_name, 'r+b')
    mm = mmap.mmap(f.fileno(), nb, access=mmap.ACCESS_WRITE)
    return tmp_name, f, mm, nb

def set_present(pmm, idx):
    # idx: 1-based ID
    i = idx - 1
    byte_idx = i >> 3
    bit = 1 << (i & 7)
    cur = pmm[byte_idx]
    pmm[byte_idx] = (cur | bit)

def is_present(pmm, idx):
    i = idx - 1
    byte_idx = i >> 3
    bit = 1 << (i & 7)
    return (pmm[byte_idx] & bit) != 0

def create_recon_file(N, recsize):
    tmpf = tempfile.NamedTemporaryFile(prefix="recon_", delete=False)
    path = tmpf.name
    tmpf.close()
    with open(path, 'wb') as f:
        f.truncate(N * recsize)
    f = open(path, 'r+b')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)  # map whole file
    return path, f, mm

def write_recon_record_mm(mm, node_id, doubles, n):
    # node_id: 1-based. doubles: iterable of 2n python floats
    recsize = 2 * n * DOUBLE_SIZE
    offset = (node_id - 1) * recsize
    fmt = '<' + 'd' * (2 * n)
    struct.pack_into(fmt, mm, offset, *doubles)

def read_recon_record_mm(mm, node_id, n):
    recsize = 2 * n * DOUBLE_SIZE
    offset = (node_id - 1) * recsize
    fmt = '<' + 'd' * (2 * n)
    vals = struct.unpack_from(fmt, mm, offset)
    # return as list of (low, high) tuples
    res = []
    for i in range(n):
        low = vals[2*i]
        high = vals[2*i+1]
        res.append((low, high))
    return res

def fast_load_leaves_to_recon(leaf_path, recon_mm, pres_mm, n):
    rec_double_count = 2 * n
    fmt_hdr = '<i'
    fmt_vals = '<' + 'd' * rec_double_count
    count = 0
    with open(leaf_path, 'rb') as f:
        while True:
            hdr = f.read(INT32_SIZE)
            if not hdr:
                break
            if len(hdr) != INT32_SIZE:
                raise RuntimeError("leaf file truncated when reading ID")
            (leaf_id,) = struct.unpack(fmt_hdr, hdr)
            data = f.read(DOUBLE_SIZE * rec_double_count)
            f.read(INT32_SIZE * 2) # Read auxiliary data
            if len(data) != DOUBLE_SIZE * rec_double_count:
                raise RuntimeError(f"leaf file truncated for ID {leaf_id}")
            vals = struct.unpack(fmt_vals, data)
            for v in vals:
                if math.isnan(v):
                    raise RuntimeError(f"leaf ID {leaf_id} contains NaN")
                # allow +inf, but finite values must be >= 0
                if v < 0.0:
                    raise RuntimeError(f"leaf ID {leaf_id} contains negative value {v}")
            write_recon_record_mm(recon_mm, leaf_id, vals, n)
            set_present(pres_mm, leaf_id)
            count += 1
            if (count & 0x3FFFF) == 0:
                print(f"[info] loaded {count} leaves")
    print(f"[info] done loading {count} leaves")

def verify_merge_children_float(childA, childB, n):
    """
    childA and childB: list of n (low,high) floats. Return parent list of n (low,high) floats or raise RuntimeError.
    Uses float arithmetic and equality checks.
    """
    diff_axes = []
    for d in range(n):
        a_low, a_high = childA[d]
        b_low, b_high = childB[d]
        if (a_low == b_low) and (a_high == b_high):
            continue
        diff_axes.append(d)
    if len(diff_axes) == 0:
        raise RuntimeError("children identical in all axes")
    if len(diff_axes) > 1:
        raise RuntimeError(f"children differ in multiple axes: {diff_axes}")

    d = diff_axes[0]
    a_low, a_high = childA[d]
    b_low, b_high = childB[d]

    # order by low
    if a_low < b_low:
        left_low, left_high = a_low, a_high
        right_low, right_high = b_low, b_high
    else:
        left_low, left_high = b_low, b_high
        right_low, right_high = a_low, a_high

    # adjacency: left_high == right_low
    if not (left_high == right_low):
        raise RuntimeError(f"children not adjacent on axis {d}: left_high={left_high}, right_low={right_low}")

    # finite split: both highs finite
    if math.isfinite(left_high) and math.isfinite(right_high):
        a = left_low
        m = left_high
        b = right_high
        # check 2*m == a + b
        if not (2.0 * m == (a + b)):
            # if strict equality failed, try a safer check via nextafter tolerance (optional)
            raise RuntimeError(f"finite-split midpoint rule fails at axis {d}: 2*m != a+b ({2.0*m} != {a+b})")
        parent_interval = (a, b)
    else:
        # infinite split: right_high is +inf expected
        if math.isfinite(right_high):
            raise RuntimeError("unexpected split pattern (left or right highs)")
        # right_high is +inf, left_high must equal 2*left_low
        if not (left_high == 2.0 * left_low):
            raise RuntimeError(f"infinite-split check failed: left_high {left_high} != 2*left_low {2.0*a}")
        parent_interval = (left_low, float('inf'))

    # build parent list: same as childA on other axes
    parent = []
    for i in range(n):
        if i == d:
            parent.append(parent_interval)
        else:
            parent.append(childA[i])
    return parent

def verify(leaf_file, children_file, n, reversed_children=False):
    N = get_N_from_children(children_file)
    print(f"[info] nodes N = {N}, dims = {n}")

    recsize = 2 * n * DOUBLE_SIZE
    recon_path, recon_f, recon_mm = create_recon_file(N, recsize)
    pres_path, pres_f, pres_mm, pres_bytes = make_presence_files(N)
    try:
        # load leaves into recon
        print("[info] loading leaves into on-disk recon...")
        fast_load_leaves_to_recon(leaf_file, recon_mm, pres_mm, n)

        # prepare children access
        fmt_child = '<ii'
        if reversed_children:
            fchild = open(children_file, 'rb')
            # we'll stream fchild from start (which represents node N) as we iterate node N..1
            stream_children = True
        else:
            # load all children into memory (8 bytes per node)
            print("[info] loading children into memory (8 bytes per node)...")
            children = [None] * (N + 1)
            with open(children_file, 'rb') as f:
                for id in range(1, N + 1):
                    data = f.read(8)
                    if len(data) != 8:
                        raise RuntimeError(f"children file truncated at id {id}")
                    left, right = struct.unpack(fmt_child, data)
                    children[id] = (left, right)
            stream_children = False
            print("[info] children loaded")

        # descending pass
        print("[info] starting descending pass...")
        processed = 0
        if stream_children:
            with fchild:
                # children file is N..1, so first read corresponds to node N
                for node_id in range(N, 0, -1):
                    data = fchild.read(8)
                    if len(data) != 8:
                        raise RuntimeError(f"children stream ended early at node {node_id}")
                    left, right = struct.unpack(fmt_child, data)
                    if left == 0 and right == 0:
                        if not is_present(pres_mm, node_id):
                            raise RuntimeError(f"leaf ID {node_id} missing in leaf file")
                    else:
                        if left == 0 or right == 0:
                            raise RuntimeError(f"internal node {node_id} has a zero child")
                        if left <= node_id or right <= node_id:
                            raise RuntimeError(f"BFS violated at parent {node_id}: child <= parent")
                        if not is_present(pres_mm, left) or not is_present(pres_mm, right):
                            raise RuntimeError(f"child not present when processing parent {node_id} (left={left}, right={right})")
                        childA = read_recon_record_mm(recon_mm, left, n)
                        childB = read_recon_record_mm(recon_mm, right, n)
                        parent = verify_merge_children_float(childA, childB, n)
                        # flatten parent to 2n floats
                        flat = []
                        for (lo,hi) in parent:
                            flat.append(lo)
                            flat.append(hi)
                        write_recon_record_mm(recon_mm, node_id, flat, n)
                        set_present(pres_mm, node_id)
                    processed += 1
                    if processed % 2000000 == 0:
                        print(f"[info] processed {processed}/{N}")
        else:
            for node_id in range(N, 0, -1):
                left, right = children[node_id]
                if left == 0 and right == 0:
                    if not is_present(pres_mm, node_id):
                        raise RuntimeError(f"leaf ID {node_id} missing in leaf file")
                else:
                    if left == 0 or right == 0:
                        raise RuntimeError(f"internal node {node_id} has a zero child")
                    if left <= node_id or right <= node_id:
                        raise RuntimeError(f"BFS violated at parent {node_id}: child <= parent")
                    if not is_present(pres_mm, left) or not is_present(pres_mm, right):
                        raise RuntimeError(f"child missing when processing parent {node_id}: left={left}, right={right}")
                    childA = read_recon_record_mm(recon_mm, left, n)
                    childB = read_recon_record_mm(recon_mm, right, n)
                    parent = verify_merge_children_float(childA, childB, n)
                    flat = []
                    for (lo,hi) in parent:
                        flat.append(lo)
                        flat.append(hi)
                    write_recon_record_mm(recon_mm, node_id, flat, n)
                    set_present(pres_mm, node_id)
                processed += 1
                if processed % 2000000 == 0:
                    print(f"[info] processed {processed}/{N}")

        print("[info] descending pass finished. verifying root regions (IDs 1..2^n)...")
        R = 1 << n
        if R > N:
            raise RuntimeError(f"expected at least {R} nodes for roots but N={N}")
        for root_id in range(1, R + 1):
            if not is_present(pres_mm, root_id):
                raise RuntimeError(f"root region ID {root_id} missing")
            root = read_recon_record_mm(recon_mm, root_id, n)
            mask = root_id - 1
            for d, (l, h) in enumerate(root):
                bit = (mask >> d) & 1
                if bit == 0:
                    # expect [0,1]
                    if not (l == 0.0 and h == 1.0):
                        raise RuntimeError(f"root region {root_id} axis {d} expected [0,1] but got [{l},{h}]")
                else:
                    if not (l == 1.0 and (not math.isfinite(h))):
                        raise RuntimeError(f"root region {root_id} axis {d} expected [1,+inf) but got [{l},{h}]")

        print(f"SUCCESS: verified tree; all {R} root regions OK and merges valid.")
    except Exception as e:
        print("[error]", e)
    finally:
        try:
            recon_mm.close()
            recon_f.close()
            os.remove(recon_path)
        except Exception:
            pass
        try:
            pres_mm.close()
            pres_f.close()
            os.remove(pres_path)
        except Exception:
            pass

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('leaf_file')
    p.add_argument('children_file')
    p.add_argument('n', type=int)
    p.add_argument('--reversed', action='store_true',
                   help='children file is stored in descending order (N..1) so it can be streamed')
    args = p.parse_args()
    verify(args.leaf_file, args.children_file, args.n, reversed_children=args.reversed)
