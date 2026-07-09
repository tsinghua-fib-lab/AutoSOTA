# Certificate Verification Guide

This directory contains all materials needed to verify the proof of the Steiner ratio lower bound. The verification process involves validating splits, checking monotonicity properties, generating domain partitions, and verifying certificates using interval arithmetic.

## Overview

The proof verification consists of **two main cases** (`d_regular` and `d_steiner`), each with **two subcases** (`f ≥ d` and `f ≤ d`). All verification steps must be completed in **both** the `d_regular/` and `d_steiner/` directories.

## Prerequisites

Ensure you have the following installed:
- **Python 3.x** (for validation scripts)
- **C++ compiler** with C++23 support (e.g., g++ 13+)
- **Julia** with the `IntervalArithmetic` package (for certificate verification)
- **Make** (for building C++ programs)

## Quick Start

For the impatient, run these commands in both `d_regular/` and `d_steiner/`:

```bash
# Steps 1-2: Validation
python split_validation.py
python mono_check.py

# Step 3: Generate partitions
make
./plot_f_ge_d
./plot_f_le_d

# Step 4: Verify partitions (replace <n> with values from table below)
python verify_partition.py "certificate_rho=0.8559_f_ge_d.bin" "child_rho=0.8559_f_ge_d.bin" <n> --reversed
python verify_partition.py "certificate_rho=0.8559_f_le_d.bin" "child_rho=0.8559_f_le_d.bin" <n> --reversed

# Step 5: Verify certificates
julia --threads auto verify_certificate.jl --f-ge-d
julia --threads auto verify_certificate.jl --f-le-d
```

---

## Detailed Verification Steps

### Step 1: Split Validation

**Purpose:** Verify that all splits defined in `splits.txt` are well-formed and valid.

```bash
python split_validation.py
```

**Expected output:** The script should complete without errors, confirming all splits are valid.

---

### Step 2: Monotonicity Check

**Purpose:** Verify that the `mono_vars` field for each split is correctly computed. This checks which variables the baseline functions (lemma 0) are non-increasing with respect to.

```bash
python mono_check.py
```

**Expected output:** Confirmation that all monotonicity properties are correctly specified.

---

### Step 3: Generate the Partition

**Purpose:** Compile the C++ programs and generate the domain partitions for both subcases.

```bash
make
./plot_f_ge_d
./plot_f_le_d
```

**What this does:**
- Compiles `plot_f_ge_d.cpp` and `plot_f_le_d.cpp` with optimization flags
- Generates partition data for the `f ≥ d` case
- Generates partition data for the `f ≤ d` case

**Generated files:**
- `certificate_rho=0.8559_f_ge_d.bin` and `child_rho=0.8559_f_ge_d.bin`
- `certificate_rho=0.8559_f_le_d.bin` and `child_rho=0.8559_f_le_d.bin`

**Time requirement:** Approximately **30 hours total** across both cases (most time is spent on the `f ≤ d, d_steiner` case)

**Alternative: Download pre-generated files**

If you prefer to skip the generation step, you can download the pre-generated partition and certificate files from:

**https://huggingface.co/datasets/keyisi/steiner-ratio**

Simply download the files and place them in the appropriate `d_regular/` or `d_steiner/` directories, then proceed directly to Step 4.

---

### Steps 4-5: Verification

**Time requirement:** Approximately **6 hours total** for partition and certificate verification across both cases.

---

### Step 4: Partition Verification

**Purpose:** Verify that the generated partitions completely cover the domain $[0,+\infty)^n$ without gaps or overlaps.

The partition scheme works as follows:
- Initially, each dimension is split into $[0,1]$ and $[1,+\infty)$, creating $2^n$ hyper-boxes
- Iteratively, finite edges $[a,b]$ are split into $[a,(a+b)/2]$ and $[(a+b)/2,b]$
- Infinite edges $[a,+\infty)$ are split into $[a,2a]$ and $[2a,+\infty)$

Each region is assigned a unique ID, and the `child_{suffix}.bin` file records the parent-child relationships. Each record in `child_{suffix}.bin` contains two `int32` values denoting the IDs of the two children of that region. By default, records are ordered as: N-th region, (N-1)-th region, ..., 1st region. The `--reversed` flag in `verify_partition.py` indicates that the records are stored in this reversed order (N, ..., 1). The `plot_{suffix}.cpp` programs generate files in reversed format by default, and the pre-generated files also use this format.

```bash
python verify_partition.py "certificate_rho=0.8559_f_ge_d.bin" "child_rho=0.8559_f_ge_d.bin" <n> --reversed
python verify_partition.py "certificate_rho=0.8559_f_le_d.bin" "child_rho=0.8559_f_le_d.bin" <n> --reversed
```

**Important:** Replace `<n>` with the appropriate value from the table below.

**Expected output:** Confirmation that the partition is complete and valid.

---

### Step 5: Certificate Verification

**Purpose:** Verify that each region's certificate is valid by performing vertex checks and monotonicity checks using interval arithmetic.

Each certificate record in `certificate_{suffix}.bin` has the following binary format (little-endian, no padding):

```
Offset   | Type    | Description
---------|---------|----------------------------------
0        | int32   | region_ID
4        | float64 | low[1], high[1]
4+16     | float64 | low[2], high[2]
...      | ...     | ...
4+16(n-1)| float64 | low[n], high[n]
4+16n    | int32   | split_ID
8+16n    | int32   | lemma_ID
```

The verification uses Julia's `IntervalArithmetic` package for rigorous validated numerics.

```bash
julia --threads auto verify_certificate.jl --f-ge-d
julia --threads auto verify_certificate.jl --f-le-d
```

**Verification checks performed** (see `verify_record()` function in `verify_certificate.jl`):

1. **Box filtering:**

   - **Symmetry filtering (d_steiner case only):** Due to structural symmetry of the tree, we can assume without loss of generality that `max(u, v) ≤ max(1, b)`. Therefore, boxes where `max(u_low, v_low) > max(1, b_high)` are skipped.

   - **Case-specific filtering (f ≤ d case):** When verifying the `f ≤ d` case, boxes where `f_low > d_high` are skipped, as these boxes satisfy `f > d` throughout and belong to the other case.

2. **Boundary optimization (f ≥ d case):** In the `f ≥ d` case, the script verifies that `f` is in the split's `mono_vars`. This ensures the function is non-increasing with respect to `f`, allowing verification only at the boundary `f = d` rather than checking the entire box.

3. **S_plus size constraint (f ≤ d case):** In the `f ≤ d` case, the script enforces that the used split must have `|S_plus| ≤ 3`. This is a requirement from the paper, as larger S_plus sets require special lemmas that are only valid in the `f ≥ d` case.

4. **Unbounded interval handling:** For any variable with an unbounded interval $[a, +\infty)$, the script verifies that:

   - The variable is in the split's `mono_vars` (function is non-increasing w.r.t. this variable)
   - The function uses lemma 0 (i.e., is a baseline function)

   This ensures that the function value at infinity is bounded by its value at the finite boundary.
   
5. **Vertex check:** Finally, the script evaluates the function at all finite vertices of the hyper-box (up to $2^n$ vertices) and verifies that the function value is non-positive at each vertex. Combined with the monotonicity properties, this guarantees the function is non-positive throughout the entire hyper-box.

**Expected output:** All certificates should pass verification, confirming that the covering function is non-positive on each hyper-box.

---

## Dimension Values by Case

The number of variables `n` varies by case and subcase:

| Subcase    | d_regular | d_steiner |
|:----------:|:---------:|:---------:|
| **f ≥ d** |     5     |     7     |
| **f ≤ d** |     6     |     8     |

Use these values when running `verify_partition.py` in Step 4.

---

## Directory Structure

```
certificate/
├── d_regular/              # Regular case verification
│   ├── splits.txt          # Split definitions
│   ├── lemmas/             # Lemma implementations (Julia)
│   ├── formulas/           # Formula definitions
│   ├── split_validation.py # Step 1 script
│   ├── mono_check.py       # Step 2 script
│   ├── plot_f_ge_d.cpp     # Partition generator (f ≥ d)
│   ├── plot_f_le_d.cpp     # Partition generator (f ≤ d)
│   ├── verify_partition.py # Step 4 script
│   └── verify_certificate.jl # Step 5 script
│
└── d_steiner/              # Steiner case verification
    └── (same structure as d_regular/)
```

---

## Notes

- All verification steps must be completed in **both** `d_regular/` and `d_steiner/` directories
- The verification is fully deterministic and should produce consistent results
- If any step fails, the proof verification is incomplete
- The C++ programs use multi-threading for performance; compilation requires C++23 support
- Julia's interval arithmetic provides rigorous bounds, ensuring no floating-point errors affect correctness

---

## Troubleshooting

**Issue:** `make` fails with compiler errors
**Solution:** Ensure you have g++ 13 or later with C++23 support. Try: `g++ --version`

**Issue:** Julia package not found
**Solution:** Install IntervalArithmetic: `julia -e 'using Pkg; Pkg.add("IntervalArithmetic")'`

**Issue:** Python script fails
**Solution:** Ensure you're running the script from within the `d_regular/` or `d_steiner/` directory

---

## Summary

Upon successful completion of all steps in both directories, you will have verified:
1. ✓ All splits are well-formed
2. ✓ Monotonicity properties are correct
3. ✓ Domain partitions are complete and valid
4. ✓ All certificates prove non-positivity on their respective regions

This constitutes a complete computer-assisted verification of the proof.
