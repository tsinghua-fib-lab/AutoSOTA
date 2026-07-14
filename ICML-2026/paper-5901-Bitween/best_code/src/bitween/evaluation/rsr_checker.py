"""
Approximate RSR checker for verified properties.

A property is approximately classified as an RSR if:
1. It contains random variables (variables beyond the function's formal parameters)
   that appear INSIDE f() calls (as query points), enabling correlated evaluation.
2. It allows recovering f at the input point from f at randomized points.

A property is NOT an RSR if:
- It's a pure symmetry (e.g., f(a,b,c,d) = f(a,c,b,d)) -- just permuting inputs
- It's a scaling property (e.g., f(2x) = 2f(x)) -- no randomness
- It's a trivial evaluation (e.g., f(0,0,0,0) = 0)
- Random variables appear only OUTSIDE f() calls (e.g., f(x) - r = 0)
- It involves only constants inside f() (e.g., f(0,b,c,0) = 2bc)

Usage:
    python -m bitween.evaluation.rsr_checker \
        --results_dir results/agentic/remote_bedrock/...

    # Or check a single equation:
    python -m bitween.evaluation.rsr_checker \
        --eq "Eq(f(a+r, b, c, d) + f(a-r, b, c, d) - 2*f(a,b,c,d) - 2*f(r,0,0,0), 0)"
"""

import argparse
import inspect
import os
import re
import sys
from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    """Information about a function used in the equations."""

    name: str
    # The formal parameter names from the function definition
    formal_params: list[str] = field(default_factory=list)
    # Number of parameters
    arity: int = 0


# Known function definitions from the algebraic benchmark
# Maps function name -> list of formal parameter names
KNOWN_FUNCTIONS = {
    # A01, A02: det 2x2
    "f(a,b,c,d)": FunctionInfo("f", ["a", "b", "c", "d"], 4),
    # A02: det product
    "det_product": FunctionInfo(
        "det_product",
        ["a1", "b1", "c1", "d1", "a2", "b2", "c2", "d2"],
        8,
    ),
    # A03: trace 2x2 (same signature as det)
    # A04: quaternion norm sq (same signature)
    # A05: quaternion mult
    "qprod_norm_sq": FunctionInfo(
        "qprod_norm_sq",
        ["a1", "b1", "c1", "d1", "a2", "b2", "c2", "d2"],
        8,
    ),
    "norm_sq": FunctionInfo("norm_sq", ["a", "b", "c", "d"], 4),
    # A07, A08, A09: 3-variable functions
    "f(x,y,z)": FunctionInfo("f", ["x", "y", "z"], 3),
    # A10: 2-variable
    "f(x,y)": FunctionInfo("f", ["x", "y"], 2),
    # A11: cross product (6-variable)
    "f(x1,y1,z1,x2,y2,z2)": FunctionInfo(
        "f", ["x1", "y1", "z1", "x2", "y2", "z2"], 6
    ),
}


def extract_function_calls(eq_str: str) -> list[tuple[str, list[str]]]:
    """Extract all function calls from an equation string.

    Returns list of (func_name, [arg1, arg2, ...]) tuples.
    """
    # Match function calls like f(...), norm_sq(...), det_product(...)
    # Handle nested parentheses
    calls = []
    # Find function name followed by opening paren
    pattern = r"(\w+)\("
    for m in re.finditer(pattern, eq_str):
        func_name = m.group(1)
        # Skip common non-function names
        if func_name in ("Eq", "Abs", "sqrt", "log", "exp", "sin", "cos",
                         "tan", "sinh", "cosh", "tanh", "atan", "asin",
                         "acos", "pi"):
            continue

        # Extract the arguments by counting parentheses
        start = m.end()
        depth = 1
        pos = start
        while pos < len(eq_str) and depth > 0:
            if eq_str[pos] == "(":
                depth += 1
            elif eq_str[pos] == ")":
                depth -= 1
            pos += 1

        args_str = eq_str[start : pos - 1]
        # Split by top-level commas (not inside nested parens)
        args = split_top_level(args_str)
        calls.append((func_name, [a.strip() for a in args]))

    return calls


def split_top_level(s: str) -> list[str]:
    """Split string by commas, respecting parentheses nesting."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def extract_variables(expr: str) -> set[str]:
    """Extract all variable names from an expression."""
    # Match word characters that aren't function names or numbers
    tokens = re.findall(r"\b([a-zA-Z_]\w*)\b", expr)
    # Filter out known non-variables
    skip = {
        "Eq", "f", "norm_sq", "qprod_norm_sq", "det_product",
        "oct_norm_sq", "oct_prod_norm_sq", "cl20_det", "cl20_prod_det",
        "Abs", "sqrt", "log", "exp", "sin", "cos", "tan",
        "sinh", "cosh", "tanh", "atan", "asin", "acos",
        "pi", "E", "I", "oo", "zoo", "nan",
    }
    return {t for t in tokens if t not in skip and not t.isdigit()}


def extract_variables_in_f_calls(eq_str: str) -> set[str]:
    """Extract variables that appear INSIDE function calls."""
    calls = extract_function_calls(eq_str)
    vars_in_calls = set()
    for func_name, args in calls:
        for arg in args:
            vars_in_calls |= extract_variables(arg)
    return vars_in_calls


def is_constant_expr(expr: str) -> bool:
    """Check if an expression is a constant (no variables)."""
    return len(extract_variables(expr)) == 0


def detect_input_variables(
    eq_str: str,
    input_params: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Detect input vs random/extra variables in an equation.

    If input_params is provided, any variable NOT in input_params is
    considered "extra" (random or constant). Otherwise, uses heuristics.

    Returns (input_vars, extra_vars)
    """
    all_vars = extract_variables(eq_str)

    if input_params is not None:
        input_vars = all_vars & input_params
        extra_vars = all_vars - input_params
        return input_vars, extra_vars

    # Fallback heuristic when input_params not known
    random_names = set()
    for var in all_vars:
        if re.match(r"^r\d*$|^s\d*$|^t\d*$|^u\d*$", var):
            random_names.add(var)
        elif re.match(r"^k\d*$|^n$", var):
            random_names.add(var)

    # Check for +r/-r patterns
    for var in all_vars:
        in_plus = re.search(
            rf"\w+\([^)]*\b\w+\s*[+]\s*{re.escape(var)}\b", eq_str
        )
        in_minus = re.search(
            rf"\w+\([^)]*\b\w+\s*[-]\s*{re.escape(var)}\b", eq_str
        )
        if in_plus and in_minus:
            random_names.add(var)

    return all_vars - random_names, random_names


# Map benchmark ID -> set of input parameter names for the PRIMARY function.
# The primary function is 'f' (or 'norm_sq' for A05).
# Variables NOT in this set are "extra" -- either random shares or
# parameters of secondary functions (det_product, qprod_norm_sq).
#
# Key insight: when f(a,b,c,d) is the function, and the agent writes
# f(a1+a2, b1+b2, ...), then a1,a2 are random shares of the input 'a'.
# The input to f is the SUM, and the shares are the randomness.
# So for f's perspective, only {a,b,c,d} are input; a1,a2 etc. are extra.
BENCHMARK_INPUT_PARAMS = {
    # A01: det(A) additive -- f(a,b,c,d)
    "A01": {"a", "b", "c", "d"},
    # A02: det(A) multiplicative -- f(a,b,c,d), det_product is secondary
    "A02": {"a", "b", "c", "d"},
    # A03: trace -- f(a,b,c,d)
    "A03": {"a", "b", "c", "d"},
    # A04: quaternion norm sq -- f(a,b,c,d)
    "A04": {"a", "b", "c", "d"},
    # A05: quaternion norm sq mult -- norm_sq(a,b,c,d), qprod_norm_sq is secondary
    "A05": {"a", "b", "c", "d"},
    # A06: trace squared -- f(a,b,c,d)
    "A06": {"a", "b", "c", "d"},
    # A07: e2(x,y,z)
    "A07": {"x", "y", "z"},
    # A08: e3(x,y,z)
    "A08": {"x", "y", "z"},
    # A09: p2(x,y,z)
    "A09": {"x", "y", "z"},
    # A10: f(x,y)
    "A10": {"x", "y"},
    # A11: f(x1,y1,z1,x2,y2,z2)
    "A11": {"x1", "y1", "z1", "x2", "y2", "z2"},
    # A12: octonion norm sq -- f(a,b,c,d,p,q,r,s)
    "A12": {"a", "b", "c", "d", "p", "q", "r", "s"},
    # A13: octonion norm sq mult -- oct_norm_sq(a,b,c,d,p,q,r,s)
    "A13": {"a", "b", "c", "d", "p", "q", "r", "s"},
    # A14: Cl(3,0) conj norm -- f(s,v1,v2,v3,b1,b2,b3,t)
    "A14": {"s", "v1", "v2", "v3", "b1", "b2", "b3", "t"},
    # A15: Cl(2,0) det mult -- cl20_det(s,a,b,c)
    "A15": {"s", "a", "b", "c"},
    # A16: Lie bracket tr([A,B]^2) -- f(a1,b1,c1,d1,a2,b2,c2,d2)
    "A16": {"a1", "b1", "c1", "d1", "a2", "b2", "c2", "d2"},
    # A17: sl(2) Killing -- f(a,b,c)
    "A17": {"a", "b", "c"},
}


@dataclass
class RSRCheckResult:
    """Result of RSR classification."""

    equation: str
    is_rsr: bool
    reason: str
    confidence: str  # "high", "medium", "low"


def check_rsr(
    eq_str: str,
    input_params: set[str] | None = None,
) -> RSRCheckResult:
    """Check if a verified property is approximately an RSR.

    Args:
        eq_str: The equation string (e.g., "Eq(f(x+r) - f(x) - f(r), 0)")
        input_params: Known input parameter names for the function(s).
            Variables not in this set are treated as random/extra.

    Returns RSRCheckResult with classification and reasoning.
    """
    eq_str = eq_str.strip()

    # Extract function calls and variables inside them
    all_func_calls = extract_function_calls(eq_str)
    vars_in_calls = extract_variables_in_f_calls(eq_str)

    # ---- Rule 1: Trivial evaluations ----
    # e.g., f(0,0,0,0) = 0, norm_sq(0,0,0,0) = 0
    if len(all_func_calls) == 1:
        _, args = all_func_calls[0]
        if all(is_constant_expr(a) for a in args):
            return RSRCheckResult(
                eq_str, False,
                "Single function call with only constants",
                "high",
            )

    # ---- Rule 2: Pure permutation/symmetry ----
    # e.g., f(a,b,c,d) - f(a,c,b,d) = 0
    # Check across all same-named function pairs
    from collections import Counter
    func_names = Counter(name for name, _ in all_func_calls)
    for fname, count in func_names.items():
        if count == 2:
            fcalls = [(n, a) for n, a in all_func_calls if n == fname]
            args1_vars = set()
            args2_vars = set()
            for a in fcalls[0][1]:
                args1_vars |= extract_variables(a)
            for a in fcalls[1][1]:
                args2_vars |= extract_variables(a)
            if args1_vars == args2_vars and len(all_func_calls) == 2:
                all_simple = all(
                    re.match(r"^-?\s*[a-zA-Z_]\w*$", a.strip())
                    for a in fcalls[0][1] + fcalls[1][1]
                )
                if all_simple:
                    return RSRCheckResult(
                        eq_str, False,
                        f"Permutation symmetry of {fname}()",
                        "high",
                    )

    # ---- Rule 3: Detect input vs extra variables ----
    input_vars, extra_vars = detect_input_variables(eq_str, input_params)

    # Check if any extra variables appear inside function calls
    extra_in_calls = extra_vars & vars_in_calls
    # ---- Rule 4: Scaling/homogeneity ----
    # e.g., f(k*a,...) - k**2*f(a,...) = 0
    # k is extra but only used as a scalar multiplier, not a query point
    if extra_vars and not extra_in_calls:
        # All extra vars are outside f-calls -- recovery-only vars
        # Check if they're scaling constants (k, n, numeric)
        if all(re.match(r"^[kn]\d*$", v) for v in extra_vars):
            return RSRCheckResult(
                eq_str, False,
                f"Scaling/homogeneity with constant(s) {extra_vars}",
                "high",
            )
        # Extra vars appear in the equation but not in any f-call
        return RSRCheckResult(
            eq_str, False,
            f"Extra variables {extra_vars} outside function calls only",
            "medium",
        )

    # ---- Rule 5: No extra variables at all ----
    if not extra_vars:
        # Check for arithmetic in f-args (self-relations like row ops)
        has_arithmetic = False
        for _, args in all_func_calls:
            for a in args:
                if re.search(r"[+\-]", a) and extract_variables(a):
                    has_arithmetic = True
                    break

        if not has_arithmetic:
            return RSRCheckResult(
                eq_str, False,
                "No extra variables; structural property",
                "high",
            )

        # Arithmetic on input vars only (e.g., row operations)
        return RSRCheckResult(
            eq_str, False,
            "Arithmetic on input variables only; no randomness",
            "medium",
        )

    # ---- Rule 6: Extra variables inside f-calls = RSR ----
    if extra_in_calls:
        # Count function calls using extra variables vs input variables
        calls_with_extra = 0
        calls_with_input = 0
        for _, args in all_func_calls:
            call_vars = set()
            for a in args:
                call_vars |= extract_variables(a)
            if call_vars & extra_vars:
                calls_with_extra += 1
            if call_vars & input_vars:
                calls_with_input += 1

        # Strong RSR: extra vars in query points AND input recoverable
        if calls_with_extra >= 1 and calls_with_input >= 1:
            return RSRCheckResult(
                eq_str, True,
                f"Extra variables {extra_in_calls} in {calls_with_extra} "
                f"query point(s); input present in {calls_with_input} call(s)",
                "high",
            )

        # All function calls use only extra vars (no input present).
        # This could still be RSR if the "input" is implicitly a combination
        # of extra vars, e.g., f(a1+a2,b1+b2,...) where a=a1+a2 is the input.
        # Heuristic: if any f-arg contains addition/subtraction of extra vars,
        # those are correlated query points (additive secret sharing).
        if calls_with_extra >= 1 and calls_with_input == 0:
            has_combined_args = False
            has_simple_args = False
            for _, args in all_func_calls:
                for a in args:
                    a_vars = extract_variables(a)
                    if a_vars and a_vars <= extra_vars:
                        if re.search(r"[+\-]", a) and len(a_vars) >= 2:
                            has_combined_args = True
                        elif len(a_vars) == 1:
                            has_simple_args = True

            # f(a1+a2,...) with f(a1,...) and f(a2,...) = additive sharing RSR
            if has_combined_args and has_simple_args and len(all_func_calls) >= 3:
                return RSRCheckResult(
                    eq_str, True,
                    f"Additive sharing: input as sum of extra variables "
                    f"{extra_in_calls}; {calls_with_extra} query point(s)",
                    "high",
                )

            # f(r1,0,0,r4) - r1**2 - r4**2 = 0
            # Property of f at random points, not self-reduction
            return RSRCheckResult(
                eq_str, False,
                f"Extra variables in queries but no input variables; "
                f"property of f at random points, not self-reduction",
                "medium",
            )

        # Fallback
        return RSRCheckResult(
            eq_str, True,
            f"Extra variables {extra_in_calls} in query points",
            "medium",
        )

    # ---- Default: uncertain ----
    return RSRCheckResult(
        eq_str, False,
        "Unable to classify confidently",
        "low",
    )


def parse_equations_from_file(filepath: str) -> list[str]:
    """Parse verified equations from a result file."""
    equations = []
    in_verified = False
    with open(filepath) as fd:
        for line in fd:
            line = line.strip()
            if line.startswith("Verified"):
                in_verified = True
                continue
            if in_verified:
                if line.startswith("Eq("):
                    # Handle "Eq(...) | error_msg" format
                    eq_part = line.split("|")[0].strip()
                    equations.append(eq_part)
                elif line == "" or line.startswith(("Unverified", "Faulty",
                                                    "Unknown", "Took time")):
                    in_verified = False
    return equations


def get_benchmark_id(filename: str) -> str | None:
    """Extract benchmark ID (e.g., 'A01') from filename."""
    m = re.match(r"(A\d+)", filename)
    return m.group(1) if m else None


def process_file(filepath: str, verbose: bool = False) -> tuple[int, int]:
    """Process a single result file and count RSRs.

    Returns (total_verified, rsr_count).
    """
    equations = parse_equations_from_file(filepath)
    basename = os.path.basename(filepath)

    # Look up input params for this benchmark
    bench_id = get_benchmark_id(basename)
    input_params = BENCHMARK_INPUT_PARAMS.get(bench_id)

    rsr_count = 0
    non_rsr_count = 0
    rsr_eqs = []
    non_rsr_eqs = []

    for eq in equations:
        result = check_rsr(eq, input_params=input_params)
        if result.is_rsr:
            rsr_count += 1
            rsr_eqs.append(result)
        else:
            non_rsr_count += 1
            non_rsr_eqs.append(result)

    total = len(equations)
    print(f"{basename}: {rsr_count}/{total} RSRs "
          f"({non_rsr_count} non-RSR)")

    if verbose:
        if rsr_eqs:
            print(f"  RSRs ({rsr_count}):")
            for r in rsr_eqs:
                print(f"    [RSR] [{r.confidence}] {r.equation}")
                print(f"           {r.reason}")
        if non_rsr_eqs:
            print(f"  Non-RSRs ({non_rsr_count}):")
            for r in non_rsr_eqs:
                print(f"    [---] [{r.confidence}] {r.equation}")
                print(f"           {r.reason}")
        print()

    return total, rsr_count, rsr_eqs, non_rsr_eqs


# Friendly names for benchmark IDs
BENCHMARK_NAMES = {
    "A01": "det(A) additive",
    "A02": "det(A) multiplicative",
    "A03": "tr(A)",
    "A04": "‖q‖² additive",
    "A05": "‖q₁q₂‖² multiplicative",
    "A06": "tr(A²)",
    "A07": "e₂(x,y,z)",
    "A08": "e₃(x,y,z)",
    "A09": "p₂(x,y,z)",
    "A10": "‖v‖² (2D)",
    "A11": "‖a × b‖²",
    "A12": "‖o‖² (octonion)",
    "A13": "‖o₁o₂‖² (Moufang)",
    "A14": "Cl(3,0) conj norm",
    "A15": "Cl(2,0) det mult",
    "A16": "tr([A,B]²) gl(2)",
    "A17": "sl(2) Killing",
}


def generate_report(
    output_path: str,
    all_file_results: list[tuple],
    total_verified: int,
    total_rsr: int,
):
    """Generate a markdown report of RSR classification results."""
    with open(output_path, "w") as f:
        f.write("# RSR Classification Report\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| # | Benchmark | RSR | Non-RSR | Total | RSR % |\n")
        f.write("|---|---|:---:|:---:|:---:|:---:|\n")

        for fname, verified, rsrs, rsr_eqs, non_rsr_eqs in all_file_results:
            bench_id = get_benchmark_id(fname)
            name = BENCHMARK_NAMES.get(bench_id, fname)
            non_rsrs = verified - rsrs
            pct = f"{100 * rsrs / verified:.0f}%" if verified > 0 else "N/A"
            f.write(f"| {bench_id} | {name} | {rsrs} | {non_rsrs} "
                    f"| {verified} | {pct} |\n")

        total_non_rsr = total_verified - total_rsr
        total_pct = (
            f"{100 * total_rsr / total_verified:.0f}%"
            if total_verified > 0 else "N/A"
        )
        f.write(f"| | **Total** | **{total_rsr}** | **{total_non_rsr}** "
                f"| **{total_verified}** | **{total_pct}** |\n")
        f.write("\n---\n\n")

        # Per-file details
        f.write("## Per-Benchmark Details\n\n")

        for fname, verified, rsrs, rsr_eqs, non_rsr_eqs in all_file_results:
            bench_id = get_benchmark_id(fname)
            name = BENCHMARK_NAMES.get(bench_id, fname)
            non_rsrs = verified - rsrs

            f.write(f"### {bench_id}: {name} ({rsrs} RSR, "
                    f"{non_rsrs} non-RSR, {verified} total)\n\n")

            if rsr_eqs:
                f.write(f"**RSRs ({len(rsr_eqs)}):**\n\n")
                for i, r in enumerate(rsr_eqs, 1):
                    f.write(f"{i}. `{r.equation}`\n")
                    f.write(f"   - *{r.reason}* [{r.confidence}]\n")
                f.write("\n")

            if non_rsr_eqs:
                f.write(f"**Non-RSRs ({len(non_rsr_eqs)}):**\n\n")
                for i, r in enumerate(non_rsr_eqs, 1):
                    f.write(f"{i}. `{r.equation}`\n")
                    f.write(f"   - *{r.reason}* [{r.confidence}]\n")
                f.write("\n")

            f.write("---\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Approximate RSR checker for verified properties"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing result .txt files",
    )
    parser.add_argument(
        "--eq",
        type=str,
        help="Check a single equation string",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show classification for each equation",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Generate markdown report at this path",
    )
    args = parser.parse_args()

    if args.eq:
        result = check_rsr(args.eq)
        marker = "RSR" if result.is_rsr else "NOT RSR"
        print(f"[{marker}] [{result.confidence}] {args.eq}")
        print(f"  Reason: {result.reason}")
        return

    if args.results_dir:
        total_verified = 0
        total_rsr = 0
        all_file_results = []

        files = sorted(
            f
            for f in os.listdir(args.results_dir)
            if f.endswith(".txt") and not f.endswith("_trace.csv")
        )

        for fname in files:
            filepath = os.path.join(args.results_dir, fname)
            verified, rsrs, rsr_eqs, non_rsr_eqs = process_file(
                filepath, verbose=args.verbose
            )
            total_verified += verified
            total_rsr += rsrs
            all_file_results.append((fname, verified, rsrs, rsr_eqs, non_rsr_eqs))
            if args.verbose:
                print()

        print(f"\nTotal: {total_rsr}/{total_verified} RSRs across {len(files)} files")

        # Generate markdown report if requested
        if args.report:
            generate_report(args.report, all_file_results, total_verified, total_rsr)
            print(f"Report written to {args.report}")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
