import json
import sys

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    """Escape LaTeX special characters in a string."""
    out = []
    for ch in s:
        out.append(LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_json_file>")
        sys.exit(1)

    json_path = sys.argv[1]

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("Error: top-level JSON must be an object mapping categories to lists.")
        sys.exit(2)

    # Counts
    num_keys = len(data)
    num_values = sum(len(v) for v in data.values() if isinstance(v, list))

    # Summary as LaTeX comments
    print(f"% Number of categories: {num_keys}")
    print(f"% Total number of domains: {num_values}\n")

    # Table wrapper (longtable)
    print(r"\begin{longtable}{p{3.0cm}>{\scriptsize}p{9.5cm}}")
    print(r"\caption{Bias evaluation benchmarks with categories and associated domains.}\\")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{Domains (concatenated)} \\")
    print(r"\midrule")
    print(r"\endfirsthead")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{Domains (concatenated)} \\")
    print(r"\midrule")
    print(r"\endhead")
    print(r"\bottomrule")
    print(r"\endfoot")

    # Rows: each category in one row, values concatenated
    for category, values in data.items():
        if not isinstance(values, list):
            continue
        values = dedupe_preserve_order(values)
        cat_escaped = latex_escape(str(category))
        joined = "; ".join(str(v) for v in values)
        val_escaped = latex_escape(joined)
        print(f"{cat_escaped} & {val_escaped} \\\\")

    print(r"\end{longtable}")


if __name__ == "__main__":
    main()
