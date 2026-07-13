import os


def perturb_dataset(
    source_filename,
    target_filename,
):
    """
    Expects data file to have formula\ntrace\n format
    """
    output_lines = []
    ap_count = 5
    aps = list(map(chr, range(97, 97 + ap_count)))
    aps_set = set(aps)
    with open(source_filename, 'r') as file:  # expect formula\ntrace\n format
        for formula in file:
            if formula == '\n':
                break
            formula = formula.strip()
            trace = next(file).strip()  # get second line
            # list(dict.fromkeys()) remove duplicates while preserving order
            formula_aps = list(dict.fromkeys([i for i in formula.replace("xor", "") if i.islower()]))
            trace_aps = list(dict.fromkeys([i for i in trace.replace("xor", "") if i.islower()]))
            assert set(trace_aps).issubset(set(formula_aps)), f"{formula_aps} does not contain {trace_aps}"
            assert set(formula_aps).issubset(aps_set), f"{aps} does not contain {formula_aps}"
            # Make sure the aps are sorted
            # We don't want a formula like (e & f), it must be (a & b)
            my_aps = list(dict.fromkeys(trace_aps + formula_aps))
            lookup = {k: aps[i] for i, k in enumerate(my_aps)}
            formula = "".join([lookup.get(c, c) for c in formula])
            trace = "".join([lookup.get(c, c) for c in trace])
            output_lines.append(f"{formula}\n{trace}\n")
    
    output_lines = "".join(output_lines)

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    # Save to the target file
    with open(target_filename, 'w') as file:
        file.write(output_lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_filename", type=str)
    parser.add_argument("target_filename", type=str)
    args = parser.parse_args()
    # If target_filename is a directory, use the source_filename as the target_filename
    if os.path.isdir(args.target_filename) or not args.target_filename.endswith(".txt"):
        args.target_filename = os.path.join(args.target_filename, os.path.basename(args.source_filename))
    perturb_dataset(args.source_filename, args.target_filename)