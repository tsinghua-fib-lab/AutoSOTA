""" Update verifiers """

import sys
from dataclasses import replace

from watermark_benchmark.utils import get_output_file, load_config
from watermark_benchmark.utils.classes import (
    Generation,
    VerifierSpec,
    WatermarkSpec,
)


def patch_generation(config_file, gamma_values, combined):
    """Patch generation verifiers.
    We want to try multiple values of gamma for the edit distance.
    We also do not need the regular empirical distances for internal randomness.
    (and only the edit distance for hash length of 1)"""
    try:
        config = load_config(config_file)
        output_file = get_output_file(config)
        generations = Generation.from_file(output_file)
    except Exception:
        output_file = config_file
        generations = Generation.from_file(output_file)

    if gamma_values is None:
        edit = False
        regular = True
    elif combined:
        edit = True
        regular = True
    else:
        edit = True
        regular = False

    for idx in range(len(generations)):
        watermark = generations[idx].watermark
        if watermark is None:
            continue

        is_trivial = watermark.hash_len == 0 or watermark.key_len == 1
        is_internal = watermark.rng == "Internal"
        is_editable = watermark.hash_len == 1 or not is_internal

        verifiers = []
        if (is_internal or is_trivial) and regular:
            if watermark.generator != "its":
                verifiers.append(
                    VerifierSpec("Theoretical", "regular", True, 0)
                )
            else:
                verifiers.append(VerifierSpec("Empirical", "regular", True, 0))

        if not is_internal and not is_trivial and regular:
            verifiers.append(VerifierSpec("Empirical", "regular", True, 0))

        if edit and is_editable:
            for g in gamma_values:
                verifiers.append(
                    VerifierSpec("Empirical", "levenshtein", True, g)
                )

        generations[idx] = replace(
            generations[idx], watermark=replace(watermark, verifiers=verifiers)
        )

    Generation.to_file(output_file, generations)


def patch_watermarks(f, gamma_values, combined):
    """Similar function, but operates on watermarks"""
    with open(f) as infile:
        watermarks = [
            WatermarkSpec.from_str(l.strip())
            for l in infile.read().split("\n")
            if len(l)
        ]

    if gamma_values is None:
        edit = False
        regular = True
    elif combined:
        edit = True
        regular = True
    else:
        edit = True
        regular = False

    for idx in range(len(watermarks)):
        watermark = watermarks[idx]

        is_trivial = watermark.hash_len == 0 or watermark.key_len == 1
        is_internal = watermark.rng == "Internal"
        is_editable = watermark.hash_len == 1 or not is_internal

        verifiers = []
        if is_trivial and regular:
            verifiers.append(VerifierSpec("Theoretical", "regular", True, 0))
            # verifiers.append(VerifierSpec('Empirical', 'regular', True, 0))

        elif is_internal and regular:
            if watermark.generator != "its":
                verifiers.append(
                    VerifierSpec("Theoretical", "regular", True, 0)
                )
                # verifiers.append(VerifierSpec('Empirical', 'regular', True, 0))
            else:
                verifiers.append(VerifierSpec("Empirical", "regular", True, 0))

        if not is_internal and not is_trivial and regular:
            verifiers.append(VerifierSpec("Empirical", "regular", True, 0))

        if edit and is_editable:
            # only one setting
            if (
                watermark.delta == 2.5
                and watermark.generator == "distributionshift"
                and watermark.temp == 1.0
                and watermark.rng != "Internal"
            ):
                for g in gamma_values:
                    verifiers.append(
                        VerifierSpec("Empirical", "levenshtein", True, g)
                    )

        watermarks[idx] = replace(watermark, verifiers=verifiers)

    with open(f + "_2", "w") as outfile:
        outfile.write("\n".join(str(w) for w in watermarks) + "\n")


def main_gen():
    if "-h" in "".join(sys.argv[1:]):
        print("SYNTAX: patch.py CONFIG COMBINED[bool] DELTAS+")
        return

    print(sys.argv)
    if len(sys.argv) > 3 and sys.argv[2] == "True":
        patch_generation(sys.argv[1], [float(v) for v in sys.argv[3:]], True)
    elif len(sys.argv) > 3:
        patch_generation(sys.argv[1], [float(v) for v in sys.argv[3:]], False)
    else:
        patch_generation(sys.argv[1], None, False)


def main_wat():
    if "-h" in "".join(sys.argv[1:]):
        print("SYNTAX: patch.py FILE COMBINED[bool] DELTAS+")
        return

    print(sys.argv)
    if len(sys.argv) > 3 and sys.argv[2] == "True":
        patch_watermarks(sys.argv[1], [float(v) for v in sys.argv[3:]], True)
    elif len(sys.argv) > 3:
        patch_watermarks(sys.argv[1], [float(v) for v in sys.argv[3:]], False)
    else:
        patch_watermarks(sys.argv[1], None, False)


if __name__ == "__main__":
    main_gen()
