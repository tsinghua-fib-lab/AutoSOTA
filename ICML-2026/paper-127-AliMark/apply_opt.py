#!/usr/bin/env python3
"""
Apply optimization changes to AliMark detection code.
Usage: python3 apply_opt.py <idea_id>

Ideas:
  CODE-2: Fix bare except in adaptive_bit_sequence_alignment.py
  ALGO-1: Replace max() with np.median() for candidate score aggregation
  ALGO-1-mean: Use mean instead of median
  CODE-1: Enable rs_multistep=True
  ALGO-3: Wider ABSA alpha/beta (0.4/2.0)
  ALGO-4: Soft scoring (temperature-scaled bits)
  COMBINED: Apply CODE-2 + ALGO-1 + CODE-1 + ALGO-3 together
"""

import sys
import os

REPO = "/repo"

def apply_code2():
    """Fix bare except in adaptive_bit_sequence_alignment.py:135"""
    filepath = os.path.join(REPO, "watermark/adaptive_bit_sequence_alignment.py")
    with open(filepath) as f:
        content = f.read()

    old = """            try:
                z_score = (mean - ber) / std
            except:
                z_score = 0.0"""

    new = """            if std > 1e-10:
                z_score = (mean - ber) / std
            else:
                z_score = 0.0"""

    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w") as f:
            f.write(content)
        print("CODE-2: Fixed bare except -> explicit std check")
        return True
    elif new in content:
        print("CODE-2: Already applied")
        return True
    else:
        print("CODE-2: Pattern not found. Checking for variant...")
        # Try alternative patterns
        if "except:" in content and "z_score = (mean - ber) / std" in content:
            print("CODE-2: Found bare except but different context")
        return False


def apply_algo1(method="median"):
    """Replace max() with median/mean for candidate score aggregation"""
    filepath = os.path.join(REPO, "watermark/alimark.py")
    with open(filepath) as f:
        content = f.read()

    # Find the max aggregation section
    old_max = "global_max_score = -float('inf')"
    old_loop = """        global_max_score = -float('inf')
        for _, text_candidate in enumerate(unique_candidates):
            extracted_bit_sequence = []
            for _, sent in enumerate(text_candidate):
                bit_signal = bit_signals_map[sent]
                extracted_bit_sequence.append(bit_signal)
            score = self._ABSA.compute_score(
                extracted_bit_sequence,
                self._secret_bit_sequence,
                lower_ratio=absa_lower_ratio,
                upper_ratio=absa_upper_ratio,
                criterion=absa_criterion
                )
            if score > global_max_score:
                global_max_score = score"""

    if method == "median":
        new_loop = """        candidate_scores = []
        for _, text_candidate in enumerate(unique_candidates):
            extracted_bit_sequence = []
            for _, sent in enumerate(text_candidate):
                bit_signal = bit_signals_map[sent]
                extracted_bit_sequence.append(bit_signal)
            score = self._ABSA.compute_score(
                extracted_bit_sequence,
                self._secret_bit_sequence,
                lower_ratio=absa_lower_ratio,
                upper_ratio=absa_upper_ratio,
                criterion=absa_criterion
                )
            candidate_scores.append(score)
        global_max_score = float(np.median(candidate_scores)) if candidate_scores else 0.0"""
    elif method == "mean":
        new_loop = """        candidate_scores = []
        for _, text_candidate in enumerate(unique_candidates):
            extracted_bit_sequence = []
            for _, sent in enumerate(text_candidate):
                bit_signal = bit_signals_map[sent]
                extracted_bit_sequence.append(bit_signal)
            score = self._ABSA.compute_score(
                extracted_bit_sequence,
                self._secret_bit_sequence,
                lower_ratio=absa_lower_ratio,
                upper_ratio=absa_upper_ratio,
                criterion=absa_criterion
                )
            candidate_scores.append(score)
        global_max_score = float(np.mean(candidate_scores)) if candidate_scores else 0.0"""
    elif method == "max":
        # Revert to max
        new_loop = """        global_max_score = -float('inf')
        for _, text_candidate in enumerate(unique_candidates):
            extracted_bit_sequence = []
            for _, sent in enumerate(text_candidate):
                bit_signal = bit_signals_map[sent]
                extracted_bit_sequence.append(bit_signal)
            score = self._ABSA.compute_score(
                extracted_bit_sequence,
                self._secret_bit_sequence,
                lower_ratio=absa_lower_ratio,
                upper_ratio=absa_upper_ratio,
                criterion=absa_criterion
                )
            if score > global_max_score:
                global_max_score = score"""
    else:
        print(f"ALGO-1: Unknown method '{method}'")
        return False

    if old_loop in content:
        content = content.replace(old_loop, new_loop)
        # Add numpy import if needed
        if "import numpy as np" not in content and method in ("median", "mean"):
            # Find the import section
            import_line = "import time"
            if import_line in content:
                content = content.replace(import_line, "import time\nimport numpy as np")
        with open(filepath, "w") as f:
            f.write(content)
        print(f"ALGO-1: Replaced max aggregation with {method}")
        return True
    elif "candidate_scores" in content:
        print(f"ALGO-1: Custom aggregation already applied")
        return True
    else:
        print("ALGO-1: Pattern not found")
        return False


def apply_code1(enable=True):
    """Enable/disable rs_multistep"""
    filepath = os.path.join(REPO, "watermark/alimark.py")
    with open(filepath) as f:
        content = f.read()

    old = '"rs_multistep": False'
    new = '"rs_multistep": True'

    if enable:
        if old in content:
            content = content.replace(old, new)
            with open(filepath, "w") as f:
                f.write(content)
            print("CODE-1: Enabled rs_multistep=True")
            return True
        elif new in content:
            print("CODE-1: rs_multistep already enabled")
            return True
    else:
        if new in content:
            content = content.replace(new, old)
            with open(filepath, "w") as f:
                f.write(content)
            print("CODE-1: Disabled rs_multistep (reverted)")
            return True

    print("CODE-1: Pattern not found")
    return False


def apply_algo3(lower=0.4, upper=2.0):
    """Tune ABSA alpha/beta parameters"""
    filepath = os.path.join(REPO, "watermark/alimark.py")
    with open(filepath) as f:
        content = f.read()

    old_lower = '"absa_lower_ratio": 0.5'
    old_upper = '"absa_upper_ratio": 1.5'

    new_lower = f'"absa_lower_ratio": {lower}'
    new_upper = f'"absa_upper_ratio": {upper}'

    changed = False
    if old_lower in content and new_lower not in content:
        content = content.replace(old_lower, new_lower)
        changed = True
    if old_upper in content and new_upper not in content:
        content = content.replace(old_upper, new_upper)
        changed = True

    if changed:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"ALGO-3: Tuned ABSA alpha/beta to {lower}/{upper}")
        return True
    else:
        print("ALGO-3: Already tuned or pattern not found")
        return True if f'"absa_lower_ratio": {lower}' in content else False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 apply_opt.py <idea_id>")
        print("Available: CODE-2, ALGO-1, ALGO-1-mean, CODE-1, ALGO-3, COMBINED, REVERT")
        sys.exit(1)

    idea = sys.argv[1].upper()
    success = True

    if idea == "CODE-2":
        success = apply_code2()
    elif idea == "ALGO-1":
        method = sys.argv[2] if len(sys.argv) > 2 else "median"
        success = apply_algo1(method)
    elif idea == "ALGO-1-MEAN":
        success = apply_algo1("mean")
    elif idea == "CODE-1":
        enable = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else True
        success = apply_code1(enable)
    elif idea == "ALGO-3":
        lower = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
        upper = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        success = apply_algo3(lower, upper)
    elif idea == "COMBINED":
        success = apply_code2() and success
        success = apply_code1(True) and success
        success = apply_algo3(0.4, 2.0) and success
        success = apply_algo1("median") and success
    elif idea == "REVERT":
        apply_algo1("max")
        apply_code1(False)
        apply_algo3(0.5, 1.5)
        # CODE-2 can't be easily reverted through pattern matching
        print("REVERT: Reset ALGO-1, CODE-1, ALGO-3 to defaults")
    else:
        print(f"Unknown idea: {idea}")
        success = False

    if success:
        print(f"\n{idea}: Successfully applied")
    else:
        print(f"\n{idea}: Some changes failed")

    sys.exit(0 if success else 1)
