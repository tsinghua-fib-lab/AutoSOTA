#  Quine-McCluskey Boolean minimization with XOR/XNOR support.
#
#  Based on the implementation by Thomas Pircher <tehpeh-web@tty1.net>
#  (c) 2006-2016, MIT License. Extended with DNF construction,
#  truth table enumeration, and complexity scoring.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.

"""Quine-McCluskey Boolean minimization, DNF construction, and truth table utilities."""

from __future__ import annotations

import itertools
import math
import re

import numpy as np


# =============================================================================
# Truth table enumeration
# =============================================================================


def enumerate_truth_table(n_bits: int, weights: list[float], bias: float, threshold: float = 0.0) -> list[int]:
    """Return minterm indices where weighted_sum + bias > threshold."""
    n_rows = 1 << n_bits
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int32)
    indices = np.arange(n_rows, dtype=np.int32)
    bits = ((indices[:, np.newaxis] >> shifts) & 1).astype(np.float64)
    sums = bits @ np.asarray(weights, dtype=np.float64) + bias
    return indices[sums > threshold].tolist()


# =============================================================================
# Quine-McCluskey minimizer
# =============================================================================


class QuineMcCluskey:
    """Minimize Boolean functions using the Quine-McCluskey algorithm.

    If instantiated with use_xor=True, the resulting expression may
    contain XOR ('^') and XNOR ('~') operators for more compact rules.
    """

    def __init__(self, use_xor: bool = False):
        self.use_xor = use_xor
        self.n_bits = 0

    def simplify(self, ones: list[int], dc: list[int] | None = None, num_bits: int | None = None) -> list[str] | None:
        """Minimize a truth table given ON-set minterms and optional don't-cares.

        Args:
            ones: Minterm indices where the function outputs 1.
            dc: Don't-care minterm indices.
            num_bits: Force a specific bit width (inferred from max term if None).

        Returns:
            Sorted list of implicant strings (chars: '0','1','-','^','~'),
            or None if the input is empty.
        """
        if dc is None:
            dc = []
        terms = ones + dc
        if not terms:
            return None

        if num_bits is not None:
            self.n_bits = num_bits
        else:
            self.n_bits = max(1, int(math.ceil(math.log2(max(terms) + 1))))

        ones_s = [self._i2b(i) for i in ones]
        dc_s = [self._i2b(i) for i in dc]
        return self._run(ones_s, dc_s)

    def _i2b(self, v: int) -> str:
        return "".join("1" if v & (1 << k) else "0" for k in range(self.n_bits - 1, -1, -1))

    def _run(self, ones: list[str], dc: list[str]) -> list[str] | None:
        terms = set(ones) | set(dc)
        if not terms:
            return None
        primes = self._get_prime_implicants(terms)
        essential = self._get_essential_implicants(primes, set(dc))
        return self._reduce_implicants(essential, set(dc))

    def _reduce_simple_xor_terms(self, t1: str, t2: str) -> str | None:
        difft10 = 0
        difft20 = 0
        ret = []
        for t1c, t2c in zip(t1, t2):
            if t1c in ('^', '~') or t2c in ('^', '~'):
                return None
            elif t1c != t2c:
                ret.append('^')
                if t2c == '0':
                    difft10 += 1
                else:
                    difft20 += 1
            else:
                ret.append(t1c)
        if difft10 == 1 and difft20 == 1:
            return "".join(ret)
        return None

    def _reduce_simple_xnor_terms(self, t1: str, t2: str) -> str | None:
        difft10 = 0
        difft20 = 0
        ret = []
        for t1c, t2c in zip(t1, t2):
            if t1c in ('^', '~') or t2c in ('^', '~'):
                return None
            elif t1c != t2c:
                ret.append('~')
                if t1c == '0':
                    difft10 += 1
                else:
                    difft20 += 1
            else:
                ret.append(t1c)
        if (difft10 == 2 and difft20 == 0) or (difft10 == 0 and difft20 == 2):
            return "".join(ret)
        return None

    def _get_prime_implicants(self, terms: set[str]) -> set[str]:
        n_groups = self.n_bits + 1
        marked: set[str] = set()

        groups: list[set[str]] = [set() for _ in range(n_groups)]
        for t in terms:
            groups[t.count('1')].add(t)

        if self.use_xor:
            for gi, group in enumerate(groups):
                for t1 in sorted(group):
                    for t2 in sorted(group):
                        t12 = self._reduce_simple_xor_terms(t1, t2)
                        if t12 is not None:
                            terms.add(t12)
                    if gi < n_groups - 2:
                        for t2 in sorted(groups[gi + 2]):
                            t12 = self._reduce_simple_xnor_terms(t1, t2)
                            if t12 is not None:
                                terms.add(t12)

        done = False
        groups_dict: dict[tuple, set[str]] = {}
        while not done:
            groups_dict = {}
            for t in terms:
                n_ones = t.count('1')
                n_xor = t.count('^')
                n_xnor = t.count('~')
                key = (n_ones, n_xor, n_xnor)
                if key not in groups_dict:
                    groups_dict[key] = set()
                groups_dict[key].add(t)

            terms = set()
            used: set[str] = set()

            for key in sorted(groups_dict.keys()):
                key_next = (key[0] + 1, key[1], key[2])
                if key_next in groups_dict:
                    group_next = groups_dict[key_next]
                    for t1 in sorted(groups_dict[key]):
                        for i, c1 in enumerate(t1):
                            if c1 == '0':
                                t2 = t1[:i] + '1' + t1[i + 1:]
                                if t2 in group_next:
                                    t12 = t1[:i] + '-' + t1[i + 1:]
                                    used.add(t1)
                                    used.add(t2)
                                    terms.add(t12)

            for key in sorted(k for k in groups_dict if k[1] > 0):
                key_complement = (key[0] + 1, key[2], key[1])
                if key_complement in groups_dict:
                    for t1 in sorted(groups_dict[key]):
                        t1_complement = t1.replace('^', '~')
                        for i, c1 in enumerate(t1):
                            if c1 == '0':
                                t2 = t1_complement[:i] + '1' + t1_complement[i + 1:]
                                if t2 in groups_dict[key_complement]:
                                    t12 = t1[:i] + '^' + t1[i + 1:]
                                    used.add(t1)
                                    terms.add(t12)

            for key in sorted(k for k in groups_dict if k[2] > 0):
                key_complement = (key[0] + 1, key[2], key[1])
                if key_complement in groups_dict:
                    for t1 in sorted(groups_dict[key]):
                        t1_complement = t1.replace('~', '^')
                        for i, c1 in enumerate(t1):
                            if c1 == '0':
                                t2 = t1_complement[:i] + '1' + t1_complement[i + 1:]
                                if t2 in groups_dict[key_complement]:
                                    t12 = t1[:i] + '~' + t1[i + 1:]
                                    used.add(t1)
                                    terms.add(t12)

            for g in groups_dict.values():
                marked |= g - used

            done = len(used) == 0

        for g in groups_dict.values():
            marked |= g
        return marked

    def _get_essential_implicants(self, terms: set[str], dc: set[str]) -> set[str]:
        perms = {t: set(p for p in self.permutations(t) if p not in dc) for t in sorted(terms)}

        ei_range: set[str] = set()
        ei: set[str] = set()
        groups: dict[int, set[str]] = {}
        for t in sorted(terms):
            n = self._term_rank(t, len(perms[t]))
            groups.setdefault(n, set()).add(t)
        for n in sorted(groups.keys(), reverse=True):
            for g in sorted(groups[n]):
                if not perms[g] <= ei_range:
                    ei.add(g)
                    ei_range |= perms[g]
        if not ei:
            ei = {'-' * self.n_bits}
        return ei

    def _term_rank(self, term: str, term_range: int) -> int:
        n = 0
        for t in term:
            if t == '-':
                n += 8
            elif t == '^':
                n += 4
            elif t == '~':
                n += 2
            elif t == '1':
                n += 1
        return 4 * term_range + n

    def permutations(self, value: str) -> list[str]:
        """Generate all concrete bit-strings from an implicant pattern.

        Input characters: '0', '1', '-', '^', '~'.
        Yields strings containing only '0' and '1'.
        """
        n_bits = len(value)
        n_xor = value.count('^') + value.count('~')
        xor_value = 0
        seen_xors = 0
        res = ['0'] * n_bits
        results: list[str] = []
        i = 0
        direction = +1
        while i >= 0:
            if value[i] in ('0', '1'):
                res[i] = value[i]
            elif value[i] == '-':
                if direction == +1:
                    res[i] = '0'
                elif res[i] == '0':
                    res[i] = '1'
                    direction = +1
            elif value[i] == '^':
                seen_xors += direction
                if direction == +1:
                    if seen_xors == n_xor and xor_value == 0:
                        res[i] = '1'
                    else:
                        res[i] = '0'
                else:
                    if res[i] == '0' and seen_xors < n_xor - 1:
                        res[i] = '1'
                        direction = +1
                        seen_xors += 1
                if res[i] == '1':
                    xor_value ^= 1
            elif value[i] == '~':
                seen_xors += direction
                if direction == +1:
                    if seen_xors == n_xor and xor_value == 1:
                        res[i] = '1'
                    else:
                        res[i] = '0'
                else:
                    if res[i] == '0' and seen_xors < n_xor - 1:
                        res[i] = '1'
                        direction = +1
                        seen_xors += 1
                if res[i] == '1':
                    xor_value ^= 1

            i += direction
            if i == n_bits:
                direction = -1
                i = n_bits - 1
                results.append("".join(res))

        return results

    def _reduce_implicants(self, implicants: set[str], dc: set[str]) -> list[str]:
        def complexity(imp: str) -> float:
            c = 0.0
            for ch in imp:
                if ch == '1':
                    c += 1.0
                elif ch == '0':
                    c += 1.5
                elif ch == '^':
                    c += 1.25
                elif ch == '~':
                    c += 1.75
            return c

        def combine(a: str, b: str) -> str | None:
            perms_a = set(p for p in self.permutations(a) if p not in dc)
            perms_b = set(p for p in self.permutations(b) if p not in dc)
            a_dcs = [i for i, c in enumerate(a) if c == '-']
            b_dcs = [i for i, c in enumerate(b) if c == '-']
            a_pot, b_pot = list(a), list(b)
            for idx in a_dcs:
                a_pot[idx] = b[idx]
            for idx in b_dcs:
                b_pot[idx] = a[idx]
            valid = []
            for cand in [''.join(a_pot), ''.join(b_pot)]:
                cand_perms = set(p for p in self.permutations(cand) if p not in dc)
                if cand_perms == perms_a | perms_b:
                    valid.append(cand)
            if valid:
                return sorted(valid, key=complexity)[0]
            return None

        while True:
            for a, b in itertools.combinations(sorted(implicants), 2):
                replacement = combine(a, b)
                if replacement:
                    implicants.discard(a)
                    implicants.discard(b)
                    implicants.add(replacement)
                    break
            else:
                break

        coverage = {
            imp: set(p for p in self.permutations(imp) if p not in dc)
            for imp in implicants
        }

        while True:
            redundant = []
            for imp in sorted(coverage):
                others = set().union(*(coverage[o] for o in coverage if o != imp))
                if coverage[imp] <= others:
                    redundant.append(imp)
            if redundant:
                worst = sorted(redundant, key=complexity, reverse=True)[0]
                del coverage[worst]
            else:
                break

        if not coverage:
            return ['-' * self.n_bits]
        return sorted(coverage.keys())


# =============================================================================
# DNF expression construction
# =============================================================================


def implicants_to_dnf(implicants: list[str] | set[str] | None, var_prefix: str = "x") -> str:
    """Convert QMC implicant strings to a DNF expression.

    Handles characters: '1' (asserted), '0' (negated), '-' (don't care),
    '^' (XOR group), '~' (XNOR group / negated XOR).
    """
    if not implicants:
        return "False"

    terms: list[str] = []
    for term in implicants:
        normal_parts: list[str] = []
        xor_positions: list[str] = []
        xnor_positions: list[str] = []

        for idx, ch in enumerate(term):
            var = f"{var_prefix}{idx}"
            if ch == '1':
                normal_parts.append(var)
            elif ch == '0':
                normal_parts.append(f"~{var}")
            elif ch == '^':
                xor_positions.append(var)
            elif ch == '~':
                xnor_positions.append(var)

        parts: list[str] = []
        if normal_parts:
            parts.extend(normal_parts)
        if xor_positions:
            if len(xor_positions) == 1:
                parts.append(xor_positions[0])
            else:
                parts.append(f"({' ^ '.join(xor_positions)})")
        if xnor_positions:
            if len(xnor_positions) == 1:
                parts.append(xnor_positions[0])
            else:
                parts.append(f"~({' ^ '.join(xnor_positions)})")

        if parts:
            terms.append("(" + " & ".join(parts) + ")")
        else:
            terms.append("True")

    return " | ".join(terms)


def dnf_cost(implicants: list[str] | set[str] | None) -> float:
    """Count total literals in a DNF expression (complexity metric).

    Each non-'-' character in each implicant counts as one literal.
    Returns 0 for constant functions.
    """
    if not implicants:
        return 0.0
    total = 0
    for term in implicants:
        n = sum(1 for ch in term if ch != '-')
        if n == 0:
            return 0.0
        total += n
    return float(total)


# =============================================================================
# Negation simplification
# =============================================================================


_FLIP = {">=": "<", ">": "<=", "<=": ">", "<": ">=", "==": "!=", "!=": "=="}


def simplify_negations(expr: str) -> str:
    """Simplify ~(feature op value) into (feature flipped_op value)."""
    pat = r"~\(([^)]+?)\s*(>=|>|<=|<|==|!=)\s*([^)]+?)\)"
    prev = None
    while prev != expr:
        prev = expr
        expr = re.sub(
            pat,
            lambda m: f"({m.group(1).strip()} {_FLIP[m.group(2)]} {m.group(3).strip()})",
            expr,
        )
    return expr
