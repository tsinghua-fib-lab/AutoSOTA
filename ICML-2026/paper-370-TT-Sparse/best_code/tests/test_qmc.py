"""QMC minimization tests.

Validates that:
1. Known Boolean functions produce correct minimal forms.
2. XOR/XNOR detection produces more compact representations.
3. enumerate_truth_table matches hand-computed outputs.
4. Edge cases (tautology, contradiction, single minterm) are handled.
5. DNF string construction from implicant patterns.
"""

import pytest

from tt_sparse.qmc import (
    QuineMcCluskey,
    dnf_cost,
    enumerate_truth_table,
    implicants_to_dnf,
)


class TestEnumerateTruthTable:
    """Verify truth table enumeration matches hand-computed outputs."""

    def test_single_input_positive_weight(self):
        minterms = enumerate_truth_table(1, [1.0], -0.5)
        assert minterms == [1]

    def test_single_input_negative_weight(self):
        minterms = enumerate_truth_table(1, [-1.0], 0.5)
        assert minterms == [0]

    def test_and_gate(self):
        minterms = enumerate_truth_table(2, [1.0, 1.0], -1.5)
        assert minterms == [3]

    def test_or_gate(self):
        minterms = enumerate_truth_table(2, [1.0, 1.0], -0.5)
        assert sorted(minterms) == [1, 2, 3]

    def test_tautology(self):
        minterms = enumerate_truth_table(2, [0.0, 0.0], 1.0)
        assert sorted(minterms) == [0, 1, 2, 3]

    def test_contradiction(self):
        minterms = enumerate_truth_table(2, [0.5, 0.5], -10.0)
        assert minterms == []

    def test_three_input_majority(self):
        minterms = enumerate_truth_table(3, [1.0, 1.0, 1.0], -1.5)
        assert sorted(minterms) == [3, 5, 6, 7]

    def test_threshold_boundary(self):
        minterms = enumerate_truth_table(2, [1.0, -1.0], 0.0)
        assert minterms == [2]


class TestQMCMinimization:
    """Verify QMC produces correct minimal forms for known functions."""

    def test_and_function(self):
        qm = QuineMcCluskey()
        result = qm.simplify([3], num_bits=2)
        assert result == ["11"]

    def test_or_function(self):
        qm = QuineMcCluskey()
        result = qm.simplify([1, 2, 3], num_bits=2)
        assert sorted(result) == ["-1", "1-"]

    def test_single_variable(self):
        qm = QuineMcCluskey()
        result = qm.simplify([2, 3], num_bits=2)
        assert result == ["1-"]

    def test_negated_variable(self):
        qm = QuineMcCluskey()
        result = qm.simplify([0, 1], num_bits=2)
        assert result == ["0-"]

    def test_tautology(self):
        qm = QuineMcCluskey()
        result = qm.simplify([0, 1, 2, 3], num_bits=2)
        assert result == ["--"]

    def test_three_variable_majority(self):
        qm = QuineMcCluskey()
        result = qm.simplify([3, 5, 6, 7], num_bits=3)
        assert sorted(result) == ["-11", "1-1", "11-"]

    def test_xor_detection(self):
        qm = QuineMcCluskey(use_xor=True)
        result = qm.simplify([1, 2], num_bits=2)
        assert result is not None
        assert len(result) <= 2
        covered = set()
        for imp in result:
            covered.update(qm.permutations(imp))
        assert covered == {"01", "10"}

    def test_xnor_detection(self):
        qm = QuineMcCluskey(use_xor=True)
        result = qm.simplify([0, 3], num_bits=2)
        assert result is not None
        covered = set()
        for imp in result:
            covered.update(qm.permutations(imp))
        assert covered == {"00", "11"}

    def test_empty_input(self):
        qm = QuineMcCluskey()
        result = qm.simplify([], num_bits=2)
        assert result is None

    def test_dct_reduces_cost(self):
        """Don't-care terms reduce rule complexity."""
        qm = QuineMcCluskey()
        ones = [1, 3]
        dc = [4, 5, 6, 7]

        result_no_dc = qm.simplify(ones, dc=[], num_bits=3)
        result_with_dc = qm.simplify(ones, dc=dc, num_bits=3)

        assert dnf_cost(result_with_dc) <= dnf_cost(result_no_dc)

    def test_dct_allows_full_simplification(self):
        """When all non-ON patterns are don't-cares, result is tautology."""
        qm = QuineMcCluskey()
        result = qm.simplify([0, 1], dc=[2, 3], num_bits=2)
        assert result == ["--"]


class TestImplicantsToDNF:
    """Verify DNF string construction from implicant patterns."""

    def test_single_positive_literal(self):
        expr = implicants_to_dnf(["1-"])
        assert "x0" in expr
        assert "x1" not in expr or "~x1" not in expr

    def test_single_negative_literal(self):
        expr = implicants_to_dnf(["0-"])
        assert "~x0" in expr

    def test_and_term(self):
        expr = implicants_to_dnf(["11"])
        assert "x0" in expr and "x1" in expr
        assert "&" in expr

    def test_or_of_terms(self):
        expr = implicants_to_dnf(["-1", "1-"])
        assert "|" in expr

    def test_tautology(self):
        expr = implicants_to_dnf(["--"])
        assert expr == "True"

    def test_xor_implicant(self):
        expr = implicants_to_dnf(["^^"])
        assert "^" in expr

    def test_xnor_implicant(self):
        expr = implicants_to_dnf(["~~"])
        assert "~(" in expr and "^" in expr

    def test_empty(self):
        expr = implicants_to_dnf(None)
        assert expr == "False"
        expr = implicants_to_dnf([])
        assert expr == "False"
