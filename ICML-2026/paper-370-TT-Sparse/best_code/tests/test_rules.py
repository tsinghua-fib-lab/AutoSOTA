"""Rule evaluation tests.

Validates that:
1. _strip_outer_parens handles nested, balanced, and unbalanced cases correctly.
2. _eval_rule evaluates AND, OR, XOR, XNOR, negation expressions.
3. _split_or / _split_and respect parenthesis nesting.
4. Comparison operators evaluate correctly against DataFrames.
5. Negation handling (both _get_negated_form and simplify_negations).
6. Don't-care identification from training data.
"""

import numpy as np
import pandas as pd
import pytest

from tt_sparse.qmc import simplify_negations
from tt_sparse.rules import (
    _dont_cares,
    _eval_comparison,
    _eval_rule,
    _eval_xor_term,
    _get_negated_form,
    _resolve_literal,
    _split_and,
    _split_or,
    _strip_outer_parens,
)


class TestStripOuterParens:
    """Verify balanced parenthesis stripping."""

    def test_no_parens(self):
        assert _strip_outer_parens("x >= 1") == "x >= 1"

    def test_single_outer_pair(self):
        assert _strip_outer_parens("(x >= 1)") == "x >= 1"

    def test_double_outer_pair(self):
        assert _strip_outer_parens("((x >= 1))") == "x >= 1"

    def test_non_wrapping_parens(self):
        expr = "(a >= 1) & (b < 2)"
        assert _strip_outer_parens(expr) == expr

    def test_nested_non_wrapping(self):
        expr = "(a) | (b)"
        assert _strip_outer_parens(expr) == expr

    def test_complex_nested(self):
        expr = "((a >= 1) & (b < 2))"
        assert _strip_outer_parens(expr) == "(a >= 1) & (b < 2)"

    def test_deeply_nested(self):
        assert _strip_outer_parens("(((x)))") == "x"

    def test_empty_string(self):
        assert _strip_outer_parens("") == ""

    def test_whitespace(self):
        assert _strip_outer_parens("  (x)  ") == "x"

    def test_regression_bug(self):
        """The original bug: strip('()') would break this expression."""
        expr = "((s1 < -0.0428) & (s3 >= 0.0192) & (s5 < 0.0560))"
        result = _strip_outer_parens(expr)
        assert result == "(s1 < -0.0428) & (s3 >= 0.0192) & (s5 < 0.0560)"
        parts = _split_and(result)
        assert len(parts) == 3


class TestSplitOperators:
    """Verify parenthesis-respecting splitting."""

    def test_split_or_simple(self):
        assert _split_or("a | b | c") == ["a", "b", "c"]

    def test_split_or_with_parens(self):
        assert _split_or("(a & b) | (c & d)") == ["(a & b)", "(c & d)"]

    def test_split_or_nested_parens(self):
        assert _split_or("(a | b) | c") == ["(a | b)", "c"]

    def test_split_and_simple(self):
        assert _split_and("a & b & c") == ["a", "b", "c"]

    def test_split_and_with_parens(self):
        assert _split_and("(a | b) & (c | d)") == ["(a | b)", "(c | d)"]

    def test_split_and_nested_or(self):
        assert _split_and("(x0 | x1) & x2") == ["(x0 | x1)", "x2"]

    def test_no_operator(self):
        assert _split_or("single_term") == ["single_term"]
        assert _split_and("single_term") == ["single_term"]

    def test_xor_not_split_by_and(self):
        assert _split_and("(a ^ b) & c") == ["(a ^ b)", "c"]


class TestEvalRule:
    """Verify Boolean expression evaluation against known inputs."""

    @pytest.fixture
    def bits_and_df(self):
        df = pd.DataFrame({
            "a": [0, 0, 1, 1],
            "b": [0, 1, 0, 1],
        })
        ltt_bits = {
            "a >= 0.5": np.array([False, False, True, True]),
            "b >= 0.5": np.array([False, True, False, True]),
        }
        return ltt_bits, df, 4

    def test_simple_literal(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("(a >= 0.5)", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [False, False, True, True])

    def test_and_expression(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("((a >= 0.5) & (b >= 0.5))", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [False, False, False, True])

    def test_or_expression(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("((a >= 0.5)) | ((b >= 0.5))", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [False, True, True, True])

    def test_negation(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("((a < 0.5))", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [True, True, False, False])

    def test_xor_expression(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("((a >= 0.5) ^ (b >= 0.5))", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_xnor_expression(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("~((a >= 0.5) ^ (b >= 0.5))", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [True, False, False, True])

    def test_complex_dnf(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        expr = "((a >= 0.5) & (b >= 0.5)) | ((a < 0.5) & (b < 0.5))"
        result = _eval_rule(expr, ltt_bits, df, n)
        np.testing.assert_array_equal(result, [True, False, False, True])

    def test_true_constant(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("True", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [True, True, True, True])

    def test_false_constant(self, bits_and_df):
        ltt_bits, df, n = bits_and_df
        result = _eval_rule("False", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [False, False, False, False])

    def test_and_with_xor(self):
        n = 4
        ltt_bits = {
            "x": np.array([False, False, True, True]),
            "y": np.array([False, True, False, True]),
            "z": np.array([True, True, True, False]),
        }
        df = pd.DataFrame()
        result = _eval_rule("((x ^ y) & z)", ltt_bits, df, n)
        np.testing.assert_array_equal(result, [False, True, True, False])


class TestResolveLiteral:
    """Verify literal resolution including negation lookup."""

    def test_direct_match(self):
        ltt_bits = {"age >= 30": np.array([True, False, True])}
        df = pd.DataFrame({"age": [40, 20, 50]})
        result = _resolve_literal("age >= 30", ltt_bits, df)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_negated_match(self):
        ltt_bits = {"age >= 30": np.array([True, False, True])}
        df = pd.DataFrame({"age": [40, 20, 50]})
        result = _resolve_literal("age < 30", ltt_bits, df)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_fallback_to_dataframe(self):
        ltt_bits = {}
        df = pd.DataFrame({"score": [0.3, 0.7, 0.5]})
        result = _resolve_literal("score >= 0.5", ltt_bits, df)
        np.testing.assert_array_equal(result, [False, True, True])

    def test_missing_column(self):
        ltt_bits = {}
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = _resolve_literal("missing >= 0.5", ltt_bits, df)
        np.testing.assert_array_equal(result, [False, False, False])


class TestEvalComparison:
    """Verify comparison operators against DataFrames."""

    @pytest.mark.parametrize("expr,expected", [
        ("x >= 3", [False, False, True, True, True]),
        ("x > 3", [False, False, False, True, True]),
        ("x <= 3", [True, True, True, False, False]),
        ("x < 3", [True, True, False, False, False]),
        ("x == 3", [False, False, True, False, False]),
        ("x != 3", [True, True, False, True, True]),
    ], ids=["ge", "gt", "le", "lt", "eq", "ne"])
    def test_numeric_operators(self, expr, expected):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = _eval_comparison(expr, df)
        np.testing.assert_array_equal(result, expected)

    def test_categorical_eq(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        result = _eval_comparison("color == 'red'", df)
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_negative_float(self):
        df = pd.DataFrame({"val": [-2.0, -0.5, 0.0, 0.5, 2.0]})
        result = _eval_comparison("val >= -0.5", df)
        np.testing.assert_array_equal(result, [False, True, True, True, True])


class TestNegationHandling:
    """Verify operator flipping for both _get_negated_form and simplify_negations."""

    @pytest.mark.parametrize("input_expr,expected", [
        ("x >= 3", "x < 3"),
        ("x < 3", "x >= 3"),
        ("x == 1", "x != 1"),
        ("score >= -0.5432", "score < -0.5432"),
    ], ids=["ge_to_lt", "lt_to_ge", "eq_to_ne", "preserves_value"])
    def test_get_negated_form(self, input_expr, expected):
        assert _get_negated_form(input_expr) == expected

    def test_no_operator_returns_none(self):
        assert _get_negated_form("plain_literal") is None

    @pytest.mark.parametrize("input_expr,expected_substr", [
        ("~(age >= 30)", "(age < 30)"),
        ("~(score < 0.5)", "(score >= 0.5)"),
        ("~(x == 1)", "(x != 1)"),
    ], ids=["ge_to_lt", "lt_to_ge", "eq_to_ne"])
    def test_simplify_negations(self, input_expr, expected_substr):
        result = simplify_negations(input_expr)
        assert result == expected_substr

    def test_simplify_negations_nested(self):
        result = simplify_negations("~(a >= 1) & ~(b < 2)")
        assert "(a < 1)" in result
        assert "(b >= 2)" in result

    def test_simplify_negations_no_change(self):
        expr = "(age >= 30) & (score < 0.5)"
        assert simplify_negations(expr) == expr


class TestDontCares:
    """Verify don't-care identification from training data."""

    def test_identifies_unobserved_patterns(self):
        X_ltt = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
            [0, 1, 1],
        ], dtype=np.float32)

        dc = _dont_cares(X_ltt, [0, 1, 2], 3, min_obs=1)
        assert sorted(dc) == [4, 5, 6, 7]

    def test_min_observations_threshold(self):
        X_ltt = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 1],
        ], dtype=np.float32)

        dc = _dont_cares(X_ltt, [0, 1], 2, min_obs=2)
        assert 1 in dc
