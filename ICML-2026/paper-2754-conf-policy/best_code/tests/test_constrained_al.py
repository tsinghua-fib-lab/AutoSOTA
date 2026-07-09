import pytest

from calibrate_mfcs import sort_both_by_first, weighted_quantile


class TestSortBothByFirst:
    def test_basic_sorting(self):
        v = [3, 1, 2]
        w = ["c", "a", "b"]
        v_sorted, w_sorted = sort_both_by_first(v, w)
        assert v_sorted == [1, 2, 3]
        assert w_sorted == ["a", "b", "c"]

    def test_already_sorted(self):
        v = [1, 2, 3]
        w = [10, 20, 30]
        v_sorted, w_sorted = sort_both_by_first(v, w)
        assert v_sorted == [1, 2, 3]
        assert w_sorted == [10, 20, 30]

    def test_single_element(self):
        v_sorted, w_sorted = sort_both_by_first([5], [99])
        assert v_sorted == [5]
        assert w_sorted == [99]


class TestWeightedQuantile:
    def test_median_uniform_weights(self):
        # q=0.5 hits the q < 0.5 branch boundary — implementation returns
        # average of adjacent values when q == 0.5
        v = [1, 2, 3, 4, 5]
        w = [0.2, 0.2, 0.2, 0.2, 0.2]
        result = weighted_quantile(v, w, 0.5)
        assert result == 2.5

    def test_all_weight_on_one_value(self):
        v = [10, 20, 30]
        w = [0.0, 1.0, 0.0]
        # q=0.5, cumsum = [0.0, 1.0, 1.0], first >= 0.5 is index 1
        result = weighted_quantile(v, w, 0.5)
        assert result == 20

    def test_upper_quantile(self):
        v = [1, 2, 3, 4]
        w = [0.25, 0.25, 0.25, 0.25]
        result = weighted_quantile(v, w, 0.9)
        assert result == 4

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError, match="Invalid q"):
            weighted_quantile([1, 2], [0.5, 0.5], 1.5)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            weighted_quantile([1, 2, 3], [0.5, 0.5], 0.5)

    def test_unnormalized_weights_raises(self):
        with pytest.raises(ValueError, match="does not add to 1"):
            weighted_quantile([1, 2], [0.3, 0.3], 0.5)
