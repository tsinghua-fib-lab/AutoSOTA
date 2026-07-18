from __future__ import annotations

import fast_tsp
import pytest


def test_errors_with_wrong_signature():
    with pytest.raises(ValueError, match='Distance matrix must be 2D'):
        fast_tsp.find_tour([0])  # type: ignore


def test_errors_with_too_small_dist():
    with pytest.raises(ValueError, match='Need at least 2 nodes'):
        fast_tsp.find_tour([[0]])


def test_errors_with_non_square_matrix():
    exp_msg = r'Distance matrix must be square, but got \(3, 2\)'
    with pytest.raises(ValueError, match=exp_msg):
        fast_tsp.find_tour([[0, 1], [2, 0], [3, 0]])


def test_errors_with_floats():
    exp_msg = 'Distance matrix must contain integers, but found'
    with pytest.raises(ValueError, match=exp_msg):
        fast_tsp.find_tour([[0.0, 1.1], [1.1, 0.0]])  # type: ignore


def test_errors_with_negative_value():
    exp_msg = 'All distances must be non-negative, but found -5'
    with pytest.raises(ValueError, match=exp_msg):
        fast_tsp.find_tour([[0, -5], [-5, 0]])


def test_errors_with_too_large_value():
    too_large_val = fast_tsp._UINT16_MAX + 1
    err_msg = fast_tsp.is_valid_dist_matrix([[0, too_large_val], [too_large_val, 0]])
    assert err_msg == f'All distances must be <= {fast_tsp._UINT16_MAX:,}, but found {too_large_val:,}'


def test_errors_with_negative_duration():
    with pytest.raises(ValueError, match='Duration must be strictly positive'):
        fast_tsp.find_tour([[0, 1], [1, 0]], duration_seconds=-1)


def test_basic_tour():
    tour = fast_tsp.find_tour([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert tour == [0, 1, 2]
