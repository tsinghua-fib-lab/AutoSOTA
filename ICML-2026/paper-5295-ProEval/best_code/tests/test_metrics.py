# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for proeval.utils.metrics."""
import numpy as np
import pytest
from proeval.utils.metrics import (
    compute_samples_to_threshold,
    embedding_coverage,
    failure_rate,
    overall_diversity,
    topic_entropy,
)
def test_topic_entropy_empty():
    assert topic_entropy([]) == 0.0

@pytest.mark.parametrize("topics", [["a"], ["a", "a", "a"], ["x", "x"]])
def test_topic_entropy_single_topic(topics):
    assert topic_entropy(topics, normalize=True) == 0.0
    assert topic_entropy(topics, normalize=False) == 0.0

def test_topic_entropy_uniform():
    result = topic_entropy(["a", "b", "a", "b"])
    assert result == pytest.approx(100.0, abs=1e-6)

def test_topic_entropy_non_negative():
    for topics in ([], ["a"], ["a", "a"], ["a", "b", "c"], ["a", "a", "b"]):
        assert topic_entropy(topics, normalize=False) >= 0.0

def test_topic_entropy_skewed():
    result = topic_entropy(["a", "a", "a", "b"])
    assert 0.0 < result < 100.0

def test_embedding_coverage_too_few_rows():
    assert embedding_coverage(np.array([[1.0, 2.0]])) == 0.0
    assert embedding_coverage(np.zeros((0, 2))) == 0.0

def test_embedding_coverage_non_2d():
    assert embedding_coverage(np.array([1.0, 2.0, 3.0])) == 0.0

def test_embedding_coverage_diverse_vs_collinear():
    # normalize_to_01=True can yield out-of-bounds values, so we test 
    # the relative ordering with False to ensure diverse > collinear.
    rng = np.random.default_rng(0)
    diverse = rng.standard_normal((20, 8))
    collinear = np.ones((20, 8))
    diverse_score = embedding_coverage(diverse, normalize_to_01=False)
    collinear_score = embedding_coverage(collinear, normalize_to_01=False)
    assert diverse_score > collinear_score

def test_embedding_coverage_finite_raw_logdet():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((10, 4))
    assert np.isfinite(embedding_coverage(emb, normalize_to_01=False))


@pytest.mark.parametrize(
    "entropy,coverage,w_entropy,w_coverage,expected",
    [
        (100.0, 1.0, 0.5, 0.5, 100.0),
        (0.0, 0.0, 0.5, 0.5, 0.0),
        (50.0, 0.5, 0.5, 0.5, 50.0),
        (100.0, 0.0, 1.0, 0.0, 100.0),
        (0.0, 1.0, 0.0, 1.0, 100.0),
    ],
)
def test_overall_diversity_weights(entropy, coverage, w_entropy, w_coverage, expected):
    result = overall_diversity(entropy, coverage, w_entropy, w_coverage)
    assert result == pytest.approx(expected)

# tests for failure_rate
def test_failure_rate_empty():
    assert failure_rate([]) == 0.0

@pytest.mark.parametrize(
    "scores,expected",
    [
        ([1.0, 0.0, 1.0, 0.0, 0.0], 40.0),
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0], 100.0),
        ([1.0, 0.0, 0.0, 0.0], 25.0),
    ],
)
def test_failure_rate_calc(scores, expected):
    assert failure_rate(scores) == pytest.approx(expected)


def test_compute_samples_skips_empty_estimates():
    results = {
        "empty_method": {"estimates": []},
        "good": {"estimates": [[1.0, 0.0]]},
    }
    means, stds = compute_samples_to_threshold(results, true_mean=0.0, thresholds=[0.5])
    assert "empty_method" not in means
    assert "good" in means

def test_compute_samples_threshold_crossing():
    results = {"method": {"estimates": [[0.5, 0.3, 0.1, 0.02, 0.005]]}}
    means, stds = compute_samples_to_threshold(results, true_mean=0.0, thresholds=[0.05])
    assert means["method"][0.05] == pytest.approx(4.0)
    assert stds["method"][0.05] == 0.0

def test_compute_samples_never_reached():
    results = {"method": {"estimates": [[0.5, 0.4, 0.3]]}}
    means, _ = compute_samples_to_threshold(results, true_mean=0.0, thresholds=[1e-9])
    assert means["method"][1e-9] == pytest.approx(3.0)

def test_compute_samples_multiple_runs():
    results = {
        "method": {
            "estimates": [
                [0.5, 0.04, 0.01, 0.001],
                [0.5, 0.4, 0.3, 0.02],
            ]
        }
    }
    means, stds = compute_samples_to_threshold(results, true_mean=0.0, thresholds=[0.05])
    assert means["method"][0.05] == pytest.approx(3.0)
    assert stds["method"][0.05] > 0.0