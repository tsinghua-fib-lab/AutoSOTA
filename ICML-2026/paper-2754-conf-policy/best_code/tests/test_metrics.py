"""Tests for the experiment metrics data models."""

from __future__ import annotations

import json
import math

import pandas as pd

from cpc_llm.metrics import (
    ARSamplingMetrics,
    ARSamplingResult,
    CPCSearchMetrics,
    CPCSearchResult,
    DatasetSizes,
    RoundSummary,
    SampleQualityMetrics,
    StageTiming,
    compute_sample_quality,
)


# ---------------------------------------------------------------------------
# SampleQualityMetrics
# ---------------------------------------------------------------------------


class TestSampleQualityMetrics:
    def test_basic(self) -> None:
        m = SampleQualityMetrics(
            n_samples=10,
            n_parsable=9,
            n_feasible=8,
            frac_feasible=0.8,
            mean_score=1.5,
            min_score=0.1,
            max_score=3.0,
        )
        assert m.n_samples == 10
        assert m.frac_feasible == 0.8

    def test_json_roundtrip(self) -> None:
        m = SampleQualityMetrics(
            n_samples=5,
            n_parsable=5,
            n_feasible=4,
            frac_feasible=0.8,
            mean_score=2.0,
            min_score=1.0,
            max_score=3.0,
        )
        j = m.model_dump_json()
        m2 = SampleQualityMetrics.model_validate_json(j)
        assert m == m2


# ---------------------------------------------------------------------------
# compute_sample_quality helper
# ---------------------------------------------------------------------------


class TestComputeSampleQuality:
    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["score"])
        q = compute_sample_quality(df)
        assert q.n_samples == 0
        assert q.n_feasible == 0
        assert math.isnan(q.mean_score)

    def test_all_finite(self) -> None:
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        q = compute_sample_quality(df)
        assert q.n_samples == 3
        assert q.n_parsable == 3
        assert q.n_feasible == 3
        assert q.frac_feasible == 1.0
        assert q.mean_score == 2.0
        assert q.min_score == 1.0
        assert q.max_score == 3.0

    def test_with_nan_and_inf(self) -> None:
        df = pd.DataFrame({"score": [1.0, float("nan"), float("inf"), 3.0]})
        q = compute_sample_quality(df)
        assert q.n_samples == 4
        assert q.n_parsable == 3  # nan excluded
        assert q.n_feasible == 2  # inf excluded too
        assert q.frac_feasible == 0.5
        assert q.mean_score == 2.0  # mean of [1.0, 3.0]


# ---------------------------------------------------------------------------
# ARSamplingMetrics
# ---------------------------------------------------------------------------


class TestARSamplingMetrics:
    def _make(self, **overrides: object) -> ARSamplingMetrics:
        defaults = dict(
            n_accepted=50,
            n_calls=10,
            n_proposals_total=200,
            acceptance_rate=0.25,
            proposal_type="unconstrained",
            ar_to_imh_switch=False,
            imh_switch_call_idx=None,
            safe_prop_mix_weight=None,
            env_const=1.5,
            accepted_quality=SampleQualityMetrics(
                n_samples=50,
                n_parsable=50,
                n_feasible=45,
                frac_feasible=0.9,
                mean_score=2.0,
                min_score=0.5,
                max_score=4.0,
            ),
            rejected_quality=SampleQualityMetrics(
                n_samples=150,
                n_parsable=140,
                n_feasible=100,
                frac_feasible=0.667,
                mean_score=1.2,
                min_score=0.1,
                max_score=3.0,
            ),
        )
        defaults.update(overrides)
        return ARSamplingMetrics(**defaults)

    def test_json_roundtrip(self) -> None:
        m = self._make()
        j = m.model_dump_json()
        m2 = ARSamplingMetrics.model_validate_json(j)
        assert m == m2

    def test_with_imh_switch(self) -> None:
        m = self._make(ar_to_imh_switch=True, imh_switch_call_idx=5)
        assert m.ar_to_imh_switch is True
        assert m.imh_switch_call_idx == 5

    def test_mixture_proposal(self) -> None:
        m = self._make(proposal_type="mixture", safe_prop_mix_weight=0.6)
        assert m.safe_prop_mix_weight == 0.6


# ---------------------------------------------------------------------------
# CPCSearchMetrics
# ---------------------------------------------------------------------------


class TestCPCSearchMetrics:
    def _make(self, **overrides: object) -> CPCSearchMetrics:
        defaults = dict(
            beta_t=1.5,
            psi_hat_t=0.9,
            grid_size=100,
            grid_position_selected=42,
            risk_margin=0.05,
            w_test=0.1,
            proposal_selected="unconstrained",
            switch_to_mixture=False,
            switch_to_optimized=False,
            psi_hat_intersection_safe=0.3,
            psi_hat_intersection_unconstrained=0.7,
            envelope_const=2.0,
        )
        defaults.update(overrides)
        return CPCSearchMetrics(**defaults)

    def test_json_roundtrip(self) -> None:
        m = self._make()
        j = m.model_dump_json()
        m2 = CPCSearchMetrics.model_validate_json(j)
        assert m == m2

    def test_inf_beta(self) -> None:
        """Beta can be inf when running unconstrained."""
        m = self._make(beta_t=float("inf"))
        j = m.model_dump_json()
        m2 = CPCSearchMetrics.model_validate_json(j)
        assert m2.beta_t == float("inf")

    def test_nan_risk_margin(self) -> None:
        """Risk margin is nan when risk is not controlled."""
        m = self._make(risk_margin=float("nan"))
        j = m.model_dump_json()
        m2 = CPCSearchMetrics.model_validate_json(j)
        assert math.isnan(m2.risk_margin)


# ---------------------------------------------------------------------------
# RoundSummary
# ---------------------------------------------------------------------------


class TestRoundSummary:
    def _make_cpc(self) -> CPCSearchMetrics:
        return CPCSearchMetrics(
            beta_t=1.5,
            psi_hat_t=0.9,
            grid_size=100,
            grid_position_selected=42,
            risk_margin=0.05,
            w_test=0.1,
            proposal_selected="unconstrained",
            switch_to_mixture=False,
            switch_to_optimized=False,
            psi_hat_intersection_safe=0.3,
            psi_hat_intersection_unconstrained=0.7,
            envelope_const=2.0,
        )

    def _make_ar(self) -> ARSamplingMetrics:
        return ARSamplingMetrics(
            n_accepted=50,
            n_calls=10,
            n_proposals_total=200,
            acceptance_rate=0.25,
            proposal_type="unconstrained",
            ar_to_imh_switch=False,
            imh_switch_call_idx=None,
            safe_prop_mix_weight=None,
            env_const=1.5,
            accepted_quality=SampleQualityMetrics(
                n_samples=50,
                n_parsable=50,
                n_feasible=45,
                frac_feasible=0.9,
                mean_score=2.0,
                min_score=0.5,
                max_score=4.0,
            ),
            rejected_quality=SampleQualityMetrics(
                n_samples=150,
                n_parsable=140,
                n_feasible=100,
                frac_feasible=0.667,
                mean_score=1.2,
                min_score=0.1,
                max_score=3.0,
            ),
        )

    def test_full_assembly(self) -> None:
        summary = RoundSummary(
            round_idx=3,
            round_type="dpo",
            cpc=self._make_cpc(),
            ar_sampling=self._make_ar(),
            dataset_sizes=DatasetSizes(n_cal=100, n_train=400, n_combined=500),
            temperatures=[0.8, 1.0, 1.2],
            timing=StageTiming(
                training_s=120.0,
                seed_selection_s=5.0,
                likelihood_computation_s=30.0,
                cpc_search_s=10.0,
                ar_sampling_s=45.0,
                dataset_split_s=2.0,
                total_round_s=212.0,
            ),
        )
        j = summary.model_dump_json(indent=2)
        parsed = json.loads(j)
        assert parsed["round_idx"] == 3
        assert parsed["round_type"] == "dpo"
        assert parsed["cpc"]["beta_t"] == 1.5
        assert parsed["ar_sampling"]["n_accepted"] == 50

    def test_sft_no_ar_sampling(self) -> None:
        """SFT rounds have no AR sampling."""
        summary = RoundSummary(
            round_idx=1,
            round_type="sft",
            cpc=self._make_cpc(),
            ar_sampling=None,
            dataset_sizes=DatasetSizes(n_cal=50, n_train=200, n_combined=250),
            temperatures=[1.0],
            timing=StageTiming(
                training_s=60.0,
                seed_selection_s=3.0,
                likelihood_computation_s=15.0,
                cpc_search_s=5.0,
                ar_sampling_s=0.0,
                dataset_split_s=1.0,
                total_round_s=84.0,
            ),
        )
        j = summary.model_dump_json()
        parsed = json.loads(j)
        assert parsed["ar_sampling"] is None

    def test_roundtrip(self) -> None:
        summary = RoundSummary(
            round_idx=1,
            round_type="marge",
            cpc=self._make_cpc(),
            ar_sampling=self._make_ar(),
            dataset_sizes=DatasetSizes(n_cal=80, n_train=320, n_combined=400),
            temperatures=[1.0],
            timing=StageTiming(
                training_s=90.0,
                seed_selection_s=4.0,
                likelihood_computation_s=20.0,
                cpc_search_s=8.0,
                ar_sampling_s=35.0,
                dataset_split_s=1.5,
                total_round_s=158.5,
            ),
        )
        j = summary.model_dump_json()
        s2 = RoundSummary.model_validate_json(j)
        assert s2 == summary


# ---------------------------------------------------------------------------
# Dataclass return types
# ---------------------------------------------------------------------------


class TestCPCSearchResult:
    def test_construction(self) -> None:
        metrics = CPCSearchMetrics(
            beta_t=1.0,
            psi_hat_t=0.5,
            grid_size=50,
            grid_position_selected=20,
            risk_margin=0.03,
            w_test=0.1,
            proposal_selected="safe",
            switch_to_mixture=False,
            switch_to_optimized=False,
            psi_hat_intersection_safe=0.4,
            psi_hat_intersection_unconstrained=0.6,
            envelope_const=1.5,
        )
        result = CPCSearchResult(
            beta_t=1.0,
            psi_hat_t=0.5,
            constrained_liks_df=pd.DataFrame(),
            constrained_liks_fp="/tmp/c.jsonl",
            unconstrained_df=pd.DataFrame(),
            unconstrained_liks_fp="/tmp/u.jsonl",
            proposal="safe",
            psi_hat_intersection_safe=0.4,
            psi_hat_intersection_unconstrained=0.6,
            envelope_const=1.5,
            search_metrics=metrics,
        )
        assert result.beta_t == 1.0
        assert result.search_metrics.grid_size == 50


class TestARSamplingResult:
    def test_none_result(self) -> None:
        result = ARSamplingResult(None, None, None, None, None, None)
        assert result.unconstrained_df is None
        assert result.sampling_metrics is None

    def test_with_data(self) -> None:
        df = pd.DataFrame({"particle": ["AAA"], "score": [1.0]})
        result = ARSamplingResult(
            unconstrained_df=df,
            unconstrained_fp="/tmp/u.jsonl",
            constrained_df=df,
            constrained_fp="/tmp/c.jsonl",
            all_proposals_fp="/tmp/all.jsonl",
            sampling_metrics=None,
        )
        assert len(result.unconstrained_df) == 1
        assert result.all_proposals_fp == "/tmp/all.jsonl"
