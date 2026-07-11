import json

import pytest

from taming_the_ito_lyon.training.io import build_training_metrics_payload
from taming_the_ito_lyon.training.io import write_test_metrics
from taming_the_ito_lyon.training.results_gathering_fns import ResultsDict


def test_build_training_metrics_payload_separates_loss_and_metric() -> None:
    payload = build_training_metrics_payload(
        run_dirname="demo_run",
        model_name="bnrde",
        num_params=123,
        final_epoch=2,
        best_epoch=1,
        training_elapsed=12.5,
        time_to_best_epoch=8.0,
        inference_elapsed=0.75,
        loss_label="sigker_branched",
        eval_metric_name="median_eigenvalue_w1",
        train_loss_history=[12.0, 9.0, 10.0],
        val_loss_history=[11.0, 8.5, 8.75],
        val_metric_history=[0.4, 0.2, 0.3],
        test_loss=8.25,
        test_eval_metric=0.15,
        test_results_dict=ResultsDict(
            eval_metric=0.15,
            results_times=[128.0, 256.0],
            results=[0.1, 0.2],
        ),
        xla_scratch_size_mib=12.5,
    )

    assert payload["timings"]["time_to_best_epoch_s"] == 8.0
    assert payload["memory"]["xla_scratch_size_mib"] == 12.5
    assert "integration" not in payload
    assert payload["train"]["loss"]["history"] == [12.0, 9.0, 10.0]
    assert payload["validation"]["loss"]["history"] == [11.0, 8.5, 8.75]
    assert payload["validation"]["metric"]["name"] == "median_eigenvalue_w1"
    assert payload["validation"]["metric"]["best"] == 0.2
    assert payload["validation"]["metric"]["best_epoch"] == 1
    assert payload["test"]["loss"]["value"] == 8.25
    assert payload["test"]["metric"]["value"] == 0.15
    assert payload["test"]["results_dict"]["results"] == [0.1, 0.2]
    assert "sigker_branched" not in payload
    assert "test_results_dict" not in payload


def test_write_test_metrics_uses_nested_schema(tmp_path) -> None:
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    metrics_path = write_test_metrics(
        run_dir=str(run_dir),
        model_name="bnrde",
        num_params=123,
        inference_elapsed=0.75,
        loss_label="sigker_branched",
        eval_metric_name="median_eigenvalue_w1",
        test_loss=0.2,
        test_eval_metric=0.15,
        test_results_dict=ResultsDict(
            eval_metric=0.15,
            results_times=[128.0],
            results=[0.1],
        ),
        checkpoint_path="saved_models/demo_run/best.eqx",
        xla_scratch_size_mib=2.5,
    )

    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["test"]["loss"]["name"] == "sigker_branched"
    assert payload["memory"]["xla_scratch_size_mib"] == 2.5
    assert "integration" not in payload
    assert payload["test"]["metric"]["name"] == "median_eigenvalue_w1"
    assert payload["test"]["results_dict"]["results"] == [0.1]
    assert "sigker_branched" not in payload
    assert "test_results_dict" not in payload


def test_write_test_metrics_multi_seed_writes_mean_and_sample_std(tmp_path) -> None:
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    write_test_metrics(
        run_dir=str(run_dir),
        model_name="bnrde",
        num_params=123,
        inference_elapsed=0.75,
        loss_label="sigker_branched",
        eval_metric_name="median_eigenvalue_w1",
        test_loss=1.1,
        test_eval_metric=1.0,
        test_results_dict=ResultsDict(
            eval_metric=1.0,
            results_times=[128.0, 256.0],
            results=[0.1, 0.2],
        ),
        checkpoint_path="saved_models/demo_run/best.eqx",
        metrics_seed=1,
    )
    metrics_path = write_test_metrics(
        run_dir=str(run_dir),
        model_name="bnrde",
        num_params=123,
        inference_elapsed=0.75,
        loss_label="sigker_branched",
        eval_metric_name="median_eigenvalue_w1",
        test_loss=3.1,
        test_eval_metric=3.0,
        test_results_dict=ResultsDict(
            eval_metric=3.0,
            results_times=[128.0, 256.0],
            results=[0.3, 0.4],
        ),
        checkpoint_path="saved_models/demo_run/best.eqx",
        metrics_seed=2,
    )

    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["seeds"] == [1, 2]
    aggregate = payload["aggregate"]
    assert aggregate["std_ddof"] == 1
    assert aggregate["num_seeds"] == 2

    metric_agg = aggregate["test"]["metric"]
    assert metric_agg["name"] == "median_eigenvalue_w1"
    assert metric_agg["mean"] == pytest.approx(2.0)
    assert metric_agg["std_1sigma"] == pytest.approx(1.4142135623730951)
    assert metric_agg["count"] == 2

    results_agg = aggregate["test"]["results_dict"]
    assert results_agg["results_times"] == [128.0, 256.0]
    assert results_agg["results_mean"] == pytest.approx([0.2, 0.3])
    assert results_agg["results_std_1sigma"] == pytest.approx(
        [0.14142135623730953, 0.14142135623730953]
    )
    assert results_agg["counts"] == [2, 2]
