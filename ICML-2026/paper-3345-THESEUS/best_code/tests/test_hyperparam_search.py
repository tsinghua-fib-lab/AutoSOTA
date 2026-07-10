from __future__ import annotations

from merge_and_rebase.hyperparam_search import (
    SearchEvaluation,
    build_search_planner,
    infer_sobol_refinement_steps,
)


def test_build_search_planner_uses_legacy_alpha_sweep() -> None:
    planner = build_search_planner(
        cfg={
            "alpha_search": True,
            "alpha_min": 0.0,
            "alpha_max": 0.2,
            "alpha_step": 0.1,
        },
        base_method_params={},
    )

    batch = planner.next_batch()
    assert batch is not None
    assert [round(item.alpha, 3) for item in batch] == [0.0, 0.1, 0.2]
    assert planner.next_batch() is None


def test_sequential_search_batches_method_params_then_alpha() -> None:
    planner = build_search_planner(
        cfg={
            "alpha_search": True,
            "alpha_min": 0.0,
            "alpha_max": 0.5,
            "alpha_step": 0.5,
            "hyperparam_search": {
                "strategy": "sequential",
                "method_params": {
                    "topk": [0.25, 0.5],
                    "merging_type": ["mean", "sum"],
                },
            },
        },
        base_method_params={"keep_ratio": 1.0},
    )

    batch0 = planner.next_batch()
    batch1 = planner.next_batch()
    batch2 = planner.next_batch()
    batch3 = planner.next_batch()

    assert batch0 is not None
    assert [(item.method_params["topk"], item.method_params["merging_type"], item.alpha) for item in batch0] == [
        (0.25, "mean", 0.0),
        (0.25, "mean", 0.5),
    ]
    assert batch1 is not None
    assert [(item.method_params["topk"], item.method_params["merging_type"], item.alpha) for item in batch1] == [
        (0.25, "sum", 0.0),
        (0.25, "sum", 0.5),
    ]
    assert batch2 is not None
    assert [(item.method_params["topk"], item.method_params["merging_type"], item.alpha) for item in batch2] == [
        (0.5, "mean", 0.0),
        (0.5, "mean", 0.5),
    ]
    assert batch3 is not None
    assert [(item.method_params["topk"], item.method_params["merging_type"], item.alpha) for item in batch3] == [
        (0.5, "sum", 0.0),
        (0.5, "sum", 0.5),
    ]
    assert planner.next_batch() is None


def test_sobol_search_refines_around_best_candidate() -> None:
    planner = build_search_planner(
        cfg={
            "seed": 7,
            "hyperparam_search": {
                "strategy": "sobol",
                "num_samples": 8,
                "refinement_steps": 1,
                "refine_factor": 0.5,
                "alpha": {"min": 0.0, "max": 1.0, "step": 0.25},
                "method_params": {
                    "topk": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
            },
        },
        base_method_params={},
    )

    stage0 = planner.next_batch()
    assert stage0 is not None
    assert stage0
    for item in stage0:
        assert 0.0 <= item.alpha <= 1.0
        assert item.method_params["topk"] in {0.1, 0.3, 0.5, 0.7, 0.9}

    best = stage0[0]
    planner.observe(
        [
            SearchEvaluation(
                candidate=item,
                score=(10.0 if item.candidate_index == best.candidate_index else 0.0),
                avg_acc=0.0,
                avg_norm_acc=0.0,
                per_task_acc=[],
                per_task_norm_acc=[],
            )
            for item in stage0
        ]
    )

    stage1 = planner.next_batch()
    assert stage1 is not None
    assert stage1
    stage1_topk_values = {item.method_params["topk"] for item in stage1}
    ordered = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_idx = ordered.index(best.method_params["topk"])
    start = max(0, min(best_idx - 1, len(ordered) - 3))
    expected = set(ordered[start : start + 3])
    assert stage1_topk_values.issubset(expected)
    assert planner.search_summary()["strategy"] == "sobol"


def test_infer_sobol_refinement_steps_scales_with_coarseness() -> None:
    assert infer_sobol_refinement_steps(num_samples=4, num_dims=2) == 2
    assert infer_sobol_refinement_steps(num_samples=32, num_dims=2) == 1
    assert infer_sobol_refinement_steps(num_samples=256, num_dims=2) == 0
