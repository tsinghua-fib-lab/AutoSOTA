from __future__ import annotations

from merge_and_rebase.finetune.reference_tasks import (
    apply_reference_tags_to_out_dir,
    build_reference_task_resolution_context,
    resolve_reference_tasks_from_kwargs,
)


def test_apply_reference_tags_to_out_dir_uses_single_reference_dataset() -> None:
    context = build_reference_task_resolution_context(
        training_tasks=["Cars", "DTD"],
        suite="vision8",
    )

    tagged = apply_reference_tags_to_out_dir(
        out_dir="src/checkpoints/DELTA",
        regularization_cfg={
            "name": "ekfac_ggn",
            "reference_datasets": ["ImageNet1K"],
        },
        context=context,
    )

    assert tagged == "src/checkpoints/DELTA_imagenet1k_ref"


def test_apply_reference_tags_to_out_dir_uses_suite_fallback_for_dataset_aware_regularizer() -> None:
    context = build_reference_task_resolution_context(
        training_tasks=["Cars", "DTD"],
        suite="vision8",
    )

    tagged = apply_reference_tags_to_out_dir(
        out_dir="src/checkpoints/tak_ekfac",
        regularization_cfg={"name": "ekfac_ggn"},
        context=context,
    )

    assert tagged == "src/checkpoints/tak_ekfac_8vision_ref"


def test_apply_reference_tags_to_out_dir_deduplicates_composite_nested_reference_tags() -> None:
    context = build_reference_task_resolution_context(
        training_tasks=["Cars", "DTD"],
        suite="vision8",
    )
    regularization_cfg = {
        "name": "composite",
        "regularizers": [
            {
                "name": "distillation",
                "teacher": {
                    "regularization": {
                        "name": "ekfac_ggn",
                        "reference_datasets": ["ImageNet1K"],
                    }
                },
            },
            {
                "name": "ekfac_ggn",
                "reference_datasets": ["ImageNet1K"],
            },
        ],
    }

    tagged = apply_reference_tags_to_out_dir(
        out_dir="src/checkpoints/DELTA",
        regularization_cfg=regularization_cfg,
        context=context,
    )

    assert tagged == "src/checkpoints/DELTA_imagenet1k_ref"


def test_apply_reference_tags_to_out_dir_avoids_double_suffix() -> None:
    context = build_reference_task_resolution_context(
        training_tasks=["Cars", "DTD"],
        suite="vision8",
    )

    tagged = apply_reference_tags_to_out_dir(
        out_dir="src/checkpoints/tak_kfac_8vision_ref",
        regularization_cfg={"name": "kfac_ggn"},
        context=context,
    )

    assert tagged == "src/checkpoints/tak_kfac_8vision_ref"


def test_apply_reference_tags_to_out_dir_avoids_reappending_existing_mid_suffix() -> None:
    context = build_reference_task_resolution_context(
        training_tasks=["Cars", "DTD"],
        suite="vision8",
    )

    tagged = apply_reference_tags_to_out_dir(
        out_dir="src/checkpoints/DELTA_zero_ekfac_weight_imagenet21kp_ref_existing_suffix",
        regularization_cfg={
            "name": "ekfac_ggn",
            "reference_datasets": ["ImageNet21KP"],
        },
        context=context,
    )

    assert tagged == "src/checkpoints/DELTA_zero_ekfac_weight_imagenet21kp_ref_existing_suffix"


def test_resolve_reference_tasks_from_kwargs_excludes_current_task() -> None:
    refs = resolve_reference_tasks_from_kwargs(
        regularization_cfg={"name": "ekfac_ggn"},
        kwargs={"reference_tasks": ["Cars", "DTD"]},
        task="Cars",
        require_reference=True,
    )

    assert refs == ["DTD"]
