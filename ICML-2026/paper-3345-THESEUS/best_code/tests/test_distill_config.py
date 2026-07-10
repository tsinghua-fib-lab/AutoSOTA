from __future__ import annotations

import pytest

from merge_and_rebase.finetune.regularizers._distill_config import teacher_train_cfg


def test_teacher_train_cfg_defaults_dense_lr_to_student_default() -> None:
    cfg = teacher_train_cfg(
        None,
        defaults={
            "lr": 1e-4,
            "dense_lr": 1e-4,
            "weight_decay": 0.1,
            "optimizer_name": "adamw",
            "scheduler_name": "cosine",
            "warmup_length": 10,
            "grad_clip_norm": 1.0,
        },
    )

    assert cfg["lr"] == pytest.approx(1e-4)
    assert cfg["dense_lr"] == pytest.approx(1e-4)


def test_teacher_train_cfg_prefers_optimizer_dense_lr_over_train_dense_lr() -> None:
    cfg = teacher_train_cfg(
        {
            "lr": 1e-4,
            "dense_lr": 3e-5,
            "optimizer": {
                "name": "adamw",
                "dense_lr": 7e-5,
            },
        },
        defaults={
            "lr": 1e-4,
            "dense_lr": 1e-4,
            "weight_decay": 0.1,
            "optimizer_name": "adamw",
            "scheduler_name": "cosine",
            "warmup_length": 10,
            "grad_clip_norm": 1.0,
        },
    )

    assert cfg["dense_lr"] == pytest.approx(7e-5)
