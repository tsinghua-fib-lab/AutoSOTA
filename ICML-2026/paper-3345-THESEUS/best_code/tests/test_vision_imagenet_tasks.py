from __future__ import annotations

from merge_and_rebase.data.templates import get_templates
from merge_and_rebase.eval.datasets.vision8_14_20 import SUITES, VISION20_TASKS, VISION_SUPPORTED_TASKS, _vision_spec


def test_imagenet_tasks_resolve_with_expected_hf_specs() -> None:
    hf_path, hf_config, split_map = _vision_spec("ImageNet1K")
    assert hf_path == "ILSVRC/imagenet-1k"
    assert hf_config is None
    assert split_map == {"train": "train", "test": "validation"}

    hf_path, hf_config, split_map = _vision_spec("ImageNet21KP")
    assert hf_path == "timm/imagenet-w21-p"
    assert hf_config is None
    assert split_map == {"train": "train", "test": "validation"}


def test_imagenet_tasks_are_supported_but_not_added_to_existing_suites() -> None:
    assert "ImageNet1K" in VISION_SUPPORTED_TASKS
    assert "ImageNet21KP" in VISION_SUPPORTED_TASKS
    assert "ImageNet1K" not in VISION20_TASKS
    assert "ImageNet21KP" not in VISION20_TASKS
    assert "ImageNet1K" not in SUITES["vision20"].tasks
    assert "ImageNet21KP" not in SUITES["vision20"].tasks


def test_imagenet_tasks_reuse_imagenet_templates() -> None:
    imagenet_templates = get_templates("ImageNet")
    assert get_templates("ImageNet1K") is imagenet_templates
    assert get_templates("ImageNet21KP") is imagenet_templates
