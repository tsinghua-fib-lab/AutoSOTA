from pathlib import Path

import torch

from merge_and_rebase.eval.utils import load_vision_checkpoint_reference


def test_load_vision_checkpoint_reference_accepts_local_adapter_dir(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    ref, obj = load_vision_checkpoint_reference(ckpt_ref=str(adapter_dir))

    assert ref == str(adapter_dir)
    assert obj == {"format": "peft", "peft_adapter_dir": str(adapter_dir)}


def test_load_vision_checkpoint_reference_accepts_hf_adapter_repo(monkeypatch) -> None:
    adapter_dir = "/tmp/fake-adapter"

    def _fake_resolve(ref: str) -> str:
        assert ref == "hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars"
        return adapter_dir

    def _fail_torch_load(*args, **kwargs):
        raise AssertionError("torch.load should not be used for HF adapter refs")

    monkeypatch.setattr("merge_and_rebase.eval.utils.resolve_peft_adapter_dir", _fake_resolve)
    monkeypatch.setattr(torch, "load", _fail_torch_load)

    ref, obj = load_vision_checkpoint_reference(
        ckpt_ref="hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars"
    )

    assert ref == "hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars"
    assert obj == {"format": "peft", "peft_adapter_dir": adapter_dir}
