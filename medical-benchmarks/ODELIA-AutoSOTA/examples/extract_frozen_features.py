"""Create legacy train/validation feature caches from preprocessed ROIs.

Install third_party/meisenmeister first. This helper requires separately acquired
weights and a local examination-level split JSON; it does not download clinical data.
Test inference should use the original TTA probabilities for historical compatibility.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from odelia_autosota.data import FEATURE_KEYS, sha256_file
from odelia_autosota.features import stage_descriptors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--checkpoint-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--amp-dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    args = p.parse_args()
    from meisenmeister.dataloading import MeisenmeisterROIDataset
    from meisenmeister.architectures.nnunet_encoder import ResidualEncoderClsNetwork
    from numpy._core.multiarray import _reconstruct

    payload = json.loads(args.split.read_text())
    split = payload[0] if isinstance(payload, list) and len(payload) == 1 else payload
    train = set(split["train"])
    val = set(split["val"])
    if train & val:
        raise ValueError("train/validation examination overlap")
    dataset = MeisenmeisterROIDataset(args.dataset)
    records = []
    for i, sample in enumerate(dataset.samples):
        case = sample["case_id"]
        if case in train or case in val:
            records.append(
                {
                    "dataset_index": i,
                    "sample_id": sample["sample_id"],
                    "case_id": case,
                    "label": int(sample["label"]),
                    "partition": "train" if case in train else "val",
                }
            )
    found = {row["case_id"] for row in records}
    if found != train | val:
        raise ValueError("split contains missing examinations")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.amp_dtype]
    amp = device.type == "cuda" and amp_dtype != torch.float32
    for fold in range(7):
        ckpt = args.checkpoint_root / f"fold_{fold}/model_best.pt"
        safe = [_reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        with torch.serialization.safe_globals(safe):
            state = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        model = ResidualEncoderClsNetwork(in_channels=3, num_classes=3)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        captures = {}
        hooks = [
            model.encoder.stages[s].register_forward_hook(
                lambda m, i, o, s=s: captures.__setitem__(s, o)
            )
            for s in (3, 4, 5)
        ]
        folder = args.output / f"fold_{fold}"
        (folder / "samples").mkdir(parents=True)
        try:
            for row in records:
                image = dataset[row["dataset_index"]]["image"].unsqueeze(0).to(device)
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp
                    ),
                ):
                    model(image)
                descriptor = (
                    stage_descriptors(*(captures[s] for s in (3, 4, 5)))[0]
                    .cpu()
                    .numpy()
                )
                # Match the search cache: float32 classification readout from final mean.
                mean = captures[5].float().mean(dim=(2, 3, 4))
                logits = (
                    torch.nn.functional.linear(
                        mean,
                        model.cls_head.fc.weight.float(),
                        model.cls_head.fc.bias.float(),
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                arrays = {}
                offset = 0
                for key, dim in zip(FEATURE_KEYS, [256] * 3 + [320] * 6):
                    arrays[key] = descriptor[offset : offset + dim]
                    offset += dim
                sample_id = row["sample_id"]
                if Path(sample_id).name != sample_id:
                    raise ValueError("unsafe sample filename")
                np.savez(
                    folder / "samples" / f"{sample_id}.npz",
                    released_logits=logits,
                    **arrays,
                )
            (folder / "manifest.json").write_text(
                json.dumps({"fold": fold, "records": records}, indent=2) + "\n"
            )
            (folder / "cache_metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": sha256_file(ckpt),
                        "split_sha256": sha256_file(args.split),
                        "amp_dtype": args.amp_dtype,
                        "encoder_frozen": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            for hook in hooks:
                hook.remove()
        del state, model, captures
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Cached fold {fold}", flush=True)


if __name__ == "__main__":
    main()
