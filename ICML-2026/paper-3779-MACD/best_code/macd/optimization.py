from __future__ import annotations

import argparse
import gc
import os
from typing import Any

import torch
import torch.nn.functional as F

from macd.io import append_jsonl, ensure_dir, read_jsonl
from macd.masks import build_object_masks, build_time_grid
from macd.models import load_video_llm

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def model_device(model: torch.nn.Module) -> torch.device:
    return getattr(model, "device", None) or next(model.parameters()).device


def load_guide_model(model_path: str, torch_dtype: str = "bfloat16") -> tuple[torch.nn.Module, Any]:
    return load_video_llm(model_path, torch_dtype=torch_dtype)


def prepare_video_inputs(
    processor: Any,
    video_path: str,
    prompt: str,
    total_pixels: int,
    min_pixels: int,
    fps: float,
    device: torch.device,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "fps": fps,
                },
            ],
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_val = video_kwargs.get("fps", None)
    if isinstance(fps_val, (list, tuple)):
        fps_val = fps_val[0] if fps_val else None
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_val,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def extract_video_tensor(inputs: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    V = inputs.get("pixel_values_videos")
    if V is None:
        V = inputs.get("pixel_values")
    if isinstance(V, (list, tuple)) and V:
        V = V[0]
    if not isinstance(V, torch.Tensor):
        return None
    if V.ndim == 4:
        V = V.unsqueeze(0)
    return V.to(device)


def infer_grid_thw(inputs: dict[str, Any], V: torch.Tensor) -> torch.Tensor | None:
    grid = inputs.get("video_grid_thw")
    if grid is None:
        grid = inputs.get("image_grid_thw")
    if isinstance(grid, torch.Tensor):
        return grid.to(V.device)
    if V.ndim == 5:
        _, _, T, H, W = V.shape
        return torch.tensor([[T, H, W]], device=V.device)
    return None


def grid_shape(grid_thw: torch.Tensor | None, V: torch.Tensor) -> tuple[int, int, int] | None:
    if isinstance(grid_thw, torch.Tensor) and grid_thw.numel() >= 3:
        values = grid_thw.detach().cpu().view(-1).tolist()
        return tuple(map(int, values[-3:]))  # type: ignore[return-value]
    if V.ndim == 5:
        return int(V.shape[2]), int(V.shape[3]), int(V.shape[4])
    return None


def build_perturbation(V: torch.Tensor, mask: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor | None:
    if V.ndim == 5:
        return -V * mask.to(V.device, dtype=V.dtype)

    if V.ndim == 2:
        V = V.unsqueeze(0)
    if V.ndim == 3:
        expected = T * H * W
        if V.shape[1] != expected:
            return None
        flat_mask = mask.view(1, 1, expected).transpose(1, 2).to(V.device, dtype=V.dtype)
        return -V * flat_mask
    return None


def forward_last_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    video_values: torch.Tensor,
    grid_thw: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "pixel_values_videos": video_values,
        "use_cache": False,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if grid_thw is not None:
        kwargs["video_grid_thw"] = grid_thw
    if mm_token_type_ids is not None:
        kwargs["mm_token_type_ids"] = mm_token_type_ids
    outputs = model(**kwargs)
    return outputs.logits[:, -1, :]


def optimize_r_finite_difference(
    model: torch.nn.Module,
    V: torch.Tensor,
    Z: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
    logits_orig: torch.Tensor,
    pgd_steps: int,
    learning_rate: float,
    fd_eps: float,
    r_initial: float,
    mm_token_type_ids: torch.Tensor | None = None,
) -> float:
    device = model_device(model)
    r = torch.tensor([r_initial], device=device)

    cpu = {
        "V": V.detach().cpu(),
        "Z": Z.detach().cpu(),
        "input_ids": input_ids.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu() if isinstance(attention_mask, torch.Tensor) else None,
        "logits_orig": logits_orig.detach().cpu(),
        "mm_token_type_ids": mm_token_type_ids.detach().cpu() if isinstance(mm_token_type_ids, torch.Tensor) else None,
    }
    for _ in range(pgd_steps):
        V_gpu = cpu["V"].to(device)
        Z_gpu = cpu["Z"].to(device)
        input_ids_gpu = cpu["input_ids"].to(device)
        attention_gpu = cpu["attention_mask"].to(device) if isinstance(cpu["attention_mask"], torch.Tensor) else None
        logits_orig_gpu = cpu["logits_orig"].to(device)
        grid_gpu = grid_thw.to(device) if isinstance(grid_thw, torch.Tensor) else None
        mm_type_gpu = cpu["mm_token_type_ids"].to(device) if isinstance(cpu["mm_token_type_ids"], torch.Tensor) else None

        with torch.inference_mode():
            logits_r = forward_last_logits(model, input_ids_gpu, attention_gpu, V_gpu + r.to(V_gpu.dtype) * Z_gpu, grid_gpu, mm_type_gpu)
            logits_eps = forward_last_logits(
                model,
                input_ids_gpu,
                attention_gpu,
                V_gpu + (r + fd_eps).to(V_gpu.dtype) * Z_gpu,
                grid_gpu,
                mm_type_gpu,
            )
            kl_r = F.kl_div(F.log_softmax(logits_r, dim=-1), F.softmax(logits_orig_gpu, dim=-1), reduction="batchmean")
            kl_eps = F.kl_div(F.log_softmax(logits_eps, dim=-1), F.softmax(logits_orig_gpu, dim=-1), reduction="batchmean")
            r = r + learning_rate * ((kl_eps - kl_r) / fd_eps)

        del V_gpu, Z_gpu, input_ids_gpu, attention_gpu, logits_orig_gpu, logits_r, logits_eps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(r.item())


def clip_r(value: float, r_initial: float, clip: bool, binary_clip: bool) -> float:
    if clip:
        value = max(0.0, min(1.0, value))
    if binary_clip:
        if value > r_initial:
            return 1.0
        if value < r_initial:
            return 0.0
    return value


def optimize_manifest(args: argparse.Namespace) -> None:
    model, processor = load_guide_model(args.model_path, args.torch_dtype)
    device = model_device(model)
    entries = read_jsonl(args.manifest_file)
    ensure_dir(os.path.dirname(args.output_r_file))
    if os.path.exists(args.output_r_file) and not args.append:
        os.remove(args.output_r_file)

    for entry in tqdm(entries, desc="videos"):
        video_path = entry.get("orig_video_path")
        video_id = str(entry.get("video_id", ""))
        prompt = str(entry.get("prompt", ""))
        objects = entry.get("objects", [])
        if not video_path or not os.path.isfile(video_path) or not objects:
            continue

        inputs = prepare_video_inputs(processor, video_path, prompt, args.video_total_pixels, args.video_min_pixels, args.video_fps, device)
        V = extract_video_tensor(inputs, device)
        if V is None:
            print(f"[WARN] Could not extract video tensor for {video_id}")
            continue

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        attention_mask = attention_mask.to(device) if isinstance(attention_mask, torch.Tensor) else None
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        mm_token_type_ids = mm_token_type_ids.to(device) if isinstance(mm_token_type_ids, torch.Tensor) else None
        grid_thw = infer_grid_thw(inputs, V)
        shape = grid_shape(grid_thw, V)
        if shape is None:
            print(f"[WARN] Could not infer THW grid for {video_id}")
            continue
        T, H, W = shape

        with torch.no_grad():
            logits_orig = forward_last_logits(model, input_ids, attention_mask, V, grid_thw, mm_token_type_ids).detach().cpu()

        objects_iter = sorted(objects, key=lambda obj: obj.get("avg_score", 0.0), reverse=True)
        if args.objects_topk and args.objects_topk > 0:
            objects_iter = objects_iter[: args.objects_topk]

        t_sample = build_time_grid(T, args.video_fps)
        for obj in tqdm(objects_iter, desc=f"objects:{video_id}", leave=False):
            object_id = str(obj.get("object_id", ""))
            masks = build_object_masks([obj], t_sample, H, W)
            mask = masks.get(object_id)
            if mask is None:
                continue
            Z = build_perturbation(V, mask, T, H, W)
            if Z is None:
                print(f"[WARN] Cannot align mask with visual tensor for {video_id}/{object_id}")
                continue
            r_value = optimize_r_finite_difference(
                model=model,
                V=V,
                Z=Z,
                input_ids=input_ids,
                attention_mask=attention_mask,
                grid_thw=grid_thw,
                logits_orig=logits_orig,
                pgd_steps=args.pgd_steps,
                learning_rate=args.r_learning_rate,
                fd_eps=args.fd_eps,
                r_initial=args.r_initial,
                mm_token_type_ids=mm_token_type_ids,
            )
            r_value = clip_r(r_value, args.r_initial, args.clip, args.binary_clip)
            append_jsonl(
                args.output_r_file,
                {
                    "question_id": entry.get("question_id"),
                    "video_id": video_id,
                    "object_id": object_id,
                    "category": obj.get("category", -1),
                    "r": r_value,
                    "orig_video_path": video_path,
                },
            )

        del inputs, V, input_ids, attention_mask, logits_orig
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize MACD per-object mask strengths.")
    parser.add_argument("--model-path", required=True, help="Guide Video-LLM path, e.g. Qwen/Qwen2-VL-7B-Instruct.")
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--output-r-file", required=True)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "auto"])
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--r-learning-rate", type=float, default=0.5)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--r-initial", type=float, default=0.75)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--binary-clip", action="store_true")
    parser.add_argument("--video-total-pixels", type=int, default=20480 * 28 * 28)
    parser.add_argument("--video-min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--objects-topk", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="Append to an existing r-file instead of overwriting it.")
    optimize_manifest(parser.parse_args())


if __name__ == "__main__":
    main()
