#!/usr/bin/env python3
"""
ChordEdit comprehensive evaluation script for PIE-bench.

Usage (from /repo):
  python eval.py --model-root /sd-turbo --pie-root /pie_bench [--max-samples N] [--json-only]

Outputs:
  - Generated images → /pie_bench/output/<method>/<subdir>/
  - Final JSON metrics block → prefixed with EVAL_RESULTS_JSON:
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/repo")
sys.path.insert(0, str(REPO_ROOT))

from pipeline_chord import ChordEditPipeline
from run_pie_bench import (
    DEFAULT_EDIT_CONFIG,
    DEFAULT_SEED,
    expand_component_paths,
    load_pie_records,
    paths_from_model_root,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ROOT = "/sd-turbo"
DEFAULT_PIE_ROOT = "/pie_bench"
DEFAULT_MAPPING_FILE = "mapping_file.json"
DEFAULT_IMAGE_SUBDIR = "annotation_images"
DEFAULT_METHOD_NAME = "ChordEdit"
DEFAULT_EXPORT_ROOT = "/pie_bench"
IMG_SIZE = 512  # PIE-bench image dimension

LOGGER = logging.getLogger("chordedit_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# RLE mask decoder (COCO-style: first token = starting value, rest = run lengths)
# ---------------------------------------------------------------------------

def _decode_rle_mask(rle_str: str) -> np.ndarray:
    """Decode a COCO-style RLE string into a (512, 512) uint8 mask (0=bg, 255=fg)."""
    tokens = rle_str.strip().split()
    if not tokens:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # First token = starting value (converted to int, treat any nonzero as foreground)
    it = iter(tokens)
    current_val = 255 if int(next(it)) != 0 else 0

    mask_flat = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.uint8)
    pos = 0
    for tok in it:
        run_len = int(tok)
        if current_val:
            mask_flat[pos : pos + run_len] = current_val
        pos += run_len
        current_val = 255 if current_val == 0 else 0  # flip

    return mask_flat.reshape((IMG_SIZE, IMG_SIZE))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psnr_mse_lpips(
    original: np.ndarray,
    edited: np.ndarray,
    mask: np.ndarray | None,
    lpips_fn: Any,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Compute PSNR (dB), MSE (∈[0,1]), LPIPS on background (non-masked) region."""
    orig_t = torch.from_numpy(original).permute(2, 0, 1).float().to(device) / 255.0
    edit_t = torch.from_numpy(edited).permute(2, 0, 1).float().to(device) / 255.0

    if mask is not None and mask.sum() > 0:
        # mask > 0 = edited region; background = ~mask
        bg = torch.from_numpy((mask < 128).astype(np.float32)).to(device)
        bg_3 = bg.unsqueeze(0).repeat(3, 1, 1)
        diff = (orig_t - edit_t) * bg_3
        n_px = bg.sum() * 3
        mse = (diff ** 2).sum().item() / max(n_px.item(), 1.0)
    else:
        mse = ((orig_t - edit_t) ** 2).mean().item()

    psnr = 100.0 if mse == 0 else 10.0 * np.log10(1.0 / max(mse, 1e-12))

    with torch.no_grad():
        lpips_val = lpips_fn(edit_t.unsqueeze(0), orig_t.unsqueeze(0)).item()

    return psnr, mse, lpips_val


def compute_clip_similarity(
    image: np.ndarray,
    prompt: str,
    clip_model: Any,
    clip_processor: Any,
    device: torch.device,
) -> float:
    """CLIP cosine similarity ×100 between image and text prompt."""
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    img_inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_feat = clip_model.get_image_features(**img_inputs)
        text_feat = clip_model.get_text_features(**text_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat * text_feat).sum(dim=-1).item()

    return sim * 100.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ChordEdit PIE-bench Evaluation")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--pie-root", default=DEFAULT_PIE_ROOT)
    parser.add_argument("--export-root", default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--output-subdir", default="default")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Edit config overrides
    parser.add_argument("--noise-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--t-delta", type=float, default=None)
    parser.add_argument("--step-scale", type=float, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    if args.json_only:
        logging.getLogger().setLevel(logging.WARNING)

    # HF mirror for model downloads
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = "<REDACTED:HF_TOKEN>"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # -------------------------------------------------------------------
    # Load PIE-bench records
    # -------------------------------------------------------------------
    pie_root = Path(args.pie_root).expanduser().resolve()
    mapping_path = pie_root / DEFAULT_MAPPING_FILE
    records = load_pie_records(pie_root, mapping_path, DEFAULT_IMAGE_SUBDIR)
    if args.max_samples:
        records = records[:args.max_samples]
    LOGGER.info("Loaded %d PIE-bench records", len(records))
    if not records:
        LOGGER.error("No PIE records found!")
        sys.exit(1)

    # Pre-decode all masks from mapping file for background consistency metrics
    with open(mapping_path) as f:
        mapping = json.load(f)
    sample_masks: Dict[str, Optional[np.ndarray]] = {}
    for sample_id in mapping:
        rle_raw = mapping[sample_id].get("mask", "0 262144")
        # Handle both list format (original PIE-Bench) and string format (PIE-Bench++)
        if isinstance(rle_raw, list):
            rle_raw = " ".join(str(x) for x in rle_raw)
        decoded = _decode_rle_mask(rle_raw)
        # Only keep non-trivial masks (>0 pixels of editing region)
        sample_masks[sample_id] = decoded if decoded.sum() > 0 else None

    LOGGER.info("Masks loaded: %d with editing regions",
                sum(1 for m in sample_masks.values() if m is not None))

    # -------------------------------------------------------------------
    # Build edit config
    # -------------------------------------------------------------------
    edit_config = dict(DEFAULT_EDIT_CONFIG)
    overrides = {
        "noise_samples": args.noise_samples,
        "n_steps": args.n_steps,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "t_delta": args.t_delta,
        "step_scale": args.step_scale,
    }
    for k, v in overrides.items():
        if v is not None:
            edit_config[k] = v
    LOGGER.info("Edit config: %s", edit_config)

    # -------------------------------------------------------------------
    # Load ChordEdit pipeline
    # -------------------------------------------------------------------
    LOGGER.info("Loading ChordEdit pipeline from %s ...", args.model_root)
    torch.cuda.reset_peak_memory_stats()

    component_paths = expand_component_paths(paths_from_model_root(args.model_root))
    pipe = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device=device,
        torch_dtype=torch.float32,
    )
    LOGGER.info("Pipeline loaded.")

    # -------------------------------------------------------------------
    # Load metric models (CLIP + LPIPS)
    # -------------------------------------------------------------------
    LOGGER.info("Loading CLIP + LPIPS evaluation models ...")
    import lpips
    from transformers import CLIPModel, CLIPProcessor

    lpips_fn = lpips.LPIPS(net="squeeze").to(device)
    clip_model = CLIPModel.from_pretrained("/models/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("/models/clip-vit-large-patch14")
    LOGGER.info("Metric models loaded.")

    # -------------------------------------------------------------------
    # Run inference + compute metrics
    # -------------------------------------------------------------------
    export_root = Path(args.export_root).expanduser().resolve()
    output_dir = export_root / "output" / args.method_name / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_psnr_list: List[float] = []
    bg_mse_list: List[float] = []
    bg_lpips_list: List[float] = []
    clip_whole_list: List[float] = []
    clip_edited_list: List[float] = []
    runtimes: List[float] = []
    processed, skipped = 0, 0

    t_total = time.time()

    for idx, record in enumerate(records, start=1):
        # Load source image
        try:
            with Image.open(record.image_path) as img:
                source_image = img.convert("RGB")
        except Exception:
            skipped += 1
            continue

        # Run ChordEdit
        t0 = time.time()
        try:
            result = pipe(
                image=source_image,
                source_prompt=record.original_prompt,
                target_prompt=record.edited_prompt,
                seed=args.seed,
                output_type="pil",
            )
        except Exception:
            skipped += 1
            continue
        t1 = time.time()
        runtimes.append(t1 - t0)

        # Extract generated image
        images = result.images
        if isinstance(images, list) and images:
            generated = images[0]
        elif torch.is_tensor(images):
            generated = pipe._tensor_to_pil(images)[0]
        else:
            skipped += 1
            continue

        # Save output
        rel_path = output_dir / record.relative_path
        rel_path.parent.mkdir(parents=True, exist_ok=True)
        generated.save(rel_path)

        source_np = np.array(source_image)
        gen_np = np.array(generated)

        # --- Background consistency ---
        mask = sample_masks.get(record.sample_id)
        psnr, mse, lpips_v = compute_psnr_mse_lpips(source_np, gen_np, mask, lpips_fn, device)
        bg_psnr_list.append(psnr)
        bg_mse_list.append(mse)
        bg_lpips_list.append(lpips_v)

        # --- CLIP semantic alignment ---
        cw = compute_clip_similarity(gen_np, record.edited_prompt, clip_model, clip_processor, device)
        ce = compute_clip_similarity(gen_np, record.edit_instruction, clip_model, clip_processor, device)
        clip_whole_list.append(cw)
        clip_edited_list.append(ce)

        processed += 1
        if processed % 50 == 0:
            LOGGER.info("Progress: %d/%d samples", processed, len(records))

    t_total_elapsed = time.time() - t_total

    # -------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------
    avg_psnr   = np.mean(bg_psnr_list)   if bg_psnr_list   else 0.0
    avg_mse    = np.mean(bg_mse_list)    if bg_mse_list    else 0.0
    avg_lpips  = np.mean(bg_lpips_list)  if bg_lpips_list  else 0.0
    avg_cw     = np.mean(clip_whole_list)  if clip_whole_list  else 0.0
    avg_ce     = np.mean(clip_edited_list) if clip_edited_list else 0.0
    avg_runtime = np.mean(runtimes) if runtimes else 0.0
    vram_mb    = get_peak_vram_mb()

    metrics = {
        "psnr":          round(float(avg_psnr), 2),
        "mse":           round(float(avg_mse), 6),
        "mse_x1e3":      round(float(avg_mse * 1000), 2),
        "lpips":         round(float(avg_lpips), 4),
        "lpips_x1e3":    round(float(avg_lpips * 1000), 2),
        "clip_whole":    round(float(avg_cw), 2),
        "clip_edited":   round(float(avg_ce), 2),
        "runtime_sec":   round(float(avg_runtime), 4),
        "total_runtime_sec": round(float(t_total_elapsed), 1),
        "vram_mb":       round(float(vram_mb), 1),
        "num_samples":   processed,
        "num_skipped":   skipped,
    }

    json_str = json.dumps(metrics, indent=2)
    print("\n" + "=" * 60)
    print("EVAL_RESULTS_JSON:")
    print(json_str)
    print("=" * 60)


if __name__ == "__main__":
    main()
