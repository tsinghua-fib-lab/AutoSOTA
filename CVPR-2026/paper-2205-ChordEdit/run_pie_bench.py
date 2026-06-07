from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from PIL import Image

from pipeline_chord import ChordEditPipeline
from utils import first_param_point, load_yaml_config


LOGGER = logging.getLogger("pie_bench")

# model root + expected component subdirectories
COMPONENT_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder",
    "tokenizer_path": "tokenizer",
    "vae_path": "vae",
}
DEFAULT_MODEL_ROOT = "/sd-turbo"
DEFAULT_COMPONENT_PATHS: Dict[str, str] = {
    key: str(Path(DEFAULT_MODEL_ROOT) / subdir) for key, subdir in COMPONENT_SUBDIRS.items()
}

DEFAULT_EDIT_CONFIG = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
}

DEFAULT_SEED = 42
DEFAULT_PRECISION = "fp32"

DEFAULT_PIE_ROOT = Path(__file__).resolve().parent / "pie_bench"
DEFAULT_MAPPING_FILE = "mapping_file.json"
DEFAULT_IMAGE_SUBDIR = "annotation_images"
DEFAULT_METHOD_NAME = "ChordEdit"


@dataclass(frozen=True)
class PieRecord:
    sample_id: str
    image_path: Path
    relative_path: Path
    original_prompt: str
    edited_prompt: str
    edit_instruction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChordEdit on PIE-Bench data and export PIE-format results.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config describing edit params.")
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Root folder containing unet/scheduler/text_encoder/tokenizer/vae subfolders.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None, help="Computation precision.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed overriding the config file.")
    parser.add_argument("--noise-samples", type=int, default=None, help="Number of MC noise samples.")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of Chord iterations.")
    parser.add_argument("--t-start", type=float, default=None, help="Edit timestep start.")
    parser.add_argument("--t-end", type=float, default=None, help="Edit timestep end.")
    parser.add_argument("--t-delta", type=float, default=None, help="Edit timestep delta.")
    parser.add_argument("--step-scale", type=float, default=None, help="Edit update magnitude.")
    parser.add_argument("--cleanup", action="store_true", help="Force cleanup on.")
    parser.add_argument("--no-cleanup", action="store_true", help="Force cleanup off.")
    parser.add_argument(
        "--center-crop",
        dest="center_crop",
        action="store_true",
        default=True,
        help="Center-crop before resize for VAE preprocessing (default).",
    )
    parser.add_argument(
        "--no-center-crop",
        dest="center_crop",
        action="store_false",
        help="Disable center crop before resizing.",
    )
    parser.add_argument(
        "--use-attention-mask",
        action="store_true",
        help="Pass attention masks to the text encoder (defaults off to mirror chord/src).",
    )
    parser.add_argument(
        "--safety-checker",
        dest="use_safety_checker",
        action="store_true",
        help="Enable StableDiffusion safety checker before exporting images.",
    )
    parser.add_argument(
        "--no-safety-checker",
        dest="use_safety_checker",
        action="store_false",
        default=False,
        help="Disable safety checker (default).",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Resolution used when feeding the VAE.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only process the first N records.")

    parser.add_argument("--pie-root", type=str, default=None, help="Root directory of PIE-Bench data.")
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=DEFAULT_MAPPING_FILE,
        help="Mapping file path relative to --pie-root.",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Subdirectory (inside --pie-root) containing the original PIE annotation images.",
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=None,
        help="Directory that follows PIE-Bench layout (data/... + output/...). Defaults to --pie-root.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=DEFAULT_METHOD_NAME,
        help="Name used under export_root/output/<method_name>/annotation_images.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Subdirectory inside output/<method_name>/ for generated images.",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Subdirectory inside data/ where original images are copied when --copy-source is set.",
    )
    parser.add_argument("--copy-source", action="store_true", help="Copy the source PIE images into export_root/data.")
    parser.add_argument(
        "--mapping-dest",
        type=str,
        default="data/mapping_file.json",
        help="Relative path (from export_root) to write the mapping file.",
    )
    parser.add_argument(
        "--no-sync-mapping",
        action="store_true",
        help="Skip copying the PIE mapping file into export_root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions when present.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Progress logging interval in number of saved samples (0 disables incremental logs).",
    )

    return parser.parse_args()


def dtype_from_precision(value: Optional[str]) -> torch.dtype:
    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def expand_component_paths(path_map: Dict[str, Optional[str]]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    for key in COMPONENT_SUBDIRS:
        value = path_map.get(key)
        fallback = DEFAULT_COMPONENT_PATHS.get(key)
        final_value = value if value is not None else fallback
        if final_value is None:
            raise ValueError(f"Missing required path for '{key}'. Provide via config or CLI.")
        expanded[key] = str(Path(final_value).expanduser().resolve())
    return expanded


def paths_from_model_root(model_root: str | Path) -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    return {key: str((root / subdir).resolve()) for key, subdir in COMPONENT_SUBDIRS.items()}


def load_pipeline_config(path: Optional[str]) -> tuple[Dict[str, Any], int, Optional[str]]:
    if path is None:
        return (dict(DEFAULT_EDIT_CONFIG), DEFAULT_SEED, DEFAULT_PRECISION)

    cfg = load_yaml_config(path)
    editor_cfg = cfg.get("editor", {})
    seed_value = editor_cfg.get("seed")
    if seed_value is None:
        seed_list = editor_cfg.get("seed_list")
        if isinstance(seed_list, (list, tuple)) and seed_list:
            seed_value = seed_list[0]
        elif seed_list is not None:
            seed_value = seed_list
    seed_value = int(seed_value) if seed_value is not None else DEFAULT_SEED
    precision = editor_cfg.get("precision", DEFAULT_PRECISION)

    params_grid = editor_cfg.get("params_grid", {})
    edit_config = first_param_point(params_grid) if params_grid else dict(DEFAULT_EDIT_CONFIG)

    return edit_config, seed_value, precision


def apply_cli_overrides(args: argparse.Namespace, edit_config: Dict[str, Any], seed: Optional[int]) -> tuple[Dict[str, Any], int]:
    overrides = {
        "noise_samples": args.noise_samples,
        "n_steps": args.n_steps,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "t_delta": args.t_delta,
        "step_scale": args.step_scale,
    }
    for key, value in overrides.items():
        if value is not None:
            edit_config[key] = value

    if args.cleanup:
        edit_config["cleanup"] = True
    elif args.no_cleanup:
        edit_config["cleanup"] = False

    cli_seed = args.seed
    seed_value = seed if cli_seed is None else cli_seed
    if seed_value is None:
        seed_value = DEFAULT_SEED
    return edit_config, int(seed_value)


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate.expanduser().resolve()
    return (base / candidate).expanduser().resolve()


def load_pie_records(root: Path, mapping_path: Path, image_subdir: str) -> List[PieRecord]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"PIE mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict):
        raise ValueError(f"Expected mapping JSON to be a dict, got {type(mapping).__name__}")

    img_root = (root / image_subdir).expanduser().resolve()
    if not img_root.exists():
        raise FileNotFoundError(f"PIE image directory does not exist: {img_root}")

    records: List[PieRecord] = []
    for sample_id in sorted(mapping.keys()):
        meta = mapping[sample_id]
        rel_value = meta.get("image_path")
        if rel_value is None:
            LOGGER.warning("Sample %s is missing 'image_path'; skipping.", sample_id)
            continue
        rel_path = Path(rel_value)
        abs_path = (img_root / rel_path).expanduser().resolve()
        if not abs_path.exists():
            LOGGER.warning("Sample %s image not found at %s; skipping.", sample_id, abs_path)
            continue

        original_prompt = meta.get("original_prompt") or meta.get("source_prompt") or ""
        edited_prompt = meta.get("editing_prompt") or meta.get("edited_prompt") or meta.get("target_prompt") or ""
        edit_instruction = meta.get("editing_instruction") or meta.get("edit_prompt") or edited_prompt

        records.append(
            PieRecord(
                sample_id=sample_id,
                image_path=abs_path,
                relative_path=rel_path,
                original_prompt=original_prompt,
                edited_prompt=edited_prompt,
                edit_instruction=edit_instruction,
            )
        )

    if not records:
        raise FileNotFoundError(f"No valid PIE records found in {mapping_path}.")
    return records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    ensure_dir(dst.parent)
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)


def sync_mapping_file(mapping_path: Path, export_root: Path, dest_relative: str, *, overwrite: bool) -> None:
    dest_path = resolve_path(export_root, dest_relative)
    copy_file(mapping_path, dest_path, overwrite=overwrite)


def save_prediction(image: Image.Image, destination: Path, *, overwrite: bool) -> None:
    ensure_dir(destination.parent)
    if overwrite or not destination.exists():
        image.save(destination)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    edit_config, seed, precision = load_pipeline_config(args.config)
    edit_config, seed = apply_cli_overrides(args, edit_config, seed)
    component_paths = expand_component_paths(paths_from_model_root(args.model_root))

    precision_choice_raw = args.precision or precision or DEFAULT_PRECISION
    precision_choice = precision_choice_raw.lower()
    if precision_choice != "fp32":
        LOGGER.warning(
            "Precision '%s' requested, but PIE export forces fp32 for numerical stability.",
            precision_choice_raw,
        )
        precision_choice = "fp32"
    torch_dtype = dtype_from_precision(precision_choice)
    compute_dtype = torch.float32

    pie_root = Path(args.pie_root).expanduser().resolve() if args.pie_root else DEFAULT_PIE_ROOT
    export_root = Path(args.export_root).expanduser().resolve() if args.export_root else pie_root
    mapping_path = resolve_path(pie_root, args.mapping_file)

    records = load_pie_records(pie_root, mapping_path, args.image_subdir)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if not records:
        LOGGER.error("No PIE records to process. Check dataset paths.")
        return

    LOGGER.info(
        "Loaded %d PIE samples from %s (mapping=%s)",
        len(records),
        pie_root,
        mapping_path,
    )
    LOGGER.info("Seed %s | Edit config %s", seed, edit_config)

    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device=args.device,
        torch_dtype=torch_dtype,
        image_size=args.image_size,
        use_center_crop=args.center_crop,
        compute_dtype=compute_dtype,
        use_attention_mask=args.use_attention_mask,
        use_safety_checker=args.use_safety_checker,
    )

    output_dir = export_root / "output" / args.method_name / args.output_subdir
    source_dir = export_root / "data" / args.source_subdir
    ensure_dir(output_dir)
    if args.copy_source:
        ensure_dir(source_dir)

    if not args.no_sync_mapping:
        sync_mapping_file(mapping_path, export_root, args.mapping_dest, overwrite=args.overwrite)
        LOGGER.info("Synchronized mapping file to %s", resolve_path(export_root, args.mapping_dest))

    processed = 0
    skipped = 0

    for idx, record in enumerate(records, start=1):
        rel_output_path = output_dir / record.relative_path
        if rel_output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            with Image.open(record.image_path) as img:
                source_image = img.convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to read %s: %s", record.image_path, exc)
            skipped += 1
            continue

        try:
            result = pipeline(
                image=source_image,
                source_prompt=record.original_prompt,
                target_prompt=record.edited_prompt,
                seed=seed,
                output_type="pil",
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.error("Pipeline failed on %s: %s", record.sample_id, exc)
            skipped += 1
            continue

        images = result.images
        if isinstance(images, list) and images:
            generated = images[0]
        elif torch.is_tensor(images):
            # Fall back to tensor output if requested differently.
            generated = pipeline._tensor_to_pil(images)[0]  # type: ignore[attr-defined]
        else:
            LOGGER.warning("No images returned for sample %s; skipping.", record.sample_id)
            skipped += 1
            continue

        save_prediction(generated, rel_output_path, overwrite=args.overwrite)

        if args.copy_source:
            target_source_path = source_dir / record.relative_path
            copy_file(record.image_path, target_source_path, overwrite=args.overwrite)

        processed += 1
        if args.log_every and processed % args.log_every == 0:
            LOGGER.info("Saved %d/%d samples (skipped=%d)", processed, len(records), skipped)

    LOGGER.info(
        "Finished PIE export. Saved %d sample(s), skipped %d (existing/errors). Results: %s",
        processed,
        skipped,
        output_dir,
    )


if __name__ == "__main__":
    main()
