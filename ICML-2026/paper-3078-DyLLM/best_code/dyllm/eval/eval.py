import argparse
import json
import torch
import os
import logging
from lm_eval.evaluator import simple_evaluate
from dyllm.eval.adapter import DyLLMAdapter


def _is_ruler_task(task_name: str) -> bool:
    task_lower = task_name.strip().lower()
    return task_lower.startswith("ruler") or task_lower.startswith("niah")


def _suppress_ruler_metadata_hint_warning() -> None:
    hint = "Custom kwargs can be passed to `--metadata`"

    class _RulerMetadataHintFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return hint not in record.getMessage()

    filt = _RulerMetadataHintFilter()
    for logger_name in ("api.task", "lm_eval.api.task", "lm_eval"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(filt)
        for handler in logger.handlers:
            handler.addFilter(filt)

    root_logger = logging.getLogger()
    root_logger.addFilter(filt)
    for handler in root_logger.handlers:
        handler.addFilter(filt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--tasks", type=str, default="gsm8k")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-shot", type=int, default=5)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--ignore-eos", action="store_true", default=False)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--num-full-steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--output-file", type=str, default=None)
    ap.add_argument("--log-samples", action="store_true", default=False)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--show-ruler-metadata-warning",
        action="store_true",
        default=False,
        help="Show lm_eval's ruler metadata 안내 WARNING (기본값: 숨김)",
    )
    ap.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="JSON string for task metadata (used for ruler tasks, e.g. '{\"max_seq_lengths\":[4096,8192]}')",
    )
    args = ap.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON for --metadata: {err}") from err
        if not isinstance(metadata, dict):
            raise ValueError("--metadata must be a JSON object")

    is_ruler_requested = any(_is_ruler_task(task) for task in tasks)

    metadata_for_eval = metadata if metadata and is_ruler_requested else None
    if is_ruler_requested:
        metadata_for_eval = dict(metadata_for_eval or {})
        if "pretrained" not in metadata_for_eval and "tokenizer" not in metadata_for_eval:
            metadata_for_eval["pretrained"] = args.model_path

    if metadata and metadata_for_eval is None:
        print("Ignoring --metadata because no ruler task was requested.")

    if not args.show_ruler_metadata_warning:
        _suppress_ruler_metadata_hint_warning()

    model_args_dict = {
        "model_path": args.model_path,
        "max_new_toks": args.max_new_tokens,
        "tensor_parallel_size": args.tp_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "ignore_eos": args.ignore_eos,
        "trust_remote_code": args.trust_remote_code,
        "num_steps": args.num_steps,
        "num_full_steps": args.num_full_steps,
        "block_size": args.block_size,
        "threshold": args.threshold,
    }
    model_args = ",".join([f"{k}={v}" for k, v in model_args_dict.items()])

    with torch.inference_mode():
        results = simple_evaluate(
            model="dyllm",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=args.num_shot,
            batch_size=args.batch_size,
            device="cuda",
            limit=args.limit,
            log_samples=args.log_samples,
            confirm_run_unsafe_code=True,
            verbosity="INFO",
            metadata=metadata_for_eval,
        )

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results.get("results", results), indent=2, default=str))


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    main()
