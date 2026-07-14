import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "facebook" / "opt-1.3b"

ALGORITHMS = ["BREW"]
DATASETS = ["c4", "opengen"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--dataset", choices=["c4", "opengen", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-length", type=int, default=230)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_model(model_path, hf_token):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    token = hf_token.strip() or None
    if token is not None:
        login(token=token)

    snapshot_download(
        repo_id="facebook/opt-1.3b",
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
        token=token,
    )

    print("All models downloaded successfully.")


def read_jsonl(path, max_samples, offset):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue

            if len(rows) >= max_samples:
                break

            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_dataset(dataset_name, max_samples, offset):
    if dataset_name == "c4":
        path = BASE_DIR / "dataset" / "c4" / "processed_c4.json"
        rows = read_jsonl(path, max_samples, offset)

        return [
            {
                "dataset": "c4",
                "sample_idx": offset + i,
                "prompt": row["prompt"],
                "natural_text": row["natural_text"],
            }
            for i, row in enumerate(rows)
        ]

    if dataset_name == "opengen":
        path = BASE_DIR / "dataset" / "openGen" / "OpenGen.jsonl"
        rows = read_jsonl(path, max_samples, offset)

        samples = []
        for i, row in enumerate(rows):
            targets = row.get("targets", [])
            natural_text = targets[0] if targets else ""

            samples.append(
                {
                    "dataset": "opengen",
                    "sample_idx": offset + i,
                    "prompt": row["prefix"],
                    "natural_text": natural_text,
                }
            )

        return samples

    raise ValueError(f"Unknown dataset: {dataset_name}")


def load_samples(dataset_name, max_samples, offset):
    names = DATASETS if dataset_name == "all" else [dataset_name]

    samples = []
    for name in names:
        samples.extend(load_dataset(name, max_samples, offset))

    return samples


def merge_prompt_and_continuation(prompt, continuation):
    if not prompt:
        return continuation

    if not continuation:
        return prompt

    if prompt[-1].isspace() or continuation[0].isspace() or continuation[0] in ".,;:!?)]}":
        return prompt + continuation

    return prompt + " " + continuation


def get_output_vocab_size(model, tokenizer):
    output_embeddings = model.get_output_embeddings()

    if output_embeddings is not None:
        return int(output_embeddings.weight.shape[0])

    return int(getattr(model.config, "vocab_size", len(tokenizer)))


def load_transformers_config(model_path, device, max_new_tokens, min_length):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() and device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    model.eval()

    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=get_output_vocab_size(model, tokenizer),
        device=device,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        do_sample=True,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id,
    )


def compact_detection(result):
    keys = [
        "is_watermarked",
        "matched",
        "total",
        "matched_blocks",
        "total_blocks",
        "match_rate",
        "match_percent",
        "best_offset",
        "num_erasures",
        "erasure_rate",
        "mode",
    ]

    return {key: result[key] for key in keys if key in result}


def save_text_result(path, title, text, result):
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write(text + "\n\n")
        f.write("Detection Result:\n")
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
        f.write("\n")


def reset_watermark_state_if_available(watermark):
    """
    Reset per-sample watermark generation state.

    This is important for BREW because the logits processor stores
    codeword_queue and token_bit_log. If these states are not reset,
    the next sample may start from a later codeword/block index.

    This matters especially for block-specific vocabulary partitions
    with seed_j = H(K, j), where the block index determines the
    vocabulary partition.
    """
    if hasattr(watermark, "reset_state"):
        watermark.reset_state()
        return

    # Fallback for implementations that do not expose reset_state().
    logits_processor = getattr(watermark, "logits_processor", None)
    if logits_processor is None:
        return

    if hasattr(logits_processor, "codeword_queue"):
        logits_processor.codeword_queue = []

    if hasattr(logits_processor, "token_bit_log"):
        logits_processor.token_bit_log = []


def run_algorithm(algorithm_name, transformers_config, samples):
    config_path = BASE_DIR / "config" / f"{algorithm_name}.json"

    watermark = AutoWatermark.load(
        algorithm_name,
        algorithm_config=str(config_path),
        transformers_config=transformers_config,
    )

    total_start_time = time.time()

    output_dir = BASE_DIR / "results" / algorithm_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    with tqdm(total=len(samples), desc=f"Running {algorithm_name}", ncols=100) as pbar:
        for sample in samples:
            dataset_name = sample["dataset"]
            sample_idx = sample["sample_idx"]
            prompt = sample["prompt"]
            natural_text = merge_prompt_and_continuation(prompt, sample["natural_text"])

            sample_dir = output_dir / dataset_name / f"sample_{sample_idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Important:
            # Reset BREW state for each independent sample.
            # Without this, token_bit_log and codeword_queue may continue from
            # the previous sample, causing block indices to drift.
            reset_watermark_state_if_available(watermark)

            watermarked_text = watermark.generate_watermarked_text(prompt)

            detect_result_watermarked = watermark.detect_watermark(
                prompt,
                watermarked_text,
            )

            unwatermarked_text = watermark.generate_unwatermarked_text(prompt)

            detect_result_unwatermarked = watermark.detect_watermark(
                prompt,
                unwatermarked_text,
            )

            detect_result_natural = watermark.detect_watermark(
                prompt,
                natural_text,
            )

            save_text_result(
                sample_dir / "result_watermarked_text.txt",
                "LLM-generated watermarked text:",
                watermarked_text,
                detect_result_watermarked,
            )

            save_text_result(
                sample_dir / "result_unwatermarked_text.txt",
                "LLM-generated unwatermarked text:",
                unwatermarked_text,
                detect_result_unwatermarked,
            )

            save_text_result(
                sample_dir / "result_natural_text.txt",
                "Natural text:",
                natural_text,
                detect_result_natural,
            )

            with open(sample_dir / "detect_result_watermarked.json", "w", encoding="utf-8") as f:
                json.dump(detect_result_watermarked, f, ensure_ascii=False, indent=4)

            with open(sample_dir / "detect_result_unwatermarked.json", "w", encoding="utf-8") as f:
                json.dump(detect_result_unwatermarked, f, ensure_ascii=False, indent=4)

            with open(sample_dir / "detect_result_natural.json", "w", encoding="utf-8") as f:
                json.dump(detect_result_natural, f, ensure_ascii=False, indent=4)

            summary.append(
                {
                    "algorithm": algorithm_name,
                    "dataset": dataset_name,
                    "sample_idx": sample_idx,
                    "watermarked": compact_detection(detect_result_watermarked),
                    "unwatermarked": compact_detection(detect_result_unwatermarked),
                    "natural": compact_detection(detect_result_natural),
                }
            )

            pbar.update(1)

    total_elapsed_time = time.time() - total_start_time

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    with open(output_dir / "time_taken.txt", "w", encoding="utf-8") as f:
        f.write(f"Total time taken for {algorithm_name}: {total_elapsed_time:.4f} seconds\n")

    print(f"Finished {algorithm_name}. Total time: {total_elapsed_time:.4f} seconds")


def main():
    args = parse_args()

    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.download_model or not (Path(args.model_path) / "config.json").exists():
        download_model(args.model_path, args.hf_token)

    samples = load_samples(args.dataset, args.max_samples, args.sample_offset)

    transformers_config = load_transformers_config(
        args.model_path,
        device,
        args.max_new_tokens,
        args.min_length,
    )

    for algorithm_name in args.algorithms:
        run_algorithm(algorithm_name, transformers_config, samples)


if __name__ == "__main__":
    main()
