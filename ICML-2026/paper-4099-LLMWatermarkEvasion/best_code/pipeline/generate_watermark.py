import os
import gc
import json
import torch
import argparse
from tqdm import tqdm
import sys

sys.path.append(".")
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    for algorithm in args.algorithms:
        print(f"Processing algorithm: {algorithm}")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        transformers_config = TransformersConfig(
            model=model,
            device=device,
            tokenizer=tokenizer,
            vocab_size=model.config.vocab_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            no_repeat_ngram_size=4,
        )

        # Load watermark
        myWatermark = AutoWatermark.load(
            algorithm,
            algorithm_config=f"config/{algorithm}.json",
            transformers_config=transformers_config,
        )

        # Input/output paths
        input_path = args.input_path
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{algorithm}_response.json")

        with open(input_path, "r") as f:
            lines = f.readlines()[:args.num_data]

        # Determine how many lines already written
        if os.path.exists(output_path):
            with open(output_path, "r") as out_f:
                existing_lines = sum(1 for _ in out_f)
        else:
            existing_lines = 0

        with open(output_path, "a") as out_f:
            for i, line in enumerate(
                tqdm(lines, desc=f"Processing {algorithm}", unit="line")
            ):
                if i < existing_lines:
                    continue

                item = json.loads(line)

                prompt = item["prompt"]
                watermarked_text = myWatermark.generate_watermarked_text(prompt)
                
                unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

                response_item = {
                    "prompt": prompt,
                    "watermarked_text": watermarked_text,
                    "unwatermarked_text": unwatermarked_text,
                }
                out_f.write(json.dumps(response_item) + "\n")

                del item, response_item

        # Free memory
        del myWatermark, transformers_config, lines
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate watermarked/unwatermarked text with different algorithms."
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "SIR",
            "KGW",
            "Unigram",
            "UPV",
            "EWD",
            "DIP",
            "EXP",
        ],
        help="List of watermarking algorithms to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained language model",
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input .jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output JSON files",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0",
        help="Which GPU(s) to make visible",
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="Which CUDA device to use (e.g., 0)"
    )

    parser.add_argument(
        "--max_new_tokens", type=int, default=230, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--num_data", type=int, default=500, help="Number of input samples to watermark"
    )

    args = parser.parse_args()
    main(args)
