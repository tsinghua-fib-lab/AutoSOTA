import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-files", type=str, nargs='+')
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--layer", type=float, default=-2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--last-token-only", action='store_true')
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        output_hidden_states=True
    ).to(args.device)
    model.eval()

    sentences = []
    for input_file in args.input_files:
        df = pd.read_parquet(input_file)
        sentences += (df['prompt'] + df['predict']).tolist()

    all_embeddings = []
    all_input_ids = []

    total = len(sentences)
    print(f"Loaded {total} sentences")

    layer_id = None

    for start in tqdm(range(0, total, args.batch_size), desc="getting embedding"):
        batch_sents = sentences[start:start + args.batch_size]

        inputs = tokenizer(
            batch_sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        ).to(args.device)

        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        if layer_id is None: # need to get
            layer_num = len(hidden_states) - 1
            if args.layer < 0:
                layer_id = int(args.layer + layer_num)
            elif args.layer > 0 and args.layer < 1: # proportion
                layer_id = int(args.layer * layer_num)
            else:
                layer_id = int(args.layer)
            print(f"Get embeddings from layer {layer_id}")

        layer_hidden = hidden_states[layer_id + 1]  # (B, L, H)
        attention_mask = inputs["attention_mask"]   # (B, L)

        for i in range(layer_hidden.size(0)):
            valid_len = attention_mask[i].sum().item()

            if args.last_token_only:
                token_emb = layer_hidden[i, valid_len - 1].cpu()
            else:
                token_emb = layer_hidden[i, :valid_len].cpu()
            token_ids = inputs["input_ids"][i, :valid_len].cpu()

            all_embeddings.append(token_emb)
            all_input_ids.append(token_ids)

    torch.save(
        {
            "model_name": args.model_name,
            "layer_id": layer_id,
            "sentences": sentences,
            "input_ids": all_input_ids,
            "embeddings": all_embeddings,
        },
        args.output_file
    )

    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
