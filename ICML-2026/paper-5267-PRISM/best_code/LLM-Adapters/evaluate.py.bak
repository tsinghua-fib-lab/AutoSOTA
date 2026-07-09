import copy
import json
import os
import re
import sys
import argparse
from pathlib import Path

import fire
import torch

# Optional local PEFT path (kept for compatibility with original repo style)
_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR / "peft" / "src"))

from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer


# ---------------------------
# Device selection
# ---------------------------
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass


# ---------------------------
# Paths (robust to CWD)
# ---------------------------
DATASET_DIR = _THIS_DIR / "dataset"
EXPERIMENT_DIR = _THIS_DIR / "experiment"
TRAINED_MODELS_DIR = _THIS_DIR / "trained_models"


# ---------------------------
# Helpers
# ---------------------------
def _select_dtype_for_device() -> torch.dtype:
    """Prefer bf16 on CUDA when available for improved numerical stability."""
    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _resolve_maybe_under_script_dir(p: str) -> str:
    """
    If `p` is a local relative path that doesn't exist from current CWD,
    but exists under this script directory (LLM-Adapters), resolve it.

    This allows notebooks at repo root to pass:
      lora_weights="trained_models/..."
    without chdir into LLM-Adapters.
    """
    if p is None:
        return p
    pp = Path(p)
    if pp.exists():
        return str(pp)
    alt = _THIS_DIR / pp
    if alt.exists():
        return str(alt)
    return p


def _dataset_dir_name(name: str) -> str:
    if name is None:
        return ""
    low = name.strip().lower()
    if low == "gsm8k":
        return "gsm8k"
    if low == "aqua":
        return "AQuA"
    if low == "svamp":
        return "SVAMP"
    if low == "mawps":
        return "mawps"
    # Legacy datasets keep original casing
    return name.strip()


def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_prompt(instruction: str, input: str = None) -> str:
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def load_data(args) -> list:
    dname = _dataset_dir_name(args.dataset)
    file_path = DATASET_DIR / dname / "test.json"
    if not file_path.exists():
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    return json.load(open(file_path, "r"))


def create_batch(dataset, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    batches = []
    num_batch = len(dataset) // batch_size if len(dataset) % batch_size == 0 else len(dataset) // batch_size + 1
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "AddSub",
            "MultiArith",
            "SingleEq",
            "gsm8k",
            "GSM8K",
            "AQuA",
            "aqua",
            "SVAMP",
            "svamp",
            "mawps",
            "MAWPS",
        ],
        required=True,
    )
    parser.add_argument("--model", choices=["LLaMA-7B", "BLOOM-7B", "GPT-j-6B", "other"], required=True)
    parser.add_argument("--adapter", choices=["LoRA", "AdapterP", "AdapterH", "Parallel", "Prefix"], required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_8bit", action="store_true", default=False)

    # Generation controls (safe defaults; we force do_sample=False)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--log_every", type=int, default=0)  # 0 = silent per-example

    return parser.parse_args()


def _load_peft_adapter(base_model, lora_weights: str, dtype: torch.dtype, device_map):
    """Handle PEFT signature differences across versions."""
    try:
        return PeftModel.from_pretrained(base_model, lora_weights, torch_dtype=dtype, device_map=device_map)
    except TypeError:
        try:
            return PeftModel.from_pretrained(base_model, lora_weights, dtype=dtype, device_map=device_map)
        except TypeError:
            try:
                return PeftModel.from_pretrained(base_model, lora_weights, torch_dtype=dtype)
            except TypeError:
                return PeftModel.from_pretrained(base_model, lora_weights)


def load_model(args) -> tuple:
    base_model = args.base_model
    if not base_model:
        raise ValueError(f"can not find base model name by the value: {args.model}")

    lora_weights = _resolve_maybe_under_script_dir(args.lora_weights)
    if not lora_weights:
        raise ValueError(f"can not find lora weight, the value is: {lora_weights}")

    load_8bit = args.load_8bit
    dtype = _select_dtype_for_device()

    # Tokenizer
    if args.model == "LLaMA-7B":
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0

    # Base model
    if device == "cuda":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        model = _load_peft_adapter(model, lora_weights, dtype=dtype, device_map="auto")
    elif device == "mps":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                dtype=dtype,
                trust_remote_code=True,
            )
        model = _load_peft_adapter(model, lora_weights, dtype=dtype, device_map={"": device})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = _load_peft_adapter(model, lora_weights, dtype=dtype, device_map={"": device})

        if not load_8bit and dtype in (torch.float16, torch.bfloat16):
            model = model.to(dtype)

    # Align special token ids
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model.config, "bos_token_id", None) is None and tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    # Force deterministic decoding (avoid beam-sampling multinomial CUDA asserts)
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.do_sample = False
        except Exception:
            pass

    model.eval()
    return tokenizer, model


def _extract_answer_number(dataset_name: str, sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    try:
        return float(pred[-1])
    except Exception:
        return float("inf")


def _extract_answer_letter(sentence: str) -> str:
    # For AQuA: choose the last standalone A-E letter (more robust against "Area", etc.)
    # Example: "Answer: C" -> C
    matches = re.findall(r"(?:^|[^A-Z])([A-E])(?:$|[^A-Z])", sentence.upper())
    if not matches:
        # fallback: any A-E letter
        m2 = re.findall(r"[A-E]", sentence.upper())
        return m2[-1] if m2 else ""
    return matches[-1]


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    share_gradio: bool = False,
):
    # Keep original repo style: arguments are taken from sys.argv via argparse
    args = parse_args()

    create_dir(EXPERIMENT_DIR)

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)

    tokenizer, model = load_model(args)

    # Generation configuration (explicitly set critical fields to avoid model defaults overriding)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else getattr(model.config, "bos_token_id", None)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    pad_id = tokenizer.pad_token_id

    # NOTE: do_sample=False is mandatory for stability with Gemma in this repo setup.
    gen_cfg = GenerationConfig(
        do_sample=False,
        num_beams=max(1, int(args.num_beams)),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    # Stabilizers (supported in many versions; ignored if not present)
    if hasattr(gen_cfg, "remove_invalid_values"):
        gen_cfg.remove_invalid_values = True
    if hasattr(gen_cfg, "renormalize_logits"):
        gen_cfg.renormalize_logits = True

    def evaluate_batch(instructions, input=None):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_input_length),
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
                max_new_tokens=int(args.max_new_tokens),
                return_dict_in_generate=True,
                output_scores=False,
            )

        sequences = out.sequences
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        outputs = []
        for text in decoded:
            if "### Response:" in text:
                outputs.append(text.split("### Response:", 1)[1].strip())
            else:
                outputs.append(text.strip())
        return outputs

    # Determine dataset & output file naming
    ds_dir = _dataset_dir_name(args.dataset)
    save_file = EXPERIMENT_DIR / f"{args.model}-{args.adapter}-{ds_dir}.json"

    total_batches = len(batches)
    correct = 0
    seen = 0
    miss = 0.001  # numeric tolerance used by original repo scripts
    output_data = []
    pbar = tqdm(total=total_batches, desc=f"eval {args.dataset}")

    for bidx, batch in enumerate(batches):
        instructions = [data.get("instruction", "") for data in batch]
        outputs = evaluate_batch(instructions)

        for data, output in zip(batch, outputs):
            label = data.get("answer")
            flag = False

            if args.dataset.strip().lower() == "aqua":
                pred = _extract_answer_letter(output)
                lab = str(label).strip().upper() if label is not None else ""
                if lab == pred:
                    correct += 1
                    flag = True
            else:
                # labels may be numeric or numeric strings
                try:
                    lab = float(label)
                except Exception:
                    lab = float("inf")
                pred = _extract_answer_number(args.dataset, output)
                if abs(lab - pred) <= miss:
                    correct += 1
                    flag = True

            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = pred
            new_data["flag"] = flag
            output_data.append(new_data)

            seen += 1
            if args.log_every and (seen % int(args.log_every) == 0):
                print(f"[{args.dataset}] seen={seen} acc={correct/seen:.4f}")

        # Write after each batch (matches repo style)
        with open(save_file, "w+") as f:
            json.dump(output_data, f, indent=4)

        pbar.update(1)

    pbar.close()
    final_acc = correct / max(1, seen)
    print(f"\nFinal accuracy: {correct}/{seen} = {final_acc:.4f}")
    print("Saved:", str(save_file))


if __name__ == "__main__":
    fire.Fire(main)
