"""
LoRA-based SFT fine-tuning from the SDFT checkpoint.
Simple script to continue training with LoRA adapters.
"""
import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/models/sdft-science")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/science_data/train_data")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def format_sft_example(example, tokenizer):
    """Format a chemistry example for SFT training."""
    messages = example["messages"]
    output_text = example["output_text"]

    # Build the full conversation with the answer
    full_messages = messages + [{"role": "assistant", "content": output_text}]

    # Apply chat template
    text = tokenizer.apply_chat_template(full_messages, tokenize=False)
    return {"text": text}


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    print(f"Loading data from {args.data_path}")
    cache_path = f"/autosota_cache/datasets/science_data/train_data"
    if os.path.exists(cache_path):
        args.data_path = cache_path
    dataset = load_from_disk(args.data_path)

    print(f"Formatting {len(dataset)} examples")
    dataset = dataset.map(
        lambda x: format_sft_example(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting examples",
    )

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    dataset = dataset.map(tokenize_fn, remove_columns=["text"], desc="Tokenizing")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=100,
        bf16=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"Starting training: {args.num_epochs} epoch(s), {len(dataset)} examples")
    trainer.train()

    # Save final model (adapter)
    adapter_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    # Merge and save full model for eval
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_path = os.path.join(args.output_dir, "merged_model")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")


if __name__ == "__main__":
    main()
