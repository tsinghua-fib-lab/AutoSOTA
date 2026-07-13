import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import pandas as pd
import torch

from datasets import load_from_disk, Audio
import evaluate

from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# -----------------------
# Helpers
# -----------------------

COMPUTE_LOSS_ON_LANG_TOKEN = False

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # inputs: already feature-extracted to "input_features"
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # labels: already tokenized to "labels"
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Do not compute loss on the language token (first token in labels after slicing)
        if not COMPUTE_LOSS_ON_LANG_TOKEN:
            labels[:, 0] = -100

        batch["labels"] = labels

        return batch


# -----------------------
# Dataset preparation
# -----------------------

def prepare_dataset_function(processor):
    """
    Returns two mapping functions:
      - prepare_audio: decode audio to arrays and compute input_features
      - prepare_labels: add per-example lang tokens and encode labels
    """

    def prepare_dataset(batch):
        audio = batch["audio"]

        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        processor.tokenizer.set_prefix_tokens(language=batch["lang"], task="transcribe") 
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids

        return batch

    return prepare_dataset


# -----------------------
# Metrics
# -----------------------

def make_compute_metrics_fn(tokenizer: WhisperTokenizer):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):  # older versions may return (logits,)
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id so we can decode
        label_ids = torch.tensor(label_ids)
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_ids = label_ids.detach().cpu().numpy()

        # Decode
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # print(pred_str)
        # print(label_str)

        wer = 100.0 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100.0 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


# -----------------------
# Main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper on a preprocessed dataset")
    p.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed data file")
    p.add_argument("--output_dir", type=str, default="./whisper-out", help="Where to save checkpoints/logs")
    p.add_argument("--model_id", type=str, default="openai/whisper-small", help="Whisper checkpoint (e.g., tiny, base, small, medium, large-v3)")
    p.add_argument("--predict_lang_token", action='store_true', help="Compute loss on lang token")
    p.add_argument("--default_language", type=str, default=None, help="Optional default language code (e.g., 'en'); per-example 'lang' still applied")
    p.add_argument("--wandb_project", type=str, default=None, help="WANDB project name (enables wandb if set)")
    p.add_argument("--wandb_entity", type=str, default=None, help="WANDB entity (optional)")
    p.add_argument("--run_name", type=str, default=None, help="Experiment/run name for logging")
    p.add_argument("--max_steps", type=int, default=None, help="Use max_steps OR num_train_epochs")
    p.add_argument("--num_train_epochs", type=float, default=3.0, help="Epochs if --max_steps not set")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=1000, help="If using step-based eval")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--generation_max_length", type=int, default=225)
    p.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Evaluation schedule")
    p.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"])
    p.add_argument("--logging_strategy", type=str, default="steps", choices=["steps", "epoch"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--push_to_hub", action="store_true", help="If set, push best model to HF Hub (requires auth)")
    return p.parse_args()

def freeze_whisper_layers(model, num_layers_to_freeze=4):
    for i in range(num_layers_to_freeze):
        for param in model.model.encoder.layers[i].parameters():
            param.requires_grad = False
    print(f"Froze first {num_layers_to_freeze} encoder layers.")

def main():
    global COMPUTE_LOSS_ON_LANG_TOKEN
    args = parse_args()
    COMPUTE_LOSS_ON_LANG_TOKEN = args.predict_lang_token

    # Optional WANDB
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name)
        # Store config
        wandb.config.update(vars(args))

    # --- Load splits

    ds = load_from_disk(args.data_dir)
    train_ds = ds["train"]
    val_ds = ds["valid"]
    test_ds = ds["test"]

    # --- Processor and model
    processor = WhisperProcessor.from_pretrained(args.model_id, task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    # Do not force a single language at generation time; let per-example labels guide training
    model.generation_config.forced_decoder_ids = None
    # For safety, set task (transcribe) but we won't force language
    model.generation_config.task = "transcribe"

    freeze_whisper_layers(model, num_layers_to_freeze=4)

    # --- Map functions
    prepare_dataset = prepare_dataset_function(processor)

    # Apply per-sample processing (parallelize with num_proc for speed)
    import multiprocessing
    num_proc = multiprocessing.cpu_count() // 2  # Use half CPUs to avoid overload
    train_ds = train_ds.map(prepare_dataset, num_proc=num_proc)
    val_ds = val_ds.map(prepare_dataset, num_proc=num_proc)
    test_ds = test_ds.map(prepare_dataset, num_proc=num_proc)

    # --- Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # --- Metrics
    compute_metrics = make_compute_metrics_fn(processor.tokenizer)

    # --- Training args
    # NOTE: Newer Transformers prefer eval_strategy/save_strategy/logging_strategy over deprecated evaluation_strategy.
    use_max_steps = args.max_steps is not None and args.max_steps > 0

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,

        # >>> ensure ints, not None
        # max_steps=args.max_steps if use_max_steps else 0, # NOTE: QUICK FIX BUT IDK WHY IT DOESNT WORK UNCOMMENTED
        num_train_epochs=args.num_train_epochs if not use_max_steps else 1.0,
        # <<<

        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        eval_strategy=args.eval_strategy,          # "steps" | "epoch" | "no"
        save_strategy=args.save_strategy,          # "steps" | "epoch"
        logging_strategy=args.logging_strategy,    # "steps" | "epoch"
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        logging_steps=args.logging_steps if args.logging_strategy == "steps" else None,
        report_to=(["wandb"] if args.wandb_project else ["none"]),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
    )
    

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,  # pass feature_extractor to avoid HF warnings
    )

    # --- Train
    trainer.train()
    
    # --- Final test evaluation
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    # Persist
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics_test.json"), "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    # Also log to wandb if enabled
    if args.wandb_project:
        import wandb
        wandb.log(test_metrics)

    # Save the final processor so downstream inference matches training
    processor.save_pretrained(args.output_dir)

    # Optionally push to hub (uses the training_args.push_to_hub flag too)
    if args.push_to_hub:
        trainer.push_to_hub()

    print("Done. Test metrics:", test_metrics)


if __name__ == "__main__":
    main()