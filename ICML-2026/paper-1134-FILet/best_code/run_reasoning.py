import argparse
import logging
import math
import os
import torch

import datasets
from datasets import load_dataset
import evaluate
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from models import modeling_llama
from tools import get_logger
import transformers
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from torch.optim import AdamW
from models import (
    configuration_llama,
)
from preprocess_functions import (
    data_to_sequence_arc,
    data_to_sequence_classification,
    data_to_sequence_four_choices,
    data_to_sequence_three_choices,
    data_to_sequence_two_choices,
)
from models import filet_layer
import peft
import json

import time
from multiprocessing import Pool
from SxSy_computation import FILet_init
logger = logging.getLogger(__name__)

task_to_keys = {
    "mmlu": ("question", "choices"),
    "hellaswag": ("ctx", "endings"),
    "obqa": ("inputs", None),
    "arcc": ("question", "choices"),
    "arce": ("question", "choices"),
    "piqa": ("goal", "sol1", "sol2"),
    "winogrande": ("sentence", "option1", "option2"),
    "siqa": ("question", "context"),
    "boolq": ("passage", "question")
}

task_to_info = {
    "mmlu": ("cais/mmlu", "all", "auxiliary_train", "test", "answer"),
    "hellaswag": ("AlekseyKorshuk/hellaswag", None, "train", "validation", "label"),
    "obqa": ("rbelanec/openbookqa", None, "train", "test", "targets"),
    "arcc": ("ibragim-bad/arc_challenge", None, "train", "test", "answerKey"),
    "arce": ("ibragim-bad/arc_easy", None, "train", "test", "answerKey"),
    "piqa": ("Narpear/piqa", None, "train", "validation", "label"),
    "winogrande": ("jaypyon/Winogrande", None, "train", "validation", "answer"),
    "siqa": ("lighteval/siqa", None, "train", "validation", "label"),
    "boolq": ("google/boolq", None, "train", "validation", "answer")
}

metrics = {
    "mmlu": "accuracy",
    "hellaswag": "accuracy",
    "obqa": "accuracy",
    "arcc": "accuracy",
    "arce": "accuracy",
    "piqa": "accuracy",
    "winogrande": "accuracy",
    "siqa": "accuracy",
    "boolq": "accuracy",
}

mode_mapping = {
    "filet": "FILET",
}

package_mapping = {
    "filet": filet_layer,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="A special token to separate different versions of results.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--baseline",
        type=str,
    )
    parser.add_argument(
        "--eval_steps", type=int, default=2000, help=""
    )
    parser.add_argument(
        "--mode", type=str, default="default"
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="A dir to store log files."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="A dir to store log intermediate outputs."
    )
    parser.add_argument(
        "--scores_dir", type=str, default=None, help="A dir to store score files."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="A dir to store models/datasets from huggingface."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--data_num",
        type=int,
        default=320,
        help="Data steps.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_plus_lambda",
        type=float,
        default=1.0,
        help="LoRA+ learning rate ratio (lr_B = lr * lambda, lr_A = lr). lambda=1 means standard LoRA.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
         "--sxsy_on_gpu", type=bool, default=True, help="Whether to store Sx and Sy on GPU."
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    # parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=114514, help="Random seed")
    parser.add_argument("--time", type=int)
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def main():
    args = parse_args()
    logger = get_logger(f"{args.task_name}_{args.model_name_or_path.split('/')[-1]}_{args.mode}_r{args.lora_rank}_seed{args.seed}_alpha{args.lora_alpha}_data{args.data_num}_{args.version}", args.log_dir)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    logger.info(accelerator.state)
    

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    raw_datasets = load_dataset(task_to_info[args.task_name][0], task_to_info[args.task_name][1], cache_dir=args.cache_dir)
    
    
    if args.task_name in ["arcc", "arce"]:
        num_labels = 0
        for split in ["train", "test"]:
            for i in range(len(raw_datasets[split])):
                choices = raw_datasets[split][i][task_to_keys[args.task_name][1]]["label"]
                if len(choices) > num_labels:
                    num_labels = len(choices)
    elif args.task_name in ["hellaswag", "obqa"]:
        num_labels = 4
        label_list = ["A", "B", "C", "D"]
        label_to_id = {l: i for i, l in enumerate(label_list)}
    elif args.task_name in ["siqa"]:
        label_list = list(set(raw_datasets[task_to_info[args.task_name][2]][task_to_info[args.task_name][4]]))
        label_to_id = {l: i for i, l in enumerate(label_list)}
        num_labels = len(label_list)
    elif args.task_name in ["piqa", "winogrande"]:
        if args.task_name == "winogrande":
            raw_datasets.pop("test")
        label_list = list(set(raw_datasets[task_to_info[args.task_name][2]][task_to_info[args.task_name][4]]))
        label_to_id = {l: i for i, l in enumerate(label_list)}
        num_labels = len(label_list)
    elif args.task_name in ["boolq"]:
        label_to_id = {False: 0, True: 1}
        num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    if tokenizer.pad_token == None:
        
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    # Preprocessing the datasets
    
    if len(task_to_keys[args.task_name]) == 2:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    elif len(task_to_keys[args.task_name]) == 3:
        sentence1_key, choice1_key, choice2_key = task_to_keys[args.task_name]
    

    def preprocess_function(examples):
        
        if args.task_name in ["arcc", "arce"]:
            sentences, labels = data_to_sequence_arc(examples, sentence1_key, sentence2_key, label_key=task_to_info[args.task_name][4], args=args)
        elif args.task_name in ["hellaswag", "obqa"]:
            sentences = data_to_sequence_four_choices(examples, sentence1_key, sentence2_key, args)
        elif args.task_name in ["siqa"]:
            sentences = data_to_sequence_three_choices(examples, sentence1_key, sentence2_key, args)
        elif args.task_name in ["piqa", "winogrande"]:
            sentences = data_to_sequence_two_choices(examples, sentence1_key, choice1_key=task_to_keys[args.task_name][1], choice2_key=task_to_keys[args.task_name][2], args=args)
        elif args.task_name in ["boolq"]:
            sentences = data_to_sequence_classification(examples, sentence1_key, sentence2_key, args)

        result = tokenizer(*sentences, padding=True, truncation=True)
        if args.task_name in ["arcc", "arce"]:
            result["labels"] = labels
        elif args.task_name not in ["arcc", "arce"]:
            
            if examples[task_to_info[args.task_name][4]][0] not in label_to_id:

                result["labels"] = [int(each) for each in examples[task_to_info[args.task_name][4]]]
            else:
                result["labels"] = [label_to_id[l] for l in examples[task_to_info[args.task_name][4]]]

        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets[task_to_info[args.task_name][3]].column_names
    )
    
    config = configuration_llama.LlamaConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )


    set_seed(args.seed)
    model = modeling_llama.LlamaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    LoRAConfig = package_mapping[args.mode].LoraConfig(
        peft_type=mode_mapping[args.mode],
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    )

    model.add_adapter(LoRAConfig, adapter_name=mode_mapping[args.mode])

    num_param = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            sub_num_param = 1
            for dim in param.shape:
                sub_num_param *= dim  
            num_param += sub_num_param
    
    logger.info("Number of Trainable Parameters: %d"%(int(num_param))) 
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(dtype=torch.bfloat16)

    train_dataset = processed_datasets[task_to_info[args.task_name][2]]
    eval_dataset_name = task_to_info[args.task_name][3]
    eval_dataset = processed_datasets[eval_dataset_name]


    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # LoRA+ decoupled learning rates: lr_A = lr, lr_B = lr * lambda
    no_decay = ["bias", "LayerNorm.weight"]
    lora_plus_lambda = args.lora_plus_lambda if hasattr(args, "lora_plus_lambda") else 1.0
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "lora_A" not in n and "lora_B" not in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "lora_A" in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "lora_B" in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * lora_plus_lambda,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "lora_A" not in n and "lora_B" not in n],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "lora_A" in n],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "lora_B" in n],
            "weight_decay": 0.0,
            "lr": args.learning_rate * lora_plus_lambda,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model = model.to(accelerator.device)
    
    if args.mode == "filet":

        FILet_init(model, target_modules=LoRAConfig.target_modules, 
                    dataloader=train_dataloader, steps=args.data_num//args.per_device_train_batch_size, 
                    adapter_name=mode_mapping[args.mode], gpu=args.sxsy_on_gpu)


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    


    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("accuracy")
    
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Run {args.time} hours.")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    max_score = 0.
    loss_curve = []
    val = True
    start_time = time.time()
    for epoch in range(args.num_train_epochs):
        model.train()
        progress_bar.set_description(f"Training epoch[{epoch+1}/{args.num_train_epochs}]")
        for step, batch in enumerate(train_dataloader):
            
            outputs = model(**batch)
            
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss_curve.append(loss.item())
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.item()
                )
                completed_steps += 1
                val = True
            else:
                val = False

            if completed_steps >= args.max_train_steps:
                break
            if completed_steps % args.eval_steps == 0 and val:
                model.eval()
                eval_results = []
                with torch.no_grad():
                    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Evaluating"):
                        outputs = model(**batch)
                        
                        predictions = outputs.logits.argmax(dim=-1)
                        metric.add_batch(
                            predictions=accelerator.gather(predictions),
                            references=accelerator.gather(batch["labels"]),
                        )

                        preds = accelerator.gather(predictions).cpu().numpy().tolist()
                        labels = accelerator.gather(batch["labels"]).cpu().numpy().tolist()
                        if "input_ids" in batch:
                            input_ids = accelerator.gather(batch["input_ids"]).cpu().numpy().tolist()
                        else:
                            input_ids = ["N/A"] * len(preds)

                        for inp, lbl, pred in zip(input_ids, labels, preds):
                            eval_results.append({
                                "input": tokenizer.decode(inp), 
                                "label": lbl,
                                "prediction": pred
                            })
                    eval_metric = metric.compute()
                    logger.info(f"epoch {epoch}: {eval_metric}")
                    if eval_metric[metrics[args.task_name]] > max_score:
                        max_score = eval_metric[metrics[args.task_name]]
                        # Save best checkpoint to prevent overfitting
                        best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")
                        unwrapped = accelerator.unwrap_model(model)
                        torch.save(unwrapped.state_dict(), best_ckpt_path)
                        logger.info(f"Saved best checkpoint with accuracy {max_score:.4f}")

                    save_path = os.path.join(args.output_dir, f"{args.task_name}_{args.model_name_or_path.split('/')[-1]}_{args.mode}_r{args.lora_rank}_seed{args.seed}_alpha{args.lora_alpha}_data{args.data_num}_{args.version}_step{completed_steps}.json")
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(eval_results, f, ensure_ascii=False, indent=2)
                model.train()
                if args.time > 0 and (time.time() - start_time) / 3600 >= args.time:
                    exit()
        
    # Load best checkpoint if saved
    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pt")
    if os.path.exists(best_ckpt_path):
        logger.info(f"Loading best checkpoint from {best_ckpt_path}")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(torch.load(best_ckpt_path, map_location=accelerator.device))
    logger.info(f"FINAL_MAX_SCORE: {max_score}")
    print(f"FINAL_MAX_SCORE: {max_score}")


if __name__ == "__main__":
    main()