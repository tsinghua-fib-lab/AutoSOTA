import torch
import numpy as np
import os
import sys
import json
from glob import glob

from tqdm.auto import tqdm
from collections import defaultdict
from typing import Dict, Union, Any, Optional, Tuple, List
import traceback

import transformers
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import EvalLoopOutput

from autoregltl import dataset
from autoregltl.losses import get_loss_fct
from autoregltl.ltl import trace_check


def compute_metrics(eval_preds):
    # These are numpy arrays
    logits, labels = eval_preds
    # For ignoring pad tokens
    weights = (labels != -100)
    outputs = np.argmax(logits, axis=-1)

    # Accuracy
    correct_predictions = (outputs == labels) * weights
    acc = np.sum(correct_predictions) / np.sum(weights)

    # Accuracy per sequence
    incorrect_predictions = (outputs != labels) * weights
    correct_sequences = 1.0 - np.minimum(np.array(1.0), np.sum(incorrect_predictions, axis=-1))
    acc_per_seq = np.sum(correct_sequences) / np.size(correct_sequences)

    return {
        "acc": acc,
        "acc_per_seq": acc_per_seq,
    }


class LTLTrainer(Trainer):
    def __init__(self, loss_fct, trace_eval_dataset, **kwargs):
        super().__init__(**kwargs)
        self.loss_fct = loss_fct
        self.trace_eval_dataset = trace_eval_dataset
        self.last_custom_eval = None
        self.gen_args = {}
    
    def log(self, logs: Dict[str, float]) -> None:
        if (s := getattr(self.loss_fct, "s", None)) is not None:
            logs['loss_s'] = float(s)
        super().log(logs)

    def custom_evaluation(self):
        if self.trace_eval_dataset is None:
            return {}
        predictions = self.model.generate_predictions(self.trace_eval_dataset, 100, self.gen_args, leave_tqdm=False)
        results = trace_check.evaluate_ltl(predictions, leave_tqdm=False)
        res = defaultdict(int)
        for result in results:
            res[result["result"]] += 1
        summary = {k: res.get(k, 0) for k in ["semantically correct", "exact match", "equivalent", "incorrect", "invalid", "timeout"]}
        output = {"eval_predict/" + k: v for k, v in summary.items()}
        self.last_custom_eval = {"summary": summary, "results": results}
        return output

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        try:
            output.metrics.update(self.custom_evaluation())
        except Exception:
            trace = traceback.format_exc()
            print("Error during custom evaluation:", trace)
            self.last_custom_eval = trace
        return output

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")
        if isinstance(self.last_custom_eval, dict):
            with open(os.path.join(output_dir, "semantic_eval.json"), "w") as f:
                json.dump(self.last_custom_eval, f, indent=4)
        elif isinstance(self.last_custom_eval, str):
            # last_custom_eval contains an error message
            with open(os.path.join(output_dir, "semantic_eval_error.txt"), "w") as f:
                f.write(self.last_custom_eval)


def save_command(args, param_count):
    """
    Save the executed command to model path.
    """
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "command-log.txt"), 'a') as f:
        f.write(f"Number of parameters: {param_count:_}\n")
        f.write("Arguments:\n")
        f.write(json.dumps(vars(args), indent=4))
        f.write("\n")
        if args.device == "cuda":
            f.write(f"Using CUDA device: {torch.cuda.get_device_name()}\n")
        else:
            f.write(f"Using device: {args.device}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"numpy version: {np.__version__}\n")
        f.write(f"torch version: {torch.__version__}\n")
        f.write(f"transformers version: {transformers.__version__}\n")
        f.write("\n")


def train(args, create_model, load_model, trainer_cls):
    if glob(os.path.join(args.model_path, "*")) and not args.resume and not args.eval:
        sys.exit("Model directory is not empty. Please specify a different output directory or use --resume to continue training.")
    elif args.resume and not glob(os.path.join(args.model_path, "checkpoint-*")):
        sys.exit("No checpoints found in the specified model directory. Please check the directory or start a new training run.")
    print("Training model:", args.model_path)

    vocab = dataset.get_dataset_vocab(args)

    if args.eval:
        model = load_model(args.model_path, torch.device(args.device))
        print(f"Loaded model from {args.model_path} for evaluation.")
    elif args.resume_from:
        model = load_model(args.resume_from, torch.device(args.device))
        print(f"Resumed model from {args.resume_from}")
    else:
        model = create_model(args, vocab)
        print("Created a new model.")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:_}")

    if args.dry:
        return

    dataset_kwargs = {
        "dataset_class": dataset.DecoderLTLDataset if args.decoder_only else dataset.EncDecLTLDataset,
        "vocab": vocab,
    }
    if args.tree_pos_enc:
        dataset_kwargs['tree_pos_enc'] = True
    train_dataset = None
    if args.eval:
        val_dataset = dataset.get_dataset(args, args.val_split, max_samples=args.val_max_samples, **dataset_kwargs)
    else:
        train_dataset = dataset.get_dataset(args, 'train', max_samples=args.train_max_samples, **dataset_kwargs)
        val_dataset = dataset.get_dataset(args, args.val_split, max_samples=args.val_max_samples, **dataset_kwargs)
    trace_eval_dataset = dataset.get_dataset(args, args.val_split, dataset.RawLTLDataset, max_samples=args.trace_max_samples)
    if args.decoder_only:
        data_collator = dataset.DecoderLTLCollator()
    elif args.tree_pos_enc:
        data_collator = dataset.EncDecLTLCollator(args.d_embed_enc)
    else:
        data_collator = dataset.EncDecLTLCollator()

    trainer = trainer_cls(
        loss_fct=get_loss_fct(args.loss_fct, model),
        trace_eval_dataset=trace_eval_dataset,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            # AdamW settings
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            max_grad_norm=args.max_grad_norm,
            # Epoch
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_acc_steps,
            # Seeding
            seed=args.seed if args.seed is not None else 42,  # 42 is the default anyway
            full_determinism=True if args.seed is not None else False,
            # misc
            output_dir=args.model_path,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.eval:
        print(trainer.evaluate())
    else:
        save_command(args, param_count)
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model()
        print("Saved model:", args.model_path)
