import torch
import torch.nn.functional as F
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from train_utils import parse_args, get_dataset, load_model_and_tokenizer, compute_metrics, do_test, find_best_temperature

class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        self.loss_type = kwargs.pop('loss_type')
        super().__init__(**kwargs)  
    
    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        model.eval()
        with torch.no_grad():
            _, outputs = model(**inputs)

        dummy = torch.tensor([], dtype=outputs.dtype, device=outputs.device) # for triggering my compute_metrics()

        return (None, dummy, (inputs["advantages"], outputs))


    def pairwise_loss(self, scores, labels, margin=0.0, eps=1e-8):
        """
        scores: (B,)
        labels: (B,)
        """
        B = scores.size(0)

        # pairwise score diff: s_i - s_j
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (B, B)

        # pairwise label diff
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # (B, B)

        # valid pairs: label_i != label_j
        valid_mask = label_diff != 0

        # y_ij in {+1, -1}
        y_ij = torch.sign(label_diff)

        # weight by absolute label difference
        weight = torch.abs(label_diff)

        # RankNet logistic loss
        # log(1 + exp(-y * (s_i - s_j)))
        loss = F.softplus(-y_ij * (score_diff - margin))

        # apply mask & weight
        loss = loss * weight * valid_mask

        # normalize
        loss = loss.sum() / (weight * valid_mask).sum().clamp_min(eps)

        return loss

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        _, outputs = model(**inputs)

        advantages = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)   # (B,) → (B, 1)
        score = torch.tanh(outputs)

        # Cast advantages to match score dtype to avoid bf16/fp32 mismatch
        advantages = advantages.to(dtype=score.dtype)

        if self.loss_type == 'pointwise':
            loss = F.mse_loss(score, advantages)
        elif self.loss_type == 'pairwise':
            loss = self.pairwise_loss(score, advantages)
        else:
            loss = F.mse_loss(score, advantages) + self.pairwise_loss(score, advantages)

        return (loss, outputs) if return_outputs else loss

def main():
    model_args, data_args, training_args = parse_args()

    # Prepare model and data
    model, tokenizer = load_model_and_tokenizer(**model_args)
    train_dataset, test_dataset = get_dataset(**data_args, tokenizer=tokenizer)
    train_eval = train_dataset.train_test_split(test_size=data_args['eval_ratio'], seed=data_args['seed'])
    data_collator = DataCollatorWithPadding(tokenizer)

    loss_type = training_args.pop('loss_type')

    # Define Training Arguments
    training_args = TrainingArguments(
        **training_args,
        remove_unused_columns=False,
        metric_for_best_model="auc_norm",
        greater_is_better=True,
        eval_strategy="steps",
        report_to='none',
        load_best_model_at_end=True
    )

    # Instantiate the Rankscore Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_eval["train"],
        eval_dataset=train_eval["test"],
        compute_metrics=compute_metrics,
        loss_type=loss_type
    )

    # Start Training
    trainer.train()

    # Find best temperature on validation set
    device = next(trainer.model.parameters()).device
    best_tau = find_best_temperature(trainer.model, train_eval["test"], data_collator, device=device)
    
    # Do test
    print("\n--- Starting Testing ---")
    do_test(test_dataset, data_collator, trainer.model, training_args.output_dir, model_args['slm_name'], model_args['llm_names'], temperature=best_tau)

if __name__ == "__main__":
    main()