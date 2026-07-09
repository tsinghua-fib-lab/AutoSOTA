# train/

Policy training procedures: SFT, DPO, and MARGE.

## Key files

- **seq2seq_sft_trainer.py** — `Seq2SeqSFTTrainer`: extends TRL's SFTTrainer with custom generation metrics (ROUGE/BLEU). `S3Callback`: uploads checkpoints to S3 during training.
- **dpo.py** — Direct Preference Optimization training. Loads preference pairs (chosen vs rejected), trains with DPO loss. Features robust CUDA initialization with retry backoff and WandB logging.
- **marge.py** — MARGE (margin-based) preference optimization. Alternative to DPO with a different objective function. Same interface pattern.
- **marge_trainer.py** — Core MARGE training implementation with `MargeTrainer` and `MargeConfig`.
- **pref_tuning_trainer.py** — `DPOTrainerWithLogging`: enhanced DPO trainer with detailed metric logging.
- **data_collators.py** — `DataCollatorForCompletionOnlyLM`: vendored from TRL, masks input tokens so the loss is computed only on completion tokens during SFT.

## Training data formats

- SFT: `{"input": prompt, "target": sequence}`
- DPO/MARGE: `{"prompt": seed, "chosen": preferred_sequence, "rejected": dispreferred_sequence}`
