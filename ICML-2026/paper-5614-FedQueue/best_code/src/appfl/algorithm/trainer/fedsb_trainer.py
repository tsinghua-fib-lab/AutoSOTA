import os
import json
import torch
import wandb
import numpy as np
from datetime import datetime
from typing import Any, Optional, Dict
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from appfl.algorithm.trainer import BaseTrainer
from transformers import AdamW, TrainingArguments, Trainer
from fed_sb.utils.initialization_utils import find_and_initialize_grad
from fed_sb.utils.gradient_utils import estimate_and_process_grads_torch
from fed_sb.utils.data_utils import (
    load_and_preprocess_it,
    DataCollatorForSupervisedDataset,
)
from fed_sb.models import (
    create_model_tokenizer_it,
    create_peft_model_it,
    create_peft_FFA_model_it,
)


class FedSBTrainer(BaseTrainer):
    def __init__(
        self,
        train_configs: DictConfig,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        self.logger = logger
        self.train_configs = train_configs
        self._set_seed()

        self.train_configs.agg_type = self.train_configs.get("agg_type", "fed-sb")
        if self.train_configs.agg_type == "fed-sb":
            self.train_configs.lora_alpha = self.train_configs.lora_r

        self.run_dir = self._create_run_directory()
        self._init_wandb()

        self.model, self.tokenizer = create_model_tokenizer_it(self.train_configs)
        self.train_dataset = load_and_preprocess_it(
            tokenizer=self.tokenizer,
            args=self.train_configs,
        )
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

    def train(self):
        if self.train_configs.agg_type == "fed-sb":
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_configs.eg_bs,
                shuffle=True,
                collate_fn=self.data_collator,
            )

            total_training_steps = len(train_loader) * self.train_configs.epochs
            eff_lr = self.train_configs.lr / (
                self.train_configs.warmup_ratio * total_training_steps
            )
            named_grads = None
            named_grads = estimate_and_process_grads_torch(
                model=self.model,
                dataloader=train_loader,
                lr=eff_lr,
                num_samples=self.train_configs.num_samples,
            )

            model, lora_config = create_peft_model_it(self.model, self.train_configs)

            self.train_configs.reconstruction_configs.svd.rank = (
                self.train_configs.lora_r
            )

            adapter_name = "default"
            peft_config_dict = {adapter_name: lora_config}

            named_grads_new = {
                f"base_model.model.{k}": v for k, v in named_grads.items()
            }

            del model
            self.model, self.tokenizer = create_model_tokenizer_it(self.train_configs)

            if named_grads is not None:
                del named_grads

            # Create client dataset
            client_dataset = self.train_dataset.select(
                range(
                    self.train_configs.client_idx
                    * len(self.train_dataset)
                    // self.train_configs.num_splits,
                    (self.train_configs.client_idx + 1)
                    * len(self.train_dataset)
                    // self.train_configs.num_splits,
                )
            )

            data_module = dict(
                train_dataset=client_dataset, data_collator=self.data_collator
            )

            client_model, lora_config = create_peft_model_it(
                self.model, self.train_configs
            )

            adapter_name = "default"

            peft_config_dict = {adapter_name: lora_config}
            find_and_initialize_grad(
                model=client_model,
                peft_config=peft_config_dict,
                adapter_name=adapter_name,
                reconstr_type="svd",
                reconstruct_config=self.train_configs.reconstruction_configs,
                writer=None,
                named_grads=named_grads_new,
            )
            for param in client_model.parameters():
                param.data = param.data.contiguous()
            optimizer = AdamW(client_model.parameters(), lr=self.train_configs.lr)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.run_dir, "checkpoints"),
                num_train_epochs=self.train_configs.epochs,
                per_device_train_batch_size=self.train_configs.batch_size,
                learning_rate=self.train_configs.lr,
                weight_decay=0,
                warmup_ratio=self.train_configs.warmup_ratio,
                lr_scheduler_type=self.train_configs.scheduler,
                seed=self.train_configs.seed,
                report_to="wandb",
                gradient_accumulation_steps=32,
                save_strategy="no",
                bf16=True,
                tf32=False,
                fp16=False,
                logging_steps=1,
                logging_first_step=True,
                logging_dir=os.path.join(self.run_dir, "logs"),
            )

            # Save training arguments
            training_args_path = os.path.join(self.run_dir, "training_args.json")
            with open(training_args_path, "w") as f:
                json.dump(training_args.to_dict(), f, indent=4)

            # Create trainers
            trainer = Trainer(
                model=client_model,
                args=training_args,
                **data_module,
                optimizers=(optimizer, None),
            )

            # Save tokenizer
            self.tokenizer.save_pretrained(os.path.join(self.run_dir, "tokenizer"))

            client_model.config.use_cache = False
            trainer.train()

            final_model_path = os.path.join(
                self.run_dir, f"final_model_{self.train_configs.client_idx}"
            )  # Fixed path naming
            trainer.save_state()
            client_model.save_pretrained(final_model_path)
            print(f"Saved model {self.train_configs.client_idx} to {final_model_path}")
            self.saved_model_path = final_model_path

            # Load adapter weights from safetensors for direct transfer
            self.adapter_weights = self._load_adapter_safetensors(final_model_path)
            self.adapter_config = self._load_adapter_config(final_model_path)
        else:
            # Create client dataset
            client_dataset = self.train_dataset.select(
                range(
                    self.train_configs.client_idx
                    * len(self.train_dataset)
                    // self.train_configs.num_splits,
                    (self.train_configs.client_idx + 1)
                    * len(self.train_dataset)
                    // self.train_configs.num_splits,
                )
            )

            data_module = dict(
                train_dataset=client_dataset, data_collator=self.data_collator
            )
            # Create client model and optimizer
            if self.train_configs.agg_type == "ffa":
                client_model, lora_config = create_peft_FFA_model_it(
                    model, self.train_configs
                )
            else:
                client_model, lora_config = create_peft_model_it(
                    model, self.train_configs
                )

            optimizer = AdamW(client_model.parameters(), lr=self.train_configs.lr)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.run_dir, "checkpoints"),
                num_train_epochs=self.train_configs.epochs,
                per_device_train_batch_size=self.train_configs.batch_size,
                learning_rate=self.train_configs.lr,
                weight_decay=0,
                warmup_ratio=self.train_configs.warmup_ratio,
                lr_scheduler_type=self.train_configs.scheduler,
                seed=self.train_configs.seed,
                report_to="wandb",
                gradient_accumulation_steps=32,
                save_strategy="no",
                bf16=True,
                tf32=False,
                fp16=False,
                logging_steps=1,
                logging_first_step=True,
                logging_dir=os.path.join(self.run_dir, "logs"),
            )

            # Save training arguments
            training_args_path = os.path.join(self.run_dir, "training_args.json")
            with open(training_args_path, "w") as f:
                json.dump(training_args.to_dict(), f, indent=4)

            # Create trainers
            trainer = Trainer(
                model=model,
                args=training_args,
                **data_module,
                optimizers=(optimizer, None),
            )

            # Save tokenizer
            self.tokenizer.save_pretrained(os.path.join(self.run_dir, "tokenizer"))

            client_model.config.use_cache = False
            trainer.train()

            final_model_path = os.path.join(
                self.run_dir, f"final_model_{self.train_configs.client_idx}"
            )  # Fixed path naming
            trainer.save_state()
            client_model.save_pretrained(final_model_path)
            print(f"Saved model {self.train_configs.client_idx} to {final_model_path}")
            self.saved_model_path = final_model_path

            # Load adapter weights from safetensors for direct transfer
            self.adapter_weights = self._load_adapter_safetensors(final_model_path)
            self.adapter_config = self._load_adapter_config(final_model_path)

    def get_parameters(self):
        metadata = {
            "model_name": self.train_configs.model,
            "agg_type": self.train_configs.agg_type,
            "lora_r": self.train_configs.lora_r,
            "lora_alpha": self.train_configs.lora_alpha,
            "max_seq_length": self.train_configs.max_seq_length,
        }

        if self.train_configs.get("local_debug", False):
            return self.saved_model_path, metadata
        else:
            return {
                "adapter_weights": self.adapter_weights,
                "adapter_config": self.adapter_config,
            }, metadata

    def _set_seed(self):
        """
        Set the random seed for reproducibility.
        """
        seed = self.train_configs.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _create_run_directory(self):
        """
        Create a directory for saving the model and logs.
        """
        base_dir = self.train_configs.get("run_dir", "experiments")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.train_configs.model.split("/")[-1]
        run_name = (
            f"{model_name}__"
            + f"r{self.train_configs.lora_r}__"
            + f"lr{self.train_configs.lr}__"
            + f"train_{self.train_configs.dataset_split.replace('[:', '').replace(']', '')}"
        )
        run_dir = os.path.join(
            base_dir, model_name, self.train_configs.agg_type, f"{timestamp}_{run_name}"
        )
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        config_dict = OmegaConf.to_container(self.train_configs, resolve=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        return run_dir

    def _load_adapter_safetensors(
        self, final_model_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Load adapter weights from the saved safetensors file - exactly what the aggregator expects.
        This ensures 100% compatibility with the file-based approach.
        """
        adapter_path = os.path.join(final_model_path, "adapter_model.safetensors")

        if os.path.exists(adapter_path):
            return load_file(adapter_path, device="cpu")
        else:
            return {}

    def _load_adapter_config(self, model_path: str) -> Dict[str, Any]:
        """
        Extract adapter configuration from the saved adapter_config.json file.
        """
        adapter_config_path = os.path.join(model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                return json.load(f)
        else:
            return {}

    def _init_wandb(self):
        """
        Initialize Weights and Biases for logging.
        """
        self.enable_wandb = self.train_configs.get("enable_wandb", False)
        if self.enable_wandb:
            wandb_run = wandb.init(
                entity=self.train_configs.wandb_configs.get("entity", None),
                project=self.train_configs.wandb_configs.get("project", None),
                dir=os.path.join(self.run_dir, "logs"),
                name=self.train_configs.wandb_configs.get("exp_name", None),
                config=OmegaConf.to_container(self.train_configs, resolve=True),
            )
            with open(os.path.join(self.run_dir, "wandb_run_id.txt"), "w") as f:
                f.write(wandb_run.id)
