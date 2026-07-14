from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class DataTrainingArguments:
    path: Optional[str] = field(
        default='/data/ablation_study/data'
    )
    max_length: int = field(default=512)
    
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    save_dir: Optional[str] = field(
        default="/data/zyq/va_embedding",
        metadata={
            "help": "The place to save the trained model."
        },
    )

@dataclass
class TrainingArguments:
    weight_decay: float = field(default=1e-4)
    lr: float = field(default=1e-4)
    deepspeed:Optional[str] = field(
        default="./va_deepspeed_stage.json",
        metadata={
            "help": "Deepspeed file path."
        }
    )
    local_rank: Optional[int] = field(default=0)
    train_epoch: Optional[int] = field(default=2)
    train_batch_size: Optional[int] = field(default=1024)
    dropout_rate: float = field(default=0.0)