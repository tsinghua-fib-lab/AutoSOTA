from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class DataTrainingArguments:
    path: Optional[str] = field(
        default='/data/ablation_study/data'
    )
    max_length: int = field(default=1024)
    test_save_dir: Optional[str] = field(
        default='./va_evaluation',
    )
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    checkpoint_path: Optional[str] = field(
        default="/data/zyq/va_embedding",
        metadata={
            "help": "The place to load the trained model."
        },
    )

@dataclass
class TrainingArguments:
    local_rank: Optional[int] = field(default=0)