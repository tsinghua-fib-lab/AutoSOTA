from dataclasses import dataclass, field


@dataclass
class ConfigBenchResult:
    pretrain_task_type: str
    dataset_name: str
    prompt_type: str

    best_params: dict[str, float] = field(
        default_factory=lambda: {}
    )

    best_loss: float = float('inf')
    final_acc_mean: float = 0.
    final_acc_std: float = 0.
    final_f1_mean: float = 0.
    final_f1_std: float = 0.
    final_roc_mean: float = 0.
    final_roc_std: float = 0.
