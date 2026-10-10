from __future__ import annotations

from meisenmeister.training.trainers.mm_trainer import mmTrainer


class mmTrainer_Debug(mmTrainer):
    ARCHITECTURE_NAME = "ResNet3D18"

    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir,
        preprocessed_dataset_dir,
        results_dir,
        architecture_name: str | None = None,
        num_workers: int | None = None,
        continue_training: bool = False,
        weights_path=None,
        experiment_postfix: str | None = None,
        compile_enabled: bool = True,
        grad_cam_enabled: bool = False,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            fold=fold,
            dataset_dir=dataset_dir,
            preprocessed_dataset_dir=preprocessed_dataset_dir,
            results_dir=results_dir,
            architecture_name=architecture_name or self.ARCHITECTURE_NAME,
            num_epochs=2,
            batch_size=1,
            num_workers=0 if num_workers is None else num_workers,
            shuffle=True,
            continue_training=continue_training,
            weights_path=weights_path,
            experiment_postfix=experiment_postfix,
            compile_enabled=compile_enabled,
            grad_cam_enabled=grad_cam_enabled,
        )


class NotATrainer:
    pass
