import lightning
import wandb
from datamodules import get_datamodule
from get_arch import get_architecture
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from pc_variants import get_pc_variant


# Define training function
def wandb_run_sweep():
    logger = WandbLogger(project="XXX", entity="XXX", mode="online")
    run_config = logger.experiment.config

    # Make sure to always generate the *exact* same datasets & batches
    lightning.seed_everything(run_config["seed"], workers=True)

    # Flatten 'config' dict into run_config
    run_config.update(run_config.get("config", {}))

    # Check whether this is the final training run, where validation is disabled for maximum training data
    FINAL_TRAINING_RUN = run_config["FINAL_TRAINING_RUN"]

    # 1: load dataset as Lightning DataModule
    batch_size = 64
    datamodule = get_datamodule(run_config["dataset"], FINAL_TRAINING_RUN)(batch_size)
    print("Training on", datamodule.dataset_name)

    # 2: Set up Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=(
            [EarlyStopping(monitor="val_acc", mode="max")] if not FINAL_TRAINING_RUN else None
        ),
        max_epochs=run_config["max_epochs"],
        inference_mode=False,  # inference_mode would interfere with the state backward pass
        limit_predict_batches=1,  # enable 1-batch prediction
    )

    # 3: Get architecture
    suffix = "-deep" if run_config["USE_DEEP_MLP"] else ""
    architecture = get_architecture(
        dataset=datamodule.dataset_name + suffix,
        use_CELoss=run_config["USE_CROSSENTROPY_INSTEAD_OF_MSE"],
    )

    # 4: Initiate model and train it
    PC_type = get_pc_variant(run_config["algorithm"], run_config["USE_CROSSENTROPY_INSTEAD_OF_MSE"])
    pc = PC_type(
        architecture,
        iters=run_config["iters"],
        e_lr=run_config["e_lr"],
        w_lr=run_config["w_lr"],
    )
    trainer.fit(pc, datamodule=datamodule)

    # 5: Test results
    trainer.test(pc, datamodule=datamodule)

    # 6: Release all CUDA memory that you can
    pc = None
    trainer = None
    lightning.pytorch.utilities.memory.garbage_collection_cuda()


def main():
    wandb.login()

    # Define the search space
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "dataset": {"value": "FashionMNIST"},
            "FINAL_TRAINING_RUN": {"value": False},
            "seed": {"value": 42},
            "algorithm": {"value": "SO"},
            "USE_DEEP_MLP": {"values": [True]},
            "USE_CROSSENTROPY_INSTEAD_OF_MSE": {"values": [False, True]},
            "e_lr": {"values": [0.1, 0.3]},
            "iters": {"values": [64, 256]},
            "w_lr": {"value": 1e-4},
            "max_epochs": {"value": 5},
        },
    }

    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="PredictiveCoding")
    wandb.agent(sweep_id, function=wandb_run_sweep)


if __name__ == "__main__":
    main()
