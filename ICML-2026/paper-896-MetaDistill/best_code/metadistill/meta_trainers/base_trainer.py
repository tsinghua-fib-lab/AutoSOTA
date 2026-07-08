import os
import warnings
from tasks import bbob, cec, ef
from torch.utils.tensorboard import SummaryWriter


FUNCTION_REGISTRY = {
    "bbob": bbob,
    "cec": cec,
    "ef": ef,
}


class BaseTrainer:
    def __init__(self, config):
        # meta-trainer's basic settings
        self.expname = config["expname"]
        self.n_epochs = config["n_epochs"]

        # inner bbo's basic settings
        self.batchsize = config["batchsize"]
        self.popsize = config["popsize"]
        self.problemdim = config["problemdim"]
        self.n_generations = config["n_generations"]

        # saving paths
        self.ckpt_saving_path = f"checkpoints/generated/{self.expname}"
        self.log_saving_path = f"tb_logs/{self.expname}"
        self.data_saving_path = f"data/train/{self.expname}"
        for path in [
            self.ckpt_saving_path,
            self.log_saving_path,
            self.data_saving_path,
        ]:
            if os.path.exists(path) and len(os.listdir(path)) > 0:
                raise ValueError(
                    f"Experiment {self.expname} has existed (non-empty dir {path})."
                )
            os.makedirs(path, exist_ok=True)

        self.logger = SummaryWriter(log_dir=self.log_saving_path)

        # Training set
        self.training_set = []
        for _tf in config["training_set"].keys():
            tf = _tf.lower()
            pre = config["training_set"][_tf].get("pre", "")
            if tf not in ["bbob", "cec", "ef"]:
                warnings.warn(f"{tf} is not a valid training/evaluating function set.")
                continue
            function_module = FUNCTION_REGISTRY[tf]
            self.training_set += [
                function_module.FUNCTIONS[f"{pre}{i}"]
                for i in range(
                    config["training_set"][tf]["start"],
                    config["training_set"][tf]["end"] + 1,
                )
            ]

    def _print_config(self, ext_configs: dict = {}):
        """
        Print trainer configuration.
        """
        settings = {
            "Experiment name": self.expname,
            "Epochs": self.n_epochs,
            "Inner optimizer population shape": (
                self.batchsize,
                self.popsize,
                self.problemdim,
            ),
            "Inner generations": self.n_generations,
            "Checkpoint saving path": self.ckpt_saving_path,
            "Log saving path": self.log_saving_path,
            "Data saving path": self.data_saving_path,
            "Training set scale": len(self.training_set),
        }
        settings.update(ext_configs)

        print("========================Config========================")
        for k, v in settings.items():
            print(f"{k}: {v}")

    def train(self):
        """
        Implement training logic.
        """
        raise NotImplementedError

    def log(self):
        """
        Log training progress.
        """
        raise NotImplementedError
