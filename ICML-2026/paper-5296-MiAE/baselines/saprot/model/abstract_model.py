import torch
import abc
import os

import pytorch_lightning as pl
from utils.lr_scheduler import Esm2LRScheduler
from torch import distributed as dist


class AbstractModel(pl.LightningModule):
    def __init__(self,
                 lr_scheduler_kwargs: dict = None,
                 optimizer_kwargs: dict = None,
                 save_path: str = None,
                 from_checkpoint: str = None,
                 load_prev_scheduler: bool = False,
                 save_weights_only: bool = True,):
        """

        Args:
            lr_scheduler: Kwargs for lr_scheduler
            optimizer_kwargs: Kwargs for optimizer_kwargs
            save_path: Save trained model
            from_checkpoint: Load model from checkpoint
            load_prev_scheduler: Whether load previous scheduler from save_path
            load_strict: Whether load model strictly
            save_weights_only: Whether save only weights or also optimizer and lr_scheduler
            
        """
        super().__init__()
        self.initialize_model()
        
        self.metrics = {}
        for stage in ["train", "valid", "test"]:
            stage_metrics = self.initialize_metrics(stage)
            # Rigister metrics as attributes
            for metric_name, metric in stage_metrics.items():
                setattr(self, metric_name, metric)
                
            self.metrics[stage] = stage_metrics

        self.lr_scheduler_kwargs = {"init_lr": 0} if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.init_optimizers()

        self.save_path = save_path
        self.save_weights_only = save_weights_only
        
        self.step = 0
        self.epoch = 0
        
        self.load_prev_scheduler = load_prev_scheduler
        if from_checkpoint:
            self.load_checkpoint(from_checkpoint, load_prev_scheduler)

    @abc.abstractmethod
    def initialize_model(self) -> None:
        """
        All model initialization should be done here
        Note that the whole model must be named as "self.model" for model saving and loading
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward propagation
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_metrics(self, stage: str) -> dict:
        """
        Initialize metrics for each stage
        Args:
            stage: "train", "valid" or "test"
        
        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_func(self, stage: str, outputs, labels) -> torch.Tensor:
        """

        Args:
            stage: "train", "valid" or "test"
            outputs: model outputs for calculating loss
            labels: labels for calculating loss

        Returns:
            loss

        """
        raise NotImplementedError

    @staticmethod
    def load_weights(model, weights):
        model_dict = model.state_dict()

        unused_params = []
        missed_params = list(model_dict.keys())

        for k, v in weights.items():
            if k in model_dict.keys():
                model_dict[k] = v
                missed_params.remove(k)

            else:
                unused_params.append(k)

        if len(missed_params) > 0:
            print(f"\033[31mSome weights of {type(model).__name__} were not "
                  f"initialized from the model checkpoint: {missed_params}\033[0m")

        if len(unused_params) > 0:
            print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

        model.load_state_dict(model_dict)
    
    # Add 1 to step after each optimizer step
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int = 0,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
        )
        self.step += 1

    def on_train_epoch_end(self):
        self.epoch += 1

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_func('train', outputs, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        return self.loss_func('valid', outputs, labels)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        return self.loss_func('test', outputs, labels)

    def load_checkpoint(self, from_checkpoint, load_prev_scheduler):
        state_dict = torch.load(from_checkpoint, map_location=self.device, weights_only=False)
        self.load_weights(self.model, state_dict["model"])
        
        if load_prev_scheduler:
            try:
                self.step = state_dict["global_step"]
                self.epoch = state_dict["epoch"]
                self.best_value = state_dict["best_value"]
                self.optimizer.load_state_dict(state_dict["optimizer"])
                self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
                print(f"Previous training global step: {self.step}")
                print(f"Previous training epoch: {self.epoch}")
                print(f"Previous best value: {self.best_value}")
                print(f"Previous lr_scheduler: {state_dict['lr_scheduler']}")
            
            except Exception as e:
                print(e)
                raise KeyError("Wrong in loading previous scheduler, please set load_prev_scheduler=False")

    def save_checkpoint(self, save_info: dict = None) -> None:
        """
        Save model to save_path
        Args:
            save_info: Other info to save
        """
        state_dict = {} if save_info is None else save_info
        state_dict["model"] = self.model.state_dict()

        if not self.save_weights_only:
            state_dict["global_step"] = self.step
            state_dict["epoch"] = self.epoch
            state_dict["best_value"] = getattr(self, f"best_value", None)
            state_dict["optimizer"] = self.optimizers().optimizer.state_dict()
            state_dict["lr_scheduler"] = self.lr_schedulers().state_dict()

        torch.save(state_dict, self.save_path)

    def check_save_condition(self, now_value: float, mode: str, save_info: dict = None) -> None:
        """
        Check whether to save model. If save_path is not None and now_value is the best, save model.
        Args:
            now_value: Current metric value
            mode: "min" or "max", meaning whether the lower the better or the higher the better
            save_info: Other info to save
        """

        assert mode in ["min", "max"], "mode should be 'min' or 'max'"

        if self.save_path is not None:
            dir = os.path.dirname(self.save_path)
            os.makedirs(dir, exist_ok=True)
            
            if dist.get_rank() == 0:
                # save the best checkpoint
                best_value = getattr(self, f"best_value", None)
                if best_value:
                    if mode == "min" and now_value < best_value or mode == "max" and now_value > best_value:
                        setattr(self, "best_value", now_value)
                        self.save_checkpoint(save_info)

                else:
                    setattr(self, "best_value", now_value)
                    self.save_checkpoint(save_info)
    
    def reset_metrics(self, stage) -> None:
        """
        Reset metrics for given stage
        Args:
            stage: "train", "valid" or "test"
        """
        for metric in self.metrics[stage].values():
            metric.reset()
    
    def get_log_dict(self, stage: str) -> dict:
        """
        Get log dict for the stage
        Args:
            stage: "train", "valid" or "test"

        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric values

        """
        return {name: metric.compute() for name, metric in self.metrics[stage].items()}
    
    def log_info(self, info: dict) -> None:
        """
        Record metrics during training and testing
        Args:
            info: dict of metrics
        """
        if getattr(self, "logger", None) is not None:
            info["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            info["epoch"] = self.epoch
            self.logger.log_metrics(info, step=self.step)

    def init_optimizers(self):
        # No decay for layer norm and bias
        no_decay = ['LayerNorm.weight', 'bias']
        
        if "weight_decay" in self.optimizer_kwargs:
            weight_decay = self.optimizer_kwargs.pop("weight_decay")
        else:
            weight_decay = 0.01
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                           lr=self.lr_scheduler_kwargs['init_lr'],
                                           **self.optimizer_kwargs)

        self.lr_scheduler = Esm2LRScheduler(self.optimizer, **self.lr_scheduler_kwargs)
    
    def configure_optimizers(self):
        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.lr_scheduler,
                                 "interval": "step",
                                 "frequency": 1}
                }
