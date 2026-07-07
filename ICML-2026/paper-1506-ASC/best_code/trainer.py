import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from model_utils import *
#from postprocessing.results_analysis import plot_classifiers
from typing import Any, Callable, Dict, Literal, Optional
from collections import defaultdict
from tqdm import tqdm
from model import *

class StrategicTrainer:
    def __init__(self,
                 model,
                 loss_fn: nn.Module,
                 optimizer: Optimizer,
                 reg_classifier: float = 0.0,
                 reg_auxiliary: float = 0.0,
                 reg_alignment: float = 0.0,
                 write_metrics: bool = True,
                 device: Optional[torch.device] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.reg_classifier = reg_classifier
        self.reg_auxiliary = reg_auxiliary
        self.reg_alignment = reg_alignment
        self.write_metrics = write_metrics
        self.device = device
        self.metrics = defaultdict(list)

        if self.device:
            self.model.to(self.device)
            self.loss_fn.to(self.device)
    
    def _update_metrics(self, train_result, prefix=Literal['train', 'val', 'test']):
        assert prefix in ['train', 'val', 'test'], "Metric's prefix must be train or val"
        
        for k, v in train_result.items():
            self.metrics[f"{prefix}_{k}"].append(v)
    
    def _update_weights_metrics(self):
        self.metrics["w_chosen"].append(self.model.get_w_chosen().tolist())
        self.metrics["b_chosen"].append(self.model.get_b_chosen().item())

        # Get all classifiers' weights & biases
        if hasattr(self.model, "num_classifiers") and self.model.num_classifiers > 1:
            W, b = self.model.get_classifiers_with_bias()

            W_list = [w_row.tolist() for w_row in W]
            b_list = b.tolist()

            self.metrics.setdefault("all_W", []).append(W_list)
            self.metrics.setdefault("all_b", []).append(b_list)

        elif hasattr(self.model, "w_min") and hasattr(self.model, "w_max"):
            w_min = self.model.get_w_min()
            w_max = self.model.get_w_max()
            self.metrics.setdefault("w_min", []).append(w_min.tolist())
            self.metrics.setdefault("w_max", []).append(w_max.tolist())

            diff = w_max - w_min
            norm_type = self.model.norm_limit_type

            if norm_type == "l1":
                dist = torch.norm(diff, p=1).item()
            elif norm_type == "inf":
                dist = torch.norm(diff, p=float('inf')).item()
            else:
                dist = torch.norm(diff, p=2).item()

            self.metrics.setdefault("actual_distance", []).append(dist)
        
        elif hasattr(self.model, "w_min") and hasattr(self.model, "w_max_offset"):
            w_min = self.model.get_w_min()
            w_max = self.model.get_w_max()
            self.metrics.setdefault("w_min", []).append(w_min.tolist())
            self.metrics.setdefault("w_max", []).append(w_max.tolist())

            diff = w_max - w_min
            norm_type = self.model.norm_limit_type

            if norm_type == "l1":
                dist = torch.norm(diff, p=1).item()
            elif norm_type == "inf":
                dist = torch.norm(diff, p=float('inf')).item()
            else:
                dist = torch.norm(diff, p=2).item()

            self.metrics.setdefault("actual_distance", []).append(dist)

    def fit(self,
        dl_train: DataLoader,
        dl_val: DataLoader,
        num_epochs: int,
        early_stopping: Optional[int] = None,
        no_val = False):

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        actual_num_epochs = 0
        early_stopping = None

        for _ in tqdm(range(num_epochs)):
            actual_num_epochs += 1
            
            # Train
            train_result = self._foreach_batch(dl_train, self._train_batch)
            if self.write_metrics:
                self._update_metrics(train_result, prefix="train")
            
            if self.write_metrics:
                self._update_weights_metrics()
            
            if no_val:
                continue
            
            val_result = self._foreach_batch(dl_val, self._test_batch)
            
            if self.write_metrics:
                self._update_metrics(val_result, prefix="val")

        if self.write_metrics:
            self.metrics['actual_num_epochs'] = actual_num_epochs
        return self.metrics
    
    def predict(self, dl_test: DataLoader):
        test_result = self._foreach_batch(dl_test, self._test_batch)
        if self.write_metrics:
            self._update_metrics(test_result, prefix="test")
        return test_result

    def _foreach_batch(self, dataloader: DataLoader, func: Callable[Any, Dict[str, Any]]):
        results =defaultdict(list)
        epoch_results = defaultdict()

        for batch in tqdm(dataloader, desc="Processing batches"):
            b_result = func(batch)
            for k, v in b_result.items():
                results[k].append(v)
        
        for k, v in results.items():
            epoch_results[k] = np.mean(v)
        
        return epoch_results
    
    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        result = defaultdict(list)
        X, Y = batch['X'], batch['y']
        if self.device:
            X, Y = X.to(self.device), Y.to(self.device)
        
        metrics_forward = self.model.forward(X, Y)
        
        if isinstance(self.loss_fn, AmbiguousStrategicHingeLoss):
            values = metrics_forward["values_of_proj"]
            loss = self.loss_fn(self.model, X, Y, values)
        
        else:
            loss = self.calc_loss(X, Y)

        reg = self.compute_regularization_loss()
        objective = loss + reg
        print(f"loss in batch is {loss} and objective is {objective}")
        self.optimizer.zero_grad()
        objective.backward()

        # Record gradients BEFORE optimizer step
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_name = f"grad_{name}"
                if param.grad is not None and not torch.isnan(param.grad).any():
                    result[grad_name] = param.grad.detach().cpu().numpy().tolist()
                else:
                    result[grad_name] = np.zeros_like(param.detach().cpu().numpy()).tolist()

        else:
            self.optimizer.step()

        if isinstance(self.model, StrategicClassifierInfiniteSet):
            self.model.validate_weights_in_bounds_and_fix()

        result['loss'] = loss.item()
        result['objective'] = objective.item()
        result['acc'] = metrics_forward["accuracy"].item()
        result['pos_recall'] = metrics_forward["pos_recall"].item()
        result['neg_recall'] = metrics_forward["neg_recall"].item()
        result['total_burden_to_AOP'] = metrics_forward["total_burden_to_AOP"].item()
        result['avg_burden_to_AOP'] = metrics_forward["avg_burden_to_AOP"].item()
        result['total_burden_to_classifier'] = metrics_forward["total_burden_to_classifier"].item()
        result['avg_burden_to_classifier'] = metrics_forward["avg_burden_to_classifier"].item()
        result['total_utility'] = metrics_forward["total_utility"].item()
        result['avg_utility'] = metrics_forward["avg_utility"].item()
        max_iterations = metrics_forward.get("max_num_iterations", -1)
        result['max_num_iterations'] = max_iterations.item() if isinstance(max_iterations, torch.Tensor) else max_iterations
        pos_moving_ratio = metrics_forward.get("pos_moving_ratio", -1)
        result['pos_moving_ratio'] = pos_moving_ratio.item() if isinstance(pos_moving_ratio, torch.Tensor) else pos_moving_ratio
        neg_moving_ratio = metrics_forward.get("neg_moving_ratio", -1)
        result['neg_moving_ratio'] = neg_moving_ratio.item() if isinstance(neg_moving_ratio, torch.Tensor) else neg_moving_ratio
        return result

    def compute_regularization_loss(self):
        regs = self.model.get_regularization_loss()

        if isinstance(regs, torch.Tensor):
            regs = (regs,)

        elif not isinstance(regs, (tuple, list)):
            raise TypeError(
                f"get_regularization_loss must return Tensor or tuple, got {type(regs)}"
            )
        
        if len(regs) == 1:
            return self.reg_classifier * regs[0]
        
        elif len(regs) == 2:
            return regs[0] * self.reg_classifier + regs[1] * self.reg_auxiliary
        
        return regs[0] * self.reg_classifier + regs[1] * self.reg_auxiliary + regs[2] * self.reg_alignment
    
    def calc_loss(self, X, Y):

        if isinstance(self.loss_fn, BasicStrategicHingeLoss):
            loss = self.loss_fn(self.model, X, Y)

        else:
            loss = self.loss_fn(self.model, X, Y)
        
        return loss

    def _test_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        result = defaultdict(list)
        X, Y = batch['X'], batch['y']
        if self.device:
            X, Y = X.to(self.device), Y.to(self.device)

        with torch.no_grad():
            metrics_evaluate = self.model.evaluate(X,Y)
            
            if isinstance(self.loss_fn, AmbiguousStrategicHingeLoss):
                values = metrics_evaluate["values_of_proj"]
                loss = self.loss_fn(self.model, X, Y, values)
        
            else:
                loss = self.calc_loss(X, Y)
            
            reg = self.compute_regularization_loss()
            objective = loss + reg

            result['loss'] = loss.item()
            result['objective'] = objective.item()
            result['acc'] = metrics_evaluate["accuracy"].item()
            result['pos_recall'] = metrics_evaluate["pos_recall"].item()
            result['neg_recall'] = metrics_evaluate["neg_recall"].item()
            result['total_burden_to_AOP'] = metrics_evaluate["total_burden_to_AOP"].item()
            result['avg_burden_to_AOP'] = metrics_evaluate["avg_burden_to_AOP"].item()
            result['total_burden_to_classifier'] = metrics_evaluate["total_burden_to_classifier"].item()
            result['avg_burden_to_classifier'] = metrics_evaluate["avg_burden_to_classifier"].item()
            result['total_utility'] = metrics_evaluate["total_utility"].item()
            result['avg_utility'] = metrics_evaluate["avg_utility"].item()
            max_iterations = metrics_evaluate.get("max_num_iterations", -1)
            result['max_num_iterations'] = max_iterations.item() if isinstance(max_iterations, torch.Tensor) else max_iterations
            pos_moving_ratio = metrics_evaluate.get("pos_moving_ratio", -1)
            result['pos_moving_ratio'] = pos_moving_ratio.item() if isinstance(pos_moving_ratio, torch.Tensor) else pos_moving_ratio
            neg_moving_ratio = metrics_evaluate.get("neg_moving_ratio", -1)
            result['neg_moving_ratio'] = neg_moving_ratio.item() if isinstance(neg_moving_ratio, torch.Tensor) else neg_moving_ratio

        return result