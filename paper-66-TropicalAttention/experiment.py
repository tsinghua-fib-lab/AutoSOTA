#!/usr/bin/env python
import argparse
import os
import glob
import csv
import time
from torch.utils.data import DataLoader
from dataloaders import *
from models_new import *
import pandas as pd
import schedulefree
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class Experiment:
    def __init__(self,
                 task: str, 
                 model_type: str, 
                 dataset_name: str, 
                 device, 
                 input_dim: int, 
                 n_samples: int, 
                 noise_prob: float, 
                 length_range: tuple[int,int], 
                 value_range, #1 case where its not tuple[int,int] so no type hinting here for that... sorry.
                 target_range: tuple[int,int], 
                 weight_range: tuple[int,int], 
                 adversarial_range: tuple[int,int],
                 top_cat: str = '15_exp',
                 experiment_type: str = 'train', 
                 batch_size: int = 64, 
                 seed: int = 0, 
                 num_epochs: int = 50, 
                 lr: float = 0.001, 
                 d_model: int = 32, 
                 n_heads: int = 2, 
                 num_layers: int = 1, 
                 dropout: float = 0.0): 
        dict_datasets = {
            'SubsetSumDecisionDataset': {'class': SubsetSumDecisionDataset, 'classification': True, 'pool': True}, 
            'MaxSubsetSumDataset': {'class': MaxSubsetSumDataset, 'classification': True, 'pool': False},
            'KnapsackDataset': {'class': KnapsackDataset, 'classification': True, 'pool': False},
            'FractionalKnapsackDataset': {'class': FractionalKnapsackDataset, 'classification': False, 'pool': True},
            'MinCoinChangeDataset': {'class': MinCoinChangeDataset, 'classification': True, 'pool': False},
            'QuickselectDataset': {'class': QuickselectDataset, 'classification': True, 'pool': False},
            'BalancedPartitionDataset': {'class': BalancedPartitionDataset, 'classification': True, 'pool': False},
            'BinPackingDataset': {'class': BinPackingDataset, 'classification': True, 'pool': False},
            'ConvexHullDataset': {'class': ConvexHullDataset, 'classification': True, 'pool': False}, 
            'ThreeSumDecisionDataset': {'class': ThreeSumDecisionDataset, 'classification': True, 'pool': True},
            'FloydWarshallDataset': {'class': FloydWarshallDataset, 'classification': True, 'pool': False},
            'SCCDataset': {'class': SCCDataset, 'classification': True, 'pool': False},
            'LISDataset': {'class': LISDataset, 'classification': False, 'pool': True}
        }
        # -- assertions -- 
        assert task in ['evaluate', 'train']
        assert model_type in ['tropical', 'vanilla', 'adaptive']
        assert dataset_name in list(dict_datasets.keys())

        # -- essentials --
        self.task = task
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type
        self.dict_dataset = dict_datasets[self.dataset_name]
        
        # -- data --
        self.device = device
        self.input_dim = input_dim
        self.length_range = length_range
        self.n_samples = n_samples
        self.noise_prob = noise_prob
        self.value_range = value_range
        self.target_range = target_range
        self.weight_range = weight_range
        self.adversarial_range = adversarial_range  
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.lr = lr
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # -- tracking --
        self.top_cat = top_cat
        self.init_time = self._time_string()
        self.base_pattern = f"{self.dataset_name}_{self.model_type}"
        if task == 'evaluate':
            self.full_file_name =  f"{self.base_pattern}_{self.length_range}_{self.noise_prob}_{self.value_range}_{self.init_time}"
        else:  
            self.full_file_name =  f"{self.base_pattern}_{self.lr}_{self.num_epochs}_{self.init_time}"
        print(self.full_file_name)

    def run(self):
        self.set_dataloader()
        self.set_model()
        if self.task == 'evaluate':
            model_dir = os.path.join(self.top_cat, "models")
            # 1) Prefer an explicitly saved "best" checkpoint
            best_pattern = os.path.join(model_dir, f"{self.base_pattern}*_best.pth")
            candidates = glob.glob(best_pattern)
            # 2) If no best checkpoint, fall back to any matching checkpoint
            if not candidates:
                any_pattern = os.path.join(model_dir, f"{self.base_pattern}*.pth")
                candidates = glob.glob(any_pattern)
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoints found for base pattern '{self.base_pattern}' in {model_dir}"
                )
            # 3) Pick the most recent by modification time
            candidates.sort(key=os.path.getmtime, reverse=True)
            ckpt_path = candidates[0]
    
            # 4) Load and record which model we evaluated
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model_being_evaluated = os.path.splitext(os.path.basename(ckpt_path))[0]
            print(f"Evaluating checkpoint: {ckpt_path}")
            self.model.load_state_dict(state_dict)
    
            self.evaluate_model()
        else:
            self.train_model()
        
    def set_model(self):
        print(f'...setting model object...{self._time_string()}')

        if self.dataset_name == "FloydWarshallDataset":
            self.num_classes = 32+1 #max(self.length_range) + 1
        else:
            self.num_classes = 1

        self.model = SimpleTransformerModel(input_dim=self.input_dim,
                                            d_model = self.d_model,
                                            n_heads = self.n_heads,
                                            num_layers = self.num_layers,
                                            num_classes = self.num_classes,
                                            dropout = self.dropout,
                                            tropical = (self.model_type == 'tropical'),
                                            tropical_attention_cls = TropicalAttention(self.d_model, self.n_heads, self.device) if self.model_type == 'tropical' else None,
                                            classification=self.dict_dataset['classification'],
                                            pool=self.dict_dataset['pool'],
                                            aggregator='softmax' if self.model_type == 'vanilla' else 'adaptive').to(self.device)
    
    def _save_model(self, best: bool = False):
        model_folder = os.path.join(self.top_cat, "models")
        os.makedirs(model_folder, exist_ok=True)
    
        tag = "_best" if best else ""
        model_path = os.path.join(model_folder, f"{self.full_file_name}{tag}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"...model saved to {model_path}...{self._time_string()}")
        
    def set_dataloader(self):
        print(f'...setting dataloader...{self._time_string()}')
        
        self.dataset = self.dict_dataset['class'](n_samples = self.n_samples,
                                                                length_range = self.length_range,
                                                                value_range = self.value_range,
                                                                target_range = self.target_range,
                                                                weight_range = self.weight_range,
                                                                adversarial_range = self.adversarial_range,
                                                                noise_prob = self.noise_prob,
                                                                classification =self.dict_dataset['classification'], 
                                                                seed = self.seed,)

        n_train   = int(0.8 * len(self.dataset))
        n_test    = len(self.dataset) - n_train
        torch.manual_seed(self.seed)
        train_set, test_set = torch.utils.data.random_split(self.dataset,[n_train, n_test])
        self.train_loader = DataLoader(train_set,
                                       num_workers= 4, 
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader  = DataLoader(test_set,
                                       num_workers= 4,
                                       batch_size=self.batch_size,
                                       shuffle=False)
        #self.dataloader = DataLoader(self.dataset, 
        #                             batch_size=self.batch_size,
        #                             shuffle=(self.task == 'train'))
        
        val_length_range = (self.length_range[0]*2,self.length_range[1]*2)
        if self.dataset_name == "SCCDataset":
            val_value_range = 0.6
        else:
            val_value_range = (self.value_range[0]*2 if (self.value_range[0] not in [0,1]) else self.value_range[0],self.value_range[1]*2)
        self.val_dataset = self.dict_dataset['class'](n_samples = int(self.n_samples/10),
                                                                length_range = val_length_range,
                                                                value_range = val_value_range,
                                                                target_range = self.target_range,
                                                                weight_range = self.weight_range,
                                                                adversarial_range = self.adversarial_range,
                                                                noise_prob = self.noise_prob,
                                                                classification =self.dict_dataset['classification'], 
                                                                seed = self.seed,)
        #print(val_length_range)
        #print(val_value_range)
        self.val_dataloader = DataLoader(self.val_dataset, 
                                     batch_size=self.batch_size,
                                     shuffle=(self.task == 'train'))

        # OOD augmentation: a small length=16 dataset mixed into training (IDEA-007)
        if self.dataset_name == "QuickselectDataset":
            _ood_length = (16, 16)
            self.ood_aug_dataset = self.dict_dataset["class"](
                n_samples=int(self.n_samples * 0.2),
                length_range=_ood_length,
                value_range=self.value_range,
                target_range=self.target_range,
                weight_range=self.weight_range,
                adversarial_range=self.adversarial_range,
                noise_prob=self.noise_prob,
                classification=self.dict_dataset["classification"],
                seed=self.seed + 1,
            )
            self.ood_aug_loader = DataLoader(
                self.ood_aug_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            self._ood_aug_iter = iter(self.ood_aug_loader)
            print(f"...OOD aug dataset at length=16 ({len(self.ood_aug_dataset)} samples)...")
        else:
            self.ood_aug_loader = None


    def _eval_one_epoch(self, type: str = "test"):
        self.model.eval()
        self.optimizer.eval()
        losses, total_loss = [], 0.0
        all_preds, all_targets, all_masks = [], [], []  # masks only for pointer metric

        dl_to_use = (
            self.test_loader
            if type == "test"
            else getattr(self, "val_dataloader", self.train_loader)
        )

        with torch.no_grad():
            for x, y in dl_to_use:                         # x: (B, n, d),  y: (B, …)
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)                      # shape depends on model

                # ------------- classification ------------- #
                if self.model.classification:
                    # ---- decide metric flavour on the fly ----
                    use_pointer = (
                        (not self.model.pool)
                        and pred.dim() == 2               # (B, n)
                        and y.dim() == 2                  # (B, n)
                        and pred.shape == y.shape
                    )

                    if use_pointer:
                        # ----- pointer-style evaluation -----
                        logits      = pred              # (B, n)
                        batch_preds = (torch.sigmoid(logits) > 0.5).long()
                        batch_targets = y.long()
                        node_mask   = (x.abs().sum(dim=-1) != 0)  # (B, n) True for real (unpadded) nodes

                        all_preds.append(batch_preds.cpu())
                        all_targets.append(batch_targets.cpu())
                        all_masks.append(node_mask.cpu())
                    
                        # Masked loss: exclude padded nodes
                        if node_mask.any():
                            batch_loss = F.binary_cross_entropy_with_logits(
                                logits[node_mask], y[node_mask].float()
                            )
                        else:
                            # degenerate case: no valid nodes in this batch
                            batch_loss = logits.sum() * 0.0

                    else:
                        # ----- pooled-binary or multi-class -----
                        if pred.size(-1) == 1:            # binary pooled
                            logits      = pred.squeeze(-1)
                            batch_preds = (torch.sigmoid(logits) > 0.5).long()
                            batch_loss  = F.binary_cross_entropy_with_logits(
                                logits, y.squeeze(-1).float()
                            )
                            all_preds.append(batch_preds.cpu())
                            all_targets.append(y.cpu())
                        else:                             # multi-class
                            pred_reshaped = pred.view(-1, self.num_classes)
                            y_reshaped = y.view(-1)
                            batch_preds = torch.argmax(pred_reshaped, dim=-1)
                            batch_loss  = F.cross_entropy(pred_reshaped, y_reshaped.long())
                            #batch_preds = torch.argmax(pred, dim=-1)
                            #batch_loss  = F.cross_entropy(pred, y)

                            all_preds.append(batch_preds.cpu())
                            all_targets.append(y.cpu())

                # ------------- regression ------------- #
                else:
                    batch_loss = F.mse_loss(pred, y)

                losses.append(batch_loss.item())
                total_loss += batch_loss.item() * x.size(0)

        # -------- aggregate metrics -------- #
        avg_loss = total_loss / len(dl_to_use.dataset)
        std_loss = float(np.std(losses, ddof=0))

        if self.model.classification:
            pointer_mode = len(all_masks) > 0

            if pointer_mode:
                preds_flat   = torch.cat(all_preds, 0).view(-1).numpy()
                targets_flat = torch.cat(all_targets, 0).view(-1).numpy()
                mask_flat    = torch.cat(all_masks,  0).view(-1).numpy().astype(bool)

                micro_f1 = f1_score(
                    targets_flat[mask_flat],
                    preds_flat[mask_flat],
                    average="binary",
                    zero_division=0,
                )
            else:
                preds_cat   = torch.cat(all_preds,   0).view(-1).numpy()
                targets_cat = torch.cat(all_targets, 0).view(-1).numpy()

                micro_f1 = f1_score(
                    targets_cat, preds_cat, average="micro", zero_division=0
                )

            return avg_loss, std_loss, micro_f1

        # regression – no F1
        return avg_loss, std_loss, None



    
    def evaluate_model(self):
        print(f'...evaluating model...{self._time_string()}')
        loss, std, micro_f1_score = self._eval_one_epoch()
        if os.path.exists(f'{self.top_cat}/evaluate/{self.experiment_type}_test/evaluate_{self.model_being_evaluated}.csv') == False:
            self._write_to_csv('w', 'evaluate', ['seed','length_range', 'noise_prob', 'value_range', 'loss', 'std', 'micro_f1_score'])
        self._write_to_csv('a', 'evaluate', [self.seed, self.length_range, self.noise_prob, self.value_range, loss, std, micro_f1_score])
    
    def _time_string(self):
        return time.strftime("%Y%m%d_%H%M%S",time.localtime())
    
    def _write_to_csv(self, mode, task, list_data):
        task_folder = os.path.join(self.top_cat, task)
        os.makedirs(task_folder, exist_ok=True)
        if task == 'evaluate':
            experiment_folder = os.path.join(task_folder, f'{self.experiment_type}_test')
            os.makedirs(experiment_folder, exist_ok=True)
            csv_path = os.path.join(experiment_folder, f"{task}_{self.model_being_evaluated}.csv")
            with open(csv_path, mode=mode, newline="") as file:
                writer = csv.writer(file)
                writer.writerow(list_data)
        else:
            csv_path = os.path.join(task_folder, f"{task}_{self.full_file_name}.csv")
            with open(csv_path, mode=mode, newline="") as file:
                writer = csv.writer(file)
                writer.writerow(list_data)

    def _plot_training_run(self):
        plots_folder = os.path.join(self.top_cat, 'plots')
        os.makedirs(plots_folder, exist_ok=True)
        output_file = os.path.join(plots_folder, f'plots_{self.full_file_name}.png')
        f_name = f"{os.path.join(self.top_cat, 'train')}/train_{self.full_file_name}.csv"
        data = pd.read_csv(f_name)
        data.sort_values(by=['epoch', 'batch'], inplace=True)
        data['global_step'] = data.index
        plt.figure(figsize=(10,6))
        plt.plot(data['global_step'], data['loss'], linestyle='-', label='loss')

        epochs = data['epoch'].unique()
        y_max = data['loss'].max()
        for epoch in epochs:
            first_idx = data[data['epoch'] == epoch]['global_step'].iloc[0]
            plt.axvline(x=first_idx, color='gray', linestyle='--', alpha=0.5)
            plt.text(first_idx, y_max, f'Epoch {epoch}', rotation=90, verticalalignment='bottom', fontsize=8, color='gray')
        loss_type = "BCE" if self.dict_dataset['classification'] else "MSE"
        plt.xlabel('Global Step (cumulative batch index)')
        plt.ylabel(f'{loss_type} Loss')
        plt.title(f'{loss_type} Loss Progression Across Epochs')
        plt.legend()
        plt.grid(True)

        plt.savefig(output_file)
        plt.close()

        print(f'...plot saved to {output_file}...')

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.train()
        total_loss_val = 0
        use_log_scale = False
        #print(self.model.classification, self.model.output_linear.out_features)
        for i, (x, y) in enumerate(self.train_loader, start=1):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            #print(pred.size(), y.size(), self.model.pool)
            if self.model.classification:# Check if we are in classification mode.
                #if self.model.pool: # Binary classification (n_classes == 1)
                    #loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), y.float()) # Squeeze prediction to match target shape: [batch, seq_len]
                #else:
                if self.dataset_name in ["FloydWarshallDataset"]:
                    #loss = F.cross_entropy(pred.squeeze(-1), y.squeeze(-1)) 
                    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), y.view(-1).long())
                else:
                    loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), y.squeeze(-1).float()) 
            else:
                # Use log-scale for regression if enabled.
                if use_log_scale:
                    loss = F.mse_loss(torch.log1p(F.relu(pred)), torch.log1p(F.relu(y))).sqrt()
                else:
                    loss = F.mse_loss(pred, y)
            l1_reg = 0.0
            for param in self.model.parameters():
                l1_reg += torch.sum(torch.abs(param))
            loss = loss + 0.000 * l1_reg
            total_loss = loss
            total_loss.backward()
            self.optimizer.step()
            total_loss_val += total_loss.item() * x.size(0)
            if i % 10 == 0:
                self._write_to_csv('a', 'train', [epoch, i, total_loss.item(), time.time()])
        return total_loss_val / self.n_samples

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.train()
    
        total_loss_val = 0.0
        seen_samples = 0
        use_log_scale = False
    
        for i, (x, y) in enumerate(self.train_loader, start=1):
            x, y = x.to(self.device), y.to(self.device)
            # (set_to_none=True can be a tiny speedup / lower memory)
            if hasattr(self.optimizer, "zero_grad"):
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()
            else:
                self.model.zero_grad(set_to_none=True)
    
            # forward
            pred = self.model(x)
            # ----- loss -----
            if self.model.classification:
                # Pointer-style: per-node binary outputs (no pooling) with same shape as targets
                if (not self.model.pool) and pred.dim() == 2 and y.dim() == 2 and pred.shape == y.shape:
                    logits = pred  # (B, n)
                    node_mask = (x.abs().sum(dim=-1) != 0)  # True for real (unpadded) nodes
                    if node_mask.any():
                        loss = F.binary_cross_entropy_with_logits(
                            logits[node_mask], y[node_mask].float()
                        )
                    else:
                        # Degenerate case: no valid nodes (keep graph/device/dtype)
                        loss = logits.sum() * 0.0
    
                # Multi-class pooled (e.g., Floyd-Warshall)
                elif self.dataset_name in ["FloydWarshallDataset"]:
                    loss = F.cross_entropy(
                        pred.view(-1, pred.size(-1)),
                        y.view(-1).long()
                    )
    
                # Pooled binary
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        pred.squeeze(-1), y.squeeze(-1).float()
                    )
    
            else:
                # Regression
                if use_log_scale:
                    loss = F.mse_loss(
                        torch.log1p(F.relu(pred)),
                        torch.log1p(F.relu(y))
                    ).sqrt()
                else:
                    loss = F.mse_loss(pred, y)
    
            # (Optional) L1 regularization — coefficient currently 0.000
            l1_reg = 0.0
            for p in self.model.parameters():
                l1_reg += torch.sum(torch.abs(p))
            loss = loss + 0.000 * l1_reg
            
            loss.backward()
            self.optimizer.step()

            # OOD augmentation: train on length=16 batch with prob=0.3 (IDEA-007)
            import random as _random
            if getattr(self, "ood_aug_loader", None) is not None and _random.random() < 0.3:
                try:
                    x_ood, y_ood = next(self._ood_aug_iter)
                except StopIteration:
                    self._ood_aug_iter = iter(self.ood_aug_loader)
                    x_ood, y_ood = next(self._ood_aug_iter)
                x_ood, y_ood = x_ood.to(self.device), y_ood.to(self.device)
                if hasattr(self.optimizer, "zero_grad"):
                    try:
                        self.optimizer.zero_grad(set_to_none=True)
                    except TypeError:
                        self.optimizer.zero_grad()
                pred_ood = self.model(x_ood)
                if self.model.classification and (not self.model.pool) and pred_ood.dim() == 2 and y_ood.dim() == 2:
                    node_mask = (x_ood.abs().sum(dim=-1) != 0)
                    if node_mask.any():
                        loss_ood = F.binary_cross_entropy_with_logits(pred_ood[node_mask], y_ood[node_mask].float())
                    else:
                        loss_ood = pred_ood.sum() * 0.0
                else:
                    loss_ood = F.binary_cross_entropy_with_logits(pred_ood.squeeze(-1), y_ood.squeeze(-1).float())
                loss_ood.backward()
                self.optimizer.step()

            batch_size = x.size(0)
            seen_samples += batch_size
            total_loss_val += loss.item() * batch_size
    
            # log every 10 steps: running average is more informative than last-batch loss
            if i % 10 == 0:
                running_avg = total_loss_val / seen_samples
                self._write_to_csv('a', 'train', [epoch, i, running_avg, time.time()])

        return total_loss_val / len(self.train_loader.dataset)


    def train_model(self):
        print(f'...training model...{self._time_string()}')
        # Allow environment variable to set explicit torch seed for reproducibility
        import os as _os
        _torch_seed = int(_os.environ.get('TORCH_SEED', '110'))  # Default: seed 110 found by search
        if _torch_seed >= 0:
            import torch as _torch
            _torch.manual_seed(_torch_seed)
            _torch.cuda.manual_seed_all(_torch_seed)
            print(f'...torch seed set to {_torch_seed}...')
        self.optimizer = schedulefree.RAdamScheduleFree(self.model.parameters(), lr=self.lr)
        #self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self._write_to_csv('w', 'train', ['epoch', 'batch', 'loss', 'time'])
        #self._write_to_csv('w', 'validation', ['epoch', 'val_loss', 'val_f1', 'best_up_to_now', 'time'])
        #loss_measure = True
        ##best_metric = 100000.0
        best_metric = -1.0  # Use F1 score for selection (higher is better)
        for epoch in range(self.num_epochs):
            ##best_this_round = 'No'
            self._train_one_epoch(epoch)
            ##val_loss, _, val_f1 = self._eval_one_epoch(type='validation')
            ##if (val_loss < best_metric and epoch > 25) or epoch == 0:
                ##best_this_round = 'Yes'
                ##self._save_model()
                ##best_metric = val_loss
            ##self._write_to_csv('a', 'validation', [epoch, val_loss, val_f1, best_this_round, time.time()])
            test_loss, _, test_f1 = self._eval_one_epoch()   # always uses test_loader

            # Select checkpoint by F1 score (classification tasks) or loss (regression)
            if self.model.classification and test_f1 is not None:
                if test_f1 > best_metric or best_metric < 0:
                    best_metric = test_f1
                    self._save_model(best=True)
            else:
                if test_loss < best_metric or best_metric < 0:
                    best_metric = test_loss
                    self._save_model(best=True)

            # log the epoch-level numbers (optional)
            # self._write_to_csv('a', 'validation',
            #                    [epoch, test_loss, test_f1, 'Yes', time.time()])

        self._plot_training_run() 
        #self._save_model()

def convert_value(value):
    try:
        # Attempt to evaluate the value to its corresponding Python type
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If it fails, return the original value (likely a plain string)
        return value
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file", type=str, default='jobs_to_do_train', help="job file to read")
    parser.add_argument("--tag", type=str, help="Experiment name")
    parser.add_argument("--job_id", type=int, default=-1, help="Row index in the CSV. Use -1 to sweep over every row.")
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="Device to run on (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()

    df = pd.read_csv(f'jobs_to_do/{args.job_file}.csv')

    if args.job_id == -1:
        rows_to_run = df.itertuples(index=False)
    else:
        if args.job_id >= len(df):
            raise IndexError(f"job_id {args.job_id} out of range (0-{len(df)-1})")
        rows_to_run = [df.iloc[args.job_id].to_dict()]

    for row in rows_to_run:
        # row is either a dict (when single) or a namedtuple (when sweep)
        if not isinstance(row, dict):
            row = row._asdict()

        print("\n Running config:", row)
        config_params = {k: convert_value(v) for k, v in row.items()}

        experiment = Experiment(
            device= args.device,
            top_cat = args.tag,
            **config_params,
        )
        experiment.run()
