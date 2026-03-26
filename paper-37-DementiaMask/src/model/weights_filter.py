"""
Model Training Module

Handles:
- Model initialization
- Training/validation loops
- Performance evaluation
- Model serialization
"""

from pathlib import Path
import os
import logging
import numpy as np
import copy
import pandas as pd
from typing import Dict, Optional, Tuple, List
import torch
from tqdm import trange
from src.preprocessing.data_generator import DataConfig
from .utils import (
    track_layers, 
    cal_conditional_prob, 
    filter_control,
    compute_metrics_df
)
from scipy.special import softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    set_seed
)
from datasets import Dataset



logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int, path: Path):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        self.model_path = path / "model_best.pt"
        self.delta_path = path / "delta_weights.pt"

    def __call__(self, score: float, model: torch.nn.Module, delta: Dict[str, torch.Tensor], 
                 greater_is_better: bool = False):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, delta)
        # early stopping for model metric or loss
        elif greater_is_better:
            if score <= self.best_score:
                self.counter += 1
                logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model, delta)
                self.counter = 0
        else:
            if score >= self.best_score:
                self.counter += 1
                logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model, delta)
                self.counter = 0

    def save_checkpoint(self, score: float, model: torch.nn.Module, delta: Dict[str, torch.Tensor]):
        torch.save(model.state_dict(), self.model_path)
        logger.info(f"Model saved with metric: {score}")
        torch.save(delta, self.delta_path)
        logger.info(f"Delta weights saved to: {self.delta_path}")
    

    def load_checkpoint(self, model: torch.nn.Module):
        model.load_state_dict(torch.load(self.model_path))
        logger.info(f"Loaded model from checkpoint: {self.model_path}")
        delta = torch.load(self.delta_path)
        logger.info(f"Loaded delta weights from: {self.delta_path}")
        return model, delta
    
   

class DualFilter:
    """Handles model training and evaluation with confounder awareness"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pretrained_model = self._init_model()
        self.metrics = dict()
    
    def _init_model(self) -> torch.nn.Module:
        """Initialize pretrained model with the same weights"""
        set_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)
        logger.info(f"Initialized model: {self.model_name}")
        return model

    def _define_tracked_layers(self) -> List[str]:
        """Define layers to track during training"""
        return track_layers(n = -1, emb=True, classifier = False)

    def _get_config(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """get config from data"""
        # generate configurations for the data
        pz1_train = cal_conditional_prob(train_data, 1)
        pz0_train = cal_conditional_prob(train_data, 0)
        pz1_test = cal_conditional_prob(test_data, 1)
        pz0_test = cal_conditional_prob(test_data, 0)
        mix_probs_z1 = train_data["confounder"].mean()

        self.config = DataConfig(
            p_pos_train_z0=pz0_train,
            p_pos_train_z1=pz1_train,
            alpha_test=pz1_test / pz0_test,
            mix_probs_z1=mix_probs_z1,
            test_size=len(test_data),
        )


    def _prepare_data(self, data: pd.DataFrame, target_col: str, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Tokenize and prepare datasets"""
        

        data = data[["text", target_col]]

        data = Dataset.from_pandas(data).map(
            self._tokenize,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing data"
        ).rename_column(target_col, "labels")

       
        data.set_format('torch')

        return (
            DataLoader(data, batch_size=8, shuffle=shuffle)
        )

    def _tokenize(self, examples: Dict) -> Dict:
        """Tokenization function for dataset processing"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

    def _train_and_track(self, 
                        pre_model: torch.nn.Module, 
                        train_data: pd.DataFrame, 
                        test_data: pd.DataFrame, 
                        num_epochs: int = 20, 
                        phase: str = 'target',
                        output_dir: Path = Path("train_logs"),
                       ) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
        """model training module and evaluation on every epoch"""

        train_loader = self._prepare_data(data=train_data, target_col=phase, shuffle=True)
        test_loader = self._prepare_data(data=test_data, target_col=phase, shuffle=False)
        
        # avoid change_inplace
        model = copy.deepcopy(pre_model).to(self.device)
        
        # logging
        exp_name = f'alpha_train_{self.config.p_pos_train_z1/self.config.p_pos_train_z0:.2f}'
        savedir = output_dir / exp_name
        ckpt_path = savedir / phase
        writer = SummaryWriter(log_dir=os.path.join(savedir, 'tb_logs', phase))


        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=num_epochs * len(train_loader)
        )
        early_stopping = EarlyStopping(
            patience=8,
            path=ckpt_path
        )

        # initialize 
        pretrained_w = {name: param.data.clone().detach() for name, param in model.named_parameters() if name in self.tracked_layers}
        cum_delta = {name: torch.zeros_like(weight) for name, weight in pretrained_w.items()}

        logger.info(f"Training {phase} phase for {num_epochs} epochs")

        for epoch in trange(num_epochs):
            model.train()
            train_loss = 0.

            for batch in train_loader:   
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()

                # Track parameter changes for specified model layers
                for name, param in model.named_parameters():
                    if name in pretrained_w:
                        weights = param.data.clone().detach()
                        delta = weights - pretrained_w[name]
                        if torch.max(torch.abs(delta)) > 0: # only track non-zero changes
                            # normalized weight update for single batch
                            cum_delta[name] += torch.abs(delta)/torch.abs(delta).max()
                            pretrained_w[name] = weights
                

            # validation for each epoch
            model.eval()
            eval_loss = 0.
            full_batch_labels = []
            full_batch_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    labels = batch["labels"]
                    logits = outputs.logits
                    full_batch_labels.extend(labels.cpu().numpy())
                    full_batch_logits.extend(logits.cpu().numpy())
    
            # logits to probs
            full_batch_probs = softmax(full_batch_logits, axis=1)

            # log metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_eval_loss = eval_loss / len(test_loader)
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/eval", avg_eval_loss, epoch)

        
            eval_metrics = compute_metrics_df(probs=full_batch_probs, 
                                              labels=full_batch_labels,
                                              )

            # Log evaluation metrics to TensorBoard
            writer.add_scalar('Accuracy/eval', eval_metrics['accuracy'], epoch)
            writer.add_scalar('AUPRC/eval', eval_metrics['aps'], epoch)
            writer.add_scalar('AUROC/eval', eval_metrics['roc'], epoch)
            writer.add_scalar('F1/eval', eval_metrics['f1'], epoch)

            early_stopping(eval_metrics['aps'], model, cum_delta, greater_is_better=True)
            if early_stopping.early_stop:
                logger.info("Early stopped")
                break
        
        writer.close()
        # load best checkpoint
        best_model, cum_delta = early_stopping.load_checkpoint(model)
        return best_model, cum_delta



    def _model_pred(self, model: torch.nn.Module, 
                    test_loader: DataLoader, 
                    group_eval: bool = False) -> Dict[str, float]: 
        """Evaluate model performance on test set"""
        full_labels = []
        full_logits = []
        
        model.eval()
        # test_loss = 0.
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                full_labels.extend(batch["labels"].cpu().numpy())
                full_logits.extend(outputs.logits.cpu().numpy())
                # test_loss += outputs.loss.item()
            
            # avg_loss = test_loss / len(test_loader)
            # eval_metrics = compute_metrics_df(logits=full_batch_logits, 
                                            #   labels=full_batch_labels,
                                            #   group=group_eval)

        # get probabilities from logits
        full_probs = softmax(full_logits, axis=1)

        return np.array(full_probs), np.array(full_labels)
    
    def _binarize(self, 
                  cum_delta: Dict[str, torch.Tensor], 
                  mask_ratio: float) -> Dict[str, torch.Tensor]:
        """Binarize cumulative delta weights based on threshold"""
        if not (0 < mask_ratio <= 1):
            raise ValueError("`mask_ratio` must be in the range (0, 1].")
        # find global cutoff threshold
        all_deltas = torch.cat([delta.flatten() for delta in cum_delta.values()]).cpu().numpy() # to numpy to save memory
        threshold = np.quantile(all_deltas, 1 - mask_ratio )
        
        binarized_delta = {name: (delta > threshold).float() for name, delta in cum_delta.items()}
        
        return binarized_delta

    def _mask_model(self, 
                    mask_ratio: float, 
                    mask_type: str,
                    ablat_rate: float,
                    ) -> None:
        """
        Mask model weights based on specified mask ratio
        Args:
            model: Model to mask
            mask_ratio: Ratio of model weights to mask
            mask_type: Type of mask to apply
        """
        assert mask_type in ['D', 'I', 'A'], "Invalid mask type. Choose from ['D', 'I', 'A']"
        
        mask_dict: Dict[str, torch.Tensor] = {}

        # get binarized delta weights
        bi_delta_p = self._binarize(self.cum_delta_p, mask_ratio)
        bi_delta_c = self._binarize(self.cum_delta_c, mask_ratio)

        for (kp, wp), (kc, wc) in zip(bi_delta_p.items(), bi_delta_c.items()):
            assert kp == kc, f"Layer mismatch: {kp} != {kc}"
            assert wp.size() == wc.size(), "layer weight shape mismatch"

            if mask_type == 'D':
                mask = (wp == 0.) & (wc == 1.)
            elif mask_type == 'I':
                mask = (wp == 1.) & (wc == 1.)
            elif mask_type == 'A':
                mask = wp == 1.
            else:
                raise ValueError(f"Invalid mask type: {mask_type}")
            
             # Convert mask (flip 1s to 0s) and apply alpha effect
            mask_dict[kp] = ~mask  # Inverts mask (1 -> 0, 0 -> 1)
            mask_dict[kp][mask_dict[kp] == 0.] = ablat_rate

       

        # Calculate number of masked neuronssk)
        masked_params = np.sum([(~mask).sum().cpu() for mask in mask_dict.values()])
        total_params = self.num_model_params
        mask_perc = (masked_params / total_params) * 100 if total_params else 0

        logging.info(f"Size of the {mask_type} mask: {masked_params} neurons, "
                    f"which is {mask_perc:.2f}% of the trainable parameters in the primary model.")

        self.masked_params = masked_params

        # Apply masks to both models
        for (name_p, param_p), (name_c, param_c) in zip(
            self.ft_model_p.named_parameters(), self.ft_model_c.named_parameters()
        ):
            param_p.detach_()  
            param_c.detach_() 

            if name_p in mask_dict:
                assert name_p == name_c, f"Layer mismatch: {name_p} != {name_c}"
                # Retrieve mask and apply element-wise multiplication
                mask_layer = mask_dict[name_p].to(self.device)
                param_p.mul_(mask_layer)
                param_c.mul_(mask_layer)

        return
        
    def _evaluate_confounder(self,
                            test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test set"""
        test_loader = self._prepare_data(data=test_df, target_col = 'confounder', shuffle=False)

         # masked model prediction
        full_probs, full_labels = self._model_pred(self.ft_model_c, test_loader)

        # evaluation
        eval_metris_full = compute_metrics_df(probs =  full_probs, 
                           labels = full_labels,
                           group = False)
        results = {
            "cfull": eval_metris_full
        }

        return results
    
    def _evaluate(self,
                  test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test set"""
        test_loader = self._prepare_data(data=test_df, target_col = 'target', shuffle=False)        
        
        # masked model prediction
        full_probs, full_labels = self._model_pred(self.ft_model_p, test_loader)

        # evaluation
        # 1. full test set
        eval_metris_full = compute_metrics_df(probs =  full_probs, 
                           labels = full_labels,
                           group = False)
        
        # slicing by confounder
        confounder = test_df["confounder"].values
        pos_indices = confounder == 0
        neg_indices = confounder == 1

        # 2. confounder=1 set
        eval_metrics_pos = compute_metrics_df(probs =  full_probs[pos_indices], 
                           labels = full_labels[pos_indices],
                           group = True)
        # 3. confounder=0 set
        eval_metrics_neg = compute_metrics_df(probs =  full_probs[neg_indices], 
                           labels = full_labels[neg_indices],
                           group = True)
        
        # save results
        results = {
            "full": eval_metris_full,
            "confounder_0": eval_metrics_neg,
            "confounder_1": eval_metrics_pos,
        }
        return results

    def _append_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        self.metrics.update({f'{self.mask_label}': metrics})

        
    def train(self, 
              train_df: pd.DataFrame, 
              eval_df: pd.DataFrame, 
              test_df: pd.DataFrame, # use for base model evaluation
              num_epochs: int = 20, 
              mask_type: str = 'D',
              mask_ratio: float = None,
              ablat_rate: float = 0.,
              output_dir: Path = Path("train_logs_df")) -> None:
        """
        Main training function with early stopping
        Args:
            train_df: Training data
            eval_df: Evaluation data
            num_epochs: Number of training epochs
            mask_type: Mask type for phase 2
            mask_ratio: Mask ratio for phase 1
            ablat_ratio: Ablation ratio for masking
            output_dir: Output directory for saving models

        """

        self.tracked_layers = self._define_tracked_layers()
        self._get_config(train_df, test_df)

        # refresh mask label for evaluation key
        mask_label = f'DF_{mask_type}_{mask_ratio}_{ablat_rate}'
        self.mask_label = mask_label
        
        if not hasattr(self, 'base_model_p'): # avoid retraining
            logger.info("Training model for target prediction...")
            ft_model_p, cum_delta_p = self._train_and_track(
                pre_model=self.pretrained_model,
                train_data=train_df,
                test_data=eval_df,
                num_epochs=num_epochs,
                output_dir=output_dir,
                phase='target',
            )
            self.ft_model_p = ft_model_p
            self.cum_delta_p = cum_delta_p
            
            # deep copy of the base p model
            self.base_model_p = copy.deepcopy(self.ft_model_p)
            # record the number of trainable parameters
            self.num_model_params = np.sum([p.numel() for p in self.ft_model_p.parameters() if p.requires_grad])
        
        else: # restore ft_model_p as the base model for masking
            self.ft_model_p = copy.deepcopy(self.base_model_p)
        # Train confounder model
        if not hasattr(self, 'ft_model_c'):
            # select only controls to avoid impact from dementia data (Y = 0), comment out if not needed
            filtered_train = filter_control(train_df, label_val=0)
            filtered_eval = filter_control(eval_df, label_val=0)

            logger.info("Training model for confounder prediction...")
            ft_model_c, cum_delta_c = self._train_and_track(
                pre_model=self.pretrained_model, # starting from the same pretrained model
                train_data=filtered_train,
                test_data=filtered_eval,
                num_epochs=num_epochs,
                output_dir=output_dir,
                phase='confounder',
            )
            self.ft_model_c = ft_model_c
            self.cum_delta_c = cum_delta_c

            # deep copy of the base c model
            self.base_model_c = copy.deepcopy(self.ft_model_c)
        
        else:
            self.ft_model_c = copy.deepcopy(self.base_model_c)

        if not self.metrics.get("base"):
            logger.info("Evaluating base model...")
            # get base model predictions
            base_results = self._evaluate(test_df)
            self.metrics["base"] = base_results

        if mask_ratio: 
            # masking phase
            logger.info("Base models ready, masking base model...")
            self._mask_model(mask_ratio=mask_ratio, mask_type=mask_type, ablat_rate=ablat_rate)


    def predict(self,
                 test_df: pd.DataFrame,
                 output: bool = False) -> Dict[str, float]:
        """Evaluate target model on different test set"""
        # prediction
        metrics = self._evaluate(test_df)
        self._append_metrics(metrics)
        if output:
            return metrics

class ECFilter(DualFilter):
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    
    
    def _define_tracked_layers(self, n_layer: int = 12, emb: bool = True) -> List[str]:
        return track_layers(n = n_layer, emb = emb, classifier = True)
    

    def _freeze_layers(self) -> None:
        """Freeze layers based on specified layer index"""
        for name, param in self.ft_model_p.named_parameters():
            if name not in self.tracked_layers:
                param.requires_grad = False
        
        return

    def _mask_model(self, mask_ratio : float , ablat_rate: float) -> None:
        """
        masking model weights in ECF
        """
        mask_dict: Dict[str, torch.Tensor] = {}
        for layer_name in self.cum_delta_c.keys():
            threshold = np.quantile(self.cum_delta_c[layer_name].cpu().numpy(), 1 - mask_ratio)
            # if delta weight is greater than threshold, mask it
            mask = (self.cum_delta_c[layer_name] > threshold)
            mask_dict[layer_name] = ~mask
            mask_dict[layer_name][mask_dict[layer_name] == 0.] = ablat_rate

                # Calculate number of masked neuronssk)
        masked_params = np.sum([(~mask).sum().cpu() for mask in mask_dict.values()])
        total_params = self.num_model_params
        mask_perc = (masked_params / total_params) * 100 if total_params else 0

        logging.info(f"Size of the mask: {masked_params} neurons, "
                    f"which is {mask_perc:.2f}% of the trainable parameters in the primary model.")

        self.masked_params = masked_params

        # Apply masks to both models
        for (name_p, param_p), (name_c, param_c) in zip(
            self.ft_model_p.named_parameters(), self.ft_model_c.named_parameters()
        ):
            param_p.detach_()  
            param_c.detach_() 

            if name_p in mask_dict:
                assert name_p == name_c, f"Layer mismatch: {name_p} != {name_c}"
                # Retrieve mask and apply element-wise multiplication
                mask_layer = mask_dict[name_p].to(self.device)
                param_p.mul_(mask_layer)
                param_c.mul_(mask_layer)

    def train(self, train_df: pd.DataFrame, 
              eval_df: pd.DataFrame, 
              test_df: pd.DataFrame, 
              n_layers: int = None,
              emb: bool = True,
              mask_ratio: float = 0.15,
              ablat_rate: float = 0.0,
              num_epochs = 20,
              output_dir = Path("train_logs_ecf")):
        """
        Train ECF model with early stopping
        """
        
        self.tracked_layers = self._define_tracked_layers(n_layer = n_layers, emb=emb)
        self._get_config(train_df, test_df)

         # refresh mask label for evaluation key
        mask_label = f'ECF_{n_layers}_{int(emb)}_{ablat_rate}'
        self.mask_label = mask_label

        if not hasattr(self, 'base_model_p'): # avoid retraining
            logger.info("Training model for target prediction...")
            ft_model_p, _ = self._train_and_track(
                pre_model=self.pretrained_model,
                train_data=train_df,
                test_data=eval_df,
                num_epochs=num_epochs,
                output_dir=output_dir,
                phase='target',
            )
            self.ft_model_p = ft_model_p
            self.base_model_p = copy.deepcopy(self.ft_model_p)
        
        else: # refresh ft_model_p as the base model for masking
            logger.info("Base model ready...")
            self.ft_model_p = copy.deepcopy(self.base_model_p)

        if not self.metrics.get("base"):
            logger.info("Evaluating base model...")
            # get base model predictions
            base_results = self._evaluate(test_df)
            self.metrics["base"] = base_results 


        if n_layers:
            # freeze layers on ft_model_p
            self._freeze_layers()
            # record the number of trainable parameters
            self.num_model_params = np.sum([p.numel() for p in self.ft_model_p.parameters() if p.requires_grad])
            
            # select only controls to avoid impact from dementia data (Y = 0), comment out if not needed
            filtered_train = filter_control(train_df, label_val=0)
            filtered_eval = filter_control(eval_df, label_val=0)
            # train confounder model after feeze the layer       
            ft_model_c, cum_delta_c = self._train_and_track(
                pre_model=self.ft_model_p,
                train_data=filtered_train,
                test_data=filtered_eval,
                num_epochs=num_epochs,
                output_dir=output_dir,
                phase='confounder',
            )
            self.ft_model_c = ft_model_c
            self.cum_delta_c = cum_delta_c

            # masking phase
            logger.info("get delta weights update, masking base model...")
            self._mask_model(mask_ratio=mask_ratio, ablat_rate=ablat_rate)