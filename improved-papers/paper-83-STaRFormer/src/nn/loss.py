import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Any, Dict, Literal, Tuple

from src.utils import LossOptions, PredictionOptions, TrainingModeOptions, TaskOptions


__all__ = ['DAReMLoss']


class DAReMLoss(nn.Module):
    """
    Loss function associated with the Dynamic Attention-based Regional Masking (DAReM) introduced in 
    STaRFormer. 

    Supports reconstruction and self-supervised contrastive learning (SSCL) loss modes, in theory. 
    Reconstruction loss has not been implemented yet. The forward method handles both standard and 
    test-time behavior.

    Attributes:
        _pred_type (str): Prediction type.
        _task (str): Task type.
        loss (nn.Module): Contrastive loss module if using SSCL.
        task (nn.Module): Task loss module (e.g., CrossEntropy, BCEWithLogits, MSE).
    """
    def __init__(self, 
                 method: Literal['reconstruction', 'sscl']='sscl',                    
                 pred_type: Literal['binary', 'multiclass']=PredictionOptions.binary, 
                 loss_kwargs: Dict[str, Any] = None,
                 task: Literal['classification', 'regression', 'forecasting']=TaskOptions.classification, 
                 task_loss_fn: str=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Initialize the DAReMLoss module.

        Args:
            method (str, optional): Loss method ('reconstruction' or 'sscl').
            pred_type (str, optional): Prediction type ('binary' or 'multiclass').
            loss_kwargs (dict, optional): Additional arguments for loss construction.
            task (str, optional): Task type ('classification', 'regression', 'forecasting').
            task_loss_fn (str, optional): Task-specific loss function name.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If required loss_kwargs are missing for SSCL.
            NotImplementedError: If 'reconstruction' method is used.
            ValueError: For unsupported method or task.
            RuntimeError: For unsupported prediction type.
        """
        assert loss_kwargs is not None
        self._pred_type = pred_type
        self._task = task
        if method == LossOptions.reconstruction:
            raise NotImplementedError
        
        elif method == LossOptions.sscl:
            assert loss_kwargs.get('lambda_cl', None) is not None, f'`loss_kwargs` need `lambda_cl` attribute!'
            assert loss_kwargs.get('temp', None) is not None, f'`loss_kwargs` need `temp` attribute!'
            assert loss_kwargs.get('lambda_fuse_cl', None) is not None, f'`loss_kwargs` need `lambda_fuse_cl` attribute!'
            assert loss_kwargs.get('batch_size', None) is not None, f'`loss_kwargs` need `batch_size` attribute!'
            loss_kwargs['task'] = task
            loss_kwargs['task_loss_fn'] = task_loss_fn
            self.loss = DAReMContrastiveLoss(pred_type=pred_type, **loss_kwargs)
        else:
            raise ValueError(f'{method} is not implemented! Available Options {LossOptions.get_options()}')
        
        if task in [TaskOptions.classification, TaskOptions.anomaly_detection]:
            if pred_type == PredictionOptions.multiclass:
                self.task = nn.CrossEntropyLoss()
            elif pred_type == PredictionOptions.binary:
                self.task = nn.BCEWithLogitsLoss()
            else:
                raise RuntimeError(f'{pred_type} is not implemented!')
            
        elif task == TaskOptions.regression:
            self.task = nn.MSELoss() if task_loss_fn == 'mean_squarred_error' else nn.L1Loss()
        
        else:
            raise ValueError(f'{task}')
    
    def forward(self, 
                y_logits: Tensor, 
                unmasked: Tensor, 
                masked: Tensor, 
                y: Tensor, # labels
                targets: Tensor=None, # targets, in case of regression necessary
                seq_len: Tensor=None, 
                per_seq_element: bool=False, 
                mode: Literal['train', 'val', 'test'] = TrainingModeOptions.train, 
                **kwargs
        ) -> Tensor | Tuple[Tensor]:
        """
        Forward pass for computing the loss.

        Args:
            y_logits (Tensor): Predicted logits.
            unmasked (Tensor): Unmasked representations (for contrastive loss).
            masked (Tensor): Masked representations (for contrastive loss).
            y (Tensor): Ground truth labels.
            targets (Tensor, optional): Regression targets.
            seq_len (Tensor, optional): Sequence lengths.
            per_seq_element (bool, optional): Whether loss is per sequence element.
            mode (str, optional): Mode ('train', 'val', 'test').
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor or tuple: Total loss and individual components.
        """
        if mode == TrainingModeOptions.test:
            if self._pred_type == PredictionOptions.binary:
                if y_logits.dtype == torch.double:
                    labels = y.type(y_logits.dtype)
                else:
                    labels = y
                
                if y.dtype == torch.long or y.dtype == torch.int64:
                    labels = y.type(y_logits.dtype)

                y_logits = y_logits.flatten()
                labels = labels.flatten()
            elif self._pred_type == PredictionOptions.multiclass:
                labels = y.flatten()
            # padding mask
            padding_mask = labels != -1 

            if self._task == TaskOptions.regression:
                assert targets is not None, f'targets cannot be None!'
                labels = targets.flatten()
                padding_mask = torch.ones_like(labels, dtype=torch.bool)
            
            if padding_mask.dim() == 1 and y_logits.dim() != 1:
                padding_mask = padding_mask.unsqueeze(1)
                return self.task(y_logits*padding_mask, labels*padding_mask.squeeze())
            
            return self.task(y_logits*padding_mask, labels*padding_mask)

            #print(padding_mask.size(), labels.size(), y_logits.size())
            #print((padding_mask*y_logits).size(), (padding_mask*labels).size())
            #return self.task(y_logits*padding_mask, labels*padding_mask.squeeze())
            ##return self.task(padding_mask*y_logits, padding_mask*labels)
        else:
            return self.loss(
                y_logits=y_logits, unmasked=unmasked, masked=masked, 
                labels=y, targets=targets, seq_len=seq_len, 
                per_seq_element=per_seq_element)#.to(device)


class DAReMContrastiveLoss(nn.Module):
    """Contrastive loss module used with DAReM for semi-supervised training.

    Combines a contrastive loss with a supervised task loss for
    regression or classification, depending on the configuration.

    Attributes:
        loss_contrastive (SemiSupervisedCL): The contrastive loss module.
        task (nn.Module): The task-specific loss (CrossEntropy, BCEWithLogits, or MSE/L1).
    """
    def __init__(self, 
        lambda_cl: float = 1.0,
        temp: float = 0.5,
        lambda_fuse_cl: float = 0.5,
        batch_size: int = None,
        pred_type: Literal['binary', 'multiclass']=PredictionOptions.binary,
        task: Literal['classification', 'regression', 'forecasting', 'anomaly_detection']=TaskOptions.classification, 
        task_loss_fn: str=None,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialize the DAReMContrastiveLoss.

        Args:
            lambda_cl (float): Weight for the contrastive loss.
            temp (float): Temperature parameter for contrastive similarities.
            lambda_fuse_cl (float): Fusion weight for the contrastive terms.
            batch_size (int): Batch size used by the contrastive loss.
            pred_type (str): Prediction type ('binary' or 'multiclass').
            task (str): Task type ('classification', 'regression', etc.).
            task_loss_fn (str, optional): Name of the task-specific loss function.
        """
        assert batch_size is not None
        self._pred_type = pred_type
        self._lambda_cl = lambda_cl
        self._task = task

        self.loss_contrastive = SemiSupervisedCL(
            temp=temp, lambda_fuse_cl=lambda_fuse_cl, batch_size=batch_size)

        if task in [TaskOptions.classification, TaskOptions.anomaly_detection]:
            if pred_type == PredictionOptions.multiclass:
                self.task = nn.CrossEntropyLoss()
            elif pred_type == PredictionOptions.binary:
                self.task = nn.BCEWithLogitsLoss()
            else:
                raise RuntimeError(f'{pred_type} is not implemented! Try one of the following {PredictionOptions.get_options()}')
            
        elif task == TaskOptions.regression:
            self.task = nn.MSELoss() if task_loss_fn == 'mean_squarred_error' else nn.L1Loss()

    
    def forward(self, 
                y_logits: Tensor, 
                unmasked: Tensor, 
                masked: Tensor, 
                labels: Tensor,
                targets: Tensor=None, 
                seq_len: Tensor=None, 
                per_seq_element: bool=False
        ) -> Tensor | Tuple[Tensor]:
        """Compute loss components.

        Args:
            y_logits (Tensor): Model predictions.
            unmasked (Tensor): Unmasked embeddings for contrastive loss.
            masked (Tensor): Masked embeddings for contrastive loss.
            labels (Tensor): Class labels or ids.
            targets (Tensor, optional): Regression targets if needed.
            seq_len (Tensor, optional): Sequence lengths for masking.
            per_seq_element (bool): If True, compute elementwise losses.

        Returns:
            Tensor or tuple: Total loss and individual components.
                - loss (Tensor): Combined weighted total loss (task and contrastive loss).
                - loss_task (Tensor): Task loss.
                - loss_contrastive  (Tensor): Combined weighted contrastive loss.
                - loss_contrastive_batch_sim (Tensor): Batch-wise contrastive loss.
                - loss_contrastive_class_sim (Tensor): Class-wise contrastive loss.
        """
        if self._task == TaskOptions.regression:
            assert targets is not None
            loss_task = self.task(y_logits.flatten(), targets.flatten())
        
            loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.loss_contrastive(
                unmasked=unmasked, masked=masked, labels=labels, seq_len=seq_len, elementwise=per_seq_element)

            loss = loss_task + self._lambda_cl * loss_contrastive 

        elif self._task in [TaskOptions.classification, TaskOptions.anomaly_detection]:
                    
            if self._pred_type == 'binary':
                if not per_seq_element:
                    y_logits = y_logits.flatten() 
                if unmasked.dtype == torch.double:
                    labels = labels.type(unmasked.dtype)

                if labels.dtype == torch.long or labels.dtype == torch.int64 \
                    or labels.dtype == torch.int32:
                    labels = labels.type(unmasked.dtype)
            
            if per_seq_element:
                # here no masking for potential padded elements (-1) necessary, 
                # as seq_lengths are used to consider not padded elements 
                if y_logits.dim() > 2:
                    y_logits = y_logits.squeeze(-1) # [bs, N]
        
                sequential_labels = torch.concat([
                    labels[i, :n]
                    for i, n in enumerate(seq_len)
                ])
                sequential_logits = torch.concat([
                    y_logits[i, :n]
                    for i, n in enumerate(seq_len)
                ])

                loss_task = self.task(sequential_logits.flatten(), sequential_labels.flatten())
            else:
                loss_task = self.task(y_logits, labels.flatten())

            loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.loss_contrastive(
                unmasked=unmasked, masked=masked, labels=labels, seq_len=seq_len, elementwise=per_seq_element)

            loss = loss_task + self._lambda_cl * loss_contrastive 

        else:
            raise NotImplementedError(f'{self._task} is not implemented!')    
        
        return loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim


class SemiSupervisedCL(nn.Module):
    """Semi-supervised contrastive loss component for STaRFormer.

    Combines global or elementwise contrastive terms with a cross-entropy loss
    over class predictions.

    Attributes:
        _temp (float): Temperature used to scale the similarity scores.
        _lambda_fuse_cl (float): Weight for fusing the contrastive loss with the task loss.
        loss_fn (nn.Module): Loss function used for class predictions (cross-entropy).
        _loss_class_sim_prev (float): Stored previous class similarity loss (for stability).
        _batch_size (int, optional): Batch size used for some contrastive operations.
    """
    def __init__(self, 
                 temp: float = 0.5, 
                 lambda_fuse_cl: float = 0.5,
                 batch_size: int=None, 
                 *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        """Initialize SemiSupervisedCL.

        Args:
            temp (float): Temperature for contrastive scaling.
            lambda_fuse_cl (float): Fusion weight for contrastive terms.
            batch_size (int, optional): Batch size for contrastive computation.
            *args: Additional positional arguments.
            **kargs: Additional keyword arguments.
        """
        self._temp = temp
        self._lambda_fuse_cl = lambda_fuse_cl
        self.loss_fn = nn.CrossEntropyLoss()
        self._loss_class_sim_prev = 0.0
        self._batch_size = batch_size
    
    def forward(self, unmasked: Tensor, masked: Tensor, labels: Tensor, seq_len: Tensor=None, elementwise: bool=False) -> Tuple[Tensor]:
        """Compute semi-supervised contrastive loss components.

        Args:
            unmasked (Tensor): Unmasked embeddings from encoder, shape [N, B, D].
            masked (Tensor): Masked embeddings from encoder, shape [N, B, D].
            labels (Tensor): Class labels for each sample.
            seq_len (Tensor, optional): Sequence lengths for each sample, used for masking.
            elementwise (bool): If True, compute elementwise contrastive loss; otherwise compute global.

        Returns:
            tuple: (loss, loss_bw, loss_cw) where
                - loss (Tensor): Combined loss (weighted sum of contrastive terms).
                - loss_bw (Tensor): Batch-wise contrastive loss.
                - loss_cw (Tensor): Class-wise contrastive loss.
        """
        # create a mask if sequences in batch are padded
        padding_mask = None
        if seq_len is not None:
            N, B, D = unmasked.size()
            padding_mask = torch.zeros((N, B, 1), dtype=torch.bool, device=unmasked.device)
            for i, seq_len in enumerate(seq_len):
                padding_mask[:seq_len, i, :] = True

        if elementwise:
            loss_bw_cl, loss_cw_cl = self._forward_elementwise_contratsive(
                unmasked=unmasked, masked=masked, labels=labels, padding_mask=padding_mask
            )
        else:
            # default 
            loss_bw_cl, loss_cw_cl = self._forward_global_contratsive(
                unmasked=unmasked, masked=masked, labels=labels, padding_mask=padding_mask
            )

        
        loss = self._lambda_fuse_cl * loss_bw_cl + (1 - self._lambda_fuse_cl) * loss_cw_cl
        return loss, loss_bw_cl, loss_cw_cl


    def _forward_global_contratsive(self, unmasked: Tensor, masked: Tensor, labels: Tensor, padding_mask: Tensor=None) -> Tuple[Tensor]:
        """(Formulation 1) - Sequence-level formulation

        Compute global (batch-wide) contrastive losses.

        Args:
            unmasked (Tensor): Unmasked embeddings [N, B, D].
            masked (Tensor): Masked embeddings [N, B, D].
            labels (Tensor): Class labels for each element.
            padding_mask (Tensor, optional): Padding mask indicating valid elements.

        Returns:
            tuple: (loss_batch_sim, loss_class_sim) where
                - loss_batch_sim (Tensor): Batch-wise contrastive loss.
                - loss_class_sim (Tensor): Class-wise contrastive loss.
        """
        # reduce dimensionality, take sum across the sequence
        if padding_mask is not None:
            unmasked_rd = (unmasked*padding_mask).mean(dim=0) # [BS, D]
            masked_rd = (masked*padding_mask).mean(dim=0) # [BS, D]
        else:
            # contrative loss
            # reduce dimensionality, take sum across the sequence
            unmasked_rd = unmasked.mean(dim=0) # [BS, D]
            masked_rd = masked.mean(dim=0) # [BS, D]

        # normalize for cos similarity
        unmasked_rd_norm = F.normalize(unmasked_rd, dim=-1) # should use p=2 for cos similarity (default p=2)
        masked_rd_norm = F.normalize(masked_rd, dim=-1) # should use p=2 for cos similarity (default p=2)
        # if vector are normalized with L2 norm, dot product becomes cos similarity
        sim = torch.matmul(masked_rd_norm, unmasked_rd_norm.T) # [BS, BS]
        sim /= self._temp
    
        # positive pairs are the diagonal elements, i.e. the same sequence in the two embeddings
        labels_batch_sim = torch.arange(sim.size(0), device=unmasked.device)
        loss_batch_sim = self.loss_fn(sim, labels_batch_sim)
        
        # if bs < number of targets, leads to an error in cross entropy
        if max(labels) > sim.size(0) and \
            (self._batch_size > sim.size(0)):
            # resample label
            unique = torch.unique(labels)
            mapping = {v.int().item(): i for i, v in enumerate(unique)}
            labels = torch.tensor([mapping[label.int().item()] for label in labels.clone()])
        try:
            # get the positive mask, positive pairs have the same class label
            pos_mask = labels.flatten().unsqueeze(0) == labels.flatten().unsqueeze(1)
            pos_mask = pos_mask.to(sim.device)
            
            # compute the log softmax, converts sim to probabilities (row sums to 1) and take logarithm (num. stability)
            log_probs = F.log_softmax(sim, dim=1)

            loss = -(pos_mask * log_probs).sum(dim=1) / pos_mask.sum(dim=-1)
            loss_class_sim = loss.mean()
            
        except Exception as e:
            warnings.warn(f'[Contrastive Loss] -- {e} || sim-size: {sim.size()} -- labels-size:{labels.size()}')
            
            labels_batch_sim = torch.arange(sim.size(0), device=unmasked.device)
            loss_class_sim = self.loss_fn(sim, labels_batch_sim)

        self._loss_class_sim_prev = loss_class_sim.clone()

        return loss_batch_sim, loss_class_sim

    def _forward_elementwise_contratsive(self, unmasked: Tensor, masked: Tensor, labels: Tensor, padding_mask: Tensor=None) -> Tuple[Tensor]:
        """(Formulation 2) - Sequence element-level formulation
        
        Compute elementwise contrastive losses.

        Args:
            unmasked (Tensor): Unmasked embeddings [N, B, D].
            masked (Tensor): Masked embeddings [N, B, D].
            labels (Tensor): Class labels.
            padding_mask (Tensor, optional): Padding mask for valid positions.

         Returns:
            tuple: (loss_bw_elementwise, loss_cw_elementwise) where
                - loss_bw_elementwise (Tensor): Batch-wise contrastive loss calculated element in the sequence.
                - loss_cw_elementwise (Tensor): Class-wise contrastive loss calculated element in the sequence.
        """
        N, B, D = unmasked.shape
        # 
        unmasked_flat = unmasked.reshape(B*N, D) # (N*B, D)
        masked_flat = masked.reshape(B*N, D) # (N*B, D)

        # norm 
        # requires p=2 for cos similarity, p=2 default in F.normalize()
        if padding_mask is not None:
            unmasked_norm_flat = F.normalize(unmasked_flat*padding_mask.reshape(N*B, -1), dim=(-1))
            masked_norm_flat = F.normalize(masked_flat*padding_mask.reshape(N*B, -1), dim=(-1))
        else:
            unmasked_norm_flat = F.normalize(unmasked_flat, dim=(-1))
            masked_norm_flat = F.normalize(masked_flat, dim=(-1))

        # inter-class similarity
        sim_bw = torch.matmul(masked_norm_flat, unmasked_norm_flat.T)
        sim_bw /= self._temp

        # batch-wise contribution
        labels_bw_sim = torch.arange(N*B, device=unmasked.device)
        loss_bw_elementwise = self._bw_cl_elementwise(sim=sim_bw, labels=labels_bw_sim)

        # class-wise contribution 
        loss_cw_elementwise = self._cw_cl_elementwise(
            unmasked=unmasked, masked=masked, labels=labels, sim_bw=sim_bw)
        return loss_bw_elementwise, loss_cw_elementwise

    def _bw_cl_elementwise(self, sim: Tensor, labels: Tensor) -> Tensor:
        """Compute batch-wise contrastive loss for elementwise terms.

        Args:
            sim (Tensor): Similarity matrix for batch elements.
            labels (Tensor): Labels for positive pair selection.

        Returns:
            loss_bw_elementwise (Tensor): Batch-wise contrastive loss calculated element in the sequence.
        """
        return self.loss_fn(sim, labels)

    def _cw_cl_elementwise(self, unmasked: Tensor, masked: Tensor, labels: Tensor, sim_bw: Tensor, padding_mask: Tensor=None) -> Tensor:
        """Compute class-wise contrastive loss for elementwise terms.

        Args:
            unmasked (Tensor): Unmasked embeddings [N, B, D].
            masked (Tensor): Masked embeddings [N, B, D].
            labels (Tensor): Class labels.
            sim_bw (Tensor): Batch-wide similarity matrix.
            padding_mask (Tensor, optional): Padding mask.

        Returns:
            loss_cw_elementwise (Tensor): Class-wise contrastive loss calculated element in the sequence.
                Sum of intra- and inter-class elementwise losses.
        """
        N, B, _ = unmasked.shape

        # intra-class contribution
        loss_intra_class = self._intra_class_sim_elementwise(unmasked=unmasked,
            masked=masked, labels=labels, padding_mask=padding_mask)
        # inter-class contribution
        labels_bw_sim = torch.arange(N*B, device=unmasked.device)
        loss_inter_class = self._inter_class_sim_elementwise(N, B, sim_bw, labels_bw_sim)

        return loss_intra_class + loss_inter_class


    def _intra_class_sim_elementwise(self, unmasked: Tensor, masked: Tensor, labels: Tensor, padding_mask: Tensor=None) -> Tensor:
        """Intra-class contrastive loss for elementwise terms.

        Args:
            unmasked (Tensor): Unmasked embeddings [N, B, D].
            masked (Tensor): Masked embeddings [N, B, D].
            labels (Tensor): Class labels.
            padding_mask (Tensor, optional): Padding mask.

        Returns:
            Intra-class loss (Tensor)
        """
        temp = 1.0 # set specific temperature
        N, _, _ = unmasked.shape

        # requires p=2 for cos similarity, p=2 default in F.normalize()
        if padding_mask is not None:
            unmasked_norm = F.normalize(unmasked*padding_mask, dim=(0,2)) # (N, B, D)
            masked_norm = F.normalize(masked*padding_mask, dim=(0,2)) # (N, B, D)
        else:
            unmasked_norm = F.normalize(unmasked, dim=(0,2)) # (N, B, D)
            masked_norm = F.normalize(masked, dim=(0,2)) # (N, B, D)
    

        #  bmm (B, N, D) x (B, D, N) --> (B, N, N)
        sim = torch.bmm(masked_norm.permute(1,0,2), unmasked_norm.permute(1,2,0))
        sim /= temp

        # get labels
        labels_dim1 = labels.unsqueeze(2) # (B, N, 1)
        labels_dim2 = labels.unsqueeze(1) # (B, 1, N)

        # labels have -1 indicating padding
        labels_dim1_bool = labels_dim1 != -1 
        labels_dim2_bool = labels_dim2 != -1

        # True where labels match and i not equal to j (same element) 
        pos_mask = (labels_dim1 == labels_dim2) & (~torch.eye(N, dtype=bool, device=unmasked.device).unsqueeze(0)) & (labels_dim1_bool & labels_dim2_bool)

        # calc loss
        return self.custom_contrastive_loss(sim, pos_mask)

    def _inter_class_sim_elementwise(self, N, B, sim, labels) -> Tensor:
        """Inter-class contrastive loss for elementwise terms.

        Args:
            N (int): Sequence length dimension.
            B (int): Batch size.
            sim (Tensor): Batch-wise similarity matrix.
            labels (Tensor): Class labels.

        Returns:
            Inter-class loss (Tensor)
        """
        # disregard the same sequence (intra class)
        # (0,0,0..., 1,1,..., 2,2,...)
        seq_id = torch.repeat_interleave(torch.arange(B), N).to(labels.device)

        seq_id_i = seq_id.unsqueeze(1)
        seq_id_j = seq_id.unsqueeze(0)
        inter_mask = seq_id_i != seq_id_j

        # get labels
        labels_dim1 = labels.flatten().unsqueeze(1) # (N*B, 1)
        labels_dim2 = labels.flatten().unsqueeze(0) # (1, N*B)

        # labels have -1 indicating padding
        labels_dim1_bool = labels_dim1 != -1 
        labels_dim2_bool = labels_dim2 != -1

        # create positive mask, exluced -1 label (padding) and only consider inter-class
        pos_mask = (labels_dim1 == labels_dim2) & (labels_dim1_bool & labels_dim2_bool) & inter_mask
        
        # calc loss
        return self.custom_contrastive_loss(sim, pos_mask)


    @staticmethod
    def custom_contrastive_loss(sim, pos_mask, eps: float=1e-06) -> Tensor:
        """Compute a generic contrastive loss from a similarity matrix and a positive-mask.

        Args:
            sim (Tensor): Similarity matrix.
            pos_mask (Tensor): Boolean mask of positive pairs.
            eps (float, optional): Small value to avoid division by zero.

        Returns:
            los (Tensor): Scalar loss value.
        """
        log_probs = F.log_softmax(sim, dim=-1)
        loss = -(pos_mask * log_probs).sum(dim=1) / (pos_mask.sum(dim=-1) + eps)
        return loss.mean()

class DAReMReconstructionLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


def check_tensor_elements_are_equal_to_each_other(tensor):
    repeated = tensor[0].repeat(len(tensor))
    return torch.all(repeated == tensor)