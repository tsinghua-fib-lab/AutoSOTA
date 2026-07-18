import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from logdiff.cs_classifier.metrics import MultiLabelAcc, MultiLabelRecall

class ClassifierTrainer(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 num_classes_per_label:list[int],
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 epochs: int = 100,
                 loss_weights: list[list[float]] = None):
        super().__init__()
        
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.val_acc = MultiLabelAcc(num_classes_per_label)
        self.train_acc = MultiLabelAcc(num_classes_per_label)
        self.val_recall = MultiLabelRecall(num_classes_per_label, average='none')
        self.num_classes_per_label = num_classes_per_label
        self.epochs = epochs

        # weights for weighted loss
        self.use_weights = False
        if loss_weights is not None:
            self.use_weights = True
            # Store names of weight buffers
            self.weight_names = [] 
            for i, weights in enumerate(loss_weights):
                weight_tensor = torch.tensor(weights, dtype=torch.float32)
                name = f'loss_weight_{i}'
                self.register_buffer(name, weight_tensor)
                self.weight_names.append(name)

    
    def configure_optimizers(self):
        """
        optimizer and scheduler
        """
        optimizer = self.hparams.optimizer(self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer, T_max=self.epochs)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _step(self, batch, batch_idx):
        """
        Shared logic for forward pass and loss calculation.
        """
        x, y = batch['X'], batch['label']
        logits = self.model(x)
        loss = 0
        
        for i in range(len(self.num_classes_per_label)):
            weight = None 
            if self.use_weights and i < len(self.weight_names):
                weight = getattr(self, self.weight_names[i])
            loss += F.cross_entropy(logits[i], y[:, i], weight=weight)
            
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, batch_idx)
        
        self.train_acc.update(logits, y)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, batch_idx)
        
        self.val_acc.update(logits, y)
        self.val_recall.update(logits, y)
        self.log('val_loss', loss)
        
        return loss
    
    def on_validation_epoch_end(self):
        for i in range(len(self.num_classes_per_label)):
            self.log(f'val_accuracy_{i}', self.val_acc[i].compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.val_acc.reset()

        all_recall_scores = self.val_recall.compute() 
        for label_idx, scores in enumerate(all_recall_scores):
            for class_idx in range(scores.shape[0]):
                self.log(
                    f'val_recall_label{label_idx}_class{class_idx}', 
                    scores[class_idx], 
                    prog_bar=True, 
                    sync_dist=True
                )
        self.val_recall.reset()
            
        return {}
    
    def on_train_epoch_end(self):
        for i in range(len(self.num_classes_per_label)):
            self.log(f'train_accuracy_{i}', self.train_acc[i].compute(), on_epoch=True, sync_dist=True)
        self.train_acc.reset()
        return {}
    
    def on_save_checkpoint(self, checkpoint):
        #save only the model
        checkpoint['state_dict'] = self.model.state_dict()
        return checkpoint