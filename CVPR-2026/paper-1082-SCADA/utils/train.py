import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from tllib.utils.data import ForeverDataIterator

from utils.utils import validate
from utils.sfda import SFDA2




def train_source(classifier, train_dl, test_dl, config, save=True, smooth=True):
    """
    Train Source Domain Classifier (With Label Smoothing)

    """
    
    
    if smooth:
        loss_fn = CrossEntropyLabelSmooth(config)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optim = SGD(classifier.get_parameters(),
                    lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optim, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.75)

    train_dl = ForeverDataIterator(train_dl)
    
    for epoch in range(config['source_epochs']):
        classifier.train()
        loss_val = 0

        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            images, labels = next(train_dl)
            images, labels = images.to(config['device']), labels.to(config['device'])
            logits = classifier(images)[0]
            loss = loss_fn(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()

            loss_val += loss.item()

        loss_val /= config['iter_per_epoch']

        if (not config['fast_train']) or (epoch == config['source_epochs'] - 1):

            if save:
                source_path = os.path.join(config['save_path'], config['dataset'], f"{config['source']}.pt")
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                torch.save(classifier.state_dict(), source_path)
        
            acc = validate(classifier, test_dl, config)        

            print(f"----------\n"
                    f"Loss     : {loss_val:.2f}\n"
                    f"Accuracy : {acc:.2f}\n" 
                    f"----------\n")

    return classifier


def train_sfda2(classifier, target_train_dl, target_test_dl, source_test_dl, config, save=None):
    """
    SFDA^2 Domain Adaptation

    """


    sfda2 = SFDA2(classifier, config)
    indexed_target_train_dl = sfda2.label_dataset(target_train_dl)
    sfda2.create_banks(indexed_target_train_dl)

    indexed_target_train_dl = ForeverDataIterator(indexed_target_train_dl)
    optimizer = SGD(classifier.get_parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: 1e-2 * (1. + 1e-3 * float(x)) ** -0.90)

    for epoch in range(config['epochs']):
        classifier.train()
        loss_val = 0
        
        for i in tqdm(range(config['iter_per_epoch']), desc=f"Epoch {epoch+1}"):
            classifier.train()
            images, indices = next(indexed_target_train_dl)
            images = images.cuda()

            loss = sfda2.loss(images, indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            loss_val += loss.item()

        if (not config['fast_train']) or (epoch == config['epochs'] - 1):
            
            if save:
                path = os.path.join(config['save_path'], config['dataset'], save, f"{config['source']}_{config['target']}.pt")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(classifier.state_dict(), path)
        
    
            loss_val /= config['iter_per_epoch']
            source_acc = validate(classifier, source_test_dl, config)
            target_acc = validate(classifier, target_test_dl, config)

            print(f"----------\n"
                  f"Epoch                  : {epoch+1}\n"
                  f"Train Loss             : {loss_val:.2f}\n"
                  f"Source Accuracy        : {source_acc:.2f}\n" 
                  f"Target Accuracy        : {target_acc:.2f}\n" 
                  f"----------\n")

    return classifier


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing regularizer.
    """


    def __init__(self, config, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.config = config
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """


        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.to(self.config['device'])
        targets = (1 - self.config['smooth']) * targets + self.config['smooth'] / self.config['num_classes']
        loss = (-targets * log_probs).sum(dim=1)
        
        if self.reduction:
            return loss.mean()

        return loss
