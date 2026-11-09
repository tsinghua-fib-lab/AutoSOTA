# Train CIFAR-10
## Environment
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.01    Driver Version: 418.87.01    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1B.0 Off |                    0 |
| N/A   38C    P0    65W / 300W |   2302MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:00:1C.0 Off |                    0 |
| N/A   37C    P0    63W / 300W |   2370MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:00:1D.0 Off |                    0 |
| N/A   40C    P0    66W / 300W |   2370MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   41C    P0    66W / 300W |   2382MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
```

## Code
```python
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.callbacks import EarlyStopping, SaveSnapshot
from kuma_utils.torch.hooks import SimpleHook
from kuma_utils.torch.model_zoo import se_resnext50_32x4d
from kuma_utils.metrics import Accuracy


@dataclass
class Config:
    num_workers: int = 32
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_rounds: int = 5


def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train = torchvision.datasets.CIFAR10(
        root='input', train=True, download=False, transform=transform)
    test = torchvision.datasets.CIFAR10(
        root='input', train=False, download=False, transform=transform)
    return train, test


def split_dataset(dataset, index):
    new_dataset = deepcopy(dataset)
    new_dataset.data = new_dataset.data[index]
    new_dataset.targets = np.array(new_dataset.targets)[index]
    return new_dataset


def get_model(num_classes):
    model = se_resnext50_32x4d(pretrained='imagenet')
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":

    cfg = Config(
        num_workers=32, 
        batch_size=2048,
        num_epochs=20,
        early_stopping_rounds=5,
    )

    train, test = get_dataset()
    print('classes', train.classes)
    
    predictions = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(train.targets, train.targets)):

        print(f'fold{fold} starting')

        valid_fold = split_dataset(train, valid_idx)
        train_fold = split_dataset(train, train_idx)

        print(f'train: {len(train_fold)} / valid: {len(valid_fold)}')

        loader_train = D.DataLoader(
            train_fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
            shuffle=True, pin_memory=True)
        loader_valid = D.DataLoader(
            valid_fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
            shuffle=False, pin_memory=True)
        loader_test = D.DataLoader(
            test, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
            shuffle=False, pin_memory=True)

        model = get_model(num_classes=len(train.classes))
        optimizer = optim.Adam(model.parameters(), lr=2e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2)
        logger = TorchLogger(
            path='results/demo/log', 
            log_items='epoch train_loss valid_loss valid_metric learning_rate early_stop', 
            file=True)
        
        trn = TorchTrainer(model, serial=f'fold{fold}')
        trn.train(
            loader=loader_train,
            loader_valid=loader_valid,
            loader_test=loader_test,
            criterion=nn.CrossEntropyLoss(),
            eval_metric=Accuracy().torch, 
            monitor_metrics=[
                Accuracy().torch
            ],
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_target='valid_loss', # ReduceLROnPlateau reads metric each epoch
            num_epochs=cfg.num_epochs,
            hook=SimpleHook(evaluate_in_batch=False),
            callbacks=[
                EarlyStopping(
                    patience=cfg.early_stopping_rounds, 
                    target='valid_metric', 
                    maximize=True),
                SaveSnapshot() # Default snapshot path: {export_dir}/{serial}.pt
            ],
            logger=logger, 
            export_dir='results/demo',
            parallel='ddp',
            fp16=True,
            deterministic=True, 
            random_state=0, 
            progress_bar=True # Progress bar shows batches done
        )

        oof = trn.predict(valid_loader)
        predictions.append(trn.predict(test_loader))

        score = Accuracy()(valid_fold.targets, oof)
        print(f'Folf{fold} score: {score:.6f}')
        break
```
## Results
```
classes ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
fold0 starting
train: 40000 / valid: 10000
TorchLogger created at 20/12/11:07:11:15
07:11:15 DDP on tcp://127.0.0.1:35563
07:11:31 All processes initialized.
07:11:31 Mixed precision training on torch amp.
07:11:34 DistributedDataParallel on devices [0, 1, 2, 3]
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.16it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.13it/s]
07:11:44 Epoch  1/20 | train_loss=1.437213 | valid_loss=5.349814 | valid_metric=0.555800 | learning_rate=0.002000 |  | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.27it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.29it/s]
07:11:55 Epoch  2/20 | train_loss=0.576135 | valid_loss=0.640557 | valid_metric=0.794100 | learning_rate=0.002000 |  | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.26it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.03it/s]
07:12:06 Epoch  3/20 | train_loss=0.273112 | valid_loss=0.754993 | valid_metric=0.783700 | learning_rate=0.002000 | (*1) | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.26it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.14it/s]
...
...
07:14:36 Epoch 17/20 | train_loss=0.000053 | valid_loss=0.873186 | valid_metric=0.856700 | learning_rate=0.001000 |  | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.34it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.29it/s]
07:14:47 Epoch 18/20 | train_loss=0.000046 | valid_loss=0.881786 | valid_metric=0.856900 | learning_rate=0.001000 |  | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.22it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.36it/s]
07:14:58 Epoch 19/20 | train_loss=0.000040 | valid_loss=0.889885 | valid_metric=0.857100 | learning_rate=0.001000 |  | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:08<00:00,  2.24it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  4.40it/s]
07:15:10 Epoch 20/20 | train_loss=0.000036 | valid_loss=0.897552 | valid_metric=0.857000 | learning_rate=0.001000 | (*1) | 
Folf0 score: 0.857100
```

# Module description
## `kuma_utils.torch.TorchTrainer`
WIP

## `kuma_utils.torch.callbacks.EarlyStopping`
```python
# Sample
EarlyStopping(
    patience: int = 5, 
    target: str = 'valid_metric', 
    maximize: bool = False, 
    skip_epoch: int = 0 
)
```
| argument   | description                                                                                         |
|------------|-----------------------------------------------------------------------------------------------------|
| patience   | Epochs to wait before early stop                                                                    |
| target     | Variable name to watch (choose from  `['train_loss', 'train_metric', 'valid_loss', 'valid_metric']`) |
| maximize   | Whether to maximize the target                                                                      |
| skip_epoch | Epochs to skip before early stop counter starts                                                     |


## `kuma_utils.torch.TorchLogger`
```python
TorchLogger(
    path: (str, pathlib.Path),
    log_items: (list, str) = [
        'epoch', 'train_loss', 'valid_loss', 'train_metric', 'valid_metric',
        'train_monitor', 'valid_monitor', 'learning_rate', 'early_stop'
        ],
    verbose_eval: int = 1,
    stdout: bool = True, 
    file: bool = False
)
```
| argument     | description                                                                                                                                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| path         | Path to export log.                                                                                                                                                                                                |
| log_items    | Items to be shown in log. Must be a combination of the following items:  `['epoch',  'train_loss', 'valid_loss', 'train_metric' , 'valid_metric', 'train_monitor',  'valid_monitor', 'learning_rate', 'early_stop', 'gpu_memory']`. List or string separated by space (e.g. `'epoch valid_loss learning_rate'`).| 
| verbose_eval | Frequency of log (unit: epoch).                                                                                                                                                                              |
| stdout       | Whether to print log.                                                                                                                                                                            |
| file         | Whether to export log file to the path. (False by default)                                                                                                                                                                          |


## Hook
Hook is used to specify detailed training and evaluation process.
Basically it is not necessary to modify this part, but in some special cases such like

- training a Graph Neural Network which takes multiple arguments in `.forward`
- training with a special metric which requires extra variables (other than predictions and targets)
- calculate metrics on whole dataset (not in each mini-batch)

A Hook class contains the following functions:
```python
class SampleHook:
    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        approx = trainer.model(inputs[0])
        loss = trainer.criterion(approx, target)
        return loss, approx

    def forward_test(self, trainer, inputs):
        approx = trainer.model(inputs[0])
        return approx

    def evaluate_batch(self, trainer, inputs, approx):
        storage = trainer.epoch_storage
        storage['approx'].append(approx)
        storage['target'].append(target)

    def evaluate_epoch(self, trainer):
        storage = trainer.epoch_storage
        metric_total = trainer.eval_metric(
                    storage['approx'], storage['target'])
                
        monitor_metrics_total = []
        for monitor_metric in trainer.monitor_metrics:
            monitor_metrics_total.append(
                monitor_metric(storage['approx'], storage['target']))
        return metric_total, monitor_metrics_total
```

`.forward_train()` is called in each mini-batch in training and validation loop. 
This method returns loss and prediction tensors.

`.forward_test()` is called in each mini-batch in inference loop. 
This method returns prediction values tensor.

`.evaluate_batch()` is called in each mini-batch after back-propagation and optimizer.step(). 
This method returns nothing.

`.evaluate_epoch()` is called at the end of each training and validation loop. 
This method returns eval_metric (scaler) and monitor metrics (list).

Note that `trainer.epoch_storage` is a dicationary object you can use freely. 
In `SampleHook`,  predictions and targets are added to storage in each mini-batch, 
and at the end of loop, metrics are calculated on the whole dataset 
(tensors are concatenated batch-wise automatically).

## Callback
TODO