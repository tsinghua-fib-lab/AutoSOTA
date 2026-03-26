#
# adapted from https://github.com/ChangWeiTan/TS-Extrinsic-Regression/tree/master/models
#
import os
import time
import torch
import math
import wandb

import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_ema import ExponentialMovingAverage as EMA

from src.models.tsr.time_series_models import TimeSeriesRegressor
from src.utils.runtime.tsr.tools import save_train_duration, save_test_duration
from src.utils.datasets.tsr import TSRData, TSRBatchData
from src.utils.runtime.tsr.tools import get_synthetic_class_label
from src.nn.loss import DAReMLoss
from src.utils import TrainingModeOptions, DatasetOptions, TaskOptions, LossOptions, ModelOptions
from tqdm.auto import tqdm



class History:
    def __init__(self, history: dict):
        self.history = history

def plot_epochs_metric(hist, file_name, model, metric='loss'):
    """
    Plot the train/test metrics of Deep Learning models

    Inputs:
        hist: training history
        file_name: save file name
        model: model name
        metric: metric
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist.history[metric], label="train")
    if "val_" + metric in hist.history.keys():
        plt.plot(hist.history['val_' + metric], label="val")

    if metric != 'lr':
        min_train = np.min(hist.history[metric])
        idx_train = np.argmin(hist.history[metric])
        plt.plot(idx_train, min_train, "rx", label="best train epoch")
    
    if "val_" + metric in hist.history.keys():
        plt.plot(idx_train, hist.history['val_' + metric][idx_train], "r+", label="best val epoch")

    plt.title(model + " " + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


class TSRDataset(Dataset):
    """A dataset class for time series regression tasks.

    This dataset holds input data, targets, and label information for each sample, 
    and returns a TSRData object upon indexing.

    Attributes:
        x: Array-like or tensor containing the input data for each sample.
        y: Array-like or tensor containing the target values for each sample.
        labels: Array-like or tensor containing the label information for each sample.
    """
    def __init__(self, x, y, labels):
        super().__init__()
        """Initialize the TSRDataset.

        Args:
            x: Array-like or tensor containing the input data for each sample.
            y: Array-like or tensor containing the target values for each sample.
            labels: Array-like or tensor containing the label information for each sample.
        """

        self.x = x
        self.y = y
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.y)
    
    def __getitem__(self, index):
        """Retrieve a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            TSRData: A data object containing the input data, label, and target for the sample.
        """
        return TSRData(
            data=self.x[index],
            label=self.labels[index],
            target=self.y[index],
        )
    

def collate_tsr(batch):
    """Collate a batch of TSRData objects into a TSRBAtchData object.

    This function is designed to be used as a collate_fn in a PyTorch DataLoader
    for time series regression/classification tasks. It pads sequences in the batch
    to the same length, concatenates labels and targets, and collects sequence lengths.

    Args:
        batch (list of TSRData): A list of TSRData objects, each containing data, label,
            target, and seq_len attributes.

    Returns:
        TSRBatchData: A batched data object containing:
            - data (Tensor): Padded sequence batch of shape (max_seq_len, batch_size, feature_dim).
            - label (Tensor): Tensor of concatenated labels for the batch.
            - target (Tensor): Tensor of concatenated targets for the batch.
            - seq_len (Tensor): Tensor of sequence lengths for each sample in the batch.

    Notes:
        - Each sample in the batch is assumed to be a TSRData object.
        - `data` is padded along the sequence length dimension (using pad_sequence from PyTorch).
        - `label`, `target`, and `seq_len` are concatenated across the batch.
        - Output is suitable for direct input to a model expecting batched time series data.

    Example:
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_tsr)
        >>> batch = next(iter(loader))
        >>> batch.data.shape  # (max_seq_len, batch_size, feature_dim)
    """
    data = []
    label = []
    target = []
    seq_len = []
    for sample in batch:
        # sample is TSRData Type
        data.append(sample.data) # add additional dimension for concatenation
        label.append(sample.label.unsqueeze(0)) # add additional dimension for concatenation
        target.append(sample.target.unsqueeze(0)) # add additional dimension for concatenation
        seq_len.append(sample.seq_len.unsqueeze(0)) # add additional dimension for concatenation

    
    #data = torch.concat(data, dim=0).permute(1, 0, 2) # [bs, N, D] --> [N, D, bs]
    data = pad_sequence(data, batch_first=False, padding_value=0)
    label = torch.concat(label)
    target = torch.concat(target).to(data.device)
    seq_len = torch.concat(seq_len).to(data.device)
    return TSRBatchData(data=data, label=label, target=target, seq_len=seq_len) #, scaler=sample.scaler)



class DLRegressorPyTorch(TimeSeriesRegressor):
    """Deep learning regressor for time series data using PyTorch.

    This class handles model construction, training, validation, and prediction
    for deep learning approaches for time series regression tasks.

    Attributes:
        name (str): Name of the regressor.
        model_init_file (str): Path to save the initial model parameters.
        best_model_file (str): Path to save the best model parameters.
        X_train: Training input data.
        y_train: Training target data.
        X_val: Validation input data.
        y_val: Validation target data.
        callbacks: Training callbacks.
        hist: Training history.
        verbose (bool): Verbosity flag.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        loss (str or callable): Loss function.
        metrics (list): List of metrics for evaluation.
        config (DictConfig): Configuration object.
        model (nn.Module): The constructed deep learning model.
        _y_scale_ds (list): Datasets requiring target scaling.
        _val_ds_eval_batch (list): Datasets evaluated in batch mode during validation.
    """

    name = "DeepLearningTSRPyTorch"
    model_init_file = "model_init.pt"
    best_model_file = "best_model.pt"

    def __init__(self, 
                 output_directory,
                 input_shape,
                 verbose=False,
                 epochs=200,
                 batch_size=16,
                 loss="mean_squared_error",
                 metrics=None,
                 config: DictConfig=None
            ):
        super().__init__(output_directory)
        """
        Initialize the regressor and set up model and training configuration.

        Args:
            output_directory (str): Directory to save model checkpoints and results.
            input_shape (tuple): Shape of input data (num_examples, num_timesteps, num_channels).
            verbose (bool, optional): Verbosity flag. Default is False.
            epochs (int, optional): Number of training epochs. Default is 200.
            batch_size (int, optional): Training batch size. Default is 16.
            loss (str or callable, optional): Loss function. Default is "mean_squared_error".
            metrics (list, optional): List of metrics for evaluation.
            config (DictConfig, optional): Configuration object.
        """


        print('[{}] Creating Regressor'.format(self.name))
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.callbacks = None
        self.hist = None

        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        #print('\n\n\n',self.batch_size,'\n\n\n')
        self.loss = loss
        if metrics is None:
            metrics = ["mae", "rmse", "lr"]
        self.metrics = metrics
        self.config = config

        self.model = self.build_model()

        if self.model is not None:
            print(self.model)
            torch.save(self.model.state_dict(), self.output_directory + self.model_init_file)
            #self.model.save_weights(self.output_directory + self.model_init_file)
        if not config.logger.sweep:
            wandb.init(
                entity=config.logger.entity,
                project=config.logger.project,
                name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                config=OmegaConf.to_container(config, resolve=True),
            )
        
        self._y_scale_ds = [
            DatasetOptions.australiarainfall,
            DatasetOptions.beijingpm10quality,
            DatasetOptions.beijingpm25quality,
            DatasetOptions.benzeneconcentration, 
            DatasetOptions.bidmcrr,
            DatasetOptions.bidmchr, 
            DatasetOptions.bidmcspo2,
            DatasetOptions.covid3month, 
            DatasetOptions.floodmodeling1, 
            DatasetOptions.floodmodeling2, 
            DatasetOptions.floodmodeling3, 
            DatasetOptions.householdpowerconsumption1, 
            DatasetOptions.householdpowerconsumption2, 
            DatasetOptions.ieeeppg,
            DatasetOptions.livefuelmoisturecontent,
            DatasetOptions.newsheadlinesentiment,
            DatasetOptions.newstitlesentiment,
            DatasetOptions.ppgdalia
        ]

        self._val_ds_eval_batch = [
            DatasetOptions.australiarainfall,
            DatasetOptions.bidmcrr,
            DatasetOptions.bidmchr, 
            DatasetOptions.bidmcspo2,
            DatasetOptions.ieeeppg, 
            DatasetOptions.householdpowerconsumption1, 
            DatasetOptions.householdpowerconsumption2,
            DatasetOptions.livefuelmoisturecontent,
            DatasetOptions.newsheadlinesentiment,
            DatasetOptions.newstitlesentiment,
            DatasetOptions.ppgdalia
        ]

        wandb.watch(self.model, log_freq=100)
    
    def build_model(self, **kwargs):
        """
        Build the DL models

        Inputs:
            input_shape: input shape for the models
        """
        pass

    
    def fit(self, x_train, y_train, x_val=None, y_val=None, monitor_val=True):
        """
        Orchestrates the fiting of DL models

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
            monitor_val: boolean indicating if model selection should be done on validation
        """
        #if self.config.model.sequ == 'SequenceModel':
        if self.config.model.sequence_model.name == ModelOptions.starformer:
            self._fit_starformer(x_train, y_train, x_val, y_val, monitor_val)
        else:
            self._fit(x_train, y_train, x_val, y_val, monitor_val)

            
    def _fit(self, x_train, y_train, x_val=None, y_val=None, monitor_val=True):
        """
        Train logic for all model, excluding STaRFormer.

        Args:
            x_train (array-like): Training input data.
            y_train (array-like): Training targets.
            x_val (array-like, optional): Validation input data.
            y_val (array-like, optional): Validation targets.
            monitor_val (bool, optional): Whether to use validation for early stopping.

        Returns:
            None
        """

        print('[{}] Training'.format(self.name))

        start_time = time.perf_counter()

        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'

        self.X_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if x_val is not None:
            self.X_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            self.y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
    
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.config.training.learning_rate,
                               betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                               eps=self.config.optimizer.eps,
                               weight_decay=self.config.optimizer.weight_decay
        )
        self.loss_fn = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config.callbacks.lr_scheduler.mode,
            factor=self.config.callbacks.lr_scheduler.factor,
            patience=self.config.callbacks.lr_scheduler.patience,
            min_lr=self.config.callbacks.lr_scheduler.min_lr,
        )
        
        self.best_loss = np.inf
        self.model.to(self.device)
        self.best_model_path = None
        loss_hist, val_loss_hist  = [], []
        mae_hist, val_mae_hist  = [], [] 
        rmse_hist, val_rmse_hist  = [], [] 
        lr_hist = []
        early_stop_counter = 0
        for epoch in range(self.epochs):
            self.model.train()
            loss_epoch = []
            mae_epoch = []
            rmse_epoch = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs.squeeze(), batch_y)
                mae = self.mae(outputs.squeeze(), batch_y)
                rmse = math.sqrt(loss.item())
                loss.backward()
                optimizer.step()

                loss_epoch.append(loss.item())
                mae_epoch.append(mae.item())
                rmse_epoch.append(rmse)
            
            if x_val is not None and monitor_val:
                val_loss, val_mae, val_rmse = self.validate(epoch)
                
                if val_rmse <= self.best_loss:
                    #print(f'Val RMSE improved at Epoch {epoch}: previous: {self.best_loss} --> new best: {val_rmse}!')
                    self.best_loss = val_rmse
                    if self.best_model_path is not None and os.path.exists(self.best_model_path):
                        os.remove(self.best_model_path)
                    self.best_model_path = os.path.join(self.output_directory, self.best_model_file.replace('best_model', f'best_model_{epoch}'))
                    torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, self.best_model_path)
                    early_stop_counter = 0
                else:
                    early_stop_counter +=1
            
            before_lr = optimizer.param_groups[0]["lr"]

            loss = sum(loss_epoch) / len(train_loader)
            mae = sum(mae_epoch) / len(train_loader)
            rmse = sum(rmse_epoch) / len(train_loader)

            # logging
            log_dict = {
                f'{TrainingModeOptions.train}/loss_task': loss,
                f'{TrainingModeOptions.train}/mae': mae,
                f'{TrainingModeOptions.train}/rmse': rmse,
                'epoch': time.perf_counter() - start_time,
                'lr-Adam': before_lr,
            }
            wandb.log(log_dict, step=epoch)

            loss_hist.append(loss), mae_hist.append(mae), rmse_hist.append(rmse)
            val_loss_hist.append(val_loss), val_mae_hist.append(val_mae), val_rmse_hist.append(val_rmse)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} | train/loss: {loss:.4f} -- train/mae: {mae:.4f} -- train/rmse: {rmse:.4f} -- lr: {before_lr:.6f}")
            
            # apply lr scheduler
            scheduler.step(metrics=val_loss)
            after_lr = optimizer.param_groups[0]["lr"]
            lr_hist.append(before_lr)

            # early stopping
            if early_stop_counter == self.config.callbacks.early_stop.patience:
                print(f'[{self.name}] Monitoring metric val/rmse has not improved for {self.config.callbacks.early_stop.patience} Epochs, Stopping training at Epoch {epoch}!')
                break
        
        self.train_duration = time.perf_counter() - start_time
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

        self.hist = History(history={
            'loss': loss_hist,
            'mae': loss_hist,
            'val_loss': val_loss_hist, 
            'val_mae': val_mae_hist,
            'lr': lr_hist
        })
        plot_epochs_metric(self.hist,
                           self.output_directory + 'epochs_loss.png',
                           metric='loss',
                           model=self.name)
        for m in self.metrics:
            plot_epochs_metric(self.hist,
                               self.output_directory + 'epochs_{}.png'.format(m),
                               metric=m,
                               model=self.name)

    def _fit_starformer(self, x_train, y_train, x_val=None, y_val=None, monitor_val=True):
        """
        Train logic for the STaRFormer model.

        Args:
            x_train (array-like): Training input data.
            y_train (array-like): Training targets.
            x_val (array-like, optional): Validation input data.
            y_val (array-like, optional): Validation targets.
            monitor_val (bool, optional): Whether to use validation for early stopping.

        Returns:
            None
        """

        print('[{}] Training'.format(self.name))

        start_time = time.perf_counter()

        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'

        if self.config.dataset in self._y_scale_ds:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            if self.config.norm == 'standard' or self.config.norm is None:
                self.y_scaler = StandardScaler()
                # to be consistent with data processing
                self.y_scaler_test = StandardScaler()
            elif self.config.norm == 'minmax':
                self.y_scaler = MinMaxScaler()
                # to be consistent with data processing
                self.y_scaler_test = MinMaxScaler()
            
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
            y_val = self.y_scaler_test.fit_transform(y_val.reshape(-1,1))

        self.X_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n_clusters = 6 if self.config.get('synthetic_labels', None) is None else self.config.synthetic_labels.n_clusters
        #print(f'\n\n\n{n_clusters=}\n\n\n')
        synthetic_labels = get_synthetic_class_label(y_train.flatten(), y_val.flatten(), 
                                                     seed=42, kmeans_kwargs={'n_clusters': n_clusters})

        if x_val is not None:
            self.X_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            self.y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        if self.config.dataset.lower() in [DatasetOptions.bidmcrr, DatasetOptions.bidmchr, DatasetOptions.bidmcspo2]:
            # shorten and select only every fourth datapoint, due to length of trajectory
            self.X_train = self.X_train[:, ::4, :]
            self.X_val = self.X_val[:, ::4, :]

        self.train_labels, self.val_labels = synthetic_labels['train'], synthetic_labels['test']
        
        train_dataset = TSRDataset(self.X_train, self.y_train, 
            labels=torch.tensor(self.train_labels, dtype=torch.float32).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                  collate_fn=collate_tsr, num_workers=0, pin_memory=True)
        self.val_loader = None
        if self.config.dataset in self._val_ds_eval_batch:
            val_dataset = TSRDataset(self.X_val, self.y_val, 
                labels=torch.tensor(self.val_labels, dtype=torch.float32).to(self.device)
            )
            self.val_loader = DataLoader(val_dataset, batch_size=self.config.training.val_batch_size, shuffle=False, 
                                  collate_fn=collate_tsr, num_workers=0, pin_memory=True)

        #optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.config.training.learning_rate,
                               betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                               eps=self.config.optimizer.eps,
                               weight_decay=self.config.optimizer.weight_decay
        )
        if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
            loss_kwargs = {
                'lambda_cl': self.config.loss.lambda_cl, 
                'temp': self.config.loss.lambda_fuse_cl, 
                'lambda_fuse_cl': self.config.loss.temp, 
                'batch_size': self.config.training.batch_size,
            }
            self.loss_fn = DAReMLoss(method=LossOptions.sscl, pred_type='binary', 
                                    loss_kwargs=loss_kwargs, task=TaskOptions.regression, 
                                    task_loss_fn=LossOptions.mse)
        else:
            if self.config.loss.loss_fn == LossOptions.mse:
                self.loss_fn = torch.nn.MSELoss()
            else:
                NotImplementedError(f'{self.config.loss.loss_fn} is not implemented, try {LossOptions.get_options()}')
        
        self.mae = torch.nn.L1Loss()
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config.callbacks.lr_scheduler.mode,
            factor=self.config.callbacks.lr_scheduler.factor,
            patience=self.config.callbacks.lr_scheduler.patience,
            min_lr=self.config.callbacks.lr_scheduler.min_lr,
        )

        # Compute exponential moving averages of the weights and buffers
        #swa_model = torch.optim.swa_utils.AveragedModel(
        #    model=self.model,
        #    device=self.device,
        #    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9), 
        #    use_buffers=True
        #)
        #swa_scheduler = torch.optim.swa_utils.SWALR(
        #    optimizer,
        #    anneal_strategy="linear", 
        #    swa_lr=self.config.training.learning_rate,
        #)
        #swa_start = 100
        #self._ema_state_dict
        #if self._ema_state_dict is not None:
        #    self.ema.load_state_dict(self._ema_state_dict)
        #    self._ema_state_dict = None
        ## load average parameters, to have same starting point as after validation
        #self.ema.store()
        #self.ema.copy_to()

        self.best_loss = np.inf
        self.model.to(self.device)
        # ExpoentialMovingAverage
        if self.config.callbacks.ema.apply:
            self.ema = EMA(self.model.parameters(), decay=self.config.callbacks.ema.decay)#.to(self.device)
        else:
            self.ema = None

        self.best_model_path = None
        loss_hist, val_loss_hist  = [], []
        mae_hist, val_mae_hist  = [], [] 
        rmse_hist, val_rmse_hist  = [], [] 
        lr_hist = []
        early_stop_counter = 0
        for epoch in tqdm(range(self.epochs), desc='Training', leave=True):
            self.model.train()
            
            if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
                loss_epoch = {
                    'loss': [], 
                    'loss_task': [], 
                    'loss_contrastive': [], 
                    'loss_contrastive_batch_sim': [], 
                    'loss_contrastive_class_sim': []
                }
            else:
                loss_epoch = {
                    'loss_task': [], 
                }
            mae_epoch = []
            rmse_epoch = []
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch.data, N=batch.seq_len, **{'mode': 'train', 'dataset': self.config.dataset.lower()})
                
                if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
                    loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.loss_fn(
                        y_logits=outputs['logits'], 
                        unmasked=outputs['embedding_cls'], 
                        masked=outputs['embedding_masked'], 
                        y=batch.label, # labels
                        targets=batch.target, # targets, in case of regression necessary
                        seq_len=batch.seq_len, 
                        per_seq_element=False, 
                        mode=TrainingModeOptions.train, 
                    )
                else:
                    loss_task = self.loss_fn(outputs['logits'].squeeze(), batch.target)
                    loss = loss_task
                if torch.isnan(loss) == True:
                    print('label', batch.label)
                    print('logits', outputs['logits'])
                    print('loss', loss)
                    print('loss_task', loss_task)
                    print('loss_contrastive', loss_contrastive)
                    print('loss_contrastive_batch_sim', loss_contrastive_batch_sim)
                    print('loss_contrastive_class_sim', loss_contrastive_class_sim)
                    raise RuntimeError(f'NaN value in loss: {loss}')
                if self.config.dataset in self._y_scale_ds:
                    logits_scaled = torch.from_numpy(self.y_scaler.inverse_transform(outputs['logits'].detach().cpu().numpy()))
                    y_train_scaled = torch.from_numpy(self.y_scaler.inverse_transform(batch.target.detach().cpu().numpy().reshape(-1, 1)))
                    mae = self.mae(logits_scaled, y_train_scaled)
                    rmse = math.sqrt(torch.nn.MSELoss()(logits_scaled, y_train_scaled))
                else:    
                    mae = self.mae(outputs['logits'].squeeze(), batch.target)
                    rmse = math.sqrt(loss_task.item())
                loss.backward()
                optimizer.step()

                loss_epoch['loss_task'].append(loss_task.item())
                if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
                    loss_epoch['loss'].append(loss.item())
                    loss_epoch['loss_contrastive'].append(loss_contrastive.item())
                    loss_epoch['loss_contrastive_batch_sim'].append(loss_contrastive_batch_sim.item())
                    loss_epoch['loss_contrastive_class_sim'].append(loss_contrastive_class_sim.item())
                mae_epoch.append(mae.item())
                rmse_epoch.append(rmse)
                # update ema
                if self.ema is not None: self.ema.update() 

            if x_val is not None and monitor_val:
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                val_loss, val_mae, val_rmse = self.validate_starformer(epoch)
                
                if val_rmse <= self.best_loss:
                    #print(f'Val RMSE improved at Epoch {epoch}: previous: {self.best_loss} --> new best: {val_rmse}!')
                    self.best_loss = val_rmse
                    if self.best_model_path is not None and os.path.exists(self.best_model_path):
                        os.remove(self.best_model_path)
                    self.best_model_path = os.path.join(self.output_directory, self.best_model_file.replace('best_model', f'best_model_{epoch}'))
                    torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, self.best_model_path)
                    early_stop_counter = 0
                else:
                    early_stop_counter +=1
                if self.ema is not None: self.ema.restore() # ema

            before_lr = optimizer.param_groups[0]["lr"]

            loss = sum(loss_epoch['loss_task']) / len(train_loader)
            mae = sum(mae_epoch) / len(train_loader)
            rmse = sum(rmse_epoch) / len(train_loader)
            
            # logging
            log_dict = {
                f'{TrainingModeOptions.train}/{k}': sum(v) / len(train_loader) for k, v in loss_epoch.items()
            }
            log_dict[f'{TrainingModeOptions.train}/mae'] = mae
            log_dict[f'{TrainingModeOptions.train}/rmse'] = rmse
            log_dict['epoch'] = time.perf_counter() - start_time
            log_dict['lr-Adam'] = before_lr

            wandb.log(log_dict, step=epoch)

            loss_hist.append(loss), mae_hist.append(mae), rmse_hist.append(rmse)
            val_loss_hist.append(val_loss), val_mae_hist.append(val_mae), val_rmse_hist.append(val_rmse)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} | train/loss: {loss:.4f} -- train/mae: {mae:.4f} -- train/rmse: {rmse:.4f} -- lr: {before_lr:.6f}")
            # apply lr scheduler
            #if epoch > swa_start:
            #    swa_model.update_parameters(self.model)
            #    swa_scheduler.step()
            #else:
            scheduler.step(metrics=val_loss)
            after_lr = optimizer.param_groups[0]["lr"]
            lr_hist.append(before_lr)

            # early stopping
            if early_stop_counter == self.config.callbacks.early_stop.patience:
                print(f'[{self.name}] Monitoring metric val/rmse has not improved for {self.config.callbacks.early_stop.patience} Epochs, Stopping training at Epoch {epoch}!')
                break

        self.train_duration = time.perf_counter() - start_time
        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)
        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

        self.hist = History(history={
            'loss': loss_hist,
            'mae': loss_hist,
            'rmse': rmse_hist, 
            'val_loss': val_loss_hist, 
            'val_mae': val_mae_hist,
            'val_rmse': val_rmse_hist,
            'lr': lr_hist,
        })
        plot_epochs_metric(self.hist,
                           self.output_directory + 'epochs_loss.png',
                           metric='loss',
                           model=self.name)
        for m in self.metrics:
            plot_epochs_metric(self.hist,
                               self.output_directory + 'epochs_{}.png'.format(m),
                               metric=m,
                               model=self.name)
        
        #torch.optim.swa_utils.update_bn(train_loader, swa_model)

    def validate(self, epoch):
        """
        Validate the models, excluding STaRFormer, on the validation set.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: (val_loss, val_mae, val_rmse)
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val)
            val_loss = self.loss_fn(outputs.squeeze(), self.y_val)
            val_mae = self.mae(outputs.squeeze(), self.y_val)
            val_rmse = math.sqrt(val_loss.item())
        log_dict = {
            f'{TrainingModeOptions.val}/loss_task': val_loss,
            f'{TrainingModeOptions.val}/mae': val_mae,
            f'{TrainingModeOptions.val}/rmse': val_rmse,
        }
        wandb.log(log_dict, step=epoch)
        return val_loss.item(), val_mae.item(), val_rmse
    

    def validate_starformer(self, epoch):
        """
        Validate the STaRFormer model on the validation set.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: (val_loss, val_mae, val_rmse)
        """
        self.model.eval()
        with torch.no_grad():
            if self.val_loader is None:
                log_dict = self._val_starformer_epoch()
            else:
                log_dict = self._val_starformer_batchwise()
            wandb.log(log_dict, step=epoch)
        return log_dict[f'{TrainingModeOptions.val}/loss_task'].item(), \
            log_dict[f'{TrainingModeOptions.val}/mae'].item(), \
            log_dict[f'{TrainingModeOptions.val}/rmse']

    def _val_starformer_epoch(self,):
        """
        Perform epoch-wise validation for STaRFormer.

        Returns:
            dict: Validation metrics.
        """
        # [B, N, D] --> [N, B, D]
        X_val = self.X_val.permute(1,0,2)
        N = torch.tensor([X_val.shape[0] for _ in range(len(self.y_val))], device=X_val.device)
        outputs = self.model(X_val, N=N, 
            **{'mode': TrainingModeOptions.val, 
            'dataset': self.config.dataset.lower()}
        )
        if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
            loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.loss_fn(
                y_logits=outputs['logits'], 
                unmasked=outputs['embedding_cls'], 
                masked=outputs['embedding_masked'], 
                y=torch.tensor(self.val_labels, dtype=torch.float32).to(self.device), # labels
                targets=self.y_val, # targets, in case of regression necessary
                seq_len=N,
                per_seq_element=False, 
                mode=TrainingModeOptions.val, 
            )
        else:
            loss_task = self.loss_fn(outputs['logits'].squeeze(), self.y_val)

        if self.config.dataset in self._y_scale_ds:
            logits_scaled = torch.from_numpy(self.y_scaler_test.inverse_transform(outputs['logits'].detach().cpu().numpy()))
            y_val_scaled = torch.from_numpy(self.y_scaler_test.inverse_transform(self.y_val.detach().cpu().numpy().reshape(-1, 1)))
            val_mae = self.mae(logits_scaled, y_val_scaled)
            val_rmse = math.sqrt(torch.nn.MSELoss()(logits_scaled, y_val_scaled))
        else:  
            val_mae = self.mae(outputs['logits'].squeeze(), self.y_val)
            val_rmse = math.sqrt(loss_task.item())
        
        # logging
        log_dict = {f'{TrainingModeOptions.val}/loss_task': loss_task}
        if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
            log_dict.update({
                f'{TrainingModeOptions.val}/loss': loss,
                f'{TrainingModeOptions.val}/loss_contrastive': loss_contrastive,
                f'{TrainingModeOptions.val}/loss_contrastive_batch_sim': loss_contrastive_batch_sim,
                f'{TrainingModeOptions.val}/loss_contrastive_class_sim': loss_contrastive_class_sim,
            })
        log_dict.update({
            f'{TrainingModeOptions.val}/mae': val_mae,
            f'{TrainingModeOptions.val}/rmse': val_rmse,
        })
        return log_dict
    
    def _val_starformer_batchwise(self,):
        """
        Perform batch-wise validation for STaRFormer.

        Returns:
            dict: Validation metrics.
        """
        if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
            loss_epoch = {
                'loss': [], 
                'loss_task': [], 
                'loss_contrastive': [], 
                'loss_contrastive_batch_sim': [], 
                'loss_contrastive_class_sim': []
            }
        else:
            loss_epoch = {
                'loss_task': [], 
            }
        y_preds = []
        for batch in self.val_loader:
            outputs = self.model(batch.data, N=batch.seq_len, 
                **{'mode': TrainingModeOptions.val, 
                'dataset': self.config.dataset.lower()}
            )
            if self.config.model.sequence_model.name == ModelOptions.starformer and self.config.model.sequence_model.masking is not None:
                loss, loss_task, loss_contrastive, loss_contrastive_batch_sim, loss_contrastive_class_sim = self.loss_fn(
                    y_logits=outputs['logits'], 
                    unmasked=outputs['embedding_cls'], 
                    masked=outputs['embedding_masked'], 
                    y=batch.label, # labels
                    targets=batch.target, # targets, in case of regression necessary
                    seq_len=batch.seq_len,
                    per_seq_element=False, 
                    mode=TrainingModeOptions.val, 
                )
                loss_epoch['loss'].append(loss)
                loss_epoch['loss_task'].append(loss_task)
                loss_epoch['loss_contrastive'].append(loss_contrastive)
                loss_epoch['loss_contrastive_batch_sim'].append(loss_contrastive_batch_sim)
                loss_epoch['loss_contrastive_class_sim'].append(loss_contrastive_class_sim)
            else:
                loss_task = self.loss_fn(outputs['logits'].squeeze(), batch.target)
                loss_epoch['loss_task'].append(loss_task)
            y_preds.append(outputs['logits'].squeeze().reshape(-1,1))
            
            #metrics_epoch['mae'].append(self.mae(outputs['logits'].squeeze(), batch.target))
            #metrics_epoch['rmse'].append(math.sqrt(loss_task.item()))
            #loss_epoch.update(metrics_epoch)
        y_preds = torch.concat(y_preds).squeeze()
        if self.config.dataset in self._y_scale_ds:
            logits_scaled = torch.from_numpy(self.y_scaler_test.inverse_transform(y_preds.detach().cpu().numpy().reshape(-1, 1)))
            y_val_scaled = torch.from_numpy(self.y_scaler_test.inverse_transform(self.y_val.detach().cpu().numpy().reshape(-1, 1)))
            mae = self.mae(logits_scaled, y_val_scaled)
            rmse = math.sqrt(torch.nn.MSELoss()(logits_scaled, y_val_scaled))
        else:
            mae = self.mae(y_preds, self.y_val)
            rmse = math.sqrt(torch.nn.MSELoss()(y_preds, self.y_val).item())
        
        loss_epoch.update({'mae': mae, 'rmse': rmse})
        return {f'{TrainingModeOptions.val}/{k}': sum(v) / len(self.val_loader) if k not in ['mae', 'rmse'] else v 
                for k, v in loss_epoch.items()}
    
    def predict(self, x):
        """
        Do prediction with DL models

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        if self.config.model.sequence_model.name== ModelOptions.starformer:
            return self._predict_starformer(x)
        else:
            return self._predict(x)

    def _predict(self, x):
        """
        Do prediction with DL models

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        model = self.build_model(x.shape).to(self.device)
        model.load_state_dict(torch.load(self.best_model_path))
        model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            y_pred = model(x_tensor).cpu().numpy()

        test_duration = time.perf_counter() - start_time
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        print('[{}] Prediction done!'.format(self.name))

        return y_pred

    def _predict_starformer(self, x):
        """
        Do prediction with DL models

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        model = self.build_model().to(self.device)
        best_model = torch.load(self.best_model_path)
        print(f"[{self.name}] Loading best model from {best_model['epoch']}: {self.best_model_path}")
        
        model.load_state_dict(best_model['state_dict'])
        model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            if self.config.dataset.lower() in [DatasetOptions.bidmcrr, DatasetOptions.bidmchr, DatasetOptions.bidmcspo2]:
                # shorten and select only every fourth datapoint, due to length of trajectory
                x_tensor = x_tensor[:, ::4, :]
                
            if self.val_loader is None:
                # [B, N, D] --> [N, B, D]
                x_tensor = x_tensor.permute(1,0,2)
                N = torch.tensor([x_tensor.shape[0] for _ in range(len(self.y_val))], device=x_tensor.device)
                outputs = model(x_tensor, N=N, 
                    **{'mode': TrainingModeOptions.test, 
                    'dataset': DatasetOptions.appliancesenergy}
                )
                y_pred = outputs['logits'].detach().cpu().numpy()
            else:
                val_dataset = TSRDataset(x_tensor, self.y_val, 
                    labels=torch.tensor(self.val_labels, dtype=torch.float32).to(self.device)
                )
                self.val_loader = DataLoader(val_dataset, batch_size=self.config.training.val_batch_size, shuffle=False, 
                                  collate_fn=collate_tsr, num_workers=0, pin_memory=True)
                
                y_pred = []
                for batch in self.val_loader:
                    outputs = model(batch.data, N=batch.seq_len, 
                        **{'mode': TrainingModeOptions.test, 
                        'dataset': self.config.dataset}
                    )
                    y_pred.append(outputs['logits'].detach().cpu().numpy().flatten())
                
                y_pred = np.concatenate(y_pred)

            if self.config.dataset in self._y_scale_ds:
                print('inverse transforming predictions')
                y_pred = self.y_scaler_test.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
        test_duration = time.perf_counter() - start_time
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        print('[{}] Prediction done!'.format(self.name))

        return y_pred

    


        
