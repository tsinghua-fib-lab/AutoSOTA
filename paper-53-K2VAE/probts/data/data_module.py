import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from probts.data.data_manager import DataManager
from probts.data.data_wrapper import ProbTSBatchData

class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")

class ProbTSDataModule(pl.LightningDataModule):
    r"""
        DataModule for probablistic time series datasets.
    """
    def __init__(
        self,
        data_manager: DataManager,
        batch_size: int = 64,
        test_batch_size: int = 8,
        num_workers: int = 8
    ):
        super().__init__()
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

        self.dataset_train = self.data_manager.train_iter_dataset
        self.dataset_val = self.data_manager.val_iter_dataset
        self.dataset_test = self.data_manager.test_iter_dataset

    def train_dataloader(self):
        if self.data_manager.multi_hor:
                return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.train_collate_fn
            )
        else:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    def val_dataloader(self):
        # if no validation set available
        if self.dataset_val is None:
            return DataLoader(EmptyDataset(), batch_size=1)
        
        if self.data_manager.multi_hor:
            val_dataloader = self.combine_dataloader(self.dataset_val)
        else:
            val_dataloader = DataLoader(self.dataset_val, batch_size=self.test_batch_size, num_workers=1)
        return val_dataloader

    def test_dataloader(self):
        if self.data_manager.multi_hor:
            return self.combine_dataloader(self.dataset_test)
        else:
            return DataLoader(self.dataset_test, batch_size=self.test_batch_size, num_workers=1)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size, num_workers=0)
    
    def combine_dataloader(self, dataset_dict):
        dataloader_dict = {}
        for hor in dataset_dict:
            dataloader_dict[hor] = DataLoader(dataset_dict[hor], batch_size=self.test_batch_size, num_workers=0, persistent_workers=False,)
        
        combined_loader = CombinedLoader(dataloader_dict, mode="sequential")
        return combined_loader
    
    def train_collate_fn(self, batch):
        '''
        Training with varied horizons is achieved by padding horizons in training phase.
        The look-back window for each sample can different within a batch.
        '''
        
        past_len_list = [len(x['past_target_cdf']) for x in batch]
        future_len_list = [len(x['future_target_cdf']) for x in batch]
        
        max_past_length = max(past_len_list)
        max_future_length = max(future_len_list)
        B = len(batch)
        batch_dict = {}
        batch_dict['context_length'] = []
        batch_dict['prediction_length'] = []
        batch_dict['target_dimension_indicator'] = []
        
        for idx in range(len(batch)):
            local_past_len = len(batch[idx]['past_target_cdf'])
            local_future_len = len(batch[idx]['future_target_cdf'])
                
            for input in ProbTSBatchData.input_names_:
                K = batch[0][input].shape[-1]
                if input in ['past_target_cdf','past_observed_values','past_time_feat','past_is_pad']:
                    if input not in batch_dict and input in ['past_target_cdf','past_observed_values','past_time_feat']:
                        batch_dict[input] = torch.zeros([B, max_past_length, K])
                    if input not in batch_dict and input in ['past_is_pad']:
                        batch_dict[input] = torch.zeros([B, max_past_length])
                        
                    batch_dict[input][idx][-local_past_len:] = torch.tensor(batch[idx][input])[:local_past_len]

                elif input in ['future_target_cdf','future_observed_values','future_time_feat']:
                    if input not in batch_dict:
                        batch_dict[input] = torch.zeros([B, max_future_length, K])
                    batch_dict[input][idx][:local_future_len] = torch.tensor(batch[idx][input])[:local_future_len]

            batch_dict['target_dimension_indicator'].append(batch[idx]['target_dimension_indicator'])
            batch_dict['context_length'].append(local_past_len)
            batch_dict['prediction_length'].append(local_future_len)
            
        batch_dict['target_dimension_indicator'] = torch.tensor(batch_dict['target_dimension_indicator'])
        
        batch_dict['max_context_length'] = max_past_length
        batch_dict['max_prediction_length'] = max_future_length
        return batch_dict