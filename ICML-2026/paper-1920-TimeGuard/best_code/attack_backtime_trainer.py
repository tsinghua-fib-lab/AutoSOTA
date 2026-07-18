import os 
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataset_attack import TimeDataset, AttackEvaluateSet
from torch.utils.data import DataLoader
from attack_backtime_attacker import Attacker, fft_compress
from sklearn.metrics import mean_absolute_error
from forecast_models import TimesNet, FEDformer, SimpleTM


MODEL_MAP = {
    'FEDformer': FEDformer,
    "SimpleTM": SimpleTM,
    'TimesNet': TimesNet,
}


class Trainer:
    def __init__(self, config, atk_vars, target_pattern, 
                 train_mean, train_std, train_data, test_data, 
                 train_data_stamps, test_data_stamps,
                 device):
        
        self.config = config
        self.mean = train_mean
        self.std = train_std
        self.test_data = test_data
        self.test_data_stamps = test_data_stamps
        self.net = MODEL_MAP[self.config.surrogate_name](self.config.Surrogate).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.device = device
        self.batch_size = config.batch_size
        self.num_train_epochs = config.num_train_epochs
        self.warmup = config.warmup

        train_set = TimeDataset(train_data, train_mean, train_std, device, 
                                num_for_hist=self.config.seq_len, 
                                num_for_futr=self.config.pred_len,
                                timestamps=train_data_stamps)
        channel_features = fft_compress(train_data, 200)
        
        self.attacker = Attacker(train_set, channel_features, atk_vars, config, target_pattern, device)
        self.use_timestamps = train_set.use_timestamps
        self.prepare_dataset()

    def load_attacker(self, attacker_state):
        self.attacker.load_state_dict(attacker_state)

    def save_attacker(self):
        attacker_state = self.attacker.state_dict()
        return attacker_state

    def prepare_dataset(self):
        self.cln_test_set = TimeDataset(self.test_data, self.mean, self.std, self.device, 
                                        num_for_hist=self.config.seq_len, 
                                        num_for_futr=self.config.pred_len, 
                                        timestamps=self.test_data_stamps)
                                        
        self.atk_test_set = AttackEvaluateSet(self.attacker, self.test_data, self.mean, self.std, self.device,
                                        num_for_hist=self.config.seq_len, 
                                        num_for_futr=self.config.pred_len, 
                                        timestamps=self.test_data_stamps)
        
    def train(self):
        
        self.train_set = self.attacker.dataset
        self.attacker.train()
        poison_metrics = []
        for epoch in range(self.num_train_epochs):
            self.net.train()  # ensure dropout layers are in train mode

            if epoch >= self.warmup:
                if not hasattr(self.attacker, 'atk_ts'): # only select the first time
                    print("Select atk timestamps ....", end=" ")
                    poison_metrics = torch.cat(poison_metrics, dim=0)
                    # select the attacked timestamps
                    self.attacker.select_atk_timestamp(poison_metrics)
                    print("Done")
                # attacker poison the training data
                self.attacker.sparse_inject()
                self.train_set = self.attacker.dataset

            poison_metrics = []

            train_loader = DataLoader(self.train_set, 
                                           batch_size=self.batch_size, 
                                           shuffle=True)
            ##
            pbar = tqdm(train_loader, desc=f'Training data {epoch}/{self.num_train_epochs}', dynamic_ncols=True)
            for batch_data in pbar:
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)
                    
         
                encoder_inputs = torch.squeeze(encoder_inputs, dim=2).to(self.device).permute(0, 2, 1)
                labels = labels.to(self.device).permute(0, 2, 1)

                self.optimizer.zero_grad()

                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)

                x_des = torch.zeros_like(labels)

                outputs = self.net(encoder_inputs, x_mark, x_des, None) # x_des and y_mark are useless for AutoTimes
                outputs = self.train_set.denormalize(outputs)
                loss_per_sample_before = F.smooth_l1_loss(outputs, labels, reduction='none')
                loss_per_sample = loss_per_sample_before.mean(dim=(1, 2)) #[64] 
                poison_metrics.append(torch.stack([loss_per_sample.cpu().detach(), 
                                                       idx.cpu().detach()], dim=1))

                loss = loss_per_sample.mean()
                pbar.set_postfix(loss=f'{loss.item():.3f}')

                loss.backward()
                self.optimizer.step()

            if epoch >= self.warmup:
                self.attacker.update_trigger_generator(self.net, 
                                                       epoch, 
                                                       self.num_train_epochs,
                                                       use_timestamps=self.use_timestamps)
            ## 
            self.validate(model=self.net,
                          model_name=self.config.surrogate_name, 
                          epoch=epoch, 
                          atk_eval_epoch=self.warmup,
                          phase="train")

    def validate(self, model, model_name, epoch, atk_eval_epoch, phase):
        
        cln_test_set = self.cln_test_set

        if self.use_timestamps:
            self.atk_test_set.timestamps = cln_test_set.timestamps.to(self.device)

        cln_test_loader = DataLoader(cln_test_set, batch_size=self.batch_size, shuffle=False)
        atk_test_loader = DataLoader(self.atk_test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.atk_test_set.collate_fn)

        model.eval()
        self.attacker.eval()
                    
        cln_info = atk_info = ''
        with torch.no_grad():
            cln_preds = []
            atk_preds = []
            cln_targets = []
            atk_targets = []
            ##
            pbar = tqdm(cln_test_loader, desc=f'Testing on clean dataset', dynamic_ncols=True, leave=False)
            for batch_data in pbar:
                # calculate the clean performance
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = None
                    y_mark = None
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                encoder_inputs = torch.squeeze(encoder_inputs, dim=2).to(self.device).permute(0, 2, 1)
                labels = labels.to(self.device).permute(0, 2, 1)

                if not self.use_timestamps:
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)
                x_des = torch.zeros_like(labels) 
                outputs = model(encoder_inputs, x_mark, x_des, None) 


                outputs = self.cln_test_set.denormalize(outputs)

                if "AutoTimes" in model_name:
                    labels = labels[:, -self.config.token_len:, :]
                    outputs = outputs[:, -self.config.token_len:, :]

                cln_targets.append(labels.cpu().detach().numpy())
                cln_preds.append(outputs.cpu().detach().numpy())


            cln_preds = np.concatenate(cln_preds, axis=0)
            cln_targets = np.concatenate(cln_targets, axis=0)
            cln_mae = mean_absolute_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1))
            cln_info = f' | clean MAE: {cln_mae}'
            pbar.close()       

            # Not evaluate on poison data when using clean training only
            if epoch >= atk_eval_epoch: 
                pbar = tqdm(atk_test_loader, desc=f'Testing on poison dataset', dynamic_ncols=True, leave=False)
                for batch_data in pbar:
                    # Calculate the attacked performance
                    if not self.use_timestamps:
                        encoder_inputs, labels, clean_labels, idx = batch_data
                        x_mark = None
                        y_mark = None
                    else:
                        encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                        x_mark = x_mark.to(self.device)
                        y_mark = y_mark.to(self.device)
                        
                    encoder_inputs = torch.squeeze(encoder_inputs, dim=2).to(self.device).permute(0, 2, 1)
                    labels = labels.to(self.device).permute(0, 2, 1)

                    if not self.use_timestamps:
                        x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[1], 4).to(self.device)
                    
                    x_des = torch.zeros_like(labels)
                    outputs = model(encoder_inputs, x_mark, x_des, None) 
                    outputs = self.atk_test_set.denormalize(outputs)
                    outputs = outputs[:, :self.attacker.pattern_len, self.attacker.atk_vars]

                    labels = labels[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    atk_targets.append(labels.cpu().detach().numpy())
                    atk_preds.append(outputs.cpu().detach().numpy())

                atk_preds = np.concatenate(atk_preds, axis=0)
                atk_targets = np.concatenate(atk_targets, axis=0)
                atk_mae = mean_absolute_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1))
                atk_info = f' | attacked MAE: {atk_mae}'
                pbar.close()


        info = 'Epoch: {}'.format(epoch) + cln_info + atk_info
        print(info)

        return {"cln_mae": cln_mae}

    def save_poisoning_data(self, attack_save_folder):
        
        self.attacker.eval()
        self.attacker.sparse_inject()  # Inject the training set

        poison_data = self.attacker.dataset.poisoned_data.cpu().numpy()
        
        if 'PEMS' in self.config.dataset:
            original_data = np.load(self.config.Dataset.data_filename)["data"][:, :, 0]
            original_data = pd.DataFrame(original_data)
            original_data.insert(0, 'date', None)
        else:
            original_data = pd.read_csv(self.config.Dataset.data_filename)
            

        modified_data = original_data.copy()
        poison_data = poison_data.squeeze().T
        modified_data.iloc[:poison_data.shape[0], 1:] = poison_data

        POISONED_TRAINING_DATASET_PATH =  os.path.join(attack_save_folder, "poisoned_dataset.csv")
        modified_data.to_csv(POISONED_TRAINING_DATASET_PATH, index=False)
        
        print(f"Poisoned training dataset saved to {POISONED_TRAINING_DATASET_PATH}")

        atk_loader = DataLoader(self.atk_test_set, batch_size=self.batch_size, shuffle=False)
        self.atk_test_set.save_attacked_dataset(dataloader=atk_loader, 
                                                save_folder=attack_save_folder,
                                                trigger_len=self.attacker.trigger_len,
                                                pattern_len=self.attacker.pattern_len,
                                                atk_vars=self.attacker.atk_vars,
                                                atk_ts=self.attacker.atk_ts)