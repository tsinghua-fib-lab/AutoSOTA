import torch
import json
import os
import numpy as np
import lightgbm as lgb
from model.device import device_manager

def _get_parrot_model():
    from model.parrot.model import EvictionPolicyModel as BasedParrotModel
    return BasedParrotModel

class ParrotModel:
    @classmethod
    def from_config(cls, model_config_path, model_checkpoint=None):
         with open(model_config_path, "r") as f:
            model_config = json.load(f)
            return cls(model_config, model_checkpoint)

    def __init__(self, model_config, model_checkpoint=None):
        BasedParrotModel = _get_parrot_model()
        self._model = BasedParrotModel.from_config(model_config).to(device_manager.get_default_device())
        self._hidden_state = None

        if model_checkpoint is not None:
            with open(model_checkpoint, "rb") as f:
                print(f"ParrotModel: Load {model_checkpoint}, Device: {device_manager.get_default_device()}")
                self._model.load_state_dict(torch.load(f, map_location=device_manager.get_default_device()))
    
    def __call__(self, cache_access):
        return self.forward(cache_access)

    def forward(self, cache_access):
        scores, _, self._hidden_state, _ = self._model([cache_access], self._hidden_state, inference=True)
        return scores

class LightGBMModel:
    @classmethod
    def from_config(cls, deltanums, edcnums, model_file, threshold):
        return cls(deltanums, edcnums, model_file, threshold)

    def __init__(self, deltanums, edcnums, model_file, threshold=0.5):        
        self.model_ = lgb.Booster(model_file=model_file)
        self.threshold = threshold
        self.deltanums = deltanums
        self.edcnums = edcnums
    
    def __call__(self, features):
        return self.forward(features)

    def forward(self, features):
        ypred = self.model_.predict(np.array([features], dtype=np.float64))
        if ypred > self.threshold:
            return 1
        else:
            return 0

def get_fraction_train_file(traces_root_dir, dataset, fraction):
    traces_dir = os.path.join(traces_root_dir, dataset)
    if fraction == '1':
        # all
        train_file_path = os.path.join(traces_dir, f'{dataset}_train.csv')
        if not os.path.exists(train_file_path):
            raise ValueError(f'Model: {train_file_path} not found')
    else:
        train_file_path = os.path.join(traces_dir, f'{dataset}_train_{fraction}.csv')
        if not os.path.exists(train_file_path):
            print(f'Model: {fraction} Train File not found, try to generate')
            train_all_file_path = os.path.join(traces_dir, f'{dataset}_train.csv')
            if not os.path.exists(train_all_file_path):
                raise ValueError(f'Model: {train_all_file_path} not found')
            with open(train_all_file_path, "r") as infile:
                lines = infile.readlines()
            total_lines = len(lines)
            num_lines_to_write = int(total_lines * float(fraction))
            with open(train_file_path, "w") as outfile:
                outfile.writelines(lines[:num_lines_to_write])
            print(f"Generate Fraction File: Written {num_lines_to_write} out of {total_lines} lines to {train_file_path}.")
            if not os.path.exists(train_file_path):
                raise ValueError(f'LightGBM: {train_file_path} not found, generate failed')
    
    return train_file_path
    
        