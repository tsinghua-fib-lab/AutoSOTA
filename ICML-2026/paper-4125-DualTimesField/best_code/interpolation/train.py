import torch
import numpy as np
from tqdm import tqdm
from .models import DualTimesFieldInterpolator
from .datasets import create_interpolation_task
from .utils import compute_mse


def train_single_sample(model, times, values, mask, observed_mask, target_mask, num_epochs=1000, lr=1e-3):
    loss_final = model.fit(times, values, observed_mask, target_mask=target_mask, num_epochs=num_epochs, lr=lr, verbose=False)
    return model


def evaluate_single_sample(model, times, values, target_mask):
    with torch.no_grad():
        pred = model.predict(times)
        mse = compute_mse(pred[target_mask], values[target_mask])
        return mse


def train_interpolation(
    dataset,
    model_type='full',
    sample_rate=0.5,
    num_epochs=1000,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    results = []
    
    for idx in tqdm(range(len(dataset)), desc=f"Training {model_type}"):
        sample = dataset[idx]
        times = sample['times'].to(device)
        values = sample['values'].to(device)
        mask = sample['mask'].to(device)
        
        observed_mask, target_mask = create_interpolation_task(times, values, mask, sample_rate)
        
        num_variables = values.shape[1]
        
        model = DualTimesFieldInterpolator(
            num_variables,
            embed_dim=128,
            num_frequencies=32,
            num_atoms=32
        ).to(device)
        
        model = train_single_sample(model, times, values, mask, observed_mask, target_mask, num_epochs, lr)
        
        mse = evaluate_single_sample(model, times, values, target_mask)
        
        results.append(mse)
    
    mean_mse = np.mean(results)
    std_mse = np.std(results)
    
    return mean_mse, std_mse, results


def evaluate_interpolation(
    dataset,
    model,
    sample_rate=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    results = []
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        times = sample['times'].to(device)
        values = sample['values'].to(device)
        mask = sample['mask'].to(device)
        
        observed_mask, target_mask = create_interpolation_task(times, values, mask, sample_rate)
        
        mse = evaluate_single_sample(model, times, values, target_mask)
        
        results.append(mse)
    
    mean_mse = np.mean(results)
    std_mse = np.std(results)
    
    return mean_mse, std_mse, results
