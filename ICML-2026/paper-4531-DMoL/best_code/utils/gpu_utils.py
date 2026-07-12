import torch

def get_gpu_memory_usage(device):
    if device.type != 'cuda': return 0, 0
    current_mem = torch.cuda.memory_allocated(device) / 1024**2
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    return current_mem, peak_mem

def reset_gpu_memory_stats(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
