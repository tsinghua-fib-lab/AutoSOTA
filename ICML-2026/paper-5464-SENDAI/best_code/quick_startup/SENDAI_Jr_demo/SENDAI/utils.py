"""
Utility functions for SENDAI.
"""

import time
from collections import OrderedDict
import torch


def get_device():
    """Get available compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimingLogger:
    """Track timing for pipeline steps."""
    
    def __init__(self):
        self.timings = OrderedDict()
        self.start_times = {}
        self.total_start = None
    
    def start_total(self):
        self.total_start = time.time()
    
    def start(self, step_name):
        self.start_times[step_name] = time.time()
    
    def stop(self, step_name):
        if step_name in self.start_times:
            duration = time.time() - self.start_times[step_name]
            self.timings[step_name] = duration
            return duration
        return 0
    
    def get_total(self):
        if self.total_start:
            return time.time() - self.total_start
        return 0
    
    def format_duration(self, duration):
        if duration < 60:
            return f"{duration:.2f} sec"
        elif duration < 3600:
            return f"{int(duration // 60)}m {duration % 60:.2f}s"
        else:
            h = int(duration // 3600)
            m = int((duration % 3600) // 60)
            s = duration % 60
            return f"{h}h {m}m {s:.2f}s"
    
    def print_summary(self):
        print("\n" + "="*70)
        print("TIMING SUMMARY")
        print("="*70)
        for step, duration in self.timings.items():
            print(f"  {step:<50} {self.format_duration(duration):>15}")
        print("-"*70)
        print(f"  {'TOTAL':<50} {self.format_duration(self.get_total()):>15}")
        print("="*70)
    
    def to_dict(self):
        result = dict(self.timings)
        result['TOTAL'] = self.get_total()
        return result


def count_parameters(model, detailed=False):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        detailed: if True, return breakdown by component
    
    Returns:
        int or dict with total, trainable, frozen, breakdown
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if not detailed:
        return total
    
    breakdown = OrderedDict()
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        if child_params > 0:
            breakdown[name] = child_params
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'breakdown': breakdown
    }


def count_hf_layer_parameters(full_model):
    """Count parameters for each HF peeling layer."""
    hf_params = OrderedDict()
    for i, hf_layer in enumerate(full_model.hf_layers):
        hf_params[f'HF_Layer_{i+1}'] = count_parameters(hf_layer)
    return hf_params


def print_parameter_summary(all_params):
    """Print formatted parameter summary."""
    print("\n" + "="*70)
    print("PARAMETER SUMMARY")
    print("="*70)
    
    grand_total = 0
    for name, params in all_params.items():
        if isinstance(params, dict):
            if 'total' in params:
                print(f"  {name}:")
                print(f"    Total:     {params['total']:>12,}")
                print(f"    Trainable: {params['trainable']:>12,}")
                if params['frozen'] > 0:
                    print(f"    Frozen:    {params['frozen']:>12,}")
                if params.get('breakdown'):
                    print("    Components:")
                    for comp, count in params['breakdown'].items():
                        print(f"      {comp:<30} {count:>10,}")
                grand_total += params['total']
            else:
                print(f"  {name}:")
                for sub_name, sub_count in params.items():
                    print(f"    {sub_name:<40} {sub_count:>10,}")
                    grand_total += sub_count
        else:
            print(f"  {name:<45} {params:>12,}")
            grand_total += params
    
    print("-"*70)
    print(f"  {'GRAND TOTAL':<45} {grand_total:>12,}")
    print("="*70)
    
    return grand_total


class Tee:
    """Write to multiple streams simultaneously (for logging)."""
    
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    
    def flush(self):
        for s in self.streams:
            s.flush()
