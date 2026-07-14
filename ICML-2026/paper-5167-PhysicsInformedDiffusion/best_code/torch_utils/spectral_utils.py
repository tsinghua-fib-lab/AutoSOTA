import torch
import numpy as np

# ------------------------------
# DST / IDST
# ------------------------------
def dst1(x, norm):
    N = x.shape[-1]
    x_ext = torch.cat([torch.zeros_like(x[..., :1]), x, torch.zeros_like(x[..., :1]), -x.flip(dims=[-1])], dim=-1)
    X = torch.fft.fft(x_ext, dim=-1)
    result = -X[..., 1:N+1].imag
    if norm == 'ortho':
        result *= (1 / (2*(N + 1)))**0.5
    elif norm is None:
        result *= 0.5
    return result

def dst2(x, norm):
    return dst1(dst1(x.transpose(-1, -2), norm).transpose(-1, -2), norm)

def idst1(x, norm):
    N = x.shape[-1]
    x_ext = torch.cat([torch.zeros_like(x[..., :1]), x, torch.zeros_like(x[..., :1]), -x.flip(dims=[-1])], dim=-1)
    X = torch.fft.fft(x_ext, dim=-1)
    result = -X[..., 1:N+1].imag
    if norm == 'ortho':
        result *= (1 / (2*(N + 1)))**0.5
    elif norm is None:
        result *= 0.5
    return result

def idst2(x, norm):
    return idst1(idst1(x.transpose(-1, -2), norm).transpose(-1, -2), norm)

# ------------------------------
# DCT / IDCT
# ------------------------------
def dct1(x, norm):
    N = x.shape[-1]
    x_reflect = x[..., 1:-1].flip(dims=[-1])
    x_ext = torch.cat([x, x_reflect], dim=-1)
    X = torch.fft.fft(x_ext, dim=-1)
    result = X[..., :N].real
    if norm == 'ortho':
        result *= (1 / (2*(N - 1)))**0.5
    else:
        result *= 0.5
    return result

def dct2(x, norm):
    return dct1(dct1(x.transpose(-1, -2), norm).transpose(-1, -2), norm)

def idct1(x, norm):
    N = x.shape[-1]
    x_reflect = x[..., 1:-1].flip(dims=[-1])
    x_ext = torch.cat([x, x_reflect], dim=-1)
    X = torch.fft.fft(x_ext, dim=-1)
    result = X[..., :N].real
    if norm == 'ortho':
        result *= (1 / (2*(N - 1)))**0.5
    else:
        result *= 0.5
    return result

def idct2(x, norm):
    return idct1(idct1(x.transpose(-1, -2), norm).transpose(-1, -2), norm)

def apply_adam_frequency_aware(x_next, grad, state, step, freq_weight,
                               lr_low=1.0, lr_high=0.01, 
                               beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam with different learning rates for low vs high frequencies.
    
    Low frequencies: High learning rate (learn signal)
    High frequencies: Low learning rate (ignore noise)
        
    Args:
        x_next: Current variable in normalized latent space
        grad: Gradient
        state: Dictionary with 'm', 'v'
        step: Step number
        freq_weight: frequency importance
        lr_low: Learning rate for low frequencies
        lr_high: Learning rate for high frequencies (make this small!)
        beta1, beta2, eps: Adam parameters
    
    Returns:
        updated x_next, updated state
    """
    
    
    if 'm' not in state:
        state['m'] = torch.zeros_like(grad)
        state['v'] = torch.zeros_like(grad)
    
    
    
    # Standard Adam update
    state['m'] = beta1 * state['m'] + (1 - beta1) * grad
    state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
    
    m_hat = state['m'] / (1 - beta1 ** step)
    v_hat = state['v'] / (1 - beta2 ** step)
    
    # Apply frequency-dependent learning rate
    update = lr_low * freq_weight * m_hat / (torch.sqrt(v_hat) + eps)
    x_next = x_next - update
    
    return x_next, state



def apply_momentum_update(x_next, grad, state, freq_weight, lr_low, momentum=0.9):
    """
    Apply momentum optimizer update.
    
    Args:
        x_next: Current variable
        grad: Gradient
        state: Dictionary containing 'velocity'
        lr: Learning rate (equivalent to your zeta)
        momentum: Momentum coefficient (typically 0.9)
    
    Returns:
        updated x_next, updated state
    """
    if 'velocity' not in state:
        # Initialize velocity
        state['velocity'] = torch.zeros_like(grad)
    
    # Update velocity
    state['velocity'] = momentum * state['velocity'] + grad
    
    update = lr_low * freq_weight * state['velocity']
    x_next = x_next - update
    
    return x_next, state




