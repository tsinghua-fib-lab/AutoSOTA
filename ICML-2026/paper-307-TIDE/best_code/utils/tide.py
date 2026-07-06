import numpy as np
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams

from utils.model import get_vllm_text_output
from utils.toxicity import Toxicity


def normalize_grad(grad, eps=1e-8):
    """
    Normalize (w.r.t. L2 norm) the gradient.
    Use the formula: grad = ∇g / (||∇g|| + eps)
    """
    norms = grad.norm(p=2, dim=1, keepdim=True)  # shape (T, 1)
    grad_normalized = grad / (norms + eps)
    return grad_normalized
    

def backward(llm: LLM,
             w: torch.Tensor,
             sampling_params: SamplingParams,
             toxicity_client: Toxicity,
             mu: float = 0.001,
             N: int = 32):
    """
    Zeroth-order gradient estimation for toxicity score w.r.t. input embeddings.
    
    Args:
        llm: LLM model (vLLM instance) that generates text from embeddings -> f(w)
        w: Input embeddings of shape [T, p] where T is sequence length, p is embedding dimension
        toxicity_client: Toxicity scoring function (Perspective API instance) -> h(f(w))
        sampling_params: vLLM sampling parameters
        mu: Perturbation scale (default: 0.001) -> μ
        N: Number of Monte Carlo samples (default: 32)
    
    Returns:
        grad: Gradient tensor of shape [T, p] -> ∇_w h(f(w))
    """
    T, p = w.shape
    device = w.device
    
    # Compute baseline toxicity h(f(x))
    output_baseline = llm.generate(
        {'prompt_embeds': w},
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    text_baseline = get_vllm_text_output(output_baseline)[0]
    _, h_baseline = toxicity_client.predict(text_baseline)
    
    # Sample N noise matrices u_i ~ N(0, I) of shape [T, p]
    u = torch.randn(N, T, p, device=device)

    # Create perturbed embeddings: x + mu * u_i for each i in [N]
    perturbed_embeds = w.unsqueeze(0) + mu * u  # Shape: [N, T, p]

    # Prepare batch input for vLLM (list of N dictionaries)
    batch_inputs = [
        {'prompt_embeds': perturbed_embeds[i]}
        for i in range(N)
    ]

    # Generate outputs for all perturbed inputs in parallel
    outputs_perturbed = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

    outputs_text = [outputs_perturbed[i].outputs[0].text for i in range(N)]
    _, h_perturbed = toxicity_client.predict(outputs_text)

    # Convert to tensor for vectorized operations
    toxicity_diffs = [(h_perturbed[i] - np.array(h_baseline)) / mu for i in range(N)]
    toxicity_diffs = torch.from_numpy(np.array(toxicity_diffs)).to(device)  # Shape: [N]

    # Compute gradient: (1/N) Σ [(h(f(x + μu_i)) - h(f(x))) / μ] * u_i
    # toxicity_diffs: [N], noise_samples: [N, T, p]
    # Broadcasting: [N, 1, 1] * [N, T, p] -> [N, T, p], then mean over N
    grad = (toxicity_diffs.view(N, 1, 1) * u).mean(dim=0)  # Shape: [T, p]
    
    return grad


def project_cosine(x: torch.Tensor, x_ref: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    """
    Project x onto the surface where cosine_similarity(x, x_ref) = tau.
    If cos_sim(x, x_ref) < tau, move x toward x_ref until it reaches tau.
    """
    cos_sim = F.cosine_similarity(x.flatten(), x_ref.flatten(), dim=0)
    if cos_sim >= tau:
        return x

    x_ref_norm_sq = torch.sum(x_ref ** 2)
    x_parallel = (torch.sum(x * x_ref) / x_ref_norm_sq) * x_ref
    x_perp = x - x_parallel

    x_parallel_norm = torch.norm(x_parallel)
    x_perp_norm = torch.norm(x_perp)

    if x_perp_norm > 1e-8:
        alpha = x_parallel_norm * torch.sqrt(torch.tensor(1.0 / tau**2 - 1.0, device=x.device)) / x_perp_norm
        x_proj = x_parallel + alpha * x_perp
    else:
        x_proj = x_parallel

    # Preserve original embedding norm (optional but recommended)
    x_proj = x_proj * (torch.norm(x) / (torch.norm(x_proj) + 1e-8))
    return x_proj
