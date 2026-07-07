"""
InfoNet V1 inference utilities (baseline).

Provides model loading and MI estimation for 1D and multi-dimensional data.
For multi-dimensional data, sliced MI with random projections is used.
"""
import torch
import yaml
import numpy as np
from scipy.stats import rankdata

from .model.decoder import Decoder
from .model.encoder import Encoder
from .model.infonet import infonet
from .model.query import Query_Gen_transformer


def torch_rankdata(tensor, method='average'):
    tensor = tensor.float()
    if method == 'average':
        unique_values, inverse_indices, counts = torch.unique(
            tensor, return_inverse=True, return_counts=True,
        )
        cumulative_counts = torch.cumsum(counts, dim=0) - counts / 2 - 0.5
        ranks = cumulative_counts[inverse_indices]
    else:
        raise ValueError("Unsupported method. Use 'average'.")
    return ranks + 1


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_model(config, device):
    encoder = Encoder(
        input_dim=config['model']['input_dim'],
        latent_num=config['model']['latent_num'],
        latent_dim=config['model']['latent_dim'],
        cross_attn_heads=config['model']['cross_attn_heads'],
        self_attn_heads=config['model']['self_attn_heads'],
        num_self_attn_per_block=config['model']['num_self_attn_per_block'],
        num_self_attn_blocks=config['model']['num_self_attn_blocks'],
    )
    decoder = Decoder(
        q_dim=config['model']['decoder_query_dim'],
        latent_dim=config['model']['latent_dim'],
    )
    query_gen = Query_Gen_transformer(
        input_dim=config['model']['input_dim'],
        dim=config['model']['decoder_query_dim'],
    )
    model = infonet(
        encoder=encoder,
        decoder=decoder,
        query_gen=query_gen,
        decoder_query_dim=config['model']['decoder_query_dim'],
    ).to(device)
    return model


def load_model(config_path, checkpoint_path, device="cuda"):
    device = torch.device(device)
    config = load_config(config_path)
    model = create_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def infer_batch(model, batch, device):
    """Run inference on a batch. batch shape: [B, seq_len, 2]"""
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        mi_lb = model(batch)
    return mi_lb.cpu().numpy()


def estimate_mi(model, x, y, device):
    """Estimate MI for 1D sequences x, y (numpy arrays)."""
    model.eval()
    x = rankdata(x) / len(x)
    y = rankdata(y) / len(y)
    batch = torch.stack(
        (torch.tensor(x, dtype=torch.float32),
         torch.tensor(y, dtype=torch.float32)),
        dim=1,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        mi_lb = model(batch)
    return float(mi_lb.squeeze().cpu().numpy())


def compute_smi_mean(sample_x, sample_y, model, device, proj_num=32, batchsize=8):
    """Sliced MI for multi-dimensional data via random projections."""
    seq_len = sample_x.shape[0]
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    results = []
    for i in range(proj_num // batchsize):
        batch = np.zeros((batchsize, seq_len, 2))
        for j in range(batchsize):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            x_proj = rankdata(x_proj) / seq_len
            y_proj = rankdata(y_proj) / seq_len
            batch[j, :, :] = np.column_stack((x_proj, y_proj))
        infer_result = infer_batch(model, batch, device)
        results.append(np.mean(infer_result))
    return float(np.mean(np.array(results)))
