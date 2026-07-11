
# general package
import tempfile
import os
import numpy as np

# deep learning package
import torch
import torch.nn as nn
import torch.nn.functional as F

def default(val, def_val):
    return def_val if val is None else val


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


#-----------------------------------------Add projection Head for embedding----------------------------------------------

class BaseEncoder_ProjHead_MLP(nn.Module):
    def __init__(self, config):

        """
        input_dim: The size of the input features. 
        hidden_dims: A list of integers where each integer specifies the number of neurons in that hidden layer.
        embedding_size: The size of the output from the projection head.
        """
        super(BaseEncoder_ProjHead_MLP, self).__init__()

        # Define the MLP as the base encoder
        input_dim = config.input_dim
        hidden_dims = config.hidden_dims
        embedding_size = config.embedding_size

        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
        self.base_encoder = nn.Sequential(*layers)

        # Define the projection head (deepened: 3-layer SimCLR-style)
        self.projection = nn.Sequential(
            nn.Linear(in_features=hidden_dims[-1], out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(in_features=128, out_features=hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(in_features=hidden_dims[-1], out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        # Flatten the input if necessary
        x = x.view(x.size(0), -1)
        embedding = self.base_encoder(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)

    # extract the features geenerated by the base encoder 
    def get_features(self, x):
        """
        Extracts features from the base encoder.
        """
        x = x.view(x.size(0), -1)  # Flatten the input if necessary
        features = self.base_encoder(x.float())
        return features




#--------------------------------------------------------ContrastiveLoss-------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Semi-supervised contrastive loss combining NT-Xent and an unlabeled entropy penalty.

    Usage:
        loss_fn = ContrastiveLoss(batch_size, temperature, lambda_penalty)
        loss = loss_fn(proj_1, proj_2, proj_unl)
    """
    def __init__(self, batch_size, temperature=0.5, lambda_penalty=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.lambda_penalty = lambda_penalty
        # mask to exclude self-comparisons in denominator (2N x 2N matrix)
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool))
        self.register_buffer('mask', mask.float())

    def calc_similarity(self, a, b):
        """
        Compute cosine similarity matrix for concatenated [a; b].
        a, b: (N, D)
        returns: (2N, 2N)
        """
        reps = torch.cat([a, b], dim=0)  # (2N, D)
        # pairwise cosine similarity
        return F.cosine_similarity(
            reps.unsqueeze(1),  # (2N,1,D)
            reps.unsqueeze(0),  # (1,2N,D)
            dim=2
        )  # (2N,2N)

    def forward(self, proj_1, proj_2, proj_unl=None, pos_weights=None):
        # Normalize projections
        z_i = F.normalize(proj_1, p=2, dim=1)  # (N, D)
        z_j = F.normalize(proj_2, p=2, dim=1)  # (N, D)
        N = z_i.shape[0]

        # NT-Xent on positives
        sim_matrix = self.calc_similarity(z_i, z_j)  # (2N,2N)
        # positive pairs in diag offsets
        sim_ij = torch.diag(sim_matrix, N)
        sim_ji = torch.diag(sim_matrix, -N)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        numerator = torch.exp(positives / self.temperature)
        denominator = self.mask * torch.exp(sim_matrix / self.temperature)
        loss_contrast = -torch.log(numerator / torch.sum(denominator, dim=1))
        # Apply expression-based weights to focus on "hard positives"
        if pos_weights is not None:
            weights = torch.cat([pos_weights, pos_weights], dim=0)  # (2N,)
            loss_contrast = torch.sum(weights * loss_contrast) / weights.sum()
        else:
            loss_contrast = torch.sum(loss_contrast) / (2 * N)

        # Entropy penalty on unlabeled
        if proj_unl is not None and self.lambda_penalty > 0:
            z_u = F.normalize(proj_unl, p=2, dim=1)  # (U, D)
            labeled = torch.cat([z_i, z_j], dim=0)    # (2N, D)
            # similarity unlabeled->labeled
            sim_ul = torch.matmul(z_u, labeled.T) / self.temperature  # (U, 2N)
            p_ul = F.softmax(sim_ul, dim=1)  # (U, 2N)
            entropy = -torch.sum(p_ul * torch.log(p_ul + 1e-8), dim=1)  # (U,)
            loss_penalty = torch.mean(entropy)
            loss = loss_contrast + self.lambda_penalty * loss_penalty
        else:
            loss_penalty = torch.tensor(0.0, device=proj_1.device)
            loss = loss_contrast

        # save for logging if desired
        self.last_contrastive = loss_contrast.detach()
        self.last_penalty = loss_penalty.detach()
        return loss


