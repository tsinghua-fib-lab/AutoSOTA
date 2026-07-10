import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Input: 
            z_i: (Batch, Dim) - The first augmented view
            z_j: (Batch, Dim) - The second augmented view
        """
        batch_size = z_i.shape[0]
        
        # 1. Concatenate all features: [Batch, Dim] -> [2*Batch, Dim]
        # We do this to compute all-vs-all similarity in one go
        features = torch.cat([z_i, z_j], dim=0)
        
        # 2. Normalize features (L2 norm) so that dot product == cosine similarity
        features = F.normalize(features, dim=1)
        
        # 3. Compute Similarity Matrix
        # (2N, Dim) @ (Dim, 2N) -> (2N, 2N)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 4. Create Labels
        # The positive pair for row i is at index (i + batch_size)
        # The positive pair for row (i + batch_size) is at index i
        labels = torch.cat([
            torch.arange(batch_size, device=z_i.device) + batch_size,
            torch.arange(batch_size, device=z_i.device)
        ], dim=0)
        
        # 5. Mask out self-similarity (diagonal)
        # A vector is perfectly similar to itself, we must ignore this in Cross Entropy
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=z_i.device)
        
        # We fill the diagonal with -inf so Softmax ignores it (e^(-inf) = 0)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # 6. Compute Cross Entropy Loss
        # We want the 'logits' at the 'labels' indices to be maximized
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
