import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading (example)
class ExampleDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.int64).to(device)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Define encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        hidden_dim = 512
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Define decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(output_dim, input_dim)
        )
    
    def forward(self, z):
        return self.layers(z)

# Define the complete model
class GSTransformModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(GSTransformModel, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

def contrastive_loss(z, labels, margin=1.0):
    """
    Contrastive loss function (vectorized implementation)
    z: [batch_size, embedding_dim] - transformed embedding points
    labels: [batch_size] - class labels of points
    margin: float - minimum distance between points of different classes
    """
    # Calculate Euclidean distances between batch points
    distances = torch.cdist(z, z, p=2)  # [batch_size, batch_size]
    
    # Construct label matrix
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [batch_size, batch_size]
    
    # Loss for same class points
    positive_distances = distances[labels_equal]
    positive_loss = positive_distances.pow(2)
    
    # Loss for different class points
    neg_mask = ~labels_equal
    negative_distances = distances[neg_mask]
    negative_loss = torch.clamp(margin - negative_distances, min=0).pow(2)
    
    # Combine all losses and calculate mean
    combined_loss = torch.cat([positive_loss, negative_loss])
    return combined_loss.mean()

