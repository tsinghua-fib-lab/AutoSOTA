import torch.nn as nn
import torch.nn.functional as F

class fastText(nn.Module):
    def __init__(self, hidden_dim = 32, padding_idx=0, vocab_size=98635, num_classes=4):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        # out = F.log_softmax(z, dim=1)

        return z