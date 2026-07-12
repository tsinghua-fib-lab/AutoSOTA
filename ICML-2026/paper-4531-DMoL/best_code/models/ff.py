import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FFLayer(nn.Module):
    def __init__(self, in_features, out_features, threshold): 
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = optim.Adam(self.parameters(), lr=0.0)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, x): 
        return self.relu(self.linear(x / (x.norm(2, 1, keepdim=True) + 1e-4)))
        
    def train_layer(self, pos_x, neg_x):
        logits = torch.cat([self(pos_x).pow(2).mean(1), self(neg_x).pow(2).mean(1)])
        labels = torch.cat([torch.ones_like(logits[:len(pos_x)]), torch.zeros_like(logits[len(pos_x):])])
        loss = self.loss_fn(logits - self.threshold, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class FF_Network(nn.Module):
    def __init__(self, num_modules, num_classes, in_channels, img_size, feature_dim=512, threshold=2.0, downstream_lr=0.01):
        super().__init__()
        first_layer_in_features = in_channels * img_size * img_size
        
        self.ff_layers = nn.ModuleList()
        for i in range(num_modules):
            in_features = first_layer_in_features if i == 0 else feature_dim
            self.ff_layers.append(FFLayer(in_features, feature_dim, threshold))
            
        self.downstream_classifier = nn.Linear(feature_dim * num_modules, num_classes)
        self.classifier_optimizer = optim.Adam(self.downstream_classifier.parameters(), lr=downstream_lr)
        self.classifier_loss_fn = nn.CrossEntropyLoss()

    def forward(self, pos_x, neg_x=None, neutral_x=None, y=None):
        if self.training:
            h_pos = pos_x.view(pos_x.size(0), -1)
            h_neg = neg_x.view(neg_x.size(0), -1)
            losses = []
            
            for layer in self.ff_layers:
                losses.append(layer.train_layer(h_pos, h_neg))
                h_pos = layer(h_pos).detach()
                h_neg = layer(h_neg).detach()
            
            with torch.no_grad():
                h = neutral_x.view(neutral_x.size(0), -1)
                features = []
                for layer in self.ff_layers:
                    h = layer(h)
                    features.append(h)

            logits = self.downstream_classifier(torch.cat(features, dim=1).detach())
            loss = self.classifier_loss_fn(logits, y)
            
            self.classifier_optimizer.zero_grad()
            loss.backward()
            self.classifier_optimizer.step()
            
            return np.mean(losses) + loss.item()
        else:
            with torch.no_grad():
                h = pos_x.view(pos_x.size(0), -1)
                features = []
                for layer in self.ff_layers:
                    h = layer(h)
                    features.append(h)
                return self.downstream_classifier(torch.cat(features, dim=1))
