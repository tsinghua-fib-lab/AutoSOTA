import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GibbsSamplingHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size:int | None):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        
        for i in range(num_classes):
            if hidden_size is None:
                setattr(self, f'head_{i}', nn.Linear(input_size + num_classes-1, 1))
            else:
                setattr(
                    self, 
                    f'head_{i}', 
                    nn.Sequential(
                        nn.Linear(input_size + num_classes-1, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1)
                    )
                )
        
        self.head_p0 = nn.Linear(input_size, 1)

    def forward_p0(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head_p0(x)).squeeze(-1)
        # all_zeros = torch.zeros(x.size(0), self.num_classes-1)
        # probs = torch.zeros(x.size(0), self.num_classes)
        # for i in range(self.num_classes):
        #     probs[:, i] = torch.sigmoid(getattr(self, f'head_{i}')(torch.cat([x, all_zeros], dim=1)).squeeze(-1))
        # return probs.median(dim=1).values

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros_like(y)
        for i in range(self.num_classes):
            outputs[:, i] = getattr(self, f'head_{i}')(torch.cat([x, y[:, :i], y[:, i+1:]], dim=1)).squeeze(-1)
        return outputs
    
    def sample(self, x: torch.Tensor, num_samples: int, temperature: float, burn_in: int):
        batch_size = x.size(0)
        sampled_labels = torch.zeros(batch_size, num_samples, self.num_classes).to(x.device)

        current_sampled_labels = torch.zeros(batch_size, self.num_classes).to(x.device)

        for j in range(burn_in + num_samples):
            for i in range(self.num_classes):
                inputs = x
                if i > 0:
                    inputs = torch.cat([inputs, current_sampled_labels[:, :i]], dim=1)
                if i < self.num_classes - 1:
                    inputs = torch.cat([inputs, current_sampled_labels[:, i+1:]], dim=1)
                current_sampled_labels[:, i] = torch.bernoulli(
                    torch.sigmoid(
                        getattr(self, f'head_{i}')(
                            inputs / temperature
                        ).squeeze()
                    )
                )
            if j >= burn_in:
                sampled_labels[:, j-burn_in] = current_sampled_labels.clone()
        return sampled_labels
        

class AutoregressiveHead(nn.Module):
    def __init__(self, input_size:int, hidden_size:int | None, topo_order: list, parents: dict[int: list]):
        super().__init__()
        self.topo_order = topo_order
        self.parents = parents

        self.num_labels = len(topo_order)

        for i in topo_order:
            if hidden_size is None:
                setattr(self, f'head_{i}', nn.Linear(input_size + len(parents[i]), 1))
            else:
                setattr(
                    self, 
                    f'head_{i}', 
                    nn.Sequential(
                        nn.Linear(input_size + len(parents[i]), hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 1)
                    )
                )

    def forward_p0(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.ones(x.size(0)).to(x.device)
        for i in self.topo_order:
            if len(self.parents[i]) == 0:
                probs *= torch.sigmoid(getattr(self, f'head_{i}')(x).squeeze())
            else:
                all_zero = torch.zeros(x.size(0), len(self.parents[i])).to(x.device)
                probs *= torch.sigmoid(getattr(self, f'head_{i}')(torch.cat([x, all_zero], dim=1)).squeeze())
        return probs

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros_like(y)
        for i in self.topo_order:
            if len(self.parents[i]) == 0:
                outputs[:, i] = getattr(self, f'head_{i}')(x).squeeze()
            else:
                parent_labels = y[:, self.parents[i]]
                outputs[:, i] = getattr(self, f'head_{i}')(torch.cat([x, parent_labels], dim=1)).squeeze()

        return outputs

    def sample(self, x: torch.Tensor, num_samples: int, temperature: float):
        batch_size = x.size(0)
        sampled_labels = torch.zeros(batch_size, num_samples, self.num_labels).to(x.device)

        # for j in range(num_samples):
        #     for i in self.topo_order:
        #         if len(self.parents[i]) == 0:
        #             sampled_labels[:, j, i] = torch.bernoulli(torch.sigmoid(getattr(self, f'head_{i}')(x).squeeze() / temperature))
        #         else:
        #             parent_labels = sampled_labels[:, j, self.parents[i]]
        #             sampled_labels[:, j, i] = torch.bernoulli(torch.sigmoid(getattr(self, f'head_{i}')(torch.cat([x, parent_labels], dim=1)).squeeze() / temperature))
        
        for i in self.topo_order:
            if len(self.parents[i]) == 0:
                prob = torch.sigmoid(getattr(self, f'head_{i}')(x) / temperature).repeat(1, num_samples)
                sampled_labels[:, :, i] = torch.bernoulli(prob)
                x = x.unsqueeze(1).repeat(1, num_samples, 1)
            else:
                parent_labels = sampled_labels[:, :, self.parents[i]]
                prob = torch.sigmoid(getattr(self, f'head_{i}')(torch.cat([x, parent_labels], dim=2)).squeeze(-1) / temperature)
                sampled_labels[:, :, i] = torch.bernoulli(prob)

        return sampled_labels

    def greedy_predict(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        predicted_labels = torch.zeros(batch_size, self.num_labels).to(x.device)

        for i in self.topo_order:
            if len(self.parents[i]) == 0:
                predicted_labels[:, i] = torch.sigmoid(getattr(self, f'head_{i}')(x).squeeze()) > 0.5
            else:
                parent_labels = predicted_labels[:, self.parents[i]]
                predicted_labels[:, i] = torch.sigmoid(getattr(self, f'head_{i}')(torch.cat([x, parent_labels], dim=1)).squeeze()) > 0.5
        return predicted_labels


class MultinomialHead(nn.Module):
    def __init__(self, input_size: int, n_classes: int):
        super().__init__()

        self.n_classes = n_classes

        # self.head_0 = nn.Linear(input_size, 1)
        for i in range(1, n_classes + 1):
            setattr(self, f'head_{i}', nn.Linear(input_size, n_classes))

        self.head_p0 = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.zeros(x.size(0), self.n_classes, self.n_classes).to(x.device)

        for i in range(1, self.n_classes + 1):
            outputs[:, i-1, :] = getattr(self, f'head_{i}')(x)

        p0 = F.sigmoid(self.head_p0(x)).squeeze(-1)

        return outputs, p0

