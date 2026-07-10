import torch
from torch import nn
import torch.nn.functional as F
import copy


class ProbModel(nn.Module):
    # Take the last penultimate fc layer
    # Add another fc layer with double size, initialized partially from old parameters
    def __init__(self, model):
        super().__init__()
        self.name = "Prob" + model.name
        self.num_classes = model.num_classes
        try:
            print(f"Creating the model {self.name}")
            # Find the penultimate linear layer
            children = list(model.net.named_children())
            for idx, (name, module) in reversed(list(enumerate(children))):
                if idx == len(children) - 1:
                    continue
                if "fc" in name:
                    old_weight = module.weight
                    old_bias = module.bias
                    self.net = nn.Sequential(model.net[:idx])
                    self.last_layer = model.net[-1]
                    break
            old_dim_out = old_weight.shape[0]
            old_dim_in = old_weight.shape[1]
            layer = nn.Linear(old_dim_in, 2 * old_dim_out)
            layer.weight.data[:old_dim_out, :] = old_weight.data
            layer.bias.data[:old_dim_out] = old_bias
            self.net.append(layer)
            self.num_features = old_dim_out
        except Exception:
            print("Creating a probabilistic ResNet")
            # ResNets only have one fully connected layer at the end
            # so we need to add another one
            self.make_prob_feat_resnet(model)
            self.last_layer = model.fc
            self.num_classes = model.num_classes

    def make_prob_feat_resnet(self, resnet):
        dim = resnet.fc.in_features
        prob_layer = nn.Linear(dim, 2 * dim)
        prob_layer.weight.data[:dim, :] = torch.eye(dim)
        # Std predictions should be small initially!
        prob_layer.weight.data[dim:, :] = 1e-4 * torch.eye(dim)
        prob_layer.bias.data[:dim] = torch.zeros(dim)
        prob_layer.bias.data[dim:] = 1e-4 * torch.ones(dim)
        self.num_features = dim
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
            prob_layer,
        )

    def forward(self, input_data):
        out = self.net(input_data)
        mean = out[:, : self.num_features]
        std = out[:, self.num_features :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(
            base_distribution=normal_dist, reinterpreted_batch_ndims=1
        )
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return output

    def forward_distr(self, input_data):
        out = self.net(input_data)
        mean = out[:, : self.num_features]
        std = out[:, self.num_features :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(
            base_distribution=normal_dist, reinterpreted_batch_ndims=1
        )
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return mean, std, sample, output, feat_dist

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())
