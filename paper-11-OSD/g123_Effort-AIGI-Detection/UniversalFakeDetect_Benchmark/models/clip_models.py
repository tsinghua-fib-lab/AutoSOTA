import math 
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
import loralib as lora


class ClipModel(nn.Module):
    def __init__(self, name, opt, num_classes=1):
        super(ClipModel, self).__init__()
        self.use_svd = opt.use_svd
        
        if self.use_svd:
            self.model = CLIPModel.from_pretrained(name)
            self.model.vision_model = apply_svd_residual_to_self_attn(self.model.vision_model, r=1024-1)
            
            for name, param in self.model.vision_model.named_parameters():
                print('{}: {}'.format(name, param.requires_grad))
            num_param = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
            num_total_param = sum(p.numel() for p in self.model.vision_model.parameters())
            print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
            
            self.fc = nn.Linear( 1024, num_classes )
        else:
            self.model = CLIPModel.from_pretrained(name)
            
            for name, param in self.model.vision_model.named_parameters():
                print('{}: {}'.format(name, param.requires_grad))
            num_param = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
            num_total_param = sum(p.numel() for p in self.model.vision_model.parameters())
            print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
            
            self.fc = nn.Linear( 1024, num_classes )

    def forward(self, x, return_feature=False):
        features = self.model.vision_model(x)['pooler_output']
        
        if return_feature:
            return features
        return self.fc(features)


# Custom module to represent the residual using SVD components
class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super(SVDResidualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # Number of top singular values to exclude

        # Main weight (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        # SVD components
        self.S_r = None
        self.U_r = None
        self.V_r = None
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None

    def forward(self, x):
        if self.S_residual is not None:
            # Reconstruct the residual weight
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            # Total weight is the fixed main weight plus the residual
            weight = self.weight_main + residual_weight
        else:
            # If residual components are not set, use only the main weight
            weight = self.weight_main

        return F.linear(x, weight, self.bias)
                   

# Function to replace nn.Linear modules within self_attn modules with SVDResidualLinear
def apply_svd_residual_to_self_attn(model, r):
    for name, module in model.named_children():
        # if ('self_attn' in name) or ('mlp' in name):
        if ('self_attn' in name):
            # Replace nn.Linear layers in this module
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    # Get parent module within self_attn
                    parent_module = module
                    sub_module_names = sub_name.split('.')
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    # Replace the nn.Linear layer with SVDResidualLinear
                    setattr(parent_module, sub_module_names[-1], replace_with_svd_residual(sub_module, r))
        else:
            # Recursively apply to child modules
            apply_svd_residual_to_self_attn(module, r)
    # After replacing, set requires_grad for residual components
    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ['S_residual', 'U_residual', 'V_residual']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# Function to replace a module with SVDResidualLinear
def replace_with_svd_residual(module, r):
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None

        # Create SVDResidualLinear module
        new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)
            
        # Calculate the frobenius norm of original weight
        new_module.weight_original_fnorm = torch.norm(module.weight.data, p='fro')

        # Perform SVD on the original weight
        U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)

        # Determine r based on the rank of the weight matrix
        r = min(r, len(S))  # Ensure r does not exceed the number of singular values

        # Keep top r singular components (main weight)
        U_r = U[:, :r]      # Shape: (out_features, r)
        S_r = S[:r]         # Shape: (r,)
        Vh_r = Vh[:r, :]    # Shape: (r, in_features)

        # Reconstruct the main weight (fixed)
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        
        # Calculate the frobenius norm of main weight
        new_module.weight_main_fnorm = torch.norm(weight_main.data, p='fro')

        # Set the main weight
        new_module.weight_main.data.copy_(weight_main)

        # Residual components (trainable)
        U_residual = U[:, r:]    # Shape: (out_features, n - r)
        S_residual = S[r:]       # Shape: (n - r,)
        Vh_residual = Vh[r:, :]  # Shape: (n - r, in_features)

        if len(S_residual) > 0:
            new_module.S_residual = nn.Parameter(S_residual.clone())
            new_module.U_residual = nn.Parameter(U_residual.clone())
            new_module.V_residual = nn.Parameter(Vh_residual.clone())
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None

        return new_module
    else:
        return module