from typing import List
import torch
import torch.nn.functional as F
from torch import nn


class PolicyTargetNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, final_act, mid_act=F.relu, target_layers_num=2):
        super().__init__()
        self.dim1 = input_dim
        self.dim2 = hidden_dim
        self.dim3 = output_dim
        self.target_layers_num = target_layers_num
        in_dim = input_dim
        for i in range(target_layers_num - 1):
            self.register_module(f"fc{i}", nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.register_module(f"fc{target_layers_num - 1}", nn.Linear(in_dim, output_dim))
        self.mid_act = mid_act
        self.final_act = final_act

    def forward(self, x):
        for i in range(self.target_layers_num - 1):
            x = self.mid_act(getattr(self, f"fc{i}")(x))
        x = self.final_act(getattr(self, f"fc{self.target_layers_num - 1}")(x))
        return x

    def get_in_dims(self) -> List[int]:
        return [self.dim1] + [self.dim2] * (self.target_layers_num - 1)

    def get_out_dims(self) -> List[int]:
        return [self.dim2] * (self.target_layers_num - 1) + [self.dim3]

    def get_submodules(self) -> List[nn.Module]:
        return [getattr(self, f"fc{i}") for i in range(self.target_layers_num)]

    def construct_opt_blocks(self, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs):
        from .opt_blocks import FIN_FOUT, OptBlock

        fin = FIN_FOUT(ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs)
        fout = FIN_FOUT(ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs)
        opt_blocks = nn.ModuleList(
            [
                OptBlock(getattr(self, f"fc{i}"), ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs)
                for i in range(self.target_layers_num)
            ]
        )
        return (opt_blocks, nn.ModuleList([fin]), nn.ModuleList([fout]))

    def get_submodule_names(self) -> List[str]:
        if not hasattr(self, "submodule_names"):
            self.submodule_names = []
            for subm in self.get_submodules():
                for name, module in self.named_modules():
                    if subm is module:
                        self.submodule_names.append(name)
                        break
        return self.submodule_names

    def merge_submodule_weights(self, weight_dicts):
        """
        convert the weight dict of the submodules to the weight dict of the target net.
        """
        weight_dict = {}
        for sub_name, wd in zip(self.get_submodule_names(), weight_dicts):
            for k, v in wd.items():
                weight_dict[sub_name + "." + k] = v

        return weight_dict
