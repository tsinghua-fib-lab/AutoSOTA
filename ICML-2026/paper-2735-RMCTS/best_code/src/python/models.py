import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib


def basic_nonlinear_block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, outputs),
        nn.BatchNorm1d(outputs),
        nn.ReLU()
    )

class SequentialAZModel1d(nn.Module):
    def __init__(self, input_size: int, num_actions: int, layer_dims: list[int]):
        super().__init__()
        assert len(layer_dims) > 0
        self.depth = len(layer_dims)
        
        layers = [basic_nonlinear_block(input_size, layer_dims[0])]
        for i in range(len(layer_dims) - 1):
            layers.append(basic_nonlinear_block(layer_dims[i], layer_dims[i + 1]))
        
        self.hidden = nn.Sequential(*layers)
        self.prepolicy = nn.Linear(layer_dims[-1], num_actions)
        self.value = nn.Linear(layer_dims[-1], 1)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden(x)
        prepolicy = self.prepolicy(x)
        value = self.value(x)
        logpolicy = self.log_softmax(prepolicy)
        return logpolicy, value
		
class SimpleResidualBlock2d(nn.Module):
    def __init__(self, channels: int = 16, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=False)
        y = x + res
        return y

class SimpleAZModel2d(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_actions: int, channels: int = 32, kernel_size: int = 3, blocks: int = 4, final_layer_dims = None):
        super().__init__()
        assert len(input_shape) == 3
        assert kernel_size % 2 == 1
        self.input_shape = input_shape
        self.channels = channels
        self.blocks = blocks
        self.kernel_size = kernel_size
        
        layers = [nn.Conv2d(in_channels = input_shape[0], 
                            out_channels = channels, 
                            kernel_size = 1, 
                            padding = 0)]
        for i in range(blocks):
            layers.append(SimpleResidualBlock2d(channels=channels, kernel_size=kernel_size))

        # prepare flattened output of last block:
        layers.append(nn.Flatten())

        # empirical size of the flattened output:
        x = torch.randn(1, *input_shape)
        y = layers[0](x)
        z = layers[1](y)
        w = layers[-1](z)
        output_size = w.shape[1]

        if final_layer_dims is None:
            N = input_shape[0] * input_shape[1] * input_shape[2] # size of flattened input
            final_layer_dims = [N, N // 2]

        assert len(final_layer_dims) > 0, "final_layer_dims must have at least one element"
        layers.append(basic_nonlinear_block(output_size, final_layer_dims[0]))
        for i in range(len(final_layer_dims) - 1):
            layers.append(basic_nonlinear_block(final_layer_dims[i], final_layer_dims[i + 1]))
        self.hidden = nn.Sequential(*layers)

        M = final_layer_dims[-1] # size of the final layer output

        self.pre_policy = nn.Linear(M, num_actions)
        self.value = nn.Linear(M, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, *self.input_shape)
        x = self.hidden(x)
        pre_policy = self.pre_policy(x)
        value = self.value(x)
        logpolicy = self.log_softmax(pre_policy)
        return logpolicy, value

def save_checkpoint(file_path: str, model: nn.Module) -> None:
    torch.save(model.state_dict(), file_path)

def load_checkpoint(file_path: str, model: nn.Module) -> None:
    model.load_state_dict(torch.load(file_path))

def export_onnx_for_inference(file_path: str | pathlib.PosixPath, model: nn.Module, input_shape: tuple[int, ...], batchsize: int) -> None:
    original_mode = model.training
    model.eval()
    original_device = next(model.parameters()).device
    model.to('cpu')
    if batchsize > 0:
        input_shape = (batchsize,) + input_shape
        x = torch.randn(input_shape, dtype=torch.float32)
        log_policy, value = model(x)
        torch.onnx.export(model,
                        x,
                        file_path,
                        export_params=True,
                        opset_version=10,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['log_policy', 'value'])
    else:
        dynamic_axes = {'input': {0: 'batch_size'},
                        'log_policy': {0: 'batch_size'},
                        'value': {0: 'batch_size'}}
        input_shape = (1,) + input_shape
        x = torch.randn(input_shape, dtype=torch.float32)
        log_policy, value = model(x)
        torch.onnx.export(model,
                        x,
                        file_path,
                        export_params=True,
                        opset_version=10,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['log_policy', 'value'],
                        dynamic_axes=dynamic_axes)

    model.to(original_device)
    if original_mode:
        model.train()
    else:
        model.eval()
    
def save_pt(file_path: str | pathlib.PosixPath, model: nn.Module, input_shape: tuple[int, ...] = None) -> None:
    model.eval()
    if input_shape is None:
        input_shape = model.input_shape
    with torch.no_grad():
        trace_model = torch.jit.trace(model, torch.randn(1, *input_shape))
    trace_model.save(file_path)
    model.train()

def load_pt(file_path: str) -> torch.jit.ScriptModule:
    return torch.jit.load(file_path)

