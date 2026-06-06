import torch

def reset_model(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_model(module)
        if module.__class__.__name__ == "OnlineNeuron" or module.__class__.__name__ == "LIFNeuron":
            module.reset()
            

def reset_BConv(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BinaryConv":
            module.test_time = True
            with torch.no_grad():
                new_weight = torch.mean(
                    torch.mean(
                        torch.mean(abs(module.weight), dim=3, keepdim=True),
                        dim=2, keepdim=True
                    ),
                    dim=1, keepdim=True
                ) * torch.sign(module.weight)
                module.weight.copy_(new_weight)
        elif module.__class__.__name__ == "TernaryConv":
            module.test_time = True
            with torch.no_grad():
                new_weight = module.quantizer.apply(module.weight, module.weight_clip_val)
                module.weight.copy_(new_weight)

            
def print_model_param_info(model):
    total_params, total_base_params = 0., 0.

    for name, param in model.named_parameters():
        layer_name = name[:len(name)-len(name.split('.')[-1])-1]
        layer = dict(model.named_modules())[layer_name]
        layer_class = layer.__class__.__name__
        # print(name, layer_name, layer_class, param.numel())
        params_count = param.numel() if (layer_class != "BatchNorm2d" and layer_class != "OnlineNeuron" and layer_class != "LIFNeuron") else 0.
        
        if layer_class == "BinaryConv":    
            params_per_byte = 0.25
        elif layer_class == "TernaryConv":
            params_per_byte = 0.375
        else:
            params_per_byte = 4.
        
        total_params += params_count * params_per_byte
        total_base_params += params_count * 4.

    total_params_info = {
        'total_params_mem': total_params,
        'total_base_params_mem': total_base_params
    }
    return total_params_info