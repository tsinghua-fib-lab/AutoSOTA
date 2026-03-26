import torch


def get_mdl_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)



def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)
    


def set_client_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)



def get_params_list_with_shape(model, param_list, device):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape).to(device))
        idx += length
    return vec_with_shape


