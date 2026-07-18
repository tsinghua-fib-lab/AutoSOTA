import torch
from tqdm import tqdm
from models.filet_layer import Linear
def FILet_init(
        model, 
        target_modules, 
        dataloader, 
        steps, 
        adapter_name="FILET", 
        gpu=False
        ):
    
    selected_modules = []
    for name, layer in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(layer, Linear):
            selected_modules.append(name)

    modules = {name: layer for name, layer in model.named_modules() if name in selected_modules}
    Sxy = {}
    for name, layer in modules.items():
        m, n = layer.weight.shape
        Sxy[name] = {"Sx": torch.zeros((n, n), device="cuda" if gpu else "cpu", dtype=layer.weight.dtype), "Sy": torch.zeros((m, m), device="cuda" if gpu else "cpu", dtype=layer.weight.dtype)}
        layer.gd = True
    count = 0
    
    for i, batch in tqdm(enumerate(dataloader), desc="Computing Sx, Sy: ", total=steps):

        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward(retain_graph=True)
        
        for name, layer in tqdm(modules.items(), total=len(modules), desc="Accumulating Sx, Sy: "):
            X = layer.input_cache.detach()
            dY = layer.output_grad_cache.grad.detach()
            if gpu:
                Sxy[name]["Sx"] += (torch.sum(torch.transpose(X, 1, 2) @ X, dim=0) / X.shape[0] / X.shape[1])
                Sxy[name]["Sy"] += (torch.sum(torch.transpose(dY, 1, 2) @ dY, dim=0) / dY.shape[0] / dY.shape[1])
            else:
                Sxy[name]["Sx"] += (torch.sum(torch.transpose(X, 1, 2) @ X, dim=0) / X.shape[0] / X.shape[1]).cpu()
                Sxy[name]["Sy"] += (torch.sum(torch.transpose(dY, 1, 2) @ dY, dim=0) / dY.shape[0] / dY.shape[1]).cpu()

        count += 1
        if count >= steps:
            break
        model.zero_grad(set_to_none=True)
    
    for name in Sxy.keys():
        Sxy[name]["Sx"] /= count
        Sxy[name]["Sy"] /= count


    for module_name, module in tqdm(modules.items(), total=len(modules)):
        module.gd = False
        module.reset_lora_parameters(adapter_name=adapter_name, init_lora_weights=True, Sx=Sxy[module_name]["Sx"].to("cuda"), Sy=Sxy[module_name]["Sy"].to("cuda"))
    return 



