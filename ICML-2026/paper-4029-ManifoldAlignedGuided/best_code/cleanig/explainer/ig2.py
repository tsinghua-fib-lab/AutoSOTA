"""
Reference:
    Zhuo & Ge, "IG2: Integrated Gradient on Iterative Gradient Path for Feature Attribution", TPAMI 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_by_2norm(x):
    batch_size = x.shape[0]
    flat = x.view(batch_size, -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)
    return x / norm.view(batch_size, *([1] * (x.dim() - 1)))


class RepresentationHook:
    def __init__(self):
        self.features = None
        
    def __call__(self, module, input, output):
        if isinstance(input, tuple):
            self.features = input[0]
        else:
            self.features = input


class IG2Explainer:
    def __init__(
        self,
        model,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        steps=201,
        step_size=0.02,
        n_references=1,
        reference_mode='random',
    ):
        self.model = model
        self.device = device
        self.exp_obj = exp_obj
        self.steps = steps
        self.step_size = step_size
        self.n_references = n_references
        self.reference_mode = reference_mode
        
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x
        
        self.rep_hook = RepresentationHook()
        self._register_hook()
        
        self.reference_bank = None
    
    def _register_hook(self):
        if hasattr(self.model, 'fc'):
            self.rep_layer = self.model.fc
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                self.rep_layer = self.model.classifier[0]
            else:
                self.rep_layer = self.model.classifier
        elif hasattr(self.model, 'linear'):
            self.rep_layer = self.model.linear
        else:
            raise ValueError("Cannot find representation layer in model")
        
        self.hook_handle = self.rep_layer.register_forward_hook(self.rep_hook)
    
    def set_reference_bank(self, reference_images):
        self.reference_bank = reference_images.to(self.device)
    
    def _get_references(self, batch_size, exclude_indices=None):
        if self.reference_bank is None:
            raise ValueError("Reference bank not set. Call set_reference_bank() first or use diffid.py which sets it automatically.")
        
        references_list = []
        for b in range(batch_size):
            if self.reference_mode == 'random':
                indices = torch.randperm(len(self.reference_bank))[:self.n_references]
            else:
                indices = torch.arange(min(self.n_references, len(self.reference_bank)))
            
            refs = self.reference_bank[indices]
            references_list.append(refs)
        
        return references_list
    
    def _get_representation(self, x):
        with torch.no_grad():
            _ = self.model(x)
        assert self.rep_hook.features is not None
        return self.rep_hook.features.clone()
    
    def _get_rep_distance_gradients(self, x, target_rep):
        x = x.clone().detach().requires_grad_(True)
        
        _ = self.model(x)
        assert self.rep_hook.features is not None
        current_rep = self.rep_hook.features
        
        loss = -F.mse_loss(current_rep, target_rep)
        loss.backward()
        
        return x.grad.detach(), loss.item()
    
    def _get_output_gradients(self, x, labels):
        x = x.clone().detach().requires_grad_(True)
        
        output = self.model(x)
        
        if self.exp_obj == 'logit':
            target = output[torch.arange(output.shape[0], device=self.device), labels]
        elif self.exp_obj == 'prob':
            probs = torch.softmax(output, dim=-1)
            target = probs[torch.arange(output.shape[0], device=self.device), labels]
        else:
            raise ValueError(f'Invalid objective function: {self.exp_obj}')
        
        grad = torch.autograd.grad(target.sum(), x)[0]
        return grad.detach()
    
    def _search_grad_path(self, x_input, references):
        n_refs = references.shape[0]
        x_repeated = x_input.unsqueeze(0).expand(n_refs, -1, -1, -1)
        
        target_reps = self._get_representation(references)
        
        delta = torch.zeros_like(x_repeated)
        path = [x_repeated.clone()]
        
        for i in range(self.steps):
            x_current = x_repeated + delta
            
            grads, loss = self._get_rep_distance_gradients(x_current, target_reps)
            
            grads_normalized = normalize_by_2norm(grads)
            delta = delta + grads_normalized * self.step_size
            
            path.append((x_repeated + delta).clone())
        
        return path
    
    def _integrate_gradients(self, path, label):
        n_refs = path[0].shape[0]
        labels_repeated = label.expand(n_refs)
        
        attr = torch.zeros_like(path[0])
        
        for i in range(1, len(path)):
            x_old = path[i - 1]
            x_new = path[i]
            
            grads = self._get_output_gradients(x_old, labels_repeated)
            attr = attr + (x_old - x_new) * grads
        
        return attr.mean(dim=0)
    
    def get_attributions(self, inputs, labels=None):
        inputs = inputs.to(self.device)
        batch_size = inputs.shape[0]
        
        if labels is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                labels = outputs.argmax(dim=1)
        
        labels = labels.to(self.device)
        
        references_list = self._get_references(batch_size)
        
        attributions = []
        for b in range(batch_size):
            x_input = inputs[b]
            label = labels[b:b+1]
            references = references_list[b]
            
            path = self._search_grad_path(x_input, references)
            
            attr = self._integrate_gradients(path, label)
            attributions.append(attr)
        
        return torch.stack(attributions, dim=0)
    
    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
