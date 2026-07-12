"""
AGI (Adversarial Gradient Integration) - Official Implementation.
Reference: Pan et al. "Explaining Deep Neural Network Models with Adversarial Gradient Integration" IJCAI 2021.
Based on: https://github.com/pd90506/AGI (AGI_main.py)

# NOTE (implementation detail / known issue):
# In the paper, the adversarial path is built by iteratively updating the *current* point:
# each step should move from the current perturbed image to the next perturbed image.
#
# However, the original "official" code calls FGSM step using the *fixed original image* as the base:
#     perturbed_image, delta = _fgsm_step(image, ...)
#
# This makes the procedure behave more like repeatedly sampling local perturbations around the original input,
# rather than truly following a progressive path in input space.
# As a result, the "path" is not accumulated step-by-step, and the attack may fail to reach the target class
# even with multiple iterations (depending on epsilon / model / clamp).
#
# If you want the path-style behavior described in the paper, the FGSM update should be applied on the
# current perturbed image, not the original image, e.g.:
#     perturbed_image, delta = _fgsm_step(perturbed_image.detach(), ...)
#
# Practical note:
# with the path-style fix, AGI attribution quality can be unstable and overall performance
# may not be strong compared to other explainers. The official implementation itself is known to be brittle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class AGIExplainer:
    """
    AGI Explainer following the official implementation exactly.
    """
    
    def __init__(
        self,
        model,
        device="cuda",
        exp_obj="prob",
        preprocess_fn=None,
        num_classes=1000,
        num_neg_cls=15,
        step_size=0.05,
        max_iter=15,
        mean=None,
        std=None,
    ):
        """
        Initialize AGI explainer.
        
        Args:
            model: Neural network model
            device: Device to use
            exp_obj: Not used (kept for compatibility, official always uses prob)
            preprocess_fn: Preprocessing function for inputs (not used, kept for compatibility)
            num_neg_cls: Number of negative classes (topk in official)
            step_size: Step size for FGSM (epsilon in official)
            max_iter: Maximum number of PGD iterations
            mean: Mean values for normalization (default: ImageNet)
            std: Std values for normalization (default: ImageNet)
        """
        self.model = model
        self.device = device
        self.epsilon = step_size  # Use official naming
        self.max_iter = max_iter
        self.num_classes = num_classes
        self.num_neg_cls = num_neg_cls
        
        # Set normalization parameters
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)
        
        # Pre-compute selected_ids like official: range(0, 999, int(1000/num_neg_cls))
        self.selected_ids = list(range(0, self.num_classes - 1, max(1, int(self.num_classes / self.num_neg_cls))))
    
    def _fgsm_step(self, image, epsilon, data_grad_adv, data_grad_lab):
        """
        FGSM step - exactly following official implementation.
        
        Args:
            image: Original image (not perturbed)
            epsilon: Step size
            data_grad_adv: Gradient w.r.t. adversarial class
            data_grad_lab: Gradient w.r.t. true class
        
        Returns:
            perturbed_rect: Clamped perturbed image
            delta: Attribution contribution (-grad_lab * delta)
        """
        # Generate the perturbed image based on steepest ascent
        delta = epsilon * data_grad_adv.sign()
        
        # + delta because we are ascending
        perturbed_image = image + delta
        
        # Calculate proper clamp bounds for normalized images
        min_vals = (0 - self.mean) / self.std  # Min pixel value after normalization
        max_vals = (1 - self.mean) / self.std  # Max pixel value after normalization
        
        # Clamp to valid normalized range
        perturbed_rect = torch.max(torch.min(perturbed_image, max_vals), min_vals)
        delta = perturbed_rect - image
        
        # Official formula
        delta = -data_grad_lab * delta
        
        return perturbed_rect, delta
    
    def _pgd_step(self, image, epsilon, model, init_pred, targeted, max_iter, collect_path=False, sample_idx=None):
        """
        PGD attack step - exactly following official implementation.
        
        Args:
            image: Original input image
            epsilon: Step size
            model: Model to attack
            init_pred: Original predicted class
            targeted: Target adversarial class
            max_iter: Maximum iterations
            collect_path: If True, collect intermediate perturbed images
            sample_idx: Optional sample index for logging
        
        Returns:
            c_delta: Cumulative attribution
            perturbed_image: Final perturbed image
            path_points: (optional) List of intermediate images if collect_path=True
        """
        perturbed_image = image.clone()
        c_delta = 0  # cumulative delta
        
        path_points = []
        if collect_path:
            # Add original image as first point
            path_points.append(image.squeeze(0).clone())
        
        # Track probability changes
        prev_prob = 0.0
        arrow = f"{init_pred.item()}→{targeted.item()}"
        base = f"AGI sample {sample_idx} {arrow}" if sample_idx is not None else f"AGI {arrow}"
        # pbar = tqdm(range(max_iter), desc=base, dynamic_ncols=True)
        for i in range(max_iter):
            # Requires grads
            perturbed_image.requires_grad = True
            output = model(perturbed_image)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
            # Calculate current probabilities
            probs = F.softmax(output, dim=1)
            target_prob = probs[0, targeted.item()].item()
            
            # Log progress every iteration
            # pbar.set_description(f"{base} ({i+1}/{max_iter})")
            # pbar.set_postfix(prev=f"{prev_prob:.4f}", now=f"{target_prob:.4f}", pred=int(pred.item()))
            prev_prob = target_prob
            
            if pred.item() == targeted.item():
                if collect_path:
                    path_points.append(perturbed_image.detach().squeeze(0).clone())

                # ✅ 성공 시: desc만 바꿔서 표시하고 따로 출력 안 함
                # pbar.set_description(f"SUCCESS {base} ({i+1}/{max_iter})")
                # pbar.set_postfix(prob=f"{target_prob:.4f}", pred=int(pred.item()))
                break
                    
            # Select the false class label
            output = probs  # Already softmaxed
            loss = output[0, targeted.item()]
            
            model.zero_grad()
            loss.backward(retain_graph=True)
            data_grad_adv = perturbed_image.grad.data.detach().clone()
            
            loss_lab = output[0, init_pred.item()]
            model.zero_grad()
            perturbed_image.grad.zero_()
            loss_lab.backward()
            data_grad_lab = perturbed_image.grad.data.detach().clone()
            
            # perturbed_image, delta = self._fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
            perturbed_image, delta = self._fgsm_step(perturbed_image.detach(), epsilon, data_grad_adv, data_grad_lab)
            c_delta += delta
            
            if collect_path:
                # Add intermediate perturbed image
                path_points.append(perturbed_image.detach().squeeze(0).clone())
        
        if collect_path:
            return c_delta, perturbed_image, path_points
        else:
            return c_delta, perturbed_image
    
    def get_attributions(self, inputs, labels=None, return_paths=False):
        """
        Compute AGI attributions following official test() function.
        
        Args:
            inputs: Input images [B, C, H, W] - should be in [0, 1] range
            labels: Target labels [B] (optional, if None uses predicted class)
            return_paths: If True, also return the adversarial paths
        
        Returns:
            attributions: Attribution maps [B, C, H, W]
            paths: (optional) Tuple of (paths_tensor, metadata) if return_paths=True
                - paths_tensor: [B, num_targets, max_iter+1, C, H, W]
                - metadata: dict with 'target_classes' and 'actual_lengths'
        """
        batch_size = inputs.shape[0]
        attributions = []
        all_paths = [] if return_paths else None
        path_metadata = {'target_classes': [], 'actual_lengths': []} if return_paths else None
        
        # Process each sample individually (like official)
        for b in range(batch_size):
            data = inputs[b:b+1].to(self.device)
            
            # Forward pass the data through the model
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
            # Use provided label or predicted
            if labels is not None:
                init_pred = labels[b:b+1].reshape(1, 1).to(self.device)
            
            # Initialize the step_grad towards all target false classes
            step_grad = torch.zeros_like(data)
            sample_paths = [] if return_paths else None  # Will store paths for each target class
            sample_targets = [] if return_paths else None  # Will store target classes
            sample_lengths = [] if return_paths else None  # Will store actual path lengths
            
            for l in self.selected_ids:
                targeted = torch.tensor([l]).to(self.device)
                if targeted.item() == init_pred.item():
                    continue  # We don't want to attack to the predicted class
                
                if return_paths:
                    # Collect path for each target class
                    result = self._pgd_step(
                        data, self.epsilon, self.model, init_pred, targeted, self.max_iter, 
                        collect_path=True, sample_idx=b
                    )
                    delta, perturbed_image, path_points = result
                    
                    # Store metadata
                    sample_targets.append(targeted.item())
                    sample_lengths.append(len(path_points))
                    
                    # Ensure we have max_iter+1 points for this target
                    if len(path_points) < self.max_iter + 1:
                        # Pad with last image if needed
                        last_img = path_points[-1] if path_points else data.squeeze(0)
                        while len(path_points) < self.max_iter + 1:
                            path_points.append(last_img.clone())
                    elif len(path_points) > self.max_iter + 1:
                        path_points = path_points[:self.max_iter + 1]
                    
                    sample_paths.append(torch.stack(path_points, dim=0))  # [max_iter+1, C, H, W]
                else:
                    delta, perturbed_image = self._pgd_step(
                        data, self.epsilon, self.model, init_pred, targeted, self.max_iter, 
                        collect_path=False, sample_idx=b
                    )
                step_grad += delta
                        
            attributions.append(step_grad)
            
            if return_paths:
                # Stack all target paths into tensor [num_targets, max_iter+1, C, H, W]
                if sample_paths:
                    path_tensor = torch.stack(sample_paths, dim=0)  # [num_targets, max_iter+1, C, H, W]
                    all_paths.append(path_tensor)
                    path_metadata['target_classes'].append(sample_targets)
                    path_metadata['actual_lengths'].append(sample_lengths)
                else:
                    # If no paths were collected (shouldn't happen), create empty tensor
                    empty_path = torch.zeros(1, self.max_iter + 1, *data.shape[1:]).to(self.device)
                    all_paths.append(empty_path)
                    path_metadata['target_classes'].append([])
                    path_metadata['actual_lengths'].append([])
        
        # Stack all attributions
        attributions = torch.cat(attributions, dim=0)
        
        if return_paths:
            # Pad all paths to have the same number of targets
            max_targets = max(path.shape[0] for path in all_paths) if all_paths else 0
            padded_paths = []
            padded_targets = []
            padded_lengths = []
            
            for i, path in enumerate(all_paths):
                if path.shape[0] < max_targets:
                    # Pad with zeros if needed
                    padding = torch.zeros(max_targets - path.shape[0], *path.shape[1:]).to(self.device)
                    path = torch.cat([path, padding], dim=0)
                padded_paths.append(path)
                
                # Pad metadata as well
                targets = path_metadata['target_classes'][i]
                lengths = path_metadata['actual_lengths'][i]
                targets_padded = targets + [-1] * (max_targets - len(targets))  # -1 for padding
                lengths_padded = lengths + [0] * (max_targets - len(lengths))  # 0 for padding
                padded_targets.append(targets_padded)
                padded_lengths.append(lengths_padded)
            
            paths = torch.stack(padded_paths, dim=0)  # [B, num_targets, max_iter+1, C, H, W]
            metadata = {
                'target_classes': padded_targets,  # List[List[int]]
                'actual_lengths': padded_lengths   # List[List[int]]
            }
            return attributions, (paths, metadata)
        else:
            return attributions