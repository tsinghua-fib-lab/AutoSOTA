import random
from tqdm import tqdm
from typing import Tuple

import torch

# This model takes original model and outputs layer of interest
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, layer_of_interest, architecture='CNN', hook_input=False):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.activations = {}
        self.architecture = architecture
        self.hook_input = hook_input
        # Define a forward hook to capture the activations
        def forward_hook(module, inputs, output):
            # inputs is always a tuple
            if self.hook_input:
                # Usually the first tensor is the actual layer input
                self.activations["output"] = inputs[0].clone() # for in place clone
            else:
                self.activations["output"] = output
     
        # Register the hook
        hook_handle = layer_of_interest.register_forward_hook(forward_hook)

        self.downsample = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        _ = self.model(x)
        if self.architecture == 'CNN':
            x = self.downsample(self.activations["output"]).squeeze(-1).squeeze(-1)
        elif self.architecture == 'TransformerMIN':
            x = torch.min(self.activations["output"], dim=1).values
        return x



# remove inplace operations
def ReLU_inplace_to_False(module):
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)

def get_random_patches(
    dataloader: torch.utils.data.DataLoader,
    patch_size: int = 64,
    stride: int = 16,
    num_patches: int = 4):
    """    
    Args:
        dataloader: PyTorch dataloader containing images
        patch_size: Size of square patches to extract
        stride: Stride of patches
        num_patches: Number of top patches to keep
    
    Returns:
        Returns patches (num_patches, 3, patch_size, patch_size)
    """
    patches = torch.zeros((num_patches, 3, patch_size, patch_size))

    total_samples = len(dataloader.dataset)

    possible_locations_x_y = list(range(0, 224-patch_size, stride))

    img_ids = random.sample(range(0, total_samples), num_patches) # get the ID of an image for each patch I want to sample

    for i,img_id in enumerate(img_ids):
        image = dataloader.dataset[img_id][0]
        patch_x = random.sample(possible_locations_x_y, 1)[0]
        patch_y = random.sample(possible_locations_x_y, 1)[0]
        patch = image[:,patch_y:patch_y+patch_size,patch_x:patch_x+patch_size]
        patches[i,:,:,:] = patch

    return patches

@torch.no_grad()
def get_lowest_highest_patches(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    output_size: int,
    patch_size: int = 64,
    stride: int = 16,
    num_patches: int = 4,
    return_images: bool = False,
    architecture: str = "cnn",
):
    """
    Combined CNN/ViT patch search.

    architecture="cnn":
      Evaluate raw patch crops directly with the model.

    architecture="vit":
      Evaluate full-size masked images where only one candidate patch is visible,
      but still return the raw cropped patches.
    """
    if architecture not in {"cnn", "vit"}:
        raise ValueError(f"Expected architecture to be 'cnn' or 'vit', got {architecture!r}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    highest_patches_scores = torch.ones((output_size, num_patches), device=device) * (-float("inf"))
    lowest_patches_scores = torch.ones((output_size, num_patches), device=device) * float("inf")

    highest_patches = torch.ones((output_size, num_patches, 3, patch_size, patch_size), device=device)
    lowest_patches = torch.ones((output_size, num_patches, 3, patch_size, patch_size), device=device)

    if return_images:
        highest_images = torch.ones((output_size, num_patches), device=device)
        lowest_images = torch.ones((output_size, num_patches), device=device)

    print(f"Finding patches ({architecture})...")
    iteration = 0

    for images, label in tqdm(dataloader):
        images = images.to(device)
        batch_size, channels, height, width = images.shape
        assert batch_size == 1 and channels == 3, f"Expected [1,3,H,W], got {images.shape}"

        if architecture == "cnn":
            patches = images.unfold(1, 3, 1).unfold(2, patch_size, stride).unfold(3, patch_size, stride)
            patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
            outputs = model(patches)

            if outputs.dim() != 2 or outputs.shape[1] != output_size:
                raise ValueError(
                    f"Expected model(patches) -> [N,{output_size}], got {tuple(outputs.shape)}."
                )

            topk_vals, topk_indices = torch.topk(outputs, k=num_patches, dim=0)
            bottomk_vals, bottomk_indices = torch.topk(outputs, k=num_patches, dim=0, largest=False)

            topk_vals = topk_vals.permute(1, 0)
            bottomk_vals = bottomk_vals.permute(1, 0)
            topk_indices = topk_indices.permute(1, 0)
            bottomk_indices = bottomk_indices.permute(1, 0)

            topk_patches = patches[topk_indices]
            bottomk_patches = patches[bottomk_indices]

        else:
            num_y = max(0, 1 + (height - patch_size) // stride)
            num_x = max(0, 1 + (width - patch_size) // stride)
            num_windows = num_y * num_x

            if num_windows == 0:
                iteration += 1
                continue

            ys = torch.arange(0, num_y * stride, step=stride, device=device)
            xs = torch.arange(0, num_x * stride, step=stride, device=device)

            raw_patches = []
            for y in ys.tolist():
                row = []
                for x in xs.tolist():
                    row.append(images[0, :, y:y + patch_size, x:x + patch_size])
                raw_patches.append(torch.stack(row, dim=0))
            raw_patches = torch.stack(raw_patches, dim=0)
            raw_patches = raw_patches.view(num_windows, 3, patch_size, patch_size)

            image_top_scores = torch.ones((output_size, num_patches), device=device) * (-float("inf"))
            image_bottom_scores = torch.ones((output_size, num_patches), device=device) * float("inf")
            image_top_indices = torch.zeros((output_size, num_patches), device=device, dtype=torch.long)
            image_bottom_indices = torch.zeros((output_size, num_patches), device=device, dtype=torch.long)

            chunk_size = max(1, min(512, num_windows))
            channel_indices = torch.arange(output_size, device=device).unsqueeze(1).expand(-1, num_patches)

            for start in range(0, num_windows, chunk_size):
                end = min(num_windows, start + chunk_size)
                current_chunk_size = end - start

                masked_images = torch.zeros((current_chunk_size, 3, height, width), device=device, dtype=images.dtype)
                window_indices = torch.arange(start, end, device=device)
                window_y = (window_indices // num_x).tolist()
                window_x = (window_indices % num_x).tolist()

                for k in range(current_chunk_size):
                    y = window_y[k] * stride
                    x = window_x[k] * stride
                    masked_images[k, :, y:y + patch_size, x:x + patch_size] = images[0, :, y:y + patch_size, x:x + patch_size]

                outputs = model(masked_images)
                if outputs.dim() != 2 or outputs.shape[0] != current_chunk_size or outputs.shape[1] != output_size:
                    raise ValueError(
                        f"Expected model(masked_images) -> [N,{output_size}], got {tuple(outputs.shape)}. "
                        "If the model returns tokens, reduce them inside intermediate_layer_model."
                    )

                k_patches = min(num_patches, current_chunk_size)
                topk_vals, topk_indices = torch.topk(outputs, k=k_patches, dim=0)
                bottomk_vals, bottomk_indices = torch.topk(outputs, k=k_patches, dim=0, largest=False)

                topk_vals = topk_vals.permute(1, 0)
                bottomk_vals = bottomk_vals.permute(1, 0)
                topk_indices = topk_indices.permute(1, 0) + start
                bottomk_indices = bottomk_indices.permute(1, 0) + start

                tmp_scores = torch.cat([image_top_scores, topk_vals], dim=1)
                tmp_indices = torch.cat([image_top_indices, topk_indices], dim=1)
                image_top_scores, selected = torch.topk(tmp_scores, k=num_patches, dim=1)
                image_top_indices = tmp_indices[channel_indices, selected]

                tmp_scores = torch.cat([image_bottom_scores, bottomk_vals], dim=1)
                tmp_indices = torch.cat([image_bottom_indices, bottomk_indices], dim=1)
                image_bottom_scores, selected = torch.topk(tmp_scores, k=num_patches, dim=1, largest=False)
                image_bottom_indices = tmp_indices[channel_indices, selected]

            topk_vals = image_top_scores
            bottomk_vals = image_bottom_scores
            topk_patches = raw_patches[image_top_indices]
            bottomk_patches = raw_patches[image_bottom_indices]

        tmp_scores_highest = torch.cat([highest_patches_scores, topk_vals], dim=1)
        tmp_patches_highest = torch.cat([highest_patches, topk_patches], dim=1)
        if return_images:
            tmp_images_highest = torch.cat(
                [highest_images, torch.full((output_size, num_patches), float(iteration), device=device)],
                dim=1,
            )

        top_patches_scores, topk_indices = torch.topk(tmp_scores_highest, k=num_patches, dim=1)

        tmp_scores_lowest = torch.cat([lowest_patches_scores, bottomk_vals], dim=1)
        tmp_patches_lowest = torch.cat([lowest_patches, bottomk_patches], dim=1)
        if return_images:
            tmp_images_lowest = torch.cat(
                [lowest_images, torch.full((output_size, num_patches), float(iteration), device=device)],
                dim=1,
            )

        bottom_patches_scores, bottomk_indices = torch.topk(tmp_scores_lowest, k=num_patches, dim=1, largest=False)

        channel_indices = torch.arange(output_size, device=device).unsqueeze(1).expand(-1, num_patches)
        top_patches = tmp_patches_highest[channel_indices, topk_indices, :, :, :]
        bottom_patches = tmp_patches_lowest[channel_indices, bottomk_indices, :, :, :]

        if return_images:
            top_images = tmp_images_highest[channel_indices, topk_indices]
            bottom_images = tmp_images_lowest[channel_indices, bottomk_indices]

        highest_patches_scores = top_patches_scores
        lowest_patches_scores = bottom_patches_scores
        highest_patches = top_patches
        lowest_patches = bottom_patches
        if return_images:
            highest_images = top_images
            lowest_images = bottom_images

        iteration += 1

    if return_images:
        return bottom_patches, bottom_patches_scores, lowest_images, top_patches, top_patches_scores, highest_images
    else:
        return bottom_patches, bottom_patches_scores, top_patches, top_patches_scores


def paste_patches(images, patches):
    B, C, H, W = images.shape
    patch_size = patches.shape[-1]

    # Step 1: Sample N random patch indices (with replacement)
    patch_indices = torch.randint(0, patches.size(0), (B,))

    # Step 2: For each image, choose a random top-left corner where patch fits
    max_x = W - patch_size
    max_y = H - patch_size
   
    xs = torch.randint(0, 2, (B,)) * (max_x) # random part outputs 0 or 1; then multiply with max vals to make sure patch is placed in corners
    ys = torch.randint(0, 2, (B,)) * (max_y)

    # Step 3: Paste patches into the images
    augmented_images = images.clone()
    #return augmented_images
    for i in range(B):
        patch = patches[patch_indices[i]]
        x, y = xs[i], ys[i]
        augmented_images[i, :, y:y+patch_size, x:x+patch_size] = patch

    return augmented_images
