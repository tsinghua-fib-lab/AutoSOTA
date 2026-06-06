import os
import torch


def save_gaussian_count(final_gaussian_count, total_epoch, final_iteration, model_path):
    """Save Gaussian count information to file."""
    os.makedirs(model_path, exist_ok=True)
    gaussian_count_file = os.path.join(model_path, "gaussian_count.txt")
    with open(gaussian_count_file, 'w') as f:
        f.write(f"Final Gaussian count: {final_gaussian_count}\n")
        f.write(f"Total epochs: {total_epoch}\n")
        f.write(f"Final iteration: {final_iteration}\n")
    print(f"Saved Gaussian count ({final_gaussian_count}) to {gaussian_count_file}")


def save_training_time(elapsed_time, model_path):
    """Save training time information to file."""
    os.makedirs(model_path, exist_ok=True)
    timing_file = os.path.join(model_path, "training_time.txt")
    with open(timing_file, 'w') as f:
        f.write(f"Training wall time: {elapsed_time:.2f} seconds\n")
        f.write(f"Training wall time: {elapsed_time/60:.2f} minutes\n")
    print(f"Saved training time ({elapsed_time:.2f}s) to {timing_file}")


def compute_gaussian_count_per_tile(tile_start_index: torch.Tensor) -> torch.Tensor:
    """
    Compute the number of Gaussians per tile from tile start indices.
    
    Args:
        tile_start_index: Tensor of shape [batch, total_tiles + 2] containing the starting 
                         index of Gaussians for each tile. Note that tile_id 0 is invalid,
                         so valid tiles are indexed from 1 to total_tiles.
    
    Returns:
        gaussian_count_per_tile: Tensor of shape [batch, total_tiles] with number of Gaussians per tile
    
    Note:
        The gaussian_count_per_tile is 0-based, so gaussian_count_per_tile[:, i] represents 
        the Gaussian count for tile i+1 (since tile_id 0 is invalid).
        
        total_tiles is derived from tile_start_index.shape[1] - 2 since tile_start_index 
        is guaranteed to have shape [batch, total_tiles + 2].
    """
    # Derive total_tiles from the shape of tile_start_index
    total_tiles = tile_start_index.shape[1] - 2
    
    # Extract start indices for valid tiles (1 to total_tiles)
    start_indices = tile_start_index[:, 1:total_tiles+1]  # [batch, total_tiles]
    
    # Extract end indices (start of next tile) for valid tiles
    end_indices = tile_start_index[:, 2:total_tiles+2]    # [batch, total_tiles]
    
    # Do subtraction immediately
    gaussian_count_per_tile = end_indices - start_indices
    
    # Handle special cases: if start_indices is -1, then gaussian_count_per_tile is 0
    start_invalid_mask = (start_indices == -1)
    gaussian_count_per_tile = torch.where(start_invalid_mask, 0, gaussian_count_per_tile)
    
    # Handle special cases: if end_indices is -1, then gaussian_count_per_tile is 0
    end_invalid_mask = (end_indices == -1)
    gaussian_count_per_tile = torch.where(end_invalid_mask, 0, gaussian_count_per_tile)
    
    return gaussian_count_per_tile


def preprocess_gt_images(train_loader, resized_image_cache, pp, allowed_scales):
    """Preprocess GT images by resizing them for different render scales and caching them."""
    
    def _get_batch_for_scale(gt_image_batch, frame_names, scale):
        """Get the appropriate image batch for a given scale."""
        if scale == 1:
            # Cache scale=1 GT images (already processed with cuda()/255.0)
            cached_images = []
            for i, frame_name in enumerate(frame_names):
                cache_key = (frame_name, scale, "gt")
                if cache_key not in resized_image_cache:
                    resized_image_cache[cache_key] = gt_image_batch[i]
                cached_images.append(resized_image_cache[cache_key])
            return torch.stack(cached_images, dim=0)
        
        # For scale > 1, get or create resized images from scale=1 cached images
        resized_images = []
        for i, frame_name in enumerate(frame_names):
            cache_key = (frame_name, scale, "gt")
            if cache_key not in resized_image_cache:
                # Get the scale=1 cached image as source
                scale_1_key = (frame_name, 1, "gt")
                if scale_1_key in resized_image_cache:
                    source_img = resized_image_cache[scale_1_key]
                else:
                    source_img = gt_image_batch[i]
                
                resized_img = torch.nn.functional.interpolate(source_img[None], scale_factor=1/scale, mode="bilinear",
                                                       recompute_scale_factor=True, antialias=True)[0]
                resized_image_cache[cache_key] = resized_img
            resized_images.append(resized_image_cache[cache_key])
        
        return torch.stack(resized_images, dim=0)

    with torch.no_grad():
        for view_matrix, proj_matrix, frustumplane, gt_image, frame_name in train_loader:
            # Process GT image exactly like in training loop
            gt_image = gt_image.cuda() / 255.0
            
            for scale in allowed_scales:
                # Get appropriate batch for this scale
                _get_batch_for_scale(gt_image, frame_name, scale)
