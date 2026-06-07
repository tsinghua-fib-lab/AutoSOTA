"""Generate 3D render views of cows"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)


def generate_renders(num_views=40, azimuth_range=(0, 360), device=None):
    """Generate cow render views
    
    Args:
        num_views: Number of views, default is 40
        azimuth_range: Azimuth angle range, default is (0, 360)
        device: Calculation device, default is None (auto selection)
        
    Returns:
        cameras, images, silhouettes: Camera parameters, rendered images, and silhouette masks
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            
    # Set rendering parameters
    image_size = 128  # Image size
    
    # Get cow 3D model (downloaded from PyTorch3D)
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    obj_filename = os.path.join(DATA_DIR, "cow.obj")
    
    # If model does not exist, use default rendering
    if not os.path.exists(obj_filename):
        print(f"Model file not found: {obj_filename}, creating simple test images")
        # Create a simple sphere instead
        return _generate_placeholder_renders(num_views, device)
    
    # Load cow 3D model
    verts, faces, aux = load_obj(obj_filename)
    
    # Adjust vertex size and center
    verts = verts * 2.0
    center = verts.mean(0)
    verts = verts - center
    
    # Create texture
    vertex_colors = torch.ones_like(verts)[None]  # Full white
    textures = TexturesVertex(verts_features=vertex_colors)
    
    # Create mesh
    cow_mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)
    
    # Set camera positions
    azim = torch.linspace(*azimuth_range, num_views)
    elev = torch.ones_like(azim) * 20.0  # Fixed altitude angle
    dist = torch.ones_like(azim) * 2.7   # Fixed distance
    
    # Get camera parameters
    R, T = look_at_view_transform(dist, elev, azim)
    
    # Create camera
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    
    # Create rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    
    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(cameras=cameras, device=device)
    )
    
    # Render images
    cow_mesh = cow_mesh.to(device)
    rendered_images = renderer(cow_mesh)
    
    # Separate RGB and A channels
    images = rendered_images[..., :3]
    silhouettes = rendered_images[..., 3:]
    
    return cameras, images, silhouettes


def _generate_placeholder_renders(num_views=40, device=torch.device("cpu")):
    """Generate placeholder renders when no 3D model is available
    
    Args:
        num_views: Number of views
        device: Calculation device
        
    Returns:
        cameras, images, silhouettes: Camera parameters, rendered images, and silhouette masks
    """
    # Create fake camera
    azim = torch.linspace(0, 360, num_views)
    elev = torch.ones_like(azim) * 20.0
    dist = torch.ones_like(azim) * 2.7
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    
    # Create fake images (128x128 red circles)
    image_size = 128
    images = []
    silhouettes = []
    
    for i in range(num_views):
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, image_size),
            torch.linspace(-1, 1, image_size),
            indexing="ij"
        )
        z = torch.sqrt(x**2 + y**2)
        mask = (z < 0.7).float()  # Circular mask, no additional dimension
        
        # Create colored image with horizontal gradient
        r = torch.ones_like(mask) * (0.8 + 0.2 * torch.cos(torch.tensor(i / num_views * 2 * np.pi)))
        g = torch.ones_like(mask) * 0.3
        b = torch.ones_like(mask) * 0.2
        
        # Create RGB image (H, W, 3) format
        rgb_image = torch.stack([r, g, b], dim=-1)  # Ensure color channels are in the last dimension
        
        # Create silhouette mask, ensure it matches RGB format but only has one channel
        mask_image = mask  # Keep as (H, W) format
        
        images.append(rgb_image)
        silhouettes.append(mask_image)
    
    # Stack all images and silhouettes
    images = torch.stack(images)
    silhouettes = torch.stack(silhouettes)
    
    return cameras, images, silhouettes


if __name__ == "__main__":
    # Test function
    cameras, images, silhouettes = generate_renders(num_views=10)
    print(f"Generated {len(images)} images/silhouettes/cameras")
    
    # Display first image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(images[0].cpu().numpy())
    plt.title("Rendered Image")
    plt.subplot(1, 2, 2)
    plt.imshow(silhouettes[0].cpu().numpy(), cmap="gray")
    plt.title("Silhouette")
    plt.show() 