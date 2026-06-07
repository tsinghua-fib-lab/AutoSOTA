"""Helpers for constructing volume renderers."""

import torch
from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    GridRaysampler
)


# NOTE: render_size accepts either an int (square output) or a (H, W) tuple/list.

def create_volume_renderer(
        render_size: int | tuple[int, int] | list[int],
        volume_extent_world: float = 3.0,
        n_pts_per_ray: int = 150,
        min_depth: float = 0.0,
        device=None,
        orthographic: bool = False):

    # Resolve device lazily
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            

    if isinstance(render_size, (tuple, list)):
        render_h, render_w = int(render_size[0]), int(render_size[1])
    else:
        render_h = render_w = int(render_size)

    if orthographic:

        half_y = volume_extent_world / 2.0
        aspect = render_w / render_h
        half_x = half_y * aspect

        raysampler = GridRaysampler(
            min_x=-half_x, max_x=half_x,
            min_y=-half_y, max_y=half_y,
            image_width=render_w,
            image_height=render_h,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=volume_extent_world,
        )
    else:

        raysampler = NDCMultinomialRaysampler(
            image_width=render_w,
            image_height=render_h,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=volume_extent_world,
        )

    raysampler = raysampler.to(device)


    raymarcher = EmissionAbsorptionRaymarcher()

    raymarcher = raymarcher.to(device)

    renderer = VolumeRenderer(
        raysampler=raysampler, raymarcher=raymarcher,
    )
    

    renderer = renderer.to(device)
    
    return renderer 
