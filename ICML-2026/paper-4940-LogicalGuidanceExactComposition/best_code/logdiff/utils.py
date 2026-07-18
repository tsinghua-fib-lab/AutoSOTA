import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from typing import Union
import os

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def true_generated_image_grid_save(true_images, generated_images, save_path):
    #log_images to a file
    fig, (ax1, ax2) = plt.subplots(1, 2)
    save_images(true_images.detach().cpu(), path=ax1, title='true_samples')
    save_images(generated_images.detach().cpu(), path=ax2, title='conditional_generated_samples')
    fig.savefig(save_path)
    plt.close(fig)

def make_grid(tensor, images_per_row=4):
    num_images = tensor.shape[0]
    img_height = tensor.shape[2]
    img_width = tensor.shape[3]
    
    # Calculate grid size
    grid_height = (num_images // images_per_row) + (num_images % images_per_row > 0)
    
    # Create an empty grid to hold all the images
    grid = np.zeros((grid_height * img_height, images_per_row * img_width, 3))
    
    for idx in range(num_images):
        row = idx // images_per_row
        col = idx % images_per_row
        # Convert from PyTorch tensor format (C, H, W) to NumPy format (H, W, C)
        img = tensor[idx].permute(1, 2, 0).numpy()
        # Clip the image values between 0 and 1 for display purposes
        img = np.clip(img, 0, 1)
        # Place the image in the correct position in the grid
        grid[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img
    
    return grid

def save_images(samples, path:Union[str,plt.Axes],title:str):
    grid = make_grid(samples, images_per_row=int(np.sqrt(samples.size(0))))
    if isinstance(path, str):
        plt.imshow(grid)
        plt.axis('off')
        plt.title(title)
        plt.savefig(path)
        plt.show()
    else:
        path.imshow(grid)
        path.axis('off')
        path.set_title(title)
        return path

def save_images_in_folder(samples, y, path:Union[str,plt.Axes],title:str,counter:int=0):
    if isinstance(path, str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    for i,sample in enumerate(samples):
        title_ = path+ f"/{counter+i}"+ "_".join([title,str(y[i])])+".png"
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        plt.imsave(title_,np.clip(sample.permute(1, 2, 0).numpy(),0,1))
        