import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img) 

import sys
sys.path.append("..")
from tinysam import sam_model_registry, SamHierarchicalMaskGenerator

model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./weights/tinysam.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
sam.eval()
mask_generator = SamHierarchicalMaskGenerator(sam)



image = cv2.imread('fig/picture2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.hierarchical_generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig("test_everthing.png")

