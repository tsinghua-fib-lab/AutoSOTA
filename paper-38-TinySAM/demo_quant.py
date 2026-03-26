import os, sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tinysam import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

sys.path.append("./tinysam")

cpt_path = "weights/tinysam_w8a8.pth"
quant_sam = torch.load(cpt_path) 
device = "cuda" if torch.cuda.is_available() else "cpu"
quant_sam.to(device=device)                                 
predictor = SamPredictor(quant_sam)

image = cv2.imread('fig/picture1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
input_point = np.array([[365, 225]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks[scores.argmax(),:,:], plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig("test_quant.png")