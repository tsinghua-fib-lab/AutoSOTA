import glob
import torch
import numpy as np
import os
from scipy import ndimage
from PIL import Image

def pil_img_to_torch(pil_img):
    pil_img = np.array(pil_img)
    if len(pil_img.shape)==2:
        pil_img = pil_img.reshape(pil_img.shape[0],pil_img.shape[1],1).repeat(3,axis=2)
    if pil_img.shape[2] == 2:
        pil_img = pil_img[:,:,0].reshape(pil_img.shape[0],pil_img.shape[1],1)
        pil_img = pil_img.repeat(3,axis=2)
        
    pil_img = pil_img[:,:,:3]
    return torch.from_numpy(pil_img.astype('float32'))/255.0

def torch_to_pil(torch_img):
    return Image.fromarray((torch_img.numpy()*255).astype(np.uint8))

def retrieve_mask_from_folder(img_folder, prefix='dl_'):
    # outfolder = 'retrieve_gen/Shaw Floors Spice It Up Tyler Taupe 00103_E9013/'
    img_files = sorted(glob.glob(f'{img_folder}{prefix}*'))
    imgs_out = [pil_img_to_torch(Image.open(file)) for file in img_files]
    # for creation of clean mask
    q_img = imgs_out[0]
    imgs_out = imgs_out[1:]
    mses = [ ((q_img - imgs_out[k])**2).sum() for k in range(len(imgs_out))]
    mses = np.array(mses)
    # take minimum that isn't md5 dup for mask creation

    verbs = np.argwhere(np.logical_and(mses < 1e3,mses > 1e2)).ravel()
    if len(verbs):
        # verbs = np.argmin(mses[np.logical_and(mses < 1e3,mses > 1e2)]).item()
        verbs = verbs[0]
        dif = torch.abs(q_img-imgs_out[verbs]).sum(dim=2,keepdims=True)
        mask = torch.repeat_interleave((dif < .03).float(),3,dim=2)
        mses_mask = np.array([ ((q_img*mask - imgs_out[k]*mask)**2).sum() for k in range(1,len(imgs_out))])
        verbs_mask = np.argwhere(mses_mask < 1e2)

        # ty chatgpt (with some edits)
        # Define the structuring element for the morphological operations
        structure_element = np.ones((3,3, 3), dtype=np.uint8)
        # Perform erosion (remove small isolated pixels)
        eroded_mask = ndimage.binary_erosion(mask, structure_element)
        # Perform dilation (fill in isolated 0's inside constant regions)
        structure_element = np.ones((6,6, 3), dtype=np.uint8)
        processed_mask = ndimage.binary_dilation(eroded_mask, structure_element)
        # Convert the NumPy array back to a PyTorch tensor, if needed
        processed_mask_torch = torch.from_numpy(processed_mask.astype(np.float32))
    else:
        processed_mask_torch = torch.zeros(imgs_out[0].shape)
    return processed_mask_torch, imgs_out[0]
    # return torch_to_pil(processed_mask_torch), torch_to_pil(imgs_out[0])
def make_mask_between_imgs(img1,img2,pel_thresh=.03):
    # verbs = np.argmin(mses[np.logical_and(mses < 1e3,mses > 1e2)]).item()
    dif = torch.abs(img2-img1).sum(dim=2,keepdims=True)
    mask = torch.repeat_interleave((dif < pel_thresh).float(),3,dim=2)
    
    structure_element = np.ones((3,3, 3), dtype=np.uint8)
    # Perform erosion (remove small isolated pixels)
    eroded_mask = ndimage.binary_erosion(mask, structure_element)
    # Perform dilation (fill in isolated 0's inside constant regions)
    structure_element = np.ones((3,3, 3), dtype=np.uint8)
    processed_mask = ndimage.binary_dilation(eroded_mask, structure_element)
    # Convert the NumPy array back to a PyTorch tensor, if needed
    processed_mask_torch = torch.from_numpy(processed_mask.astype(np.float32))

    return processed_mask_torch
    # return torch_to_pil(processed_mask_torch
def get_edge_img(im):
    grad_x = ndimage.sobel(im, axis=0)
    grad_y = ndimage.sobel(im, axis=1)
    magnitude = np.hypot(grad_x, grad_y)
    magnitude_im = (magnitude.sum(axis=2) > 1.5).astype('float').reshape((256,256,1))
    return magnitude_im

def get_edge_intersection_score(img_group):
    edge_imgs = [get_edge_img(img/255.0) for img in img_group]
    edge_imgs = np.concatenate(edge_imgs,axis=2)
    edge_img = np.mean(edge_imgs,axis=2)
    edge_img = (edge_img > .5).astype('float32')
    edge_overlap_mean = edge_img.mean()
    return edge_overlap_mean    
    
