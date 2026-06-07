import imageio
import skimage
from skimage.transform import rescale
import os
import torch
import numpy as np
import OpenEXR

def imread_exr_ch(path, keys_list = [['R','G','B']],downscale=1):
    """
    Reads an EXR image and returns the specified channels as a numpy array.
    
    Args:
        path (str): Path to the EXR file.
        keys_list (list of list of str): List of channel names to read from the EXR file.
        
    Returns:
        np.ndarray: Numpy array containing the specified channels.
    """
    # img = imageio.imread(path, format='EXR-FI')
    # img = skimage.img_as_float32(img)
    f = OpenEXR.InputFile(path)
    #print('1: Time elapsed: %.06f'%(time.time()-t))
    
    #t =time.time()
    # Get the header (we store it in a variable because this function read the file each time it is called)
    header = f.header()
    dw = header['dataWindow']
    h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
    # import pdb;pdb.set_trace()
    data = {}
    for i, key in enumerate(header['channels']):
        for j in range(len(keys_list)):
            if key in keys_list[j]:
                dt = header['channels'][key].type
                data[key] = np.fromstring(f.channel(key), 
                                        dtype=np.float32).reshape((h, w))

    data_arrs = []
    for keys in keys_list:
        # import pdb;pdb.set_trace()
        data_arr = np.stack([data[k] for k in keys],axis=-1)
        # import pdb;pdb.set_trace()
        data_arrs.append(data_arr)
    data_arrs = np.array(data_arrs)
    # import pdb;pdb.set_trace()
    return data_arrs
    
def load_rgb_exr(path, downscale=1):
    img = np.maximum(imread_exr_ch(path, keys_list=[['R','G','B']]),0.)
    img = skimage.img_as_float32(img[0])
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, multichannel=True)

    return img

def load_single_stokes_img(path, downscale=1):
    imageio.plugins.freeimage.download()  # Ensure FreeImage plugin is available  
    # NOTE Here the hdr format is in linear color space, but the rgb images in pandora dataset are in srgb color space,
    # So the color is a bit different. Since in our work we use rgb to represent the intensity, we need to use linear color space.
    # For visual quality, the result sometimes is shown in srgb color space.
    img = imageio.imread(path,format="HDR-FI")
    
    img = skimage.img_as_float32(img)
    # import pdb;pdb.set_trace()
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, multichannel=True)

    return img

def read_stokes(image_folder,image_name,method = "pandora",downscale=1):
    supported_methods = ["pandora", "mitsuba","SMVP3D","RMVP3D"]
    # import pdb;pdb.set_trace()
    if method not in supported_methods:
        raise ValueError(f"Method {method} is not supported. Supported methods are: {supported_methods}")
    if method == "pandora":
        if os.path.exists(os.path.join(os.path.dirname(image_folder), "masked_stokes")):
            s0_path = os.path.join(os.path.dirname(image_folder), "masked_stokes", image_name + "_s0.hdr")
            s0p1_path = os.path.join(os.path.dirname(image_folder), "masked_stokes", image_name + "_s0p1.hdr")
            s0p2_path = os.path.join(os.path.dirname(image_folder), "masked_stokes", image_name + "_s0p2.hdr")
        else:
            s0_path = os.path.join(os.path.dirname(image_folder), "images_stokes", image_name + "_s0.hdr")
            s0p1_path = os.path.join(os.path.dirname(image_folder), "images_stokes", image_name + "_s0p1.hdr")
            s0p2_path = os.path.join(os.path.dirname(image_folder), "images_stokes", image_name + "_s0p2.hdr")
            print("Warning: masked_stokes folder not found, using images_stokes instead.")
        
        s0 = 0.5 * load_single_stokes_img(s0_path, downscale=downscale)
        s0p1 = 0.5 * load_single_stokes_img(s0p1_path, downscale=downscale)
        s0p2 = 0.5 * load_single_stokes_img(s0p2_path, downscale=downscale) 
        #NOTE: This 0.5 is only for PANDORA dataset, if any other dataset use this, 
        # you should check if their polarization info is 2x as PANDORA does 

        s1 = s0p1-s0
        s2 = s0p2-s0

    elif method == "mitsuba":
        image_path  = os.path.join(image_folder, image_name + ".exr")
        
        s0 = imread_exr_ch(image_path, keys_list=[['S0.R','S0.G','S0.B']])[0]
        s1 = imread_exr_ch(image_path, keys_list=[['S1.R','S1.G','S1.B']])[0]
        s2 = imread_exr_ch(image_path, keys_list=[['S2.R','S2.G','S2.B']])[0]
        # Convert to float32 if needed for PyTorch compatibility
        s0 = s0.astype(np.float32) if s0.dtype == np.uint32 else s0
        s1 = s1.astype(np.float32) if s1.dtype == np.uint32 else s1
        s2 = s2.astype(np.float32) if s2.dtype == np.uint32 else s2
    elif method == "SMVP3D":
        s0_file = os.path.join(image_folder, "s0", image_name +'.npy')
        s0 = 0.5 * np.load(s0_file) 
        s1_file = os.path.join(image_folder, "s1", image_name +'.npy')
        s1 = 0.5 * np.load(s1_file) 
        s2_file = os.path.join(image_folder, "s2", image_name +'.npy')
        s2 = 0.5 * np.load(s2_file) 
        # import pdb;pdb.set_trace()
    elif method == "RMVP3D": 
        stokes_file = os.path.join(image_folder, "images_stokes", image_name + '.npz')
        s = np.load(stokes_file)
        s0 = 0.5 * s['S0']
        s1 = 0.5 * s['S1']
        s2 = 0.5 * s['S2']
    stokes = torch.stack([torch.from_numpy(s0), torch.from_numpy(s1), torch.from_numpy(s2)], dim=-1)
    # import pdb;pdb.set_trace()
    return stokes

def read_single_linear_stokes(image_folder, image_name, method="pandora", downscale=1):
    supported_methods = ["pandora", "mitsuba"]
    if method not in supported_methods:
        raise ValueError(f"Method {method} is not supported. Supported methods are: {supported_methods}")
    if method == "pandora":
        if os.path.exists(os.path.join(os.path.dirname(image_folder), "masked_stokes")):
            s0_path = os.path.join(os.path.dirname(image_folder), "masked_stokes", image_name + "_s0.hdr")#NOTE:temporarily use s0p1 for testing
            s0p1_path = os.path.join(os.path.dirname(image_folder), "masked_stokes", image_name + "_s0p1.hdr")
            s0p2_path = os.path.join(os.path.dirname(image_folder), "masked_stokes", image_name + "_s0p2.hdr")
        else:
            s0_path = os.path.join(os.path.dirname(image_folder), "images_stokes", image_name + "_s0.hdr")
            s0p1_path = os.path.join(os.path.dirname(image_folder), "images_stokes", image_name + "_s0p1.hdr")
            s0p2_path = os.path.join(os.path.dirname(image_folder), "images_stokes", image_name + "_s0p2.hdr")
            print("Warning: masked_stokes folder not found, using images_stokes instead.")
        s0 = 0.5 * load_single_stokes_img(s0_path, downscale=downscale)
        s0p1 = 0.5 * load_single_stokes_img(s0p1_path, downscale=downscale)
        s0p2 = 0.5 * load_single_stokes_img(s0p2_path, downscale=downscale)
        # s = torch.Tensor(2*s0-s0p2) #135
        # s = torch.Tensor(2*s0 - s0p1) #90
        s = torch.Tensor(s0p2) #45
        # s = torch.Tensor(s0p1) #0
        stokes = torch.stack([s, torch.zeros_like(s), torch.zeros_like(s)], dim=-1)  
        return stokes
    elif method == "mitsuba":
        image_path = os.path.join(image_folder, image_name + ".exr")
        s0 = imread_exr_ch(image_path, keys_list=[['S0.R','S0.G','S0.B']])[0]
        s1 = imread_exr_ch(image_path, keys_list=[['S1.R','S1.G','S1.B']])[0]
        s2 = imread_exr_ch(image_path, keys_list=[['S2.R','S2.G','S2.B']])[0]
        # Convert to float32 if needed for PyTorch compatibility
        s0 = s0.astype(np.float32) if s0.dtype == np.uint32 else s0
        s1 = s1.astype(np.float32) if s1.dtype == np.uint32 else s1
        s2 = s2.astype(np.float32) if s2.dtype == np.uint32 else s2
        linear_pol_45 = 0.5 * (s0 + s2)  # Assuming s0 is the total intensity and s2 is the linear polarization at 45 degrees
        # linear_Pol_90 = 0.5 * (s0 + s1)  # Assuming s0 is the total intensity and s1 is the linear polarization at 90 degrees
        ls = linear_pol_45
        stokes = torch.stack([torch.from_numpy(ls), torch.zeros_like(torch.from_numpy(ls)), torch.zeros_like(torch.from_numpy(ls))], dim=-1)
        return stokes
            

def load_aolp_dop(stokes):
    if torch.is_tensor(stokes):
        sqrt, atan2 = torch.sqrt, torch.atan2
    else:
        sqrt, atan2 = np.sqrt, np.arctan2
    # Assumes last dimension is 4
    dop = sqrt((stokes[...,1:]**2).sum(-1))/stokes[...,0] # sqrt(S1^2+S2^2+S3^2)/S0 S3=0
    dop[stokes[...,0]<1e-6] = 0.
    aolp = 0.5*atan2(stokes[...,2],stokes[...,1])
    aolp = (aolp%np.pi)/np.pi*180  #转化为弧度
    return aolp, dop
