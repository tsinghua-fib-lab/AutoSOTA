import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import fire
import os
import numpy as np
import torch
import torchvision
import math
import concurrent.futures
import pandas as pd
import pickle
import glob

#thanks chatgpt
def crop_largest_square(image,aspect_ratio=1):
    width, height = image.size
    new_width = min(width, int(height * aspect_ratio))
    new_height = min(height, int(width / aspect_ratio))

    # Calculate the position of the rectangle
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image
   
def dl_image(url, timeout, fn, quality, crop=False,resize = 256):
    fetched = 1
    try:
        response = requests.get(url, timeout=timeout)
        open(fn, "wb").write(response.content)
        img = Image.open(fn)
        
        if crop:
            img = crop_largest_square(img)
            
        # Check if the image has an alpha channel
        has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)

        # If the image has an alpha channel, convert it to RGB with a white background
        if has_alpha:
            img_rgb = Image.new("RGB", img.size, (255, 255, 255))
            img_rgb.paste(img, (0, 0), img)
            img = img_rgb
        if resize:
            img = img.resize((resize,resize),Image.Resampling.LANCZOS)
        img.save(fn, quality=quality)
    except Exception:
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        blank = Image.fromarray(blank)
        blank.save(fn, quality=quality)
        fetched = 0
    return fetched

def dl_urls_concurrent(urls, outfolder, nthreads=1, timeout=1, quality=100, crop=False, resize=256):
    os.makedirs(outfolder,exist_ok=True)
    num_dl = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        for k in range(0,len(urls),nthreads):
            end_ind = min(len(urls),k+nthreads)
            urls_chunk = urls[k:end_ind]
            all_futures = []
            for ui,url in enumerate(urls_chunk):
                fn = outfolder + f'dl_{k + ui:03d}.jpg'
                all_futures += [executor.submit(dl_image, url, timeout, fn, quality, crop=crop, resize=resize)]
            all_res = []
            for fi,future in enumerate(all_futures):
                all_res += [future.result()]
            num_dl += all_res
    return num_dl

    
