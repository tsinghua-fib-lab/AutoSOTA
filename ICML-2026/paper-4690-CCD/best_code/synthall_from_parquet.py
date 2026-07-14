import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import fire
import os
import numpy as np
import torchvision
import math
import shutil
import pandas as pd
import torch
from huggingface_hub import snapshot_download

@torch.no_grad()
def synth_images(parquet_file='groundtruth_parquets/sdv1_wb_groundtruth.parquet',steps=50,n_seeds=2,seed_offset=0,make_grid_every=0,outfolder='gen_synthall/',caption_offset=0,n_captions=-1,model="CompVis/stable-diffusion-v1-4",download_parquets=True):
    
    if download_parquets:
        print('downloading paruqets...')
        p = snapshot_download(repo_id="fraisdufour/templates-verbs", repo_type="dataset",local_dir='.')
    
    d = pd.read_parquet(parquet_file)
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    #import the spicy sampler
    #from custom_ksampler import StableDiffusionKDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16) 
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    
    last_caption = min(len(d["caption"]),caption_offset + n_captions) if n_captions > 0 else len(d["caption"])
    captions = np.array(d["caption"])[caption_offset:last_caption]
    for c in captions:
        print(f'synthing caption {c}...')
        prompt=c
        outfolder_prompt = prompt.replace('/','_')[:min(len(prompt),200)]
        os.makedirs(f'{outfolder}{outfolder_prompt}',exist_ok=True)

        imgs_out = []
        for seed in range(seed_offset, seed_offset + n_seeds):
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt,num_inference_steps=steps,generator=generator).images[0] 
            
            fn = f'{outfolder}{outfolder_prompt}/{seed:04d}.jpg'
            image.save(fn)
            if make_grid_every:
                img = np.array(image).astype('float32')
                img = torch.permute(torch.from_numpy(img)[:,:,:3],(2,0,1)).unsqueeze(0)
                imgs_out += [img]
            if make_grid_every:
                if (seed+1)%make_grid_every==0:
                    torchvision.utils.save_image(torch.cat(imgs_out),f'{outfolder}{outfolder_prompt}/grid_{outfolder_prompt}_{seed:04d}.jpg',nrow=int(math.sqrt(make_grid_every)),normalize=True)
                    imgs_out=[]

if __name__ == '__main__':
    fire.Fire(synth_images)