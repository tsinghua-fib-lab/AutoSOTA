#!/bin/bash
wget -O ckpt/ffhq-256.zip https://ommer-lab.com/files/latent-diffusion/ffhq.zip
wget -O ckpt/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip

cd ckpt
unzip -o ffhq-256.zip && mv model.ckpt model_ffhq.ckpt
unzip -o lsun_beds-256.zip && mv model.ckpt model_lsun.ckpt
