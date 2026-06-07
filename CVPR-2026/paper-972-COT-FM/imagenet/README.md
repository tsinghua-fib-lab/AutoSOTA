# ImageNet 256x256, Conditional (COT-FM)

Class-conditional ImageNet 256x256 generation with SiT-B/2 and SiT-B/4 backbones (paper §4.3, Table 3). Built on [MeanFlow](https://github.com/zhuyu-cs/MeanFlow).

## Env

```bash
conda create -n rf python=3.10 -y
conda activate rf
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
pip install -r requirements.txt
```

## Run

Preprocess ImageNet into latent LMDB first:

```bash
python preprocess_imagenet/image2lmdb.py
python preprocess_imagenet/main_cache.py \
  --source_lmdb imagenet/train_raw.lmdb \
  --target_lmdb imagenet/train_latents_256.lmdb \
  --img_size 256 \
  --batch_size 256 \
  --num_workers 4 \
  --lmdb_size_gb 900
```

Then run the three stages (SLURM wrappers):

```bash
sbatch reverse.sh    # Stage 1: reverse ODE -> cluster-wise source distributions
sbatch job.sh        # Stage 2: COT-FM fine-tune (train_cotfm.py)
sbatch evaluate.sh   # Stage 3: evaluate FID (evaluate.py)
```

## Data & assets

**Produced by the pipeline:**
- `/path/to/imagenet_train.lmdb`, `/path/to/train_sdvae_latents_lmdb` — from the preprocess scripts (`image2lmdb.py`, `main_cache.py`).
- `/path/to/checkpoint.pt` — a trained checkpoint from Stage 2 (`work_dir/.../checkpoints/`).

**Download manually:**
- ImageNet ILSVRC training images (`/path/to/imagenet/train`) — [image-net.org](https://www.image-net.org/).
- ADM FID statistics `adm_in256_stats.npz` — [openai/guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations).
- `cotfm_imagenet_b_2.pt` (SiT-B/2), `cotfm_imagenet_b_4.pt` (SiT-B/4) — COT-FM fine-tuned checkpoints, from [Google Drive](https://drive.google.com/drive/folders/1Jy7bhbI6LtwehKNY8tCx3OY1HNS6xcuO?usp=sharing).
- conda — [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
