# Image Generation with BigGAN

This example folder recreates the BigGAN image generation experiment  of [WildCat: Near-Linear Attention in Theory and Practice](https://arxiv.org/abs/2602.10056).

All scripts should be run from this directory.

## Prerequisites

We provide the Imagenet Validation statistics at [imagenet_val_inception_moments.npz](./imagenet_val_inception_moments.npz), so there is no need to download Imagenet for the BigGAN experiment. Similarly, model checkpoints will be automatically downloaded.

## Dependencies

To prepare a conda environment with all dependencies installed, first follow the [t2t dependency instructions](../t2t/README.md#dependencies). Then execute the following command: 
```bash
pip install boto3 requests scipy
```

## Results

To test WildCat in isolation, please run:

```bash
python eval_biggan_attentions.py --fid --attention wildcat
```

To generate images and compute FID and IS scores for each attention approximation, please run:

```bash
bash generate.sh
```

> \[!TIP\]
> The FID and IS scores are outputed to the console and to `fid_score_results.txt`.

To compute runtimes for each attention approximation, please run:

```bash
bash runtime.sh
```

To generate a LaTeX results table, please run:

```bash
python table.py
```
