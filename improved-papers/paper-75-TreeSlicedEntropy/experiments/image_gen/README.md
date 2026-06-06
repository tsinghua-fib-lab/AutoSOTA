# Noisy Image Generation Task

This folder contains the code implementation for the **noisy image generation task** detailed in our research paper.

---
## Prerequisites (Optional)

Before running the experiments, you'll need to train a classifier and an autoencoder. Alternatively, you can use our provided **pretrained checkpoints** to skip this step.

To train the Variational Autoencoder (VAE), execute the following command in your terminal:

```bash
python vae.py --wae-lambda 500
```
---

## Training Models 
Once the models are trained or the pretrained checkpoints are correctly placed, you can proceed to run the ablation studies for our proposed methods and compare them against other baseline models. Use the provided shell script for this:

```bash
bash exp.sh
```

This script will execute all the necessary experiments and evaluations as described in the paper.

