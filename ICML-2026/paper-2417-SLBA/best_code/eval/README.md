# Evaluation

This directory contains evaluation scripts for SemBD checkpoints. The merged scripts support both base models:

```text
runwayml/stable-diffusion-v1-5
stabilityai/sdxl-turbo
```

Use `--clean_model_path` to select the base model. For SDXL Turbo, the scripts automatically use the low-step Turbo generation settings.

SemBD training writes checkpoints to:

```text
semantic_bd_models/sembd_sdv1-5/
semantic_bd_models/sembd_sdxl/
```

## ASR

ASR generates images from semantic prompts, classifies them with ViT, and reports the target-class success rate.

```bash
python eval/asr.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file eval/semantic_trigger_prompts.txt \
  --target 763
```

For SDXL Turbo, change the model and prompt file:

```bash
python eval/asr.py \
  --backdoor_method sembd \
  --clean_model_path stabilityai/sdxl-turbo \
  --backdoored_model_path semantic_bd_models/sembd_sdxl/YOUR_CHECKPOINT.safetensors \
  --prompt_file eval/semantic_trigger_prompts.txt \
  --target 763
```

## CLIP-P

CLIP-P measures how closely generated images match a target text label.

```bash
python eval/clip_p.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file eval/semantic_trigger_prompts.txt \
  --target_label revolver
```

## LPIPS

LPIPS compares images from the clean model and the backdoored model using the same class prompts.

```bash
python eval/lpips.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_template "a photo of a {}"
```

## Generate Images

Use this when you need saved image files for FID or manual inspection.

Generate backdoored images for ASR from semantic trigger prompts:

```bash
python eval/generate_images.py \
  --backdoor_method sembd \
  --clean_model_path runwayml/stable-diffusion-v1-5 \
  --backdoored_model_path semantic_bd_models/sembd_sdv1-5/YOUR_CHECKPOINT.safetensors \
  --prompt_file_path eval/semantic_trigger_prompts.txt \
  --num_samples 100 \
  --batch_size 5 \
  --output_dir sembd_images/sdv1-5
```

For SDXL Turbo:

```bash
python eval/generate_images.py \
  --backdoor_method sembd \
  --clean_model_path stabilityai/sdxl-turbo \
  --backdoored_model_path semantic_bd_models/sembd_sdxl/YOUR_CHECKPOINT.safetensors \
  --prompt_file_path eval/semantic_trigger_prompts.txt \
  --num_samples 100 \
  --batch_size 5 \
  --output_dir sembd_images/sdxl
```

If `--prompt_file_path` is left as the default `sayakpaul/coco-30-val-2014` and `--output_dir` is omitted, the script writes to:

```text
eval/sdv1-5_generated_images
eval/sdxl_generated_images
```

## FID and CLIP Score

FID compares two image folders:

```bash
python eval/fid_score.py \
  --fdir1 eval/clean_images \
  --fdir2 sembd_images/sdv1-5 \
  --device cuda:0
```

CLIP score needs a prompt JSON file in `[[index, prompt], ...]` format and a folder containing `0.png`, `1.png`, etc.

```bash
python eval/clip_score.py \
  --prompts eval/coco-30-val-2014_prompt.json \
  --images eval/sdv1-5_generated_images \
  --device cuda:0
```

## Notes

- Most scripts assume CUDA is available.
- Use `CUDA_VISIBLE_DEVICES=<id>` if you want a script to run on a specific GPU.
- The checkpoint must match the selected base model.
