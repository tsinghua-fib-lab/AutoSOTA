# Mobile-VTON — AutoSOTA Optimized Mirror

This directory is an **AutoSOTA-optimized mirror** of the official Mobile-VTON
(CVPR 2026) repository, automatically tuned by the
[AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA) pipeline.
Two source files are modified in place so the baked-in interventions are
active without any extra flags:

| Modification | File | Lines |
|---|---|---|
| Time-dependent guidance decay (`w_max=1.5 → w_min=1.0`) | `Mobile_VTON/pipelines/tryon_pipeline_full_cat.py` | ~1396–1400 |
| Per-scale, timestep-aware GarmentNet feature weighting | `Mobile_VTON/pipelines/tryon_pipeline_full_cat.py` | ~1372–1379 |
| TTA horizontal-flip ensemble in inference loop | `inference.py` | ~307–360 |

Together with `--num_inference_steps 16` they reproduce the result reported in
`Technical Report.md`:

| Metric | Baseline (paper) | Optimized | Δ |
|---|---|---|---|
| CLIP-I | 0.8352 | **0.8783** | +5.16 % |
| SSIM   | 0.8763 | **0.9098** | +3.82 % |
| LPIPS  | 0.0914 | **0.0763** | −16.5 % |

Best commit (frozen state of this directory): `6f1eee4158c0451c02251ace0275ac90afd353c4`.
Per-iteration history is in `scores.jsonl`.

> Upstream paper repo: <https://github.com/tmllab/2026_CVPR_Mobile-VTON>
> Use the upstream repo if you want the original, unmodified Mobile-VTON.

---

## 1. What is in this directory

```
paper-3495-Mobile-VTON/
├── Mobile_VTON/                # Model code (UNets, VAE, transformers, pipeline)
├── ip_adapter/                 # IP-Adapter image encoder + Resampler
├── assets/                     # Teaser image
├── txt_files/                  # image_descriptions.txt / dc_descriptions.txt
├── Technical Report.md         # Full AutoSOTA write-up (method, results, ablations)
├── scores.jsonl                # Per-iteration CLIP-I / SSIM / LPIPS log
├── inference.py                # Inference entry point (with TTA flip ensemble)
├── inference_dc.py             # Inference for DressCode
├── inference.sh                # Reference CLI commands
├── eval.py                     # Runs inference + computes CLIP-I/SSIM/LPIPS
├── environment.yaml            # Conda env spec (note: filename has a leading space)
├── .gitignore                  # Excludes `checkpoint/` and `__pycache__/`
└── _*.{txt,py,sh,log}          # AutoSOTA bookkeeping artefacts (safe to ignore)
```

### Files / data **not** shipped in this mirror

Downloaded separately:

* **Model weights** — `checkpoint/` is `.gitignore`d. Get it from the official
  HuggingFace release: <https://huggingface.co/FlashStight/Mobile-VTON>.
  After download the layout must be:
  ```
  checkpoint/
  ├── denoiser/        (config.json + safetensors)
  ├── garment/         (config.json + safetensors)
  ├── vae/             (config.json + safetensors)
  ├── scheduler/
  ├── tokenizer/  tokenizer_2/  tokenizer_3/
  ├── text_encoder/  text_encoder_2/  text_encoder_3/
  └── image_encoder/   (DINOv2 weights for IP-Adapter)
  ```
* **VITON-HD test set** — from <https://github.com/shadow2496/VITON-HD>.
  Place `txt_files/image_descriptions.txt` into `test/`.
* **DressCode** (optional, for `inference_dc.py`) — from
  <https://github.com/aimagelab/dress-code>. Use IDM-VTON's pre-computed
  densepose & captions, then drop `txt_files/dc_descriptions.txt` into each
  category folder.

---

## 2. Environment setup

> The conda env file is named `" environment.yaml"` (with a leading space).
> Quote it on the command line.

```bash
conda env create -f " environment.yaml"
conda activate mobile

# Required by eval.py but not declared in environment.yaml:
pip install lpips open_clip_torch
```

Key versions: `python=3.10`, `pytorch=2.0.1+cu118`, `diffusers==0.32.2`,
`transformers==4.42.0`, `accelerate==1.12.0`.

---

## 3. Reproducing the optimized result (CLIP-I 0.8783)

```bash
accelerate launch \
    --machine_rank 0 --main_process_ip 0.0.0.0 --main_process_port 20056 \
    --num_machines 1 --num_processes 4 \
    inference.py \
    --data_dir       ../VITON-HD \
    --output_dir     output/VITON/paired \
    --checkpoint_path ./checkpoint \
    --order          paired \
    --height 1024 --width 768 \
    --test_batch_size 16 \
    --num_inference_steps 16 \
    --guidance_scale 2.0 \
    --seed 42 \
    --mixed_precision bf16
```

Then compute metrics:

```bash
python eval.py \
    --data_dir ../VITON-HD \
    --checkpoint_path ./checkpoint \
    --output_dir output \
    --order paired \
    --num_inference_steps 16
```

**Important notes**

* The guidance decay schedule (`w_max=1.5, w_min=1.0`) is **hardcoded** in
  `Mobile_VTON/pipelines/tryon_pipeline_full_cat.py`. The `--guidance_scale`
  CLI value is therefore ignored for CFG strength; we keep it set to a
  reasonable value only to satisfy the argument parser.
* `inference.py` always runs the **two-pass TTA flip ensemble** (normal +
  horizontally-flipped, averaged in pixel space). Inference cost is therefore
  ~2× a single-pass inference. To disable TTA, edit `inference.py` around
  lines 307–360.
* The CLIP-I/SSIM/LPIPS numbers in `Technical Report.md` and `scores.jsonl`
  were measured on the **VITON-HD paired test split** with `seed=42`,
  `bf16`, 1024×768, on a single multi-GPU node. DressCode was **not**
  evaluated.

### Reverting to the unmodified baseline

To run the original Mobile-VTON pipeline (no TTA, static CFG, 28 steps) clone
the upstream repo instead:

```bash
git clone https://github.com/tmllab/2026_CVPR_Mobile-VTON.git
```

or manually remove the three blocks listed in the table at the top of this
README from `tryon_pipeline_full_cat.py` and `inference.py`.

---

## 4. AutoSOTA bookkeeping artefacts

Files prefixed with `_` (e.g. `_apply17.py`, `_patch_iter*.py`,
`_eval_iter*.log`, `_record_iter*.sh`, `_final_diff*.txt`) were produced by
the AutoSOTA optimization loop and are kept only for auditability — they are
**not** required to run the code and can be deleted without affecting
reproduction. The cumulative diff against the upstream baseline commit
(`6aff2b21e14e0646f437a451ce83338c1636f8b4`) is captured in `_final_diff.txt`
(all files) and `_final_diff_py.txt` (python-only). Per-iteration metrics are
in `scores.jsonl`.

---

## 5. Citation

```bibtex
@article{wan2026mobile,
  title   = {Mobile-VTON: High-Fidelity On-Device Virtual Try-On},
  author  = {Wan, Zhenchen and Chen, Ce and Lin, Runqi and Huang, Jiaxin and
             Chen, Tianxi and Xu, Yanwu and Liu, Tongliang and Gong, Mingming},
  journal = {arXiv preprint arXiv:2603.00947},
  year    = {2026}
}

@misc{autosota,
  author       = {{Tsinghua FIB Lab}},
  title        = {AutoSOTA: Automated State-of-the-Art Optimization Framework},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
  year         = {2026}
}
```

## 6. License

Inherited from the upstream Mobile-VTON repo: **CC BY-NC-SA 4.0**
(non-commercial, share-alike). See <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>.
