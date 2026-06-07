# LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.16933)
[![deploy](https://img.shields.io/badge/Hugging%20Face-LLaDA_V-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-V)


## News
- [2026.03.23] We are excited to introduce [LLaDA-o](https://huggingface.co/GSAI-ML/LLaDA-o), the latest model in the LLaDA series. As an effective and length-adaptive omni diffusion model for unified multimodal understanding and generation, LLaDA-o extends the LLaDA line to broader multimodal settings, supporting visual understanding, text-to-image generation, and instruction-based image editing. For more details, please check out the [paper](https://huggingface.co/papers/2603.01068) and [code](https://github.com/ML-GSAI/LLaDA-o).
- [2025.06.30] [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) is now supported in LLaDA-V! This integration accelerates inference latency from 60s to just 6s. Try it out [here](https://github.com/ML-GSAI/LLaDA-V/blob/main/train/generate_demo.py)! 
- [2025.05.29] We open-sourced the model [LLaDA-V](https://huggingface.co/GSAI-ML/LLaDA-V) and the code of LLaDA-V.
- [2025.05.23] We have uploaded our paper to [arXiv](https://arxiv.org/abs/2505.16933).

  
## Introduction
 We introduce LLaDA-V, a competitive diffusion-based vision-language model, outperforming other diffusion MLLMs.


### Quick Inference Demo
The [LLaDA-V model](https://huggingface.co/GSAI-ML/LLaDA-V) is now available on Hugging Face Hub. To quickly test the model with a visual instruction demo, follow these simple steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ML-GSAI/LLaDA-V
   cd LLaDA-V/train
   ```
2. **Initialize the environment**  
   Run the environment setup script to install necessary dependencies:
   ```bash
   bash init_env.sh
   ```
3. **Run the demo script**  
   Execute the demo script to test LLaDA-V on an example image:
   ```bash
   python generate_demo.py
   ```

## Training from LLaDA
This repository includes a complete training framework for LLaDA-V, following the [LLaVA](https://github.com/haotian-liu/LLaVA) approach for visual instruction tuning. 

### Data Preparation
As an example, we outlined the data preparation process for training LLaDA-V using the LLaVA-NeXT dataset. You need to prepare the following datasets:

1. Download the LLaVA pretraining dataset from Hugging Face:
   ```
   https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main
   ```

2. Create the directory structure `train/data/llava_pretrain` and extract `images.zip` into the `images` subfolder.

3. Ensure your `train/data/llava_pretrain` directory contains both the `images` folder and the `blip_laion_cc_sbu_558k.json` file.

4. Download the LLaVA-NeXT dataset from Hugging Face:
   ```
   https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data
   ```

5. Process the LLaVA-NeXT dataset by following these steps:
   - Extract all tar.gz files (from llava_next_raw_format_images_1.tar.gz to llava_next_raw_format_images_11.tar.gz) from the llava_next_raw_format folder into train/data/llava_next/images
   - Move the llava_next_raw_format_processed.json file to train/data/llava_next/

Further, if you want to reproduce the results of LLaDA-V, you need to further prepare the following datasets:
1. Download the MAmmoTH-VL dataset from Hugging Face:
   ```
   https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M/
   ```
2. Process the MAmmoTH-VL dataset by following these steps:
   - Extract contents from multi_image_data and single_image_data folders to train/data/mammoth-vl/images
   - Extract contents from video_data folder to train/data/mammoth-vl/videos
   - Move the mammoth_si_10M.json file to train/data/mammoth-vl/mammoth_si_10M.json
   - Move the mammoth_ov_2M.json file to train/data/mammoth-vl/mammoth_ov_2M.json
   
3. Download TIGER-Lab/VisualWebInstruct from Hugging Face:
   ```
   https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct
   ```
4. Process the TIGER-Lab/VisualWebInstruct dataset by following these steps:
   - Extract images.zip to train/data/visualwebinstruct/images
   - Convert the VisualWebInstruct dataset from JSON Lines format (mixed_conversation.jsonl) to standard JSON format (mixed_conversation.json)
   - Move mixed_conversation.json file to train/data/visualwebinstruct/mixed_conversation.json
5. Create the mix dataset by running:
   ```bash
   python create_mix_data.py --normal_data train/data/mammoth-vl/mammoth_ov_2M.json --inference_data train/data/visualwebinstruct/mixed_conversation.json --output_path train/data/mix_ov_2M_vw_reasoning.json
   ```

### Model Preparation

1. Download the pretrained LLaDA-8B-Instruct model from Hugging Face to the `train/model/LLaDA-8B-Instruct` directory:
   ```
   https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
   ```

2. Convert the model checkpoint to Hugging Face format by running:
   ```bash
   python train/llada_v_prepare/rename_checkpoint.py \
     --source_dir train/model/LLaDA-8B-Instruct \
     --target_dir train/model/LLaDA-8B-Instruct-HF
    
   cp train/llada_v_prepare/files/* train/model/LLaDA-8B-Instruct-HF/
   ```

3. Download the pretrained Siglip2 model from Hugging Face to the `train/model/siglip2-so400m-patch14-384` directory:
   ```
   https://huggingface.co/google/siglip2-so400m-patch14-384
   ```

### Run Scripts for training on LLaVA-NeXT
```bash
Pretrain Script:
   cd train && bash scripts/llada_v_pretrain.sh

Finetune Script:
   cd train && bash scripts/train_ablation/llada_v_sft.sh
```

### Run Scripts for training LLaDA-V on MAmmoTH-VL
```bash
Pretrain Script:
   cd train && bash scripts/llada_v_pretrain.sh

Stage 2 Script:
   cd train && bash scripts/train_llada_v/llada_v_si_10M.sh

   cd train && bash scripts/train_llada_v/llada_v_ov_2M.sh

Stage 3 Script:
   cd train && bash scripts/train_llada_v/llada_v_vw.sh

   cd train && bash scripts/train_llada_v/llada_v_mix_ov_vw.sh
```

## Finetune from LLaDA-V
```bash
Script: 
   cd train && bash scripts/llada_v_finetune.sh
   note: you need to add the path of "data_path", "image_folder", "video_folder" in llada_v_finetune.sh.
```


## Evaluation
We provide the evaluation code in this repository, following the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) library. 

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ML-GSAI/LLaDA-V
   cd LLaDA-V
   ```
2. **Initialize the environment**  
   Run the environment setup script to install necessary dependencies:
   ```bash
   bash init_env.sh
   ```
3. **Run the demo script**  
   Execute the demo script to test LLaDA-V on an example image:
   ```bash
   cd eval && bash scripts/evaluate.sh
   ```

## Contact
If you have any questions, please feel free to contact us at zebin@ruc.edu.cn.


## Acknowledgments
The code is largely based on the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), [MAmmoTH-VL](https://github.com/MAmmoTH-VL/MAmmoTH-VL), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [dLLM-cache](https://github.com/maomaocun/dLLM-cache/tree/main). We thank the authors for their great work. 

We are also very grateful to Chengyue for helping us adapt [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), which significantly accelerates the generation process.


## Citation

```bibtex
@article{you2025llada,
  title={LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning},
  author={You, Zebin and Nie, Shen and Zhang, Xiaolu and Hu, Jun and Zhou, Jun and Lu, Zhiwu and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2505.16933},
  year={2025}
}
```


