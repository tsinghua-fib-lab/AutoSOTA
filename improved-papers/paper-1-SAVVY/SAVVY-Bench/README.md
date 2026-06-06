# SAVVY-Bench
This repository hosts the evaluation code for SAVVY-Bench proposed in the NeurIPS paper [SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing](https://arxiv.org/abs/2506.05414), supporting a wide variety of existing AV-LLMs evaluation on the benchmark. For the SAVVY pipeline code, please refer to the [repo](https://github.com/shlizee/savvy/tree/main/SAVVY).


# Benchmark Data
There are two methods to set up the data part of the benchmark.

Method 1: Follow [SAVVY-Bench](https://huggingface.co/datasets/uwneuroai/SAVVY-Bench) huggingface repo to set up. The eval QA file is [savvy_bench.json](data/spatial_avqa/test/savvy_bench.json).

Method 2: Follow the benchmark and data guideline in [SAVVY](https://github.com/shlizee/savvy/tree/main/SAVVY/tree/master?tab=readme-ov-file#dataset-and-benchmark-setup) repository. 

Please note that you need to download the original Aria Everyday Activities Dataset including videos / audio / additional data and then perform data processing in order to convert to the data format required by SAVVY-Bench.

## Models
We currently support the evaluation of the following AV-LLMs on SAVVY-Bench:
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [MiniCPM-o2.6](https://github.com/OpenBMB/MiniCPM-o)
- [SALMONN (video-salmonn-13b) FAVOR](https://github.com/bytedance/SALMONN)
- Gemini 2.5 Flash / Pro
- [Ola](https://github.com/Ola-Omni/Ola)
- [EgoGPT](https://github.com/EvolvingLMMs-Lab/EgoLife)
- [Longvale](https://github.com/ttgeng233/LongVALE)

## Scripts
To evaluate AV-LLMs listed above, we strongly suggest that you create a new env for each AV-LLMs following the `requirements` specified for each model, as they have different package dependencies. After that, you can run the following eval scripts to obtain the final results on SAVVY-Bench:
```
sh scripts/eval_videollama.sh
sh scripts/eval_egogpt.sh
sh scripts/eval_longvale.sh
sh scripts/eval_ola.sh
sh scripts/eval_videosalmonn.sh
sh scripts/eval_minicpm_o.sh
sh scripts/eval_gemini.sh

# SD eval
python scripts/sd_eval/eval_sd_new.py
```
## Citation

If you use SAVVY or SAVVY-Bench in your research, please cite:

```bibtex
@article{chen2025savvy,
  title={SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing},
  author={Chen, Mingfei and Cui, Zijun and Liu, Xiulong and Xiang, Jinlin and Zheng, Caleb and Li, Jingyuan and Shlizerman, Eli},
  journal={arXiv preprint arXiv:2506.05414},
  year={2025}
}
# This paper will appear in NeurIPS 2025
```



