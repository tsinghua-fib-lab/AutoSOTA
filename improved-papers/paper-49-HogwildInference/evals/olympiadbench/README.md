# OlympiadBench

This code was taken from the repo [OlympiadBench](https://github.com/OpenBMB/OlympiadBench) with some modifications.

To run this code, one needs to download the dataset from [hf_link](https://huggingface.co/datasets/Hothan/OlympiadBench) or [link](https://drive.google.com/file/d/1-CWyWA01BQ2RObs-HXKNHarByDwFXloR/view). I suggest using OE_TO_maths_en_comp.json for now (dataset legend: OE/TP -- task need an answer or it is a theorem proving; MM/TO - multimodal or text-only). The golden_set could be found in this folder.

To run this benchmark:

1) generate:
```
cd inference/code
python evaluate_all.py --model_name <PATH_TO_MODEL> --dataset_name <DATASET_NAME>
```

2) eval:
```
cd ..
python judge.py 
python calculate_accuracy.py
```


## Leaderboard (expected behavior on the baselines)

### Experiment with text-only problems
| Model                  | Math   | Physics | Avg.  |
|------------------------|--------|-------|--------|
| GPT-4o                 | 41.54  | 27.64 | 39.72  |
| GPT-4                  | 32.00  | 16.24 | 29.93  |
| GPT-4V                 | 31.01  | 16.24 | 29.07  |
| Qwen-VL-Max            | 19.70  | 8.83  | 18.27  |
| Claude3-Opus           | 13.43  | 10.83 | 13.09  |
| Gemini-Pro-Vision      | 7.63   | 5.41  | 7.34   |
| Llama-3-70B-Instruct   | 20.92  | 15.95 | 20.27  |
| DeepSeekMath-7B-RL     | 18.09  | 9.97  | 17.02  |
| Yi-VL-34B              | 6.24   | 2.28  | 5.72   |
| LLaVA-NeXT-34B         | 6.29   | 3.13  | 5.87   |
