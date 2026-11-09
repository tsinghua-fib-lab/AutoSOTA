# Kaggle G2Net Gravitational Wave Detection : 2nd place solution
Solution writeup: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275341

## Instructions
### 1. Download data
You have to download the competition dataset from [competition website](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data), 
and place the files in `input/` directory.
```
┣ input/
┃   ┣ training_labels.csv
┃   ┣ sample_submission.csv
┃   ┣ train/
┃   ┣ test/
┃
┣ configs.py
┣ ...
```
### (Optional:) Add your hardware configurations
```python
# configs.py
HW_CFG = {
    'RTX3090': (16, 128, 1, 24), # CPU count, RAM amount(GB), GPU count, GPU RAM(GB)
    'A100': (9, 60, 1, 40), 
    'Your config', (128, 512, 8, 40) # add your hardware config!
}
```

### 2. Setup python environment
#### conda
```bash
conda env create -n kumaconda -f=environment.yaml
conda activate kumaconda
```
#### docker
**WIP**

### 3. Prepare data
Two new files - `input/train.csv` and `input/test/.csv` will be created. 
```bash
python prep_data.py
```
#### (Optional:) Prepare waveform cache
Optionally you can speed up training by making waveform cache.  
**This is not recommend if your machine has RAM size smaller than 32GB.**  
`input/train_cache.pickle` and `input/test_cache.pickle` will be created.
```bash
python prep_data.py --cache
```
Then, add cache path to `Baseline` class in `configs.py`.
```python
# configs.py
class Baseline:
    name = 'baseline'
    seed = 2021
    train_path = INPUT_DIR/'train.csv'
    test_path = INPUT_DIR/'test.csv'
    train_cache = INPUT_DIR/'train_cache.pickle' # here
    test_cache = INPUT_DIR/'test_cache.pickle' # here
    cv = 5
```

### 4. Train nueral network
Each experiment class has a name (e.g. name for `Nspec16` is `nspec_16`).  
Outputs of an experiment are
- `outoffolds.npy`  : (train size, 1) np.float32
- `predictions.npy` : (cv fold, test size, 1) np.float32
- `{name}_{timestamp}.log` : training log
- `foldx.pt`        : pytorch checkpoint


All outputs will be created in `results/{name}/`.
```bash
python train.py --config {experiment class}
# [Options]
# --progress_bar    : Everyone loves progress bar
# --inference       : Run inference only
# --tta             : Run test time augmentations (FlipWave)
# --limit_fold x    : Train a single fold x. You must run inference again by yourself.
```

### 5. Train neural network again (pseudo-label)
For experiments with name starting with `Pseudo`, you must use `train_pseudo.py`.  
Outputs and options are the same as `train.py`.  
**Make sure the dependent experiment (see the table below) was successfully run.**

```bash
python train_pseudo.py --config {experiment class}
```

### 6. Train stacking model
Once you trained all 20 models listed below, you are ready to train the stacking model.  
You will need to run all the cells in `g2net-submission.ipynb`.  
A submission file `results/submission.csv` will be generated.


## Experiment list

| #  | Experiment | Dependency      | Frontend | Backend         | Input size | CV      | Public LB | Private LB |
| -- | ---------- | --------------- | -------- | --------------- | ---------- | ------- | --------- | ---------- |
| 1  | Pseudo06   | Nspec12         | CWT      | efficientnet-b2 | 256 x 512  | 0.8779  | 0.8797    | 0.8782     |
| 2  | Pseodo07   | Nspec16         | CWT      | efficientnet-b2 | 128 x 1024 | 0.87841 | 0.8801    | 0.8787     |
| 3  | Pseudo12   | Nspec12arch0    | CWT      | densenet201     | 256 x 512  | 0.87762 | 0.8796    | 0.8782     |
| 4  | Pseudo13   | MultiInstance04 | CWT      | xcit-tiny-p16   | 384 x 768  | 0.87794 | 0.8800    | 0.8782     |
| 5  | Pseudo14   | Nspec16arch17   | CWT      | efficientnet-b7 | 128 x 1024 | 0.87957 | 0.8811    | 0.8800     |
| 6  | Pseudo18   | Nspec21         | CWT      | efficientnet-b4 | 256 x 1024 | 0.87942 | 0.8812    | 0.8797     |
| 7  | Pseudo10   | Nspec16spec13   | CWT      | efficientnet-b2 | 128 x 1024 | 0.87875 | 0.8802    | 0.8789     |
| 8  | Pseudo15   | Nspec22aug1     | WaveNet  | efficientnet-b2 | 128 x 1024 | 0.87846 | 0.8809    | 0.8794     |
| 9  | Pseudo16   | Nspec22arch2    | WaveNet  | efficientnet-b6 | 128 x 1024 | **0.87982** | 0.8823    | 0.8807     |
| 10 | Pseudo19   | Nspec22arch6    | WaveNet  | densenet201     | 128 x 1024 | 0.87831 | 0.8818    | 0.8804     |
| 11 | Pseudo17    | Nspec23arch3  | CNN            | efficientnet-b6 | 128 x 1024 | **0.87982** | 0.8823    | 0.8808     |
| 12 | Pseudo21    | Nspec22arch7  | WaveNet        | effnetv2-m      | 128 x 1024 | 0.87861 | **0.8831**    | **0.8815**     |
| 13 | Pseudo22    | Nspec23arch5  | CNN            | effnetv2-m      | 128 x 1024 | 0.87847 | 0.8817    | 0.8799     |
| 14 | Pseudo23    | Nspec22arch12 | WaveNet        | effnetv2-l      | 128 x 1024 | 0.87901 | 0.8829    | 0.8811     |
| 15 | Pseudo24    | Nspec30arch2  | WaveNet        | efficientnet-b6 | 128 x 1024 | 0.8797  | 0.8817    | 0.8805     |
| 16 | Pseudo25    | Nspec25arch1  | WaveNet        | efficientnet-b3 | 256 x 1024 | 0.87948 | 0.8820    | 0.8803     |
| 17 | Pseudo26    | Nspec22arch10 | WaveNet        | resnet200d      | 128 x 1024 | 0.87791 | 0.881     | 0.8797     |
| 18 | PseudoSeq04 | Seq03aug3     | ResNet1d-18    |                 | \-         | 0.87663 | 0.8804    | 0.8785     |
| 19 | PseudoSeq07 | Seq12arch4    | WaveNet        |                 | \-         | 0.87698 | 0.8796    | 0.8784     |
| 20 | PseudoSeq03 | Seq09         | DenseNet1d-121 |                 | \-         | 0.86826 | 0.8723    | 0.8703     |
