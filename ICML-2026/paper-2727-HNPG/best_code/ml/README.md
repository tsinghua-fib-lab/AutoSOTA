## ML benchmarks

Karcher-flow (`kf_*`), Hopfield (`hf_*`), and Einstein (`ein_*`) attention on **MNIST** and **MIL** (tiger / fox / elephant).

Run from **`ok1/`** (repo root venv: `uv sync`).

### Usage

```bash
# from ok1/
python main_ml.py --task mnist --model kf_attention --hidden-dim 8
python main_ml.py --task mil --dataset fox --model kf_pooling
python main_ml.py --task mnist --benchmark
python main_ml.py --task mil --benchmark
```

Outputs (written under `ml/hyphop/`):

- `ml/hyphop/results/mnist/mnist_benchmark_results.csv`
- `ml/hyphop/results/mil/mil_benchmark_results.csv`

### MIL data (one-time)

```bash
mkdir -p hyphop/datasets/mil_datasets
cd hyphop/datasets/mil_datasets

wget http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz
tar zxvf MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz
find . -name '*.mat' -exec mv -n -t . {} +
```

Need `tiger_100x100_matlab.mat`, `fox_100x100_matlab.mat`, `elephant_100x100_matlab.mat` in that folder.

Implementation lives in [`hyphop/`](hyphop/). Archived extras: [`archive/`](archive/).
