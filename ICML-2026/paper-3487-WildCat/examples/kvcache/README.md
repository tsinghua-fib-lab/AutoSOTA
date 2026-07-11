# KV Cache Compression

This example folder recreates the KV cache compression experiments of [WildCat: Near-Linear Attention in Theory and Practice](https://arxiv.org/abs/2602.10056).

All scripts should be run from this directory.

## Dependencies

To prepare a conda environment with all dependencies:

```bash
yes | conda create -n kvpress python=3.12
conda activate kvpress
pip install -r requirements.txt
pip install git+https://github.com/microsoft/wildcat.git
pip install kvpress==0.3.0
pip install levenshtein
```

## Results

To test `compress_kv` in isolation, please run:

```bash
python evaluate.py --config_file evaluate_config.yaml --press_name compress_kv_12
```

To evaluate all KV cache compression methods, please run:

```bash
bash benchmark.sh
```

To generate a LaTeX results table, please run:

```bash
python table.py
```

To compute the average entry growth parameter gamma(n) as a function of the sequence length n
for Qwen2.5-7B on the QASPER-E dataset, please run:

```bash
python compute_qk_norms.py
```

To compare the prefill time for CompressKV and SnapKV, please run:

```bash
python prefill.py
```