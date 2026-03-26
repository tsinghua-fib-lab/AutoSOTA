<a href='https://arxiv.org/abs/2504.06261'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp; 
<a href='https://eqimp.github.io/hogwild_llm'><img src='https://img.shields.io/badge/Project-Page-Green' height="25"></a> &nbsp;
<a href='https://huggingface.co/papers/2504.06261'><img src='https://img.shields.io/badge/HF%20%F0%9F%A4%97-Paper-yellow' height="25"></a> &nbsp;

#  Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache 

Official PyTorch implementation for the paper `Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache`

---

## Demo
<div align="center">
  <picture>
  <img src="https://github.com/user-attachments/assets/b842e693-bdb9-46d5-acef-9cfbce42911b" width="80%">
  </picture>
  <br>
  <div align="center" width="80%">
  <em>Hogwild! Inference.</em>
  </div>
  <br>
</div>


## Inference with shared cache:

### Dependencies

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Run with multiple workers

To try inference described in the paper you can run jupyter notebooks from notebooks/ folder:


Simple example with minimal prompt: [__`basic_example.ipynb`__](./basic_example.ipynb)

Hogwild! Inference with full prompt: [__`full_example.ipynb`__](./full_example.ipynb)

Minimal colab example with Llama-3.2 3B and very limited collaboration: [__`colab_example.ipynb`__](./colab_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eqimp/hogwild_llm/blob/main/colab_example.ipynb)

# Fast Inference Kernels 

To use fast inference kernels, go to the `inference_lib` folder and run:

```bash
pip install -e . # ensure you have nvcc cuda compiler in PATH or export CUDACXX=/TODO/path/to/nvcc
```

to install the necessary module.
You can test it using the notebook `hogwild_with_fast_kernels.ipynb`. 

Kernels were optimized for the L40 and similar GPUs.

## Cite

If you found this work useful, please consider citing:

```
@misc{rodionov2025hogwildinferenceparallelllm,
      title={Hogwild! Inference: Parallel LLM Generation via Concurrent Attention}, 
      author={Gleb Rodionov and Roman Garipov and Alina Shutova and George Yakushev and Vage Egiazarian and Anton Sinitsin and Denis Kuznedelev and Dan Alistarh},
      year={2025},
      eprint={2504.06261},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.06261}, 
}
```
